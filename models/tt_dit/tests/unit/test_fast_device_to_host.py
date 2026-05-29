# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for fast_device_to_host with Wan 2.2 VAE output shapes.

Parametrized resolutions:
  720p: BCTHW = (1, 3, 81, 720, 1280)
  480p: BCTHW = (1, 3, 81, 480, 832)

Sharding: H (dim 3) on TP axis (mesh axis 0), W (dim 4) on SP axis (mesh axis 1).

Run with:
    pytest models/tt_dit/tests/unit/test_fast_device_to_host.py -k "bh_4x32" --timeout=300
"""

import time

import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager
from ...utils.tensor import fast_device_to_host, float_to_uint8, typed_tensor_2dshard
from ...utils.test import line_params, ring_params

# Wan 2.2 VAE output — BCTHW (H and W are parametrized per-test)
B, C, T = 1, 3, 81
TP_AXIS = 0  # mesh axis for height
SP_AXIS = 1  # mesh axis for width
H_DIM = 3  # BCTHW dimension for height
W_DIM = 4  # BCTHW dimension for width


def _make_reference_tensor(H: int, W: int) -> torch.Tensor:
    """Create a deterministic reference tensor with the Wan VAE output shape."""
    gen = torch.Generator().manual_seed(42)
    return torch.randn(B, C, T, H, W, generator=gen, dtype=torch.bfloat16)


def _shard_to_device(ref: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Shard a BCTHW tensor: H on TP axis (0), W on SP axis (1)."""
    return typed_tensor_2dshard(
        ref,
        mesh_device,
        shard_mapping={TP_AXIS: H_DIM, SP_AXIS: W_DIM},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _make_concat_dims() -> list[int | None]:
    dims: list[int | None] = [None, None]
    dims[TP_AXIS] = H_DIM
    dims[SP_AXIS] = W_DIM
    return dims


def _make_ccl_manager(mesh_device: ttnn.MeshDevice, num_links: int, topology: ttnn.Topology) -> CCLManager | None:
    return CCLManager(mesh_device, num_links=num_links, topology=topology)


# ---------------------------------------------------------------------------
# Test parametrization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mesh_device, num_links, device_params, topology",
    [[(4, 32), 2, ring_params, ttnn.Topology.Ring], [(4, 8), 2, line_params, ttnn.Topology.Linear]],
    ids=["bh_4x32", "bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "height, width",
    [(720, 1280), (480, 832)],
    ids=["720p", "480p"],
)
class TestFastDeviceToHost:
    def test_correctness(self, mesh_device, num_links, device_params, topology, height, width):
        """Round-trip: shard to device -> fast_device_to_host -> compare to reference."""
        ref = _make_reference_tensor(height, width)
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        result = fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        assert result is not None, f"Rank {rank} got None from fast_device_to_host"
        assert result.shape == ref.shape, f"Shape mismatch: {result.shape} vs {ref.shape}"
        torch.testing.assert_close(result, ref, rtol=0, atol=0)

    def test_performance(self, mesh_device, num_links, device_params, topology, height, width):
        """Measure average D2H time over 10 iterations."""
        n_iters = 10

        ref = _make_reference_tensor(height, width)
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        # Warmup
        fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)

        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()

        start = time.perf_counter()
        for _ in range(n_iters):
            video = fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)
            video = video.permute(0, 2, 3, 4, 1).float()
        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * height * width * 2
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- fast_device_to_host + permute + float performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {height}, {width}, {C}) float32 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_uint8_accuracy(self, mesh_device, num_links, device_params, topology, height, width):
        """Compare on-device uint8 conversion against host float32 path.

        Runs twice with different input tensors to verify program cache reuse
        (override_runtime_arguments must correctly update buffer addresses).
        """
        import torch

        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        for iteration, seed in enumerate([42, 123]):
            gen = torch.Generator().manual_seed(seed)
            ref = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
            tt_tensor = _shard_to_device(ref, mesh_device)

            # --- Host reference path: bf16 → float32 → ×255 → uint8 ---
            host_uint8 = ((ref.float() + 1.0) * 0.5 * 255.0).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 4, 1)

            # --- Device path: pre_transfer_fn handles bf16 → uint8 on device ---
            device_uint8 = fast_device_to_host(
                tt_tensor,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
                pre_transfer_fn=float_to_uint8,
                permute=(0, 2, 3, 4, 1),
            )

            diff = (device_uint8.int() - host_uint8.int()).abs()
            max_err = diff.max().item()
            mean_err = diff.float().mean().item()
            pct_exact = (diff == 0).float().mean().item() * 100

            label = "fresh" if iteration == 0 else "cached"
            print(f"\n--- uint8 accuracy, iter {iteration} ({label}) ---")
            print(f"  Max error:     {max_err}")
            print(f"  Mean error:    {mean_err:.4f}")
            print(f"  Exact match:   {pct_exact:.1f}%")
            assert max_err <= 1, f"Max uint8 error {max_err} > 1 on iter {iteration} ({label})"

    def test_uint8_performance(self, mesh_device, num_links, device_params, topology, height, width):
        """Measure on-device uint8 conversion + D2H + permute."""
        import torch

        n_iters = 10
        gen = torch.Generator().manual_seed(42)
        ref = torch.rand(B, C, T, height, width, generator=gen, dtype=torch.bfloat16)
        tt_tensor = _shard_to_device(ref, mesh_device)
        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        def _convert_and_transfer():
            return fast_device_to_host(
                tt_tensor,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
                pre_transfer_fn=float_to_uint8,
                permute=(0, 2, 3, 4, 1),
            )

        # Warmup
        _convert_and_transfer()

        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(n_iters):
            _convert_and_transfer()
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * height * width
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- on-device uint8 + D2H + permute performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {height}, {width}, {C}) uint8 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")
