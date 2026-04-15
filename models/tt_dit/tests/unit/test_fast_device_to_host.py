# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Tests for fast_device_to_host with Wan 2.2 VAE output shape.

VAE output shape: BCTHW = (1, 3, 81, 720, 1280)
  - H (dim 3) fractured on TP axis (mesh axis 0, size 4)
  - W (dim 4) fractured on SP axis (mesh axis 1, size 32)
  - Per-device shard: (1, 3, 81, 180, 40)

Run with:
    pytest models/tt_dit/tests/unit/test_fast_device_to_host.py -k "bh_4x32" --timeout=300
"""

import time

import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager
from ...utils.tensor import fast_device_to_host, typed_tensor_2dshard
from ...utils.test import line_params, ring_params

# Wan 2.2 VAE output — BCTHW
B, C, T, H, W = 1, 3, 81, 720, 1280
TP_AXIS = 0  # mesh axis for height
SP_AXIS = 1  # mesh axis for width
H_DIM = 3  # BCTHW dimension for height
W_DIM = 4  # BCTHW dimension for width


def _make_reference_tensor() -> torch.Tensor:
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
class TestFastDeviceToHost:
    def test_correctness(self, mesh_device, num_links, device_params, topology):
        """Round-trip: shard to device -> fast_device_to_host -> compare to reference."""
        ref = _make_reference_tensor()
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        result = fast_device_to_host(tt_tensor, mesh_device, concat_dims, ccl_manager=ccl_manager)

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        # In single-host or root=None (default), all ranks get the result.
        assert result is not None, f"Rank {rank} got None from fast_device_to_host"
        assert result.shape == ref.shape, f"Shape mismatch: {result.shape} vs {ref.shape}"
        torch.testing.assert_close(result, ref, rtol=0, atol=0)

    def test_slow_perf(self, mesh_device, num_links, device_params, topology):
        """Measure average D2H time over 10 iterations."""
        n_iters = 10

        ref = _make_reference_tensor()
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        # Warmup
        ccl_manager.device_to_host(tt_tensor, concat_dims)

        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()

        start = time.perf_counter()
        for _ in range(n_iters):
            ccl_manager.device_to_host(tt_tensor, concat_dims)
        # Sync + barrier before measurement
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * H * W * 2  # bfloat16 = 2 bytes
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- ccl_manager.device_to_host performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Tensor shape:  ({B}, {C}, {T}, {H}, {W}) bfloat16")
            print(f"  Tensor size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_performance(self, mesh_device, num_links, device_params, topology):
        """Measure average D2H time over 10 iterations."""
        n_iters = 10

        ref = _make_reference_tensor()
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
        tensor_bytes = B * C * T * H * W * 4  # float32 = 4 bytes (full pipeline output)
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- fast_device_to_host + permute + float performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {H}, {W}, {C}) float32 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_fused_correctness(self, mesh_device, num_links, device_params, topology):
        """Round-trip with fused permute + dtype conversion."""
        import torch

        ref = _make_reference_tensor()
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        result = fast_device_to_host(
            tt_tensor,
            mesh_device,
            concat_dims,
            ccl_manager=ccl_manager,
            permute=(0, 2, 3, 4, 1),
            dtype=torch.float32,
        )

        expected = ref.permute(0, 2, 3, 4, 1).float()
        assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
        assert result.dtype == torch.float32
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_fused_performance(self, mesh_device, num_links, device_params, topology):
        """Measure fused permute+dtype D2H time over 10 iterations."""
        import torch

        n_iters = 10
        ref = _make_reference_tensor()
        tt_tensor = _shard_to_device(ref, mesh_device)
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        concat_dims = _make_concat_dims()

        # Warmup
        fast_device_to_host(
            tt_tensor,
            mesh_device,
            concat_dims,
            ccl_manager=ccl_manager,
            permute=(0, 2, 3, 4, 1),
            dtype=torch.float32,
        )

        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()

        start = time.perf_counter()
        for _ in range(n_iters):
            fast_device_to_host(
                tt_tensor,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
                permute=(0, 2, 3, 4, 1),
                dtype=torch.float32,
            )
        ttnn.synchronize_device(mesh_device)
        if ttnn.using_distributed_env():
            ttnn.distributed_context_barrier()
        end = time.perf_counter()

        avg_s = (end - start) / n_iters
        tensor_bytes = B * C * T * H * W * 4  # float32 = 4 bytes
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- fast_device_to_host FUSED performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {H}, {W}, {C}) float32 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")

    def test_uint8_accuracy(self, mesh_device, num_links, device_params, topology):
        """Compare on-device uint8 conversion against host float32 path."""
        import torch

        # VAE output is in [0, 1] after add(1)*0.5 + clamp
        gen = torch.Generator().manual_seed(42)
        ref = torch.rand(B, C, T, H, W, generator=gen, dtype=torch.bfloat16)
        tt_tensor = _shard_to_device(ref, mesh_device)

        # --- Host reference path: bf16 → float32 → ×255 → uint8 ---
        host_uint8 = (ref.float() * 255).round().clamp(0, 255).to(torch.uint8)

        # --- Device path: bf16 → ×255 → clamp → typecast(uint8) → D2H ---
        tt_tile = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
        tt_scaled = ttnn.multiply(tt_tile, 255.0)
        tt_clamped = ttnn.clamp(tt_scaled, min=0.0, max=255.0)
        tt_clamped = ttnn.to_layout(tt_clamped, ttnn.ROW_MAJOR_LAYOUT)
        tt_uint8 = ttnn.typecast(tt_clamped, ttnn.uint8)
        # tt_uint8 = ttnn.to_layout(tt_uint8, ttnn.ROW_MAJOR_LAYOUT)

        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)
        device_uint8 = fast_device_to_host(tt_uint8, mesh_device, concat_dims, ccl_manager=ccl_manager)

        diff = (device_uint8.int() - host_uint8.int()).abs()
        max_err = diff.max().item()
        mean_err = diff.float().mean().item()
        pct_exact = (diff == 0).float().mean().item() * 100

        print(f"\n--- uint8 accuracy (device vs host float32 path) ---")
        print(f"  Max error:     {max_err}")
        print(f"  Mean error:    {mean_err:.4f}")
        print(f"  Exact match:   {pct_exact:.1f}%")
        assert max_err <= 1, f"Max uint8 error {max_err} > 1"

    def test_uint8_performance(self, mesh_device, num_links, device_params, topology):
        """Measure on-device uint8 conversion + D2H + permute."""
        import torch

        n_iters = 10
        gen = torch.Generator().manual_seed(42)
        ref = torch.rand(B, C, T, H, W, generator=gen, dtype=torch.bfloat16)
        tt_tensor = _shard_to_device(ref, mesh_device)
        concat_dims = _make_concat_dims()
        ccl_manager = _make_ccl_manager(mesh_device, num_links, topology)

        def _convert_and_transfer():
            tt_tile = ttnn.to_layout(tt_tensor, ttnn.TILE_LAYOUT)
            tt_scaled = ttnn.multiply(tt_tile, 255.0)
            tt_clamped = ttnn.clamp(tt_scaled, min=0.0, max=255.0)
            tt_uint8 = ttnn.typecast(tt_clamped, ttnn.uint8)
            tt_uint8 = ttnn.to_layout(tt_uint8, ttnn.ROW_MAJOR_LAYOUT)
            return fast_device_to_host(
                tt_uint8,
                mesh_device,
                concat_dims,
                ccl_manager=ccl_manager,
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
        tensor_bytes = B * C * T * H * W  # uint8 = 1 byte
        throughput_gbs = (tensor_bytes / avg_s) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            print(f"\n--- on-device uint8 + D2H + permute performance (root=0) ---")
            print(f"  Mesh shape:    {tuple(mesh_device.shape)}")
            print(f"  Output shape:  (1, {T}, {H}, {W}, {C}) uint8 (BTHWC)")
            print(f"  Output size:   {tensor_bytes / 1e6:.1f} MB")
            print(f"  Iterations:    {n_iters}")
            print(f"  Average time:  {avg_s * 1000:.1f} ms")
            print(f"  Throughput:    {throughput_gbs:.2f} GB/s")
