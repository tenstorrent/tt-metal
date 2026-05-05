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

import math
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


# ---------------------------------------------------------------------------
# TestD2HLayoutPerformance
# ---------------------------------------------------------------------------

# Dimension-ordering configs: perm applied to (C=0, T=1, H=2, W=3) dims,
# followed by the resulting positions of H and W in the permuted tensor.
_DIM_ORDER_CONFIGS: dict[str, tuple[tuple[int, ...], int, int]] = {
    #        perm from (C,T,H,W)   h_pos  w_pos
    "CTHW": ((0, 1, 2, 3),         2,     3),
    "THWC": ((1, 2, 3, 0),         1,     2),
    "CHWT": ((0, 2, 3, 1),         1,     2),
}

# (label, C, H, W); T=81 frames is fixed
_D2H_SHAPES = [
    ("720p_3c", 3, 720, 1280),   # full RGB 720p
    ("480p_3c", 3, 480,  832),   # full RGB 480p
    ("720p_1c", 1, 720, 1280),   # Y channel 720p
    ("480p_1c", 1, 480,  832),   # Y channel 480p
    ("360p_1c", 1, 360,  640),   # UV pooled 720p (H//2, W//2)
    ("240p_1c", 1, 240,  416),   # UV pooled 480p
]

_D2H_T       = 81   # frames
_D2H_TP_SIZE = 4    # mesh axis-0 devices (height sharding)
_D2H_SP_SIZE = 8    # mesh axis-1 devices (width sharding)
_BF16_BYTES  = 2    # bytes per bfloat16 element


def _d2h_shard_shape(
    C: int, H: int, W: int,
    perm: tuple[int, ...],
    h_dim: int,
    w_dim: int,
) -> tuple[int, ...]:
    """Per-device logical shard shape after permuting (C,T,H,W) and splitting H/W."""
    base = [C, _D2H_T, H, W]
    shape = [base[i] for i in perm]
    shape[h_dim] //= _D2H_TP_SIZE
    shape[w_dim] //= _D2H_SP_SIZE
    return tuple(shape)


def _d2h_alloc_bytes(shard_shape: tuple[int, ...], layout: ttnn.Layout) -> int:
    """Estimated bytes allocated per device, including padding.

    TILE: last two dims rounded up to multiples of 32 (one 32×32 bfloat16 tile = 2 048 B).
    ROW_MAJOR: no tile padding; reports exact logical size.
    """
    s = list(shard_shape)
    if layout == ttnn.TILE_LAYOUT:
        s[-2] = math.ceil(s[-2] / 32) * 32
        s[-1] = math.ceil(s[-1] / 32) * 32
    return math.prod(s) * _BF16_BYTES


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), line_params]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dim_order_name", list(_DIM_ORDER_CONFIGS.keys()))
@pytest.mark.parametrize(
    "layout",
    [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    ids=["row_major", "tile"],
)
@pytest.mark.parametrize(
    "shape_name, channels, height, width",
    _D2H_SHAPES,
    ids=[s[0] for s in _D2H_SHAPES],
)
class TestD2HLayoutPerformance:
    """Sweep dim-ordering × layout × shape to measure raw device-to-host bandwidth.

    Each parametrized case:
      - Shards a (C, T, H, W) bfloat16 tensor onto a 4×8 mesh
        (H split over axis-0, W split over axis-1).
      - Warms up, then times 10 iterations of:
            tt_tensor.cpu(blocking=False)
            ttnn.synchronize_device(mesh_device)
      - Prints per-device allocated size (with tile padding where applicable)
        and average transfer time.

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py \\
            -k "TestD2HLayoutPerformance" --timeout=600 -s
    """

    def test_transfer_speed(
        self,
        mesh_device: ttnn.MeshDevice,
        device_params,
        dim_order_name: str,
        layout: ttnn.Layout,
        shape_name: str,
        channels: int,
        height: int,
        width: int,
    ):
        perm, h_dim, w_dim = _DIM_ORDER_CONFIGS[dim_order_name]

        # Build host tensor in the chosen dim order
        gen = torch.Generator().manual_seed(42)
        host = torch.randn(channels, _D2H_T, height, width, generator=gen, dtype=torch.bfloat16)
        host = host.permute(perm).contiguous()

        # Shard onto device
        tt_tensor = typed_tensor_2dshard(
            host,
            mesh_device,
            shard_mapping={TP_AXIS: h_dim, SP_AXIS: w_dim},
            layout=layout,
            dtype=ttnn.bfloat16,
        )

        # ---- size reporting ----
        shard = _d2h_shard_shape(channels, height, width, perm, h_dim, w_dim)
        logical_per_dev  = math.prod(shard) * _BF16_BYTES
        alloc_per_dev    = _d2h_alloc_bytes(shard, layout)
        total_alloc      = alloc_per_dev * _D2H_TP_SIZE * _D2H_SP_SIZE
        logical_total    = channels * _D2H_T * height * width * _BF16_BYTES
        padding_overhead = (alloc_per_dev / logical_per_dev - 1.0) * 100.0

        # ---- warmup ----
        tt_tensor.cpu(blocking=False)
        ttnn.synchronize_device(mesh_device)

        # ---- timed loop ----
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            tt_tensor.cpu(blocking=False)
            ttnn.synchronize_device(mesh_device)
        elapsed_s = time.perf_counter() - start

        avg_ms = elapsed_s / n_iters * 1_000
        throughput_gbs = (logical_total / (elapsed_s / n_iters)) / 1e9

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank == 0:
            layout_str = "tile" if layout == ttnn.TILE_LAYOUT else "row_major"
            pad_str = f" (+{padding_overhead:.0f}% tile padding)" if layout == ttnn.TILE_LAYOUT else ""
            print(
                f"\n--- D2H layout sweep: {dim_order_name} | {layout_str} | {shape_name} ---\n"
                f"  Full tensor:       (C={channels}, T={_D2H_T}, H={height}, W={width})\n"
                f"  Per-device shard:  {shard}\n"
                f"  Per-device alloc:  {alloc_per_dev / 1e6:.2f} MB{pad_str}\n"
                f"  Total on-device:   {total_alloc / 1e6:.2f} MB "
                f"({_D2H_TP_SIZE * _D2H_SP_SIZE} devices)\n"
                f"  Logical total:     {logical_total / 1e6:.2f} MB\n"
                f"  Avg D2H time:      {avg_ms:.1f} ms\n"
                f"  Throughput:        {throughput_gbs:.2f} GB/s (logical bytes)"
            )
