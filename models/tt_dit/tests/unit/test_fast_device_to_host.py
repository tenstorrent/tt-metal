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

import numpy as np
import pytest
import torch

import ttnn

from ...parallel.manager import CCLManager
from ...utils.tensor import _reassemble_2d, fast_device_to_host, float_to_uint8, typed_tensor_2dshard
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
_DTYPE_BYTES  = 1    # bytes per uint8 element (on-device after RGB/YUV conversion)


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

    TILE: last two dims rounded up to multiples of 32 (one 32×32 uint8 tile = 1 024 B).
    ROW_MAJOR: no tile padding; reports exact logical size.
    """
    s = list(shard_shape)
    if layout == ttnn.TILE_LAYOUT:
        s[-2] = math.ceil(s[-2] / 32) * 32
        s[-1] = math.ceil(s[-1] / 32) * 32
    return math.prod(s) * _DTYPE_BYTES


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [[(4, 8), line_params]],
    ids=["bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
class TestD2HLayoutPerformance:
    """Sweep dim-ordering × layout × shape internally and report a single perf table.

    For each (shape, layout, dim_order):
      - Shards a (C, T, H, W) uint8 tensor onto a 4×8 mesh
        (H split over axis-0, W split over axis-1).
        uint8 reflects the on-device dtype after RGB/YUV color conversion.
      - Warms up, then times 10 iterations of:
            out = tt_tensor.cpu(blocking=False)
            ttnn.synchronize_device(mesh_device)
      - Records per-device allocated size (with tile padding where applicable),
        average transfer time, and throughput.

    All results are emitted in a single summary table at the end.

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py \\
            -k "TestD2HLayoutPerformance" --timeout=600 -s
    """

    def test_transfer_speed(self, mesh_device: ttnn.MeshDevice, device_params):
        n_iters = 10
        layouts = (("row_major", ttnn.ROW_MAJOR_LAYOUT), ("tile", ttnn.TILE_LAYOUT))
        results = []

        for shape_name, channels, height, width in _D2H_SHAPES:
            for layout_name, layout in layouts:
                for dim_order_name, (perm, h_dim, w_dim) in _DIM_ORDER_CONFIGS.items():
                    gen = torch.Generator().manual_seed(42)
                    host = torch.randint(
                        0, 256, (channels, _D2H_T, height, width), generator=gen, dtype=torch.uint8
                    )
                    host = host.permute(perm).contiguous()

                    tt_tensor = typed_tensor_2dshard(
                        host,
                        mesh_device,
                        shard_mapping={TP_AXIS: h_dim, SP_AXIS: w_dim},
                        layout=layout,
                        dtype=ttnn.uint8,
                    )

                    shard = _d2h_shard_shape(channels, height, width, perm, h_dim, w_dim)
                    logical_per_dev = math.prod(shard) * _DTYPE_BYTES
                    alloc_per_dev = _d2h_alloc_bytes(shard, layout)
                    logical_total = channels * _D2H_T * height * width * _DTYPE_BYTES
                    padding_overhead = (alloc_per_dev / logical_per_dev - 1.0) * 100.0

                    # Warmup. `out` must be held until after synchronize: the dispatch
                    # descriptor stores a raw pointer into the host buffer, and dropping
                    # the returned tensor before the async read completes is a UAF.
                    out = tt_tensor.cpu(blocking=False)
                    ttnn.synchronize_device(mesh_device)

                    start = time.perf_counter()
                    for _ in range(n_iters):
                        out = tt_tensor.cpu(blocking=False)
                        ttnn.synchronize_device(mesh_device)
                    elapsed_s = time.perf_counter() - start
                    del out

                    avg_ms = elapsed_s / n_iters * 1_000
                    throughput_gbs = (logical_total / (elapsed_s / n_iters)) / 1e9

                    results.append(
                        {
                            "shape": shape_name,
                            "layout": layout_name,
                            "dim_order": dim_order_name,
                            "shard": str(shard),
                            "alloc_mb": alloc_per_dev / 1e6,
                            "pad_pct": padding_overhead,
                            "total_mb": logical_total / 1e6,
                            "avg_ms": avg_ms,
                            "gbs": throughput_gbs,
                        }
                    )

                    del tt_tensor

        rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
        if rank != 0:
            return

        shard_w = max(len(r["shard"]) for r in results)
        cols = (
            ("shape", 8, "<", "s"),
            ("layout", 10, "<", "s"),
            ("dim_order", 10, "<", "s"),
            ("shard", shard_w, "<", "s"),
            ("alloc/dev (MB)", 15, ">", ".2f"),
            ("pad %", 7, ">", ".1f"),
            ("total (MB)", 11, ">", ".2f"),
            ("avg (ms)", 9, ">", ".1f"),
            ("GB/s", 7, ">", ".2f"),
        )
        keys = ("shape", "layout", "dim_order", "shard", "alloc_mb", "pad_pct", "total_mb", "avg_ms", "gbs")

        header = " ".join(f"{name:{align}{width}}" for name, width, align, _ in cols)
        print(f"\n--- D2H layout sweep: mesh {tuple(mesh_device.shape)} ---")
        print(header)
        print("-" * len(header))
        for r in results:
            cells = []
            for (_, width, align, fmt), key in zip(cols, keys):
                cells.append(f"{r[key]:{align}{width}{fmt}}")
            print(" ".join(cells))


# ---------------------------------------------------------------------------
# Host-only rearrange benchmark
# ---------------------------------------------------------------------------


def test_host_rearrange_speed():
    """Compare host-side RGB interleaved concat across several backends.

    720p, C=3, uint8. Output: (T, H, W, 3) — ffmpeg AV_PIX_FMT_RGB24 layout.

    Variants × dim_orders (CTHW, CHWT):
      - naive:          per-shard numpy strided scatter into out.
      - reassemble:     production `_reassemble_2d` (single-thread torch
                        permute().contiguous() + assign).
      - threaded:       numpy scatter parallelized across a ThreadPoolExecutor.
      - torch_threaded: same as `threaded` but using torch.Tensor.copy_() on a
                        torch.from_numpy view (different SIMD/iter backend).
      - numba (if installed): @njit(parallel=True) kernels with explicit C=3
                              unroll, separate CTHW/CHWT loop nesting.

    Mesh: 4 (TP/H) x 8 (SP/W). Per-shard (3, T, 180, 160) or (3, 180, 160, T).

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_host_rearrange_speed -s
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    C, T, H, W = 3, 81, 720, 1280
    TP, SP = 4, 8
    h_per, w_per = H // TP, W // SP

    n_iters = 10
    n_workers = min(8, os.cpu_count() or 8)
    n_shards = TP * SP
    mesh_coords = [(r, c) for r in range(TP) for c in range(SP)]

    def src_view_torch(shard, dim_order):
        """Zero-copy (T, h_per, w_per, C) torch view of the shard."""
        # CTHW: (C, T, h, w) -> permute(1,2,3,0) -> (T, h, w, C)
        # CHWT: (C, h, w, T) -> permute(3,1,2,0) -> (T, h, w, C)
        if dim_order == "CTHW":
            return shard.permute(1, 2, 3, 0)
        return shard.permute(3, 1, 2, 0)

    def src_view_numpy(shard, dim_order):
        return src_view_torch(shard, dim_order).numpy()

    # --- Variant 1: naive numpy per-shard strided scatter -------------------

    def rgb_concat_naive(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        for (r, c), shard in zip(mesh_coords, shards):
            out[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :] = src_view_numpy(
                shard, dim_order
            )
        return out

    # --- Variant 2: production `_reassemble_2d` -----------------------------

    def rgb_concat_reassemble(shards, dim_order):
        if dim_order == "CTHW":
            permute = (1, 2, 3, 0)
            concat_dims = [2, 3]
        else:  # CHWT
            permute = (3, 1, 2, 0)
            concat_dims = [1, 2]
        shard_shape = list(shards[0].shape)
        out = _reassemble_2d(
            mesh_coords, shards, shard_shape, (TP, SP), concat_dims, permute=permute
        )
        return out.numpy()  # zero-copy: out is contiguous

    # --- Variant 3: threaded numpy scatter ----------------------------------

    pool = ThreadPoolExecutor(max_workers=n_workers)

    def rgb_concat_threaded(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        srcs = [src_view_numpy(s, dim_order) for s in shards]

        def write_one(out, src, r, c):
            out[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :] = src

        futures = [
            pool.submit(write_one, out, src, r, c)
            for (r, c), src in zip(mesh_coords, srcs)
        ]
        for f in futures:
            f.result()
        return out

    # --- Variant 4: torch.copy_() + threaded --------------------------------

    def rgb_concat_torch_threaded(shards, dim_order):
        out = np.empty((T, H, W, C), dtype=np.uint8)
        out_t = torch.from_numpy(out)
        srcs = [src_view_torch(s, dim_order) for s in shards]

        def write_one(out_t, src, r, c):
            out_t[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per, :].copy_(src)

        futures = [
            pool.submit(write_one, out_t, src, r, c)
            for (r, c), src in zip(mesh_coords, srcs)
        ]
        for f in futures:
            f.result()
        return out

    # --- Variant 5: numba parallel JIT --------------------------------------

    try:
        import numba

        # CTHW source: (C, T, h, w) contiguous. Inner w loop has stride 1 in src
        # for each fixed C, and the C unroll generates 3 sequential streams the
        # hardware prefetcher handles well. Output is (T, H, W, C) contiguous.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_rgb_cthw(src, out, r_off, c_off, h_per, w_per, T_):
            for t in numba.prange(T_):
                for h in range(h_per):
                    h_g = r_off + h
                    for w in range(w_per):
                        w_g = c_off + w
                        out[t, h_g, w_g, 0] = src[0, t, h, w]
                        out[t, h_g, w_g, 1] = src[1, t, h, w]
                        out[t, h_g, w_g, 2] = src[2, t, h, w]

        # CHWT source: (C, h, w, T) contiguous. Inner t loop has stride 1 in src
        # for each fixed (C, h, w); dest writes hop by H*W*C across t. Three
        # source streams again, but with stride T=81 between (h, w) pixels.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_rgb_chwt(src, out, r_off, c_off, h_per, w_per, T_):
            for h in numba.prange(h_per):
                h_g = r_off + h
                for w in range(w_per):
                    w_g = c_off + w
                    for t in range(T_):
                        out[t, h_g, w_g, 0] = src[0, h, w, t]
                        out[t, h_g, w_g, 1] = src[1, h, w, t]
                        out[t, h_g, w_g, 2] = src[2, h, w, t]

        # Force JIT compilation up front (first call is slow).
        _dummy_src = np.zeros((C, 1, 1, 1), dtype=np.uint8)
        _dummy_out = np.zeros((1, 1, 1, C), dtype=np.uint8)
        _scatter_rgb_cthw(_dummy_src, _dummy_out, 0, 0, 1, 1, 1)
        _scatter_rgb_chwt(_dummy_src, _dummy_out, 0, 0, 1, 1, 1)

        def rgb_concat_numba(shards, dim_order):
            out = np.empty((T, H, W, C), dtype=np.uint8)
            kernel = _scatter_rgb_cthw if dim_order == "CTHW" else _scatter_rgb_chwt
            for (r, c), shard in zip(mesh_coords, shards):
                # shard contiguous: (C, T, h, w) for CTHW; (C, h, w, T) for CHWT
                kernel(shard.numpy(), out, r * h_per, c * w_per, h_per, w_per, T)
            return out

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        rgb_concat_numba = None  # type: ignore[assignment]

    variants = [
        ("naive",          rgb_concat_naive),
        ("reassemble",     rgb_concat_reassemble),
        ("threaded",       rgb_concat_threaded),
        ("torch_threaded", rgb_concat_torch_threaded),
    ]
    if HAS_NUMBA:
        variants.append(("numba", rgb_concat_numba))

    configs = [
        ("CTHW", (C, T, h_per, w_per)),
        ("CHWT", (C, h_per, w_per, T)),
    ]

    try:
        # --- Correctness check: every variant matches `naive` for both dim_orders ---
        for name, shape in configs:
            gen = torch.Generator().manual_seed(0)
            check_shards = [
                torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)
            ]
            ref = rgb_concat_naive(check_shards, name)
            for vname, fn in variants[1:]:
                got = fn(check_shards, name)
                assert np.array_equal(ref, got), f"variant '{vname}' disagrees with 'naive' for {name}"

        # --- Benchmark ---
        rows = []
        for name, shape in configs:
            shards = [torch.empty(shape, dtype=torch.uint8) for _ in range(n_shards)]
            for variant_name, fn in variants:
                # Warmup
                out = fn(shards, name)

                start = time.perf_counter()
                for _ in range(n_iters):
                    out = fn(shards, name)
                elapsed = time.perf_counter() - start

                avg_ms = elapsed / n_iters * 1_000
                gbs = (out.nbytes / (elapsed / n_iters)) / 1e9
                rows.append((variant_name, name, avg_ms, gbs, str(out.shape)))

        print()
        print(f"  threads = {n_workers}, numba = {HAS_NUMBA}")
        header = f"  {'variant':<16} {'dim_order':<10} {'avg (ms)':>10} {'GB/s':>8} {'output shape':>22}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for variant_name, name, avg_ms, gbs, shape in rows:
            print(f"  {variant_name:<16} {name:<10} {avg_ms:>10.2f} {gbs:>8.2f} {shape:>22}")
    finally:
        pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
# Host-only YUV 4:2:0 planar concat benchmark
# ---------------------------------------------------------------------------


def test_yuv_planar_concat_speed():
    """Compare host-side YUV 4:2:0 planar concat across several backends.

    Variants × two on-device dim_orders (CTHW, CHWT):
      - naive:          per-shard numpy strided scatter into planar view.
      - vectorized:     torch.stack + permute + contiguous, single big write per plane.
      - threaded:       numpy scatter parallelized across a ThreadPoolExecutor.
      - torch_threaded: same shape as `threaded` but using torch.Tensor.copy_()
                        on torch.as_strided views — different SIMD/iter backend.
      - numba (if installed): @njit(parallel=True) kernels, separate CTHW/CHWT
                              specializations with explicit loop nesting.

    Output shape: (T, H*W + 2*(H/2 * W/2)) uint8 — standard 4:2:0 planar
    [Y | U | V] layout with one row per frame.

    Mesh: 4 (TP/H) x 8 (SP/W). Y per-shard (180, 160); U/V per-shard (90, 80).

    Run with:
        pytest models/tt_dit/tests/unit/test_fast_device_to_host.py::test_yuv_planar_concat_speed -s
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    H, W, T = 720, 1280, 81
    TP, SP = 4, 8
    Hu, Wu = H // 2, W // 2

    h_per_y, w_per_y = H // TP, W // SP
    h_per_uv, w_per_uv = Hu // TP, Wu // SP

    hw = H * W
    uv = Hu * Wu
    row_stride = hw + 2 * uv  # bytes per frame (uint8)

    n_iters = 10
    n_workers = min(8, os.cpu_count() or 8)
    n_shards = TP * SP
    mesh_coords = [(r, c) for r in range(TP) for c in range(SP)]

    def to_thw_view(shard: torch.Tensor, dim_order: str) -> np.ndarray:
        """Zero-copy (T, h, w) numpy view of a per-device shard (C=1)."""
        v = shard.squeeze(0).numpy()  # CTHW -> (T, h, w); CHWT -> (h, w, T)
        if dim_order == "CHWT":
            v = v.transpose(2, 0, 1)  # strided view, no copy
        return v

    def make_views(out):
        y_view = np.lib.stride_tricks.as_strided(
            out, shape=(T, H, W), strides=(row_stride, W, 1), writeable=True
        )
        u_view = np.lib.stride_tricks.as_strided(
            out[:, hw:], shape=(T, Hu, Wu), strides=(row_stride, Wu, 1), writeable=True
        )
        v_view = np.lib.stride_tricks.as_strided(
            out[:, hw + uv :], shape=(T, Hu, Wu), strides=(row_stride, Wu, 1), writeable=True
        )
        return y_view, u_view, v_view

    # --- Variant 1: naive per-shard strided scatter -------------------------

    def planar_concat_naive(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        y_view, u_view, v_view = make_views(out)

        def scatter(shards, view, h_per, w_per):
            for (r, c), shard in zip(mesh_coords, shards):
                view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per] = to_thw_view(
                    shard, dim_order
                )

        scatter(y_shards, y_view, h_per_y, w_per_y)
        scatter(u_shards, u_view, h_per_uv, w_per_uv)
        scatter(v_shards, v_view, h_per_uv, w_per_uv)
        return out

    # --- Variant 2: torch.stack + permute + contiguous, one big write -------

    def planar_concat_vectorized(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)

        def assemble(shards, h_per, w_per):
            """Stack all shards into one (T, TP*h_per, SP*w_per) contiguous tensor."""
            stacked = torch.stack(shards).squeeze(1)  # (32, ...)
            if dim_order == "CTHW":
                # (32, T, h, w) -> (TP, SP, T, h, w) -> (T, TP, h, SP, w)
                t = stacked.reshape(TP, SP, T, h_per, w_per).permute(2, 0, 3, 1, 4)
            else:  # CHWT
                # (32, h, w, T) -> (TP, SP, h, w, T) -> (T, TP, h, SP, w)
                t = stacked.reshape(TP, SP, h_per, w_per, T).permute(4, 0, 2, 1, 3)
            return t.contiguous().reshape(T, TP * h_per, SP * w_per)

        y_full = assemble(y_shards, h_per_y, w_per_y)    # (T, H, W)
        u_full = assemble(u_shards, h_per_uv, w_per_uv)  # (T, Hu, Wu)
        v_full = assemble(v_shards, h_per_uv, w_per_uv)  # (T, Hu, Wu)

        # `*_full` is contiguous, so .reshape(T, plane) is a zero-copy view.
        # The assignment is one strided-dest write per plane (numpy collapses
        # the matching inner two strides into a per-row memcpy).
        out[:, :hw] = y_full.reshape(T, hw).numpy()
        out[:, hw : hw + uv] = u_full.reshape(T, uv).numpy()
        out[:, hw + uv :] = v_full.reshape(T, uv).numpy()
        return out

    # --- Variant 3: threaded scatter ----------------------------------------

    pool = ThreadPoolExecutor(max_workers=n_workers)

    def planar_concat_threaded(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        y_view, u_view, v_view = make_views(out)

        # Convert torch shards to numpy views on the main thread (zero-copy).
        # Doing this off-thread would risk torch/Python contention on the GIL.
        y_np = [to_thw_view(s, dim_order) for s in y_shards]
        u_np = [to_thw_view(s, dim_order) for s in u_shards]
        v_np = [to_thw_view(s, dim_order) for s in v_shards]

        def write_one(view, src, r, c, h_per, w_per):
            view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per] = src

        futures = []
        for (r, c), src in zip(mesh_coords, y_np):
            futures.append(pool.submit(write_one, y_view, src, r, c, h_per_y, w_per_y))
        for (r, c), src in zip(mesh_coords, u_np):
            futures.append(pool.submit(write_one, u_view, src, r, c, h_per_uv, w_per_uv))
        for (r, c), src in zip(mesh_coords, v_np):
            futures.append(pool.submit(write_one, v_view, src, r, c, h_per_uv, w_per_uv))
        for f in futures:
            f.result()
        return out

    # --- Variant 4: torch native copy + threaded ----------------------------

    def planar_concat_torch_threaded(y_shards, u_shards, v_shards, dim_order):
        out = np.empty((T, row_stride), dtype=np.uint8)
        out_t = torch.from_numpy(out)

        # Strided 3D views via torch.as_strided (no copy, shares storage with `out`).
        y_view = out_t.as_strided((T, H, W), (row_stride, W, 1), 0)
        u_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw)
        v_view = out_t.as_strided((T, Hu, Wu), (row_stride, Wu, 1), hw + uv)

        def src_view(shard):
            v = shard.squeeze(0)  # CTHW: (T, h, w); CHWT: (h, w, T)
            if dim_order == "CHWT":
                v = v.permute(2, 0, 1)
            return v

        # Pre-build all source views on the main thread to avoid contention.
        y_src = [src_view(s) for s in y_shards]
        u_src = [src_view(s) for s in u_shards]
        v_src = [src_view(s) for s in v_shards]

        def write_one(view, src, r, c, h_per, w_per):
            view[:, r * h_per : (r + 1) * h_per, c * w_per : (c + 1) * w_per].copy_(src)

        futures = []
        for (r, c), src in zip(mesh_coords, y_src):
            futures.append(pool.submit(write_one, y_view, src, r, c, h_per_y, w_per_y))
        for (r, c), src in zip(mesh_coords, u_src):
            futures.append(pool.submit(write_one, u_view, src, r, c, h_per_uv, w_per_uv))
        for (r, c), src in zip(mesh_coords, v_src):
            futures.append(pool.submit(write_one, v_view, src, r, c, h_per_uv, w_per_uv))
        for f in futures:
            f.result()
        return out

    # --- Variant 5: numba parallel JIT --------------------------------------

    try:
        import numba

        # CTHW source: (T, h_per, w_per) contiguous. Inner w loop is stride-1
        # in both source and dest -> LLVM emits memcpy-equivalent SIMD.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_cthw_numba(src, out_flat, plane_offset, row_stride_, W_, r_off, c_off, h_per, w_per, T_):
            for t in numba.prange(T_):
                row_base = t * row_stride_ + plane_offset + r_off * W_ + c_off
                for h in range(h_per):
                    base_dst = row_base + h * W_
                    for w in range(w_per):
                        out_flat[base_dst + w] = src[t, h, w]

        # CHWT source: (h_per, w_per, T) contiguous. Inner T loop is stride-1
        # in source and stride row_stride in dest -> sequential strided stores.
        # Compiled loop avoids numpy's per-element dispatch overhead.
        @numba.njit(parallel=True, cache=True, boundscheck=False)
        def _scatter_chwt_numba(src, out_flat, plane_offset, row_stride_, W_, r_off, c_off, h_per, w_per, T_):
            for h in numba.prange(h_per):
                row_off = plane_offset + (r_off + h) * W_ + c_off
                for w in range(w_per):
                    dst_off = row_off + w
                    for t in range(T_):
                        out_flat[t * row_stride_ + dst_off] = src[h, w, t]

        # Force JIT compilation up front (first call is slow).
        _dummy_src_cthw = np.zeros((1, 1, 1), dtype=np.uint8)
        _dummy_src_chwt = np.zeros((1, 1, 1), dtype=np.uint8)
        _dummy_out = np.zeros(row_stride, dtype=np.uint8)
        _scatter_cthw_numba(_dummy_src_cthw, _dummy_out, 0, row_stride, W, 0, 0, 1, 1, 1)
        _scatter_chwt_numba(_dummy_src_chwt, _dummy_out, 0, row_stride, W, 0, 0, 1, 1, 1)

        def planar_concat_numba(y_shards, u_shards, v_shards, dim_order):
            out = np.empty((T, row_stride), dtype=np.uint8)
            out_flat = out.reshape(-1)

            kernel = _scatter_cthw_numba if dim_order == "CTHW" else _scatter_chwt_numba

            def scatter(shards, plane_offset, plane_W, h_per, w_per):
                for (r, c), shard in zip(mesh_coords, shards):
                    src = shard.squeeze(0).numpy()  # CTHW: (T, h, w); CHWT: (h, w, T)
                    kernel(src, out_flat, plane_offset, row_stride, plane_W,
                           r * h_per, c * w_per, h_per, w_per, T)

            scatter(y_shards, 0,        W,  h_per_y,  w_per_y)
            scatter(u_shards, hw,       Wu, h_per_uv, w_per_uv)
            scatter(v_shards, hw + uv,  Wu, h_per_uv, w_per_uv)
            return out

        HAS_NUMBA = True
    except ImportError:
        HAS_NUMBA = False
        planar_concat_numba = None  # type: ignore[assignment]

    variants = [
        ("naive",          planar_concat_naive),
        ("vectorized",     planar_concat_vectorized),
        ("threaded",       planar_concat_threaded),
        ("torch_threaded", planar_concat_torch_threaded),
    ]
    if HAS_NUMBA:
        variants.append(("numba", planar_concat_numba))

    configs = [
        # name, Y shard shape, UV shard shape
        ("CTHW", (1, T, h_per_y, w_per_y), (1, T, h_per_uv, w_per_uv)),
        ("CHWT", (1, h_per_y, w_per_y, T), (1, h_per_uv, w_per_uv, T)),
    ]

    try:
        # --- Correctness check: every variant matches `naive` for both dim_orders ---
        for name, y_shape, uv_shape in configs:
            gen = torch.Generator().manual_seed(0)
            cy = [torch.randint(0, 256, y_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            cu = [torch.randint(0, 256, uv_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            cv = [torch.randint(0, 256, uv_shape, generator=gen, dtype=torch.uint8) for _ in range(n_shards)]
            ref = planar_concat_naive(cy, cu, cv, name)
            for vname, fn in variants[1:]:
                got = fn(cy, cu, cv, name)
                assert np.array_equal(ref, got), f"variant '{vname}' disagrees with 'naive' for {name}"

        # --- Benchmark ---
        rows = []
        for name, y_shape, uv_shape in configs:
            y_shards = [torch.empty(y_shape, dtype=torch.uint8) for _ in range(n_shards)]
            u_shards = [torch.empty(uv_shape, dtype=torch.uint8) for _ in range(n_shards)]
            v_shards = [torch.empty(uv_shape, dtype=torch.uint8) for _ in range(n_shards)]

            for variant_name, fn in variants:
                # Warmup
                out = fn(y_shards, u_shards, v_shards, name)

                start = time.perf_counter()
                for _ in range(n_iters):
                    out = fn(y_shards, u_shards, v_shards, name)
                elapsed = time.perf_counter() - start

                avg_ms = elapsed / n_iters * 1_000
                gbs = (out.nbytes / (elapsed / n_iters)) / 1e9
                rows.append((variant_name, name, avg_ms, gbs, str(out.shape)))

        print()
        print(f"  threads = {n_workers},  numba = {HAS_NUMBA}")
        header = f"  {'variant':<16} {'dim_order':<10} {'avg (ms)':>10} {'GB/s':>8} {'output shape':>20}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for variant_name, name, avg_ms, gbs, shape in rows:
            print(f"  {variant_name:<16} {name:<10} {avg_ms:>10.2f} {gbs:>8.2f} {shape:>20}")
    finally:
        pool.shutdown(wait=True)
