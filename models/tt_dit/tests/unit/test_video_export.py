# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Video-encode benchmarks.

Three sweeps that don't need a device:
  * ``test_pix_fmt_in``      — rgb24 vs yuv420p input to ffmpeg.
  * ``test_gop_threads_sweep`` — libx264 -g x -threads grid for yuv420p input.
  * ``test_tune_zerolatency`` — -tune zerolatency on/off.

Two single-config baselines that don't need a device:
  * ``test_best_config``      — subprocess ffmpeg with the prior-sweep winner.
  * ``test_best_config_pyav`` — same config via in-process PyAV (libavcodec
                                 bindings — no subprocess, no pipe).

One holistic test that *does* need a 4x8 (or 4x32) mesh:
  * ``test_e2e_rgb_vs_yuv``   — full pipeline cost: device tensor -> D2H ->
                                 ffmpeg encode, comparing the RGB path
                                 (``float_to_uint8`` + permute + ``rgb24``
                                 ffmpeg) vs the YUV path
                                 (``fast_device_to_host_yuv`` + ``yuv420p``
                                 ffmpeg).

Run all encode-only tests with:
    pytest models/tt_dit/tests/unit/test_video_export.py \\
        -k "not e2e" -s

Run the e2e test with mesh hardware:
    pytest models/tt_dit/tests/unit/test_video_export.py::test_e2e_rgb_vs_yuv -s
"""

import os
import statistics
import time

import numpy as np
import pytest
import torch

import ttnn
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.tensor import (
    fast_device_to_host,
    fast_device_to_host_yuv,
    float_to_uint8,
    typed_tensor_2dshard,
)
from models.tt_dit.utils.test import line_params, ring_params
from models.tt_dit.utils.video import export_to_video, export_to_video_pyav, rgb_to_yuv420p

NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
FPS = 16
PRESET = "ultrafast"

GOPS = [1, 2, 4, 8, 16, 32]
THREADS = [1, 2, 4, 8, 10, 12, 14, 16, 18, 20]

# Best encode config established in prior sweeps.
BEST_GOP = 4
BEST_THREADS = 16

# Sharding setup (matches test_fast_device_to_host.py).
B_DEV, C_DEV = 1, 3
TP_AXIS = 0  # mesh axis for height
SP_AXIS = 1  # mesh axis for width
H_DIM = 3  # BCTHW dimension for height
W_DIM = 4  # BCTHW dimension for width

# Optional pre-recorded real-frame source for the YUV-input encode benchmarks.
# Compression-related metrics (output file size) are much more representative
# on natural video than on uniform random noise (which is ~incompressible).
# The file at this path, if present, is expected to be a numpy ``.npy`` of
# shape ``(T, H*W*3//2)`` uint8 — the on-device YUV420p planar layout produced
# by ``fast_device_to_host_yuv``.  Override with the ``WAN_FRAMES_T2V_NPY``
# environment variable.  Falls back to ``rgb_to_yuv420p(frames_rgb)`` if the
# file isn't found or its shape doesn't match.
_DEFAULT_YUV_NPY_PATH = "wan_frames_t2v.npy"


def _real_yuv_planar_path() -> str | None:
    path = os.environ.get("WAN_FRAMES_T2V_NPY", _DEFAULT_YUV_NPY_PATH)
    return path if os.path.isfile(path) else None


@pytest.fixture(scope="module")
def frames_rgb():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(NUM_FRAMES, HEIGHT, WIDTH, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def frames_yuv420(frames_rgb):
    """Real-frame YUV420p planar buffer if available, else synthetic.

    Real content makes the file-size column of the GOP × threads sweep
    meaningful — random noise is ~incompressible, so every cell would land at
    the same (high) bitrate regardless of GOP choice.
    """
    path = _real_yuv_planar_path()
    if path is not None:
        arr = np.load(path)
        expected_planar = HEIGHT * WIDTH + 2 * (HEIGHT // 2) * (WIDTH // 2)
        if arr.dtype != np.uint8 or arr.ndim != 2 or arr.shape[1] != expected_planar:
            print(
                f"\n[frames_yuv420] {path} has shape={arr.shape} dtype={arr.dtype}; "
                f"expected (T, {expected_planar}) uint8 — falling back to synthetic."
            )
        else:
            t = arr.shape[0]
            if t != NUM_FRAMES:
                print(
                    f"\n[frames_yuv420] {path} has {t} frames (expected {NUM_FRAMES}); "
                    "using as-is — encode times scale with frame count."
                )
            print(f"\n[frames_yuv420] using real frames from {path} " f"(shape={arr.shape}, dtype={arr.dtype})")
            return arr

    print("\n[frames_yuv420] no real-frame npy found, using synthetic noise (size column not meaningful)")
    return rgb_to_yuv420p(frames_rgb)


def _benchmark(frames, tmp_path, prefix, num_runs, encoder=export_to_video, **kwargs):
    durations = []
    out_path = None
    for i in range(num_runs):
        out_path = str(tmp_path / f"{prefix}_{i}.mp4")
        t0 = time.perf_counter()
        encoder(frames, out_path, fps=FPS, preset=PRESET, **kwargs)
        durations.append(time.perf_counter() - t0)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    return {
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "std": statistics.stdev(durations) if num_runs > 1 else 0.0,
        "min": min(durations),
        "max": max(durations),
        "size_mb": size_mb,
    }


def _stats(xs: list[float]) -> dict:
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "min": min(xs),
        "max": max(xs),
    }


# ---------------------------------------------------------------------------
# Sweep 1: rgb24 vs yuv420p input — does feeding YUV directly skip enough work
# (libswscale RGB->YUV conversion) to matter?
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pix_fmt_in", ["rgb24", "yuv420p"])
def test_pix_fmt_in(pix_fmt_in, frames_rgb, frames_yuv420, tmp_path):
    frames = frames_rgb if pix_fmt_in == "rgb24" else frames_yuv420
    stats = _benchmark(
        frames,
        tmp_path,
        f"pix_{pix_fmt_in}",
        num_runs=10,
        pix_fmt_in=pix_fmt_in,
        width=WIDTH,
        height=HEIGHT,
    )
    print(
        f"\n[pix_fmt_in={pix_fmt_in}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Sweep 2+3: GOP × threads.  Combined into a single test: libx264's
# frame-threading parallelizes across GOPs, so the two parameters interact.
# ---------------------------------------------------------------------------


def test_gop_threads_sweep(frames_yuv420, tmp_path):
    grid = {}  # (gop, threads) -> stats
    num_runs = 10

    for gop in GOPS:
        for threads in THREADS:
            grid[(gop, threads)] = _benchmark(
                frames_yuv420,
                tmp_path,
                f"g{gop}_t{threads}",
                num_runs=num_runs,
                pix_fmt_in="yuv420p",
                width=WIDTH,
                height=HEIGHT,
                gop=gop,
                threads=threads,
            )

    lines = [
        "",
        f"gop x threads sweep (preset={PRESET}, yuv420p input, {NUM_FRAMES} frames, "
        f"{WIDTH}x{HEIGHT}, {num_runs} runs/cell)",
        "",
        "mean encode time (seconds)",
    ]
    header = "  gop\\threads | " + " | ".join(f"{t:>7d}" for t in THREADS)
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['mean']:7.3f}" for t in THREADS)
        lines.append(row)

    lines.append("")
    lines.append("stdev encode time (seconds)")
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['std']:7.3f}" for t in THREADS)
        lines.append(row)

    lines.append("")
    lines.append("output file size (MB)")
    lines.append(header)
    lines.append("-" * len(header))
    for gop in GOPS:
        row = f"  {gop:>11d} | " + " | ".join(f"{grid[(gop, t)]['size_mb']:7.2f}" for t in THREADS)
        lines.append(row)

    best_gop, best_th = min(grid, key=lambda k: grid[k]["mean"])
    best = grid[(best_gop, best_th)]
    lines.append("")
    lines.append(f"Fastest: gop={best_gop} threads={best_th} mean={best['mean']:.3f}s size={best['size_mb']:.2f} MB")

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Best config baselines.  Lock to yuv420p input + the prior-sweep winner so
# subsequent encoder/feeding-path changes are apples-to-apples.
# ---------------------------------------------------------------------------


def test_best_config(frames_yuv420, tmp_path):
    """Subprocess ffmpeg, yuv420p input, gop=4, threads=16."""
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        "best",
        num_runs=10,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        gop=BEST_GOP,
        threads=BEST_THREADS,
    )
    print(
        f"\n[best yuv420p gop={BEST_GOP} threads={BEST_THREADS}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


def test_best_config_pyav(frames_yuv420, tmp_path):
    """Same config via PyAV (in-process libavcodec) — no subprocess, no pipe."""
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        "best_pyav",
        num_runs=10,
        encoder=export_to_video_pyav,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        gop=BEST_GOP,
        threads=BEST_THREADS,
    )
    print(
        f"\n[pyav yuv420p gop={BEST_GOP} threads={BEST_THREADS}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


# ---------------------------------------------------------------------------
# tune=zerolatency: disables lookahead and B-frames in libx264.  Expected to
# speed up encode at the cost of compression efficiency (larger output).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tune", [None, "zerolatency"])
def test_tune_zerolatency(tune, frames_yuv420, tmp_path):
    label = "off" if tune is None else tune
    stats = _benchmark(
        frames_yuv420,
        tmp_path,
        f"tune_{label}",
        num_runs=10,
        pix_fmt_in="yuv420p",
        width=WIDTH,
        height=HEIGHT,
        threads=BEST_THREADS,
        tune=tune,
    )
    print(
        f"\n[tune={label}] "
        f"mean={stats['mean']:.3f}s median={stats['median']:.3f}s "
        f"std={stats['std']:.3f}s min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"size={stats['size_mb']:.2f} MB"
    )


# ---------------------------------------------------------------------------
# Holistic end-to-end: device tensor -> D2H -> ffmpeg encode.  Two paths:
#   A) RGB uint8 D2H (float_to_uint8 + permute to BTHWC) -> export_to_video(rgb24)
#   B) YUV420p D2H (on-device color conversion + downsample) -> export_to_video(yuv420p)
# ---------------------------------------------------------------------------


def _shard_to_device(ref: torch.Tensor, mesh_device) -> ttnn.Tensor:
    return typed_tensor_2dshard(
        ref,
        mesh_device,
        shard_mapping={TP_AXIS: H_DIM, SP_AXIS: W_DIM},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )


def _concat_dims() -> list[int | None]:
    dims: list[int | None] = [None, None]
    dims[TP_AXIS] = H_DIM
    dims[SP_AXIS] = W_DIM
    return dims


@pytest.mark.parametrize(
    "mesh_device, num_links, device_params, topology",
    [
        [(4, 32), 2, ring_params, ttnn.Topology.Ring],
        [(4, 8), 2, {**line_params, "l1_small_size": 2048}, ttnn.Topology.Linear],
    ],
    ids=["bh_4x32", "bh_4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "height, width",
    [(720, 1280), (480, 832)],
    ids=["720p", "480p"],
)
def test_e2e_rgb_vs_yuv(mesh_device, num_links, device_params, topology, height, width, tmp_path):
    """Compare full pipeline cost of RGB-path vs YUV-path end to end.

    Path A (RGB):
      ``fast_device_to_host(pre_transfer_fn=float_to_uint8, permute=(0,2,3,4,1))``
      -> ``(B, T, H, W, C)`` uint8 -> ``export_to_video(rgb24)``

    Path B (YUV420p):
      ``fast_device_to_host_yuv`` -> ``(T, H*W*3//2)`` uint8 planar
      -> ``export_to_video(yuv420p, width, height)``

    Both encodes use the established best config: preset=ultrafast,
    gop=BEST_GOP, threads=BEST_THREADS.  Reports per-stage means: D2H, encode,
    and total — so it's clear which leg of the pipeline benefits.
    """
    n_iters = 5

    gen = torch.Generator().manual_seed(42)
    ref = torch.rand(B_DEV, C_DEV, NUM_FRAMES, height, width, generator=gen, dtype=torch.bfloat16) * 2.0 - 1.0
    tt_tensor = _shard_to_device(ref, mesh_device)
    ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=topology)
    concat_dims = _concat_dims()

    # ----- Warmup both paths (program cache, ffmpeg subprocess start, etc.) --
    warmup_rgb = fast_device_to_host(
        tt_tensor,
        mesh_device,
        concat_dims,
        ccl_manager=ccl_manager,
        pre_transfer_fn=float_to_uint8,
        permute=(0, 2, 3, 4, 1),
    )
    export_to_video(
        warmup_rgb[0].numpy(),
        str(tmp_path / "warmup_rgb.mp4"),
        fps=FPS,
        preset=PRESET,
        gop=BEST_GOP,
        threads=BEST_THREADS,
    )

    warmup_yuv = fast_device_to_host_yuv(tt_tensor, mesh_device)
    export_to_video(
        warmup_yuv,
        str(tmp_path / "warmup_yuv.mp4"),
        fps=FPS,
        preset=PRESET,
        pix_fmt_in="yuv420p",
        width=width,
        height=height,
        gop=BEST_GOP,
        threads=BEST_THREADS,
    )

    ttnn.synchronize_device(mesh_device)

    # ----- Path A: RGB uint8 D2H + rgb24 encode --------------------------------
    rgb_d2h, rgb_enc, rgb_total = [], [], []
    for i in range(n_iters):
        out_path = str(tmp_path / f"rgb_{i}.mp4")
        t0 = time.perf_counter()
        dev_rgb = fast_device_to_host(
            tt_tensor,
            mesh_device,
            concat_dims,
            ccl_manager=ccl_manager,
            pre_transfer_fn=float_to_uint8,
            permute=(0, 2, 3, 4, 1),
        )
        frames_rgb = dev_rgb[0].numpy()  # (T, H, W, 3) uint8
        t1 = time.perf_counter()
        export_to_video(
            frames_rgb,
            out_path,
            fps=FPS,
            preset=PRESET,
            gop=BEST_GOP,
            threads=BEST_THREADS,
        )
        t2 = time.perf_counter()
        rgb_d2h.append(t1 - t0)
        rgb_enc.append(t2 - t1)
        rgb_total.append(t2 - t0)

    rgb_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    # ----- Path B: YUV420p D2H + yuv420p encode --------------------------------
    yuv_d2h, yuv_enc, yuv_total = [], [], []
    for i in range(n_iters):
        out_path = str(tmp_path / f"yuv_{i}.mp4")
        t0 = time.perf_counter()
        # ``fast_device_to_host_yuv`` returns (T, H*W*3//2) uint8 numpy directly
        # (no batch dim) — matches export_to_video(pix_fmt_in='yuv420p').
        frames_yuv = fast_device_to_host_yuv(tt_tensor, mesh_device)
        t1 = time.perf_counter()
        export_to_video(
            frames_yuv,
            out_path,
            fps=FPS,
            preset=PRESET,
            pix_fmt_in="yuv420p",
            width=width,
            height=height,
            gop=BEST_GOP,
            threads=BEST_THREADS,
        )
        t2 = time.perf_counter()
        yuv_d2h.append(t1 - t0)
        yuv_enc.append(t2 - t1)
        yuv_total.append(t2 - t0)

    yuv_size_mb = os.path.getsize(out_path) / (1024 * 1024)

    rank = int(ttnn.distributed_context_get_rank()) if ttnn.using_distributed_env() else 0
    if rank != 0:
        return

    rgb_d2h_s = _stats(rgb_d2h)
    rgb_enc_s = _stats(rgb_enc)
    rgb_tot_s = _stats(rgb_total)
    yuv_d2h_s = _stats(yuv_d2h)
    yuv_enc_s = _stats(yuv_enc)
    yuv_tot_s = _stats(yuv_total)

    def _row(label, s):
        return (
            f"  {label:<18s} mean {s['mean']*1000:7.1f} ms | "
            f"median {s['median']*1000:7.1f} ms | "
            f"std {s['std']*1000:6.1f} ms | "
            f"min {s['min']*1000:7.1f} ms | "
            f"max {s['max']*1000:7.1f} ms"
        )

    lines = [
        "",
        f"=== Holistic end-to-end: RGB vs YUV ({width}x{height}, {NUM_FRAMES} frames, "
        f"{n_iters} iters, preset={PRESET}, gop={BEST_GOP}, threads={BEST_THREADS}) ===",
        f"Mesh shape: {tuple(mesh_device.shape)}",
        "",
        "Path A — RGB uint8 D2H -> ffmpeg rgb24:",
        _row("D2H:", rgb_d2h_s),
        _row("encode:", rgb_enc_s),
        _row("TOTAL:", rgb_tot_s),
        f"  output size:       {rgb_size_mb:.2f} MB",
        "",
        "Path B — YUV420p D2H -> ffmpeg yuv420p:",
        _row("D2H:", yuv_d2h_s),
        _row("encode:", yuv_enc_s),
        _row("TOTAL:", yuv_tot_s),
        f"  output size:       {yuv_size_mb:.2f} MB",
        "",
        "Delta (YUV - RGB), mean ms (negative = YUV faster):",
        f"  D2H:               {(yuv_d2h_s['mean'] - rgb_d2h_s['mean']) * 1000:+7.1f} ms",
        f"  encode:            {(yuv_enc_s['mean'] - rgb_enc_s['mean']) * 1000:+7.1f} ms",
        f"  TOTAL end-to-end:  {(yuv_tot_s['mean'] - rgb_tot_s['mean']) * 1000:+7.1f} ms",
        "",
    ]
    print("\n".join(lines))
