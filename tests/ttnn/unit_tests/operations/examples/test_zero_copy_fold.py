# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Zero-copy same-spec sharded tilize: fold reader+writer into one compute kernel vs three kernels.

Both variants tilize a resident ROW_MAJOR sharded tensor into a same-spec TILE sharded tensor with the
CBs aliased onto the L1 shards (no DRAM, no NoC). The only difference is program structure:
`reader_compute_writer` (3 kernels + the reader->compute->writer CB handshake) vs `compute_only` (1
kernel that self-arms and self-drains). Correctness is the only pass/fail; DEVICE KERNEL DURATION [ns]
is measured and reported so the fixed per-core dispatch+handshake overhead is visible (and its
amortization as the shard grows).
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.zero_copy_fold import BASELINE, VARIANTS, run_op, sharded_memory_config

TILE = 32
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

# (H, W, ncores). HEIGHT-sharded; each core holds an [H/ncores, W] tile-aligned RM shard.
# Small shards (few tiles/core) first — where the fixed 3-kernel overhead should dominate — then
# progressively larger shards where it amortizes.
_SHAPES = [
    (64, 64, 2),  # shard 32x64  -> 1x2 = 2 tiles/core
    (128, 64, 4),  # shard 32x64  -> 1x2 = 2 tiles/core, 4 cores
    (64, 128, 2),  # shard 32x128 -> 1x4 = 4 tiles/core
    (128, 128, 2),  # shard 64x128 -> 2x4 = 8 tiles/core
    (256, 128, 8),  # shard 32x128 -> 1x4 = 4 tiles/core, 8 cores
    (512, 128, 4),  # shard 128x128 -> 4x4 = 16 tiles/core
    (1024, 256, 4),  # shard 256x256 -> 8x8 = 64 tiles/core (large: gap should shrink)
]


def _make_input(device, H, W, ncores, seed=13):
    torch.manual_seed(seed)
    data = torch.rand(H, W)
    x = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=sharded_memory_config(H, W, ncores),
    )
    return x, data.to(torch.bfloat16).to(torch.float32)


def _check(output, golden, label):
    got = ttnn.to_torch(output).to(torch.float32)
    assert list(got.shape) == list(golden.shape), f"{label}: shape {list(got.shape)} != {list(golden.shape)}"
    # Pure layout conversion, bf16->bf16, no dtype change -> must be bit-exact.
    max_abs = (got - golden).abs().max().item()
    assert max_abs == 0.0, f"{label}: not bit-exact, max_abs={max_abs}"
    return max_abs


# =============================================================================
# In-process device-kernel timing (validated pattern)
# =============================================================================
def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {key: [] for key in runners}
    for trial in range(trials + 1):
        for key, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"no profiler data for {key}"
            if trial:
                samples[key].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    if name := os.environ.get("ARCH_NAME"):
        return name
    a = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH", "GRAYSKULL": "GS"}.get(a, a)


def _int(name, default):
    return int(os.environ.get(name, default))


def _csv(name, default):
    val = os.environ.get(name)
    return [s for s in val.split(",") if s] if val else list(default)


def _shapes(name):
    val = os.environ.get(name)
    if not val:
        return None
    return [tuple(int(x) for x in tok.split(",")) for tok in val.split(";") if tok]


def _tiles_per_core(H, W, ncores):
    return (H // ncores // TILE) * (W // TILE)


# =============================================================================
# Tests
# =============================================================================
def test_zero_copy_fold_correctness(device):
    """Both variants produce the bit-exact tilized output on every shape."""
    for H, W, ncores in _SHAPES:
        x, golden = _make_input(device, H, W, ncores)
        for variant in VARIANTS:
            out = run_op(x, variant=variant, ncores=ncores, kernel_iters=2)
            _check(out, golden, f"{variant} {H}x{W}/{ncores}c")
            logger.info(f"OK {variant:22s} {H}x{W} ncores={ncores} tiles/core={_tiles_per_core(H, W, ncores)}")


def test_zero_copy_fold_device_perf(device):
    """compute_only vs reader_compute_writer: DEVICE KERNEL DURATION per launch, across shard sizes.

    CLI-driveable via ZCF_VARIANTS, ZCF_SHAPES (H,W,ncores;...), ZCF_TRIALS, ZCF_KERNEL_ITERS, ZCF_REPORT.
    """
    trials = _int("ZCF_TRIALS", "5")
    kernel_iters = _int("ZCF_KERNEL_ITERS", "100")
    variants = [v for v in _csv("ZCF_VARIANTS", VARIANTS) if v in VARIANTS]
    shapes = _shapes("ZCF_SHAPES") or _SHAPES

    inputs, goldens = {}, {}
    for H, W, ncores in shapes:
        inputs[(H, W, ncores)], goldens[(H, W, ncores)] = _make_input(device, H, W, ncores)

    # correctness gate before timing
    for shape in shapes:
        for variant in variants:
            out = run_op(inputs[shape], variant=variant, ncores=shape[2], kernel_iters=1)
            _check(out, goldens[shape], f"{variant} {shape}")

    runners = {
        (variant, shape): (
            lambda v=variant, s=shape: run_op(inputs[s], variant=v, ncores=s[2], kernel_iters=kernel_iters)
        )
        for shape in shapes
        for variant in variants
    }
    samples = _measure(device, runners, trials, kernel_iters)

    def med(v, s):
        return statistics.median(samples[(v, s)]) if (v, s) in samples else None

    def cell(v, s, base):
        ns = med(v, s)
        if ns is None:
            return "—"
        return f"{ns:.0f} ({base / ns:.2f}×)" if (base and v != BASELINE) else f"{ns:.0f}"

    lines = [
        "# Zero-copy same-spec sharded tilize — compute_only (fold) vs reader_compute_writer (3 kernels)",
        "",
        f"box={socket.gethostname()}  arch={_arch_label(device)}  placement=HEIGHT-sharded resident-L1 "
        f"(no DRAM/NoC)  N={trials} (median)  kernel-iters={kernel_iters} (steady-state)",
        "Same tilize, same aliased CBs — only the program structure differs. ns = median DEVICE KERNEL "
        "DURATION per launch. Ratio = reader_compute_writer / compute_only (>1 => fold is faster).",
        "",
        "| H×W | cores | tiles/core | reader_compute_writer ns | compute_only ns (×) |",
        "|---|---:|---:|---:|---:|",
    ]
    for shape in shapes:
        H, W, ncores = shape
        base = med(BASELINE, shape)
        lines.append(
            f"| {H}×{W} | {ncores} | {_tiles_per_core(H, W, ncores)} | "
            f"{cell(BASELINE, shape, None)} | {cell('compute_only', shape, base)} |"
        )
    lines += [
        "",
        "`reader_compute_writer` runs three kernels per core (a dataflow reader that arms the resident "
        "input CB, the compute tilize, a dataflow writer that drains the output CB) plus the "
        "reader->compute->writer circular-buffer handshake. `compute_only` folds the arm+drain into the "
        "single compute kernel. Both alias the CBs onto the resident L1 shards, so there is no NoC to hide "
        "the extra kernels' fixed dispatch+handshake cost — it shows up as latency on small shards and "
        "amortizes as tiles/core grows.",
    ]
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    if report_path := os.environ.get("ZCF_REPORT"):
        Path(report_path).write_text(report)
