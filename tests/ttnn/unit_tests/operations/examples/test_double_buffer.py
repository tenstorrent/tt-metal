# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `double_buffer` data-movement pipelining example.

Two kernel-level knobs decide whether a DRAM reader->compute->writer pipeline is
latency-bound or bandwidth-bound: (1) `block` = reads/writes issued per NoC
barrier (block=1 is the latency-bound trap), and (2) double buffering (CB depth).
See ttnn/ttnn/operations/examples/double_buffer/README.md.

    # both variants at every block produce the identical, correct relu output
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_correctness

    # device kernel duration + achieved DRAM GB/s, block x variant (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_double_buffer.py::test_double_buffer_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module (not a dir conftest) so it doesn't perturb other
# examples' measurement. setdefault -> respects an outer tracy run if present.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.examples.double_buffer import double_buffer, VARIANTS

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32
# The "transfer size" axis: bytes per tile (the NoC transaction size) is set by the
# tile dtype — bfloat8_b ~1088 B, bfloat16 2048 B, float32 4096 B. Never hard-code it;
# read the real page size back from the allocated tensor (bfloat8_b even carries an
# exponent header). PCC tolerance is looser for the lossy bfloat8_b format.
_DTYPES = {
    "bfloat8_b": (ttnn.bfloat8_b, 0.99),
    "bfloat16": (ttnn.bfloat16, 0.9999),
    "float32": (ttnn.float32, 0.9999),
}
# Defaults are overridable via env so the CLI (python -m ...double_buffer) can
# measure the caller's own shape/params through the same in-process path.
SHAPE = tuple(int(x) for x in os.environ.get("DB_SHAPE", "512,512").split(","))  # 256 tiles (single core)
NUM_CORES = int(os.environ.get("DB_CORES", "1"))  # cores running the pipeline (each independent)
BLOCK_SWEEP = tuple(int(x) for x in os.environ.get("DB_BLOCKS", "1,2,4,8,16,32").split(","))  # reads per barrier
COMPUTE_PASSES = int(os.environ.get("DB_PASSES", "1"))  # relu repeats; kept light to study data movement
KERNEL_ITERS = int(os.environ.get("DB_ITERS", "1"))  # in-kernel repeat of the tile range
DTYPE_NAME = os.environ.get("DB_DTYPE", "bfloat16")  # transfer size: bfloat8_b | bfloat16 | float32
DTYPE, DTYPE_PCC = _DTYPES[DTYPE_NAME]
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("DB_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, dtype=DTYPE):
    torch.manual_seed(0)
    # Signed so relu is non-trivial (~half the values clamp to 0, the rest pass through).
    torch_input = torch.rand(SHAPE, dtype=torch.float32) * 2.0 - 1.0
    tt_input = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return tt_input


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs dispatched since the last read.

    ReadDeviceProfiler finishes the queue before reading and *consumes* the window,
    so a flush-read then a work-read brackets exactly the ops run in between.
    """
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    """Average ns/op: warm up, flush, run N, read, divide by N."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = _read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


def _gbps(ns_per_op, num_pages, page_bytes):
    """Achieved DRAM bandwidth: read + write traffic (2 x tensor bytes) over the
    measured time. bytes / ns == GB/s (1e9 B / 1e-9 s). page_bytes is the real
    per-tile DRAM page size for this dtype, read back from the tensor."""
    traffic_bytes = 2 * num_pages * page_bytes * KERNEL_ITERS
    return traffic_bytes / ns_per_op


# (dtype_name, variant, block, passes) — covers all three transfer sizes x both
# variants, the block-remainder path (6 does not divide 256), and idempotency (passes=4).
_CORRECTNESS_CASES = [
    (name, variant, block, passes)
    for name in _DTYPES
    for variant in VARIANTS
    for (block, passes) in [(1, 1), (6, 1), (8, 4)]
]


@pytest.mark.parametrize("dtype_name,variant,block,compute_passes", _CORRECTNESS_CASES)
def test_double_buffer_correctness(device, dtype_name, variant, block, compute_passes):
    """Every (dtype, variant, block, passes) is the same relu pipeline: output == relu(input)."""
    dtype, pcc = _DTYPES[dtype_name]
    tt_input = _make_input(device, dtype=dtype)
    expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))

    out = ttnn.to_torch(
        double_buffer(
            tt_input,
            variant=variant,
            block=block,
            num_cores=NUM_CORES,
            compute_passes=compute_passes,
            kernel_iters=KERNEL_ITERS,
        )
    ).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, pcc)


def test_double_buffer_device_perf(device):
    """Measure device kernel duration + achieved DRAM GB/s over block x variant.

    Correctness lives in test_double_buffer_correctness; this test only measures
    and reports (perf is evidence, never a pass/fail — the only assertion here is
    that the profiler produced a number).
    """
    tt_input = _make_input(device)
    num_pages = (SHAPE[0] // TILE) * (SHAPE[1] // TILE)
    page_bytes = tt_input.buffer_aligned_page_size()  # real per-tile DRAM page for this dtype

    ns = {}
    for block in BLOCK_SWEEP:
        for variant in VARIANTS:
            run_fn = lambda v=variant, b=block: double_buffer(
                tt_input,
                variant=v,
                block=b,
                num_cores=NUM_CORES,
                compute_passes=COMPUTE_PASSES,
                kernel_iters=KERNEL_ITERS,
            )
            value = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} block={block} (profiler-enabled build?)"
            ns[(block, variant)] = value

    trap = ns[(BLOCK_SWEEP[0], "single_buffered")]  # block=1 single-buffered = the naive starting point
    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== double_buffer device perf (relu DRAM pipeline) — reads/barrier x buffering ===",
        f"    shape={SHAPE}  tiles={num_pages}  dtype={DTYPE_NAME}  tile_bytes={page_bytes}  cores={NUM_CORES}  passes={COMPUTE_PASSES}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    DRAM traffic = read+write = {2 * num_pages * page_bytes / 1e6:.2f} MB/launch; GB/s = traffic / kernel_ns",
        f"    {'block':>5}  {'variant':<16}  {'ns/op':>11}  {'GB/s':>7}  {'vs trap':>8}",
    ]
    for block in BLOCK_SWEEP:
        for variant in VARIANTS:
            v = ns[(block, variant)]
            ratio = trap / v if v else float("nan")
            tag = "  (trap)" if (block == BLOCK_SWEEP[0] and variant == "single_buffered") else f"  {ratio:5.2f}x"
            lines.append(f"    {block:>5}  {variant:<16}  {v:>11.1f}  {_gbps(v, num_pages, page_bytes):>7.1f}{tag}")
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# When batching + double buffering helps / when it does not (Wormhole B0 n150).
#
# Baseline = the trap (block=1, single_buffered: one read, barrier, one-tile CB).
# Lever    = block=4, double_buffered (4 reads per barrier, CB = 2*4 tiles).
#
#   helps_1core      1 Tensix core, 256 tiles bf16: latency-bound (~6.5 GB/s) -> ~2.8x
#   helps_8cores     8 Tensix cores, 512 tiles/core: aggregate trap BW ~51 GB/s, far below
#                    the DRAM ceiling, so every core still gains                  -> ~2.8x
#   does_not_help    64 Tensix cores, 64 tiles/core: the trap alone already moves ~190 GB/s
#                    (DRAM peak), so there is no idle NoC time left to recover     -> ~0.98x
#
# A trial = mean device kernel duration over _LAUNCHES_PER_TRIAL launches of one
# variant (one profiler window). Baseline/lever trials are interleaved A,B,A,B and
# the median of _N_TRIALS trials per variant is compared.
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5
_BASE_CFG = dict(variant="single_buffered", block=1)
_LEVER_CFG = dict(variant="double_buffered", block=4)

# (regime id, shape, cores, kind, threshold). helps -> 1 + 0.5*(measured-1) (floor 1.05);
# does_not_help -> <= 1.05.
_REGIMES = [
    ("helps_1core_256tiles", (512, 512), 1, "helps", 1.8),  # measured ~2.63x
    ("helps_8cores_512tiles_per_core", (2048, 2048), 8, "helps", 1.85),  # measured ~2.73x
    ("does_not_help_64cores_64tiles_per_core", (2048, 2048), 64, "does_not_help", 1.05),
]


def _trial_ns(device, run_fn):
    for _ in range(_LAUNCHES_PER_TRIAL):
        run_fn()
    total = _read_kernel_ns(device)
    assert total is not None, "profiler produced no data (profiler-enabled build?)"
    return total / _LAUNCHES_PER_TRIAL


def _interleaved(device, run_base, run_lever):
    for _ in range(N_WARMUP):
        run_base()
        run_lever()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    base, lever = [], []
    for _ in range(_N_TRIALS):
        base.append(_trial_ns(device, run_base))
        lever.append(_trial_ns(device, run_lever))
    return base, lever


@pytest.mark.parametrize("regime,shape,cores,kind,threshold", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_double_buffer_regimes(device, regime, shape, cores, kind, threshold):
    """Trap (block=1, single_buffered) vs lever (block=4, double_buffered) where the
    pipeline is latency-bound and where DRAM bandwidth is already saturated."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch.float32) * 2.0 - 1.0
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
    run_base = lambda: double_buffer(tt_input, num_cores=cores, **_BASE_CFG)
    run_lever = lambda: double_buffer(tt_input, num_cores=cores, **_LEVER_CFG)
    for run_fn in (run_base, run_lever):
        assert_with_pcc(expected, ttnn.to_torch(run_fn()).to(torch.float32), 0.9999)

    base, lever = _interleaved(device, run_base, run_lever)
    b_med, l_med = statistics.median(base), statistics.median(lever)
    speedup = b_med / l_med
    num_pages = (shape[0] // TILE) * (shape[1] // TILE)
    traffic = 2 * num_pages * tt_input.buffer_aligned_page_size()
    logger.info(
        f"[WH_B0] double_buffer {regime}: shape={shape} tiles={num_pages} cores={cores} "
        f"tiles/core={num_pages // cores} | trap median {b_med:.1f} ns (min {min(base):.1f} max {max(base):.1f}, "
        f"{traffic / b_med:.1f} GB/s) | block4+double median {l_med:.1f} ns (min {min(lever):.1f} "
        f"max {max(lever):.1f}, {traffic / l_med:.1f} GB/s) | speedup {speedup:.2f}x | "
        f"trials {_N_TRIALS}x{_LAUNCHES_PER_TRIAL} launches"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
