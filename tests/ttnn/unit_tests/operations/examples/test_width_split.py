# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `width_split` work-distribution example.

A wide, SHORT tensor (one tile-row tall, many tile-columns) has no parallelism
along its height, so a "split by tile-rows" strategy strands all the work on ONE
core. Splitting along the WIDTH (tile-columns) instead fills the compute grid.
See ttnn/ttnn/operations/examples/width_split/README.md.

    # both variants produce the identical, correct relu output at every width
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_correctness

    # device kernel duration + active cores, single_core vs width_split across widths
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.examples.width_split import width_split, VARIANTS

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32
_DTYPES = {"bfloat8_b": (ttnn.bfloat8_b, 0.99), "bfloat16": (ttnn.bfloat16, 0.9999), "float32": (ttnn.float32, 0.9999)}

# Wide-short: H fixed at one tile-row; only W (the width) sweeps.
WIDTHS = tuple(int(x) for x in os.environ.get("WS_WIDTHS", "32,256,1024,2048,4096,8192").split(","))
VARIANT_SEL = os.environ.get("WS_VARIANT", "all")
DTYPE_NAME = os.environ.get("WS_DTYPE", "bfloat16")
DTYPE, DTYPE_PCC = _DTYPES[DTYPE_NAME]
KERNEL_ITERS = int(os.environ.get("WS_ITERS", "1"))
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("WS_TRIALS", "20"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, w, dtype=DTYPE):
    torch.manual_seed(0)
    torch_input = torch.rand((TILE, w), dtype=torch.float32) * 2.0 - 1.0  # signed -> relu is non-trivial
    return ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total = _read_kernel_ns(device)
    return total / N_PROFILE_ITERS if total is not None else None


def _active_cores(device, variant, w):
    grid = device.compute_with_storage_grid_size()
    wt = w // TILE
    return 1 if variant == "single_core" else max(1, min(wt, grid.x * grid.y))


_CORRECTNESS_CASES = [(name, variant, w) for name in _DTYPES for variant in VARIANTS for w in (32, 1024, 4096)]


@pytest.mark.parametrize("dtype_name,variant,w", _CORRECTNESS_CASES)
def test_width_split_correctness(device, dtype_name, variant, w):
    """Both variants at every width produce the same relu output: out == relu(input)."""
    dtype, pcc = _DTYPES[dtype_name]
    tt_input = _make_input(device, w, dtype=dtype)
    expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
    out = ttnn.to_torch(width_split(tt_input, variant=variant, kernel_iters=KERNEL_ITERS)).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, pcc)


def test_width_split_device_perf(device):
    """Measure device kernel duration + active cores, single_core vs width_split across widths.

    Correctness lives in test_width_split_correctness; this only measures/reports
    (perf is evidence, never pass/fail — the only assertion is that the profiler produced a number)."""
    variants = VARIANTS if VARIANT_SEL == "all" else (VARIANT_SEL,)
    ns = {}
    for w in WIDTHS:
        tt_input = _make_input(device, w)
        for variant in variants:
            run_fn = lambda v=variant, t=tt_input: width_split(t, variant=v, kernel_iters=KERNEL_ITERS)
            value = _measure_ns(device, run_fn)
            assert value is not None, f"profiler produced no data for {variant} w={w} (profiler-enabled build?)"
            ns[(w, variant)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    grid = device.compute_with_storage_grid_size()
    lines = [
        "",
        "=== width_split device perf (relu, wide-short H=32) — single_core vs width_split ===",
        f"    H={TILE} (1 tile-row)  dtype={DTYPE_NAME}  grid={grid.x}x{grid.y}={grid.x*grid.y}  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    {'W':>6}  {'Wt':>4}  {'variant':<12}  {'cores':>5}  {'ns/op':>11}  {'speedup':>8}",
    ]
    for w in WIDTHS:
        base = ns.get((w, "single_core"))
        for variant in variants:
            v = ns[(w, variant)]
            cores = _active_cores(device, variant, w)
            spd = (base / v) if (base and variant != "single_core") else None
            tag = "  (base)" if variant == "single_core" else (f"  {spd:5.2f}x" if spd else "")
            lines.append(f"    {w:>6}  {w // TILE:>4}  {variant:<12}  {cores:>5}  {v:>11.1f}{tag}")
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# When width-splitting helps / when it does not (Wormhole B0 n150 calibration).
#
#   helps          Wt=64 (W=2048): single_core serializes 64 tiles; width_split puts
#                  1 tile on each of 64 Tensix cores.            measured ~6.0x
#   does_not_help  Wt=2  (W=64):   the whole tensor is 2 tiles — one 8-tile read block
#                  on one Tensix core costs ~2.1 us, dominated by the ~1.8 us fixed
#                  per-launch cost; a 2nd Tensix core only adds its own startup.  ~0.98x
#
# A trial = mean device kernel duration over _LAUNCHES_PER_TRIAL launches of one
# variant (one profiler window). Baseline/lever trials are interleaved A,B,A,B and
# the median of _N_TRIALS trials per variant is compared.
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5

# (regime id, W, kind, min speedup | max speedup). Thresholds: helps -> 1 + 0.5*(measured-1),
# floor 1.05; does_not_help -> <= 1.05.
_REGIMES = [
    ("helps_wt64", 2048, "helps", 3.5),  # measured ~6.1x -> 1 + 0.5*5.1 = 3.55, rounded down
    ("does_not_help_wt2", 64, "does_not_help", 1.05),
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


@pytest.mark.parametrize("regime,w,kind,threshold", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_width_split_regimes(device, regime, w, kind, threshold):
    """single_core (baseline) vs width_split (lever) at a width where splitting pays and
    one where it does not. Checks relu correctness for both, then asserts the speedup band."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    tt_input = _make_input(device, w, dtype=ttnn.bfloat16)
    expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
    for variant in VARIANTS:
        out = ttnn.to_torch(width_split(tt_input, variant=variant)).to(torch.float32)
        assert_with_pcc(expected, out, 0.9999)

    base, lever = _interleaved(
        device,
        lambda: width_split(tt_input, variant="single_core"),
        lambda: width_split(tt_input, variant="width_split"),
    )
    b_med, l_med = statistics.median(base), statistics.median(lever)
    speedup = b_med / l_med
    wt = w // TILE
    logger.info(
        f"[WH_B0] width_split {regime}: W={w} Wt={wt} cores={_active_cores(device, 'width_split', w)} | "
        f"single_core median {b_med:.1f} ns (min {min(base):.1f} max {max(base):.1f}) | "
        f"width_split median {l_med:.1f} ns (min {min(lever):.1f} max {max(lever):.1f}) | "
        f"speedup {speedup:.2f}x | trials {_N_TRIALS}x{_LAUNCHES_PER_TRIAL} launches"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
