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

    # reduce case (per-row mean subtracted from the row): row_split_reduce vs width_split_reduce,
    # sweeping tile-rows R_t x Wt x split factor k
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_device_perf

    # reduce case on ROW_MAJOR sticks (W bf16 datums per DRAM page): R_t x W x k, with DRAM transfer
    # size and count per configuration
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_rm_device_perf

    # helps / does-not-help regimes (elementwise, reduce and ROW_MAJOR reduce cases)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # quiet the per-read profiler histograms

import statistics

import pytest
import torch

import ttnn
from ttnn.operations.examples.width_split import (
    width_split,
    width_split_reduce,
    VARIANTS,
    REDUCE_VARIANTS,
    reduce_active_cores,
)

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
# Reduce case: y = x - mean_W(x) per row. A width split of a per-row reduction makes the k
# Tensix cores sharing a tile-row exchange partials once per row (the combine).
# ---------------------------------------------------------------------------

WSR_ROWS = tuple(int(x) for x in os.environ.get("WSR_ROWS", "1,16,64,128").split(","))
WSR_WTS = tuple(int(x) for x in os.environ.get("WSR_WTS", "4,8,16,32,64").split(","))
# k values tried per cell; each is kept only if it divides Wt, k <= Wt and R_t * k <= WSR_MAX_OVERSUB * grid
WSR_KS = tuple(int(x) for x in os.environ.get("WSR_KS", "1,2,4,8,16,32,64").split(","))
WSR_MAX_OVERSUB = float(os.environ.get("WSR_MAX_OVERSUB", "2"))


def _make_reduce_input(device, rows_t, wt, layout=ttnn.TILE_LAYOUT):
    """Rows with distinct offsets, so a wrong (or missing) cross-core combine moves the output."""
    torch.manual_seed(0)
    h, w = rows_t * TILE, wt * TILE
    offsets = torch.rand((h, 1), dtype=torch.float32) * 4.0 - 2.0
    torch_input = offsets + (torch.rand((h, w), dtype=torch.float32) * 2.0 - 1.0)
    tt_input = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.to_torch(tt_input).to(torch.float32)
    return tt_input, x - x.mean(dim=-1, keepdim=True)


def _check_reduce(expected, out, what=""):
    assert list(out.shape) == list(expected.shape), f"{what}: {out.shape} != {expected.shape}"
    max_err = (expected - out).abs().max().item()
    pcc = torch.corrcoef(torch.stack([expected.flatten(), out.flatten()]))[0, 1].item()
    assert pcc >= 0.9995, f"{what}: PCC {pcc:.6f} < 0.9995, max abs error {max_err:.4f}"
    assert max_err < 0.03, f"{what}: max abs error {max_err:.4f} (a wrong combine shifts whole rows by their offset)"


def _reduce_cells(device):
    grid = device.compute_with_storage_grid_size()
    n = grid.x * grid.y
    for rows_t in WSR_ROWS:
        for wt in WSR_WTS:
            ks = [k for k in WSR_KS if k > 1 and k <= wt and wt % k == 0 and rows_t * k <= WSR_MAX_OVERSUB * n]
            yield rows_t, wt, ks


# (R_t tile-rows, Wt tiles, k, kernel_iters): one slice per Tensix core, multi-row groups
# (cb_gather parity reuse), 1-tile slices, a 32-member group, and a repeated launch body.
_REDUCE_CORRECTNESS = [
    (1, 4, 1, 1),
    (1, 8, 2, 1),
    (1, 32, 8, 1),
    (1, 32, 32, 1),
    (4, 16, 4, 1),
    (64, 8, 2, 1),
    (128, 4, 2, 1),
    (128, 8, 4, 3),
    (3, 16, 2, 2),
]


@pytest.mark.parametrize("rows_t,wt,k,iters", _REDUCE_CORRECTNESS)
def test_width_split_reduce_correctness(device, rows_t, wt, k, iters):
    """Both reduce variants produce x - mean_W(x) (row_split_reduce ignores k)."""
    tt_input, expected = _make_reduce_input(device, rows_t, wt)
    for variant in REDUCE_VARIANTS:
        out = ttnn.to_torch(width_split_reduce(tt_input, variant=variant, k=k, kernel_iters=iters)).to(torch.float32)
        _check_reduce(expected, out, f"{variant} R_t={rows_t} Wt={wt} k={k} iters={iters}")


def test_width_split_reduce_device_perf(device):
    """Sweep R_t x Wt x k: row_split_reduce (k=1 baseline) vs width_split_reduce at every legal k.
    Every measured configuration is also checked for correctness. Perf is reported, never asserted."""
    grid = device.compute_with_storage_grid_size()
    rows = []
    for rows_t, wt, ks in _reduce_cells(device):
        tt_input, expected = _make_reduce_input(device, rows_t, wt)
        configs = [("row_split_reduce", 1)] + [("width_split_reduce", k) for k in ks]
        base = None
        for variant, k in configs:
            out = ttnn.to_torch(width_split_reduce(tt_input, variant=variant, k=k)).to(torch.float32)
            _check_reduce(expected, out, f"{variant} R_t={rows_t} Wt={wt} k={k}")
            value = _measure_ns(
                device,
                lambda v=variant, kk=k, t=tt_input: width_split_reduce(t, variant=v, k=kk, kernel_iters=KERNEL_ITERS),
            )
            assert value is not None, f"profiler produced no data for {variant} R_t={rows_t} Wt={wt} k={k}"
            base = value if base is None else base
            cores = reduce_active_cores(device, rows_t, wt, variant, k)
            rows.append((rows_t, wt, variant, k, cores, wt // k, value, base / value))
        tt_input.deallocate()

    lines = [
        "",
        "=== width_split reduce case (y = x - mean_W(x)) — row_split_reduce vs width_split_reduce ===",
        f"    dtype=bfloat16 TILE, DRAM interleaved in/out  grid={grid.x}x{grid.y}={grid.x*grid.y}  arch={device.arch()}  "
        f"iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        "    DRAM GB/s = (input + output bytes) / time = 2 * R_t * Wt * 2048 B / ns",
        f"    {'R_t':>4}  {'Wt':>3}  {'variant':<19}  {'k':>3}  {'cores':>5}  {'tiles/slice':>11}  {'ns/op':>10}  "
        f"{'GB/s':>6}  {'vs k=1':>7}",
    ]
    for rows_t, wt, variant, k, cores, slice_tiles, value, spd in rows:
        tag = "  (base)" if variant == "row_split_reduce" else f"  {spd:5.2f}x"
        gbs = 2 * rows_t * wt * 2048 / value
        lines.append(
            f"    {rows_t:>4}  {wt:>3}  {variant:<19}  {k:>3}  {cores:>5}  {slice_tiles:>11}  {value:>10.1f}  "
            f"{gbs:>6.1f}{tag}"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Reduce case on ROW_MAJOR sticks: the same y = x - mean_W(x), but the input and output are
# Layout::ROW_MAJOR (one DRAM page = one stick of W bf16 datums = W * 2 B). Each Tensix core reads
# its piece of each of the 32 sticks of a tile-row (32 reads of W * 2 / k B), tilizes, reduces /
# applies (same combine), untilizes and writes 32 stick pieces back. Splitting the row over k
# Tensix cores now also multiplies the DRAM transfer count by k and divides each transfer by k.
# ---------------------------------------------------------------------------

WSRM_ROWS = tuple(int(x) for x in os.environ.get("WSRM_ROWS", "1,4,16,47,64,128").split(","))
WSRM_WTS = tuple(int(x) for x in os.environ.get("WSRM_WTS", "2,4,8,16,32,64").split(","))
WSRM_KS = tuple(int(x) for x in os.environ.get("WSRM_KS", "1,2,4").split(","))


def _rm_cells(device):
    grid = device.compute_with_storage_grid_size()
    n = grid.x * grid.y
    for rows_t in WSRM_ROWS:
        for wt in WSRM_WTS:
            ks = [k for k in WSRM_KS if k > 1 and k <= wt and wt % k == 0 and rows_t * k <= WSR_MAX_OVERSUB * n]
            yield rows_t, wt, ks


# (R_t tile-rows, Wt tiles, k, kernel_iters): 128 B sticks, the field shapes (2048 x 256 and
# 47 tile-rows x 256), 4096 B sticks (64-tile slices), multi-row groups, a repeated launch body.
_REDUCE_RM_CORRECTNESS = [
    (1, 2, 1, 1),
    (1, 2, 2, 1),
    (64, 8, 2, 1),
    (47, 8, 2, 1),
    (16, 64, 1, 1),
    (16, 64, 4, 1),
    (128, 4, 2, 1),
    (3, 16, 4, 2),
]


@pytest.mark.parametrize("rows_t,wt,k,iters", _REDUCE_RM_CORRECTNESS)
def test_width_split_reduce_rm_correctness(device, rows_t, wt, k, iters):
    """Both reduce variants on ROW_MAJOR sticks produce x - mean_W(x), in ROW_MAJOR."""
    tt_input, expected = _make_reduce_input(device, rows_t, wt, layout=ttnn.ROW_MAJOR_LAYOUT)
    for variant in REDUCE_VARIANTS:
        out_t = width_split_reduce(tt_input, variant=variant, k=k, kernel_iters=iters)
        assert out_t.layout == ttnn.ROW_MAJOR_LAYOUT
        _check_reduce(expected, ttnn.to_torch(out_t).to(torch.float32), f"RM {variant} R_t={rows_t} Wt={wt} k={k}")


def test_width_split_reduce_rm_device_perf(device):
    """Sweep R_t x W x k on ROW_MAJOR sticks: row_split_reduce (k=1) vs width_split_reduce.
    Every measured configuration is also checked for correctness. Perf is reported, never asserted."""
    grid = device.compute_with_storage_grid_size()
    rows = []
    for rows_t, wt, ks in _rm_cells(device):
        tt_input, expected = _make_reduce_input(device, rows_t, wt, layout=ttnn.ROW_MAJOR_LAYOUT)
        configs = [("row_split_reduce", 1)] + [("width_split_reduce", k) for k in ks]
        base = None
        for variant, k in configs:
            out = ttnn.to_torch(width_split_reduce(tt_input, variant=variant, k=k)).to(torch.float32)
            _check_reduce(expected, out, f"RM {variant} R_t={rows_t} Wt={wt} k={k}")
            value = _measure_ns(
                device,
                lambda v=variant, kk=k, t=tt_input: width_split_reduce(t, variant=v, k=kk, kernel_iters=KERNEL_ITERS),
            )
            assert value is not None, f"profiler produced no data for RM {variant} R_t={rows_t} Wt={wt} k={k}"
            base = value if base is None else base
            cores = reduce_active_cores(device, rows_t, wt, variant, k)
            rows.append((rows_t, wt, variant, k, cores, value, base / value))
        tt_input.deallocate()

    lines = [
        "",
        "=== width_split reduce case on ROW_MAJOR sticks (y = x - mean_W(x)) — row_split_reduce vs width_split_reduce ===",
        f"    dtype=bfloat16 ROW_MAJOR, DRAM interleaved in/out  grid={grid.x}x{grid.y}={grid.x*grid.y}  "
        f"arch={device.arch()}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        "    stick B = W * 2; piece B = stick B / k (bytes per DRAM transfer); transfers = 2 * 32 * R_t * k (read + write)",
        "    DRAM GB/s = 2 * R_t * 32 * stick B / ns;  Mxfer/s = transfers / ns * 1000",
        f"    {'R_t':>4}  {'W':>5}  {'stick B':>7}  {'variant':<19}  {'k':>2}  {'cores':>5}  {'piece B':>7}  "
        f"{'xfers':>6}  {'ns/op':>10}  {'GB/s':>6}  {'Mxfer/s':>7}  {'vs k=1':>7}",
    ]
    for rows_t, wt, variant, k, cores, value, spd in rows:
        stick = wt * TILE * 2
        xfers = 2 * TILE * rows_t * k
        tag = "  (base)" if variant == "row_split_reduce" else f"  {spd:5.2f}x"
        lines.append(
            f"    {rows_t:>4}  {wt * TILE:>5}  {stick:>7}  {variant:<19}  {k:>2}  {cores:>5}  {stick // k:>7}  "
            f"{xfers:>6}  {value:>10.1f}  {2 * rows_t * TILE * stick / value:>6.1f}  {xfers / value * 1000:>7.0f}{tag}"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# When width-splitting helps / when it does not (numbers below: WH B0 n150; BH p150b in README.md).
#
#   helps          Wt=64 (W=2048): single_core serializes 64 tiles; width_split puts
#                  1 tile on each of 64 Tensix cores.            measured ~6.0x
#   measure        Wt=2  (W=64):   the whole tensor is 2 tiles — one 8-tile read block
#                  on one Tensix core costs ~2.1 us, dominated by the ~1.8 us fixed
#                  per-launch cost; a 2nd Tensix core only adds its own startup.  WH ~0.98x, BH 1.18x
#
# A trial = mean device kernel duration over _LAUNCHES_PER_TRIAL launches of one
# variant (one profiler window). Baseline/lever trials are interleaved A,B,A,B and
# the median of _N_TRIALS trials per variant is compared.
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5

# (regime id, W, kind, min speedup | max speedup). Thresholds: helps -> 1 + 0.5*(measured-1),
# floor 1.05; does_not_help -> <= 1.05; measure -> correctness only, speedup logged.
# Thresholds hold on every measured board (WH B0 n150, BH p150b); `measure` regimes flip between boards and only log
# - numbers per board in README.md.
_REGIMES = [
    ("helps_wt64", "eltwise", (2048,), "helps", 3.5),  # WH 6.19x, BH 7.22x
    ("measure_wt2", "eltwise", (64,), "measure", None),  # WH 0.99x, BH 1.18x - verdict differs
    ("does_not_help_wt1", "eltwise", (32,), "does_not_help", 1.05),  # WH 1.00x, BH 1.02x (one Tensix core either way)
    # reduce case (x - mean_W(x)): params = (R_t tile-rows, Wt tiles, split factor k).
    # helps: 1 tile-row on 1 -> 8 Tensix cores, 4 tiles per slice.            BH p150b 2.34-2.36x
    ("reduce_helps_rt1_wt32_k8", "reduce", (1, 32, 8), "helps", 1.68),
    # does not help: 64 tile-rows already on 64 of 130 Tensix cores (DRAM ~60% of its ceiling);
    # doubling to 128 Tensix cores (8 tiles per slice) buys nothing.               BH p150b 0.98x
    ("reduce_does_not_help_rt64_wt16_k2", "reduce", (64, 16, 2), "does_not_help", 1.05),
    # the same with 4-tile slices sits on the edge.                       BH p150b 1.03-1.04x
    ("reduce_measure_rt64_wt8_k2", "reduce", (64, 8, 2), "measure", None),
    # reduce case on ROW_MAJOR sticks: params = (R_t tile-rows, Wt tiles, k); stick = Wt * 64 B,
    # each Tensix core reads / writes 32 pieces of stick / k B per tile-row whatever k is.
    # helps: 1 tile-row of 4096 B sticks on 1 -> 4 Tensix cores, 1024 B pieces.   BH p150b 1.86-1.88x
    ("reduce_rm_helps_rt1_w2048_k4", "reduce_rm", (1, 64, 4), "helps", 1.43),
    # does not help: 64 tile-rows of 512 B sticks (2048 x 256) on 64 -> 128 Tensix cores, 256 B
    # pieces: 2x the DRAM transfers, ~1 ns each across the grid.                  BH p150b 0.65x
    ("reduce_rm_does_not_help_rt64_w256_k2", "reduce_rm", (64, 8, 2), "does_not_help", 1.05),
    # 4096 B sticks with the tile-rows on half the grid: 2048 B pieces, at the edge. BH p150b 1.04-1.08x
    ("reduce_rm_measure_rt64_w2048_k2", "reduce_rm", (64, 64, 2), "measure", None),
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


@pytest.mark.parametrize("regime,op,params,kind,threshold", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_width_split_regimes(device, regime, op, params, kind, threshold):
    """Baseline vs width split where splitting pays and where it does not. Checks correctness of
    both, then asserts the speedup band.
      eltwise: single_core vs width_split (relu, H=32, W=params[0]).
      reduce:  row_split_reduce vs width_split_reduce (x - mean_W(x), R_t x Wt, split factor k).
      reduce_rm: the same on ROW_MAJOR sticks (stick pieces read, tilize, untilize, pieces written)."""
    if op == "eltwise":
        (w,) = params
        tt_input = _make_input(device, w, dtype=ttnn.bfloat16)
        expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
        for variant in VARIANTS:
            out = ttnn.to_torch(width_split(tt_input, variant=variant)).to(torch.float32)
            assert_with_pcc(expected, out, 0.9999)
        run_base = lambda: width_split(tt_input, variant="single_core")
        run_lever = lambda: width_split(tt_input, variant="width_split")
        shape_desc = f"W={w} Wt={w // TILE} cores={_active_cores(device, 'width_split', w)}"
        names = ("single_core", "width_split")
    else:
        rows_t, wt, k = params
        layout = ttnn.ROW_MAJOR_LAYOUT if op == "reduce_rm" else ttnn.TILE_LAYOUT
        tt_input, expected = _make_reduce_input(device, rows_t, wt, layout=layout)
        for variant in REDUCE_VARIANTS:
            _check_reduce(expected, ttnn.to_torch(width_split_reduce(tt_input, variant=variant, k=k)).to(torch.float32))
        run_base = lambda: width_split_reduce(tt_input, variant="row_split_reduce")
        run_lever = lambda: width_split_reduce(tt_input, variant="width_split_reduce", k=k)
        shape_desc = (
            f"{'ROW_MAJOR stick=' + str(wt * 64) + 'B piece=' + str(wt * 64 // k) + 'B ' if op == 'reduce_rm' else ''}"
            f"R_t={rows_t} Wt={wt} k={k} tiles/slice={wt // k} cores "
            f"{reduce_active_cores(device, rows_t, wt, 'row_split_reduce', 1)} -> "
            f"{reduce_active_cores(device, rows_t, wt, 'width_split_reduce', k)}"
        )
        names = ("row_split_reduce", "width_split_reduce")

    base, lever = _interleaved(device, run_base, run_lever)
    b_med, l_med = statistics.median(base), statistics.median(lever)
    speedup = b_med / l_med
    logger.info(
        f"[{device.arch()}] width_split {regime}: {shape_desc} | "
        f"{names[0]} median {b_med:.1f} ns (min {min(base):.1f} max {max(base):.1f}) | "
        f"{names[1]} median {l_med:.1f} ns (min {min(lever):.1f} max {max(lever):.1f}) | "
        f"speedup {speedup:.2f}x | trials {_N_TRIALS}x{_LAUNCHES_PER_TRIAL} launches"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    elif kind == "measure":
        pass  # verdict differs by arch: correctness checked and speedup logged above, no perf assert
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
