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
    # sweeping tile-rows R_t x Wt x split factor k (a measurement sweep: skipped unless switched on)
    EXAMPLES_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_reduce_device_perf

    # reduce case on Layout::ROW_MAJOR sticks (W bf16 datums per DRAM page): R_t x W x k, with DRAM
    # transfer size and count per configuration (a measurement sweep: skipped unless switched on)
    EXAMPLES_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
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

import pytest
import torch

import ttnn
from ttnn.operations.examples.width_split import (
    width_split,
    width_split_reduce,
    VARIANTS,
    REDUCE_VARIANTS,
    reduce_active_cores,
    reduce_cb_bytes,
    reduce_fits_l1,
)

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    check_regime,
    grid_cores,
    grid_xy,
    interleaved_ns,
    read_kernel_ns,
    sweep_gate,
)

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


def _make_input(device, w, dtype=DTYPE):
    torch.manual_seed(0)
    torch_input = torch.rand((TILE, w), dtype=torch.float32) * 2.0 - 1.0  # signed -> relu is non-trivial
    return ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _measure_ns(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # flush warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total = read_kernel_ns(device)
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
# k values tried per cell; each is kept only if it divides Wt, k <= Wt, k <= the grid's Tensix cores,
# R_t * k <= WSR_MAX_OVERSUB * grid Tensix cores and its circular buffers fit L1 (see _sweep_cells)
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


def _sweep_cells(device, rows_list, wts, ks_list, row_major):
    """(R_t, Wt, [k > 1 ...]) per sweep cell. A k is kept when it divides Wt, k <= Wt, k <= the grid's
    Tensix cores, R_t * k <= WSR_MAX_OVERSUB * grid Tensix cores, and the reduce program's circular
    buffers fit L1 (`reduce_fits_l1`, the same check `validate_reduce` raises on). A cell whose k = 1
    baseline (row_split_reduce) does not fit is skipped with a log line, not run."""
    n = grid_cores(device)
    for rows_t in rows_list:
        for wt in wts:
            if not reduce_fits_l1(wt, 1, row_major):
                logger.info(
                    f"skip R_t={rows_t} Wt={wt}: row_split_reduce needs {reduce_cb_bytes(wt, 1, row_major)} B of "
                    "circular buffers per Tensix core, over the L1 budget (validate_reduce would reject it)"
                )
                continue
            ks = [
                k
                for k in ks_list
                if 1 < k <= min(wt, n)
                and wt % k == 0
                and rows_t * k <= WSR_MAX_OVERSUB * n
                and reduce_fits_l1(wt, k, row_major)
            ]
            yield rows_t, wt, ks


def _reduce_cells(device):
    return _sweep_cells(device, WSR_ROWS, WSR_WTS, WSR_KS, row_major=False)


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


@sweep_gate("WS_SWEEPS")
def test_width_split_reduce_device_perf(device):
    """Sweep R_t x Wt x k: row_split_reduce (k=1 baseline) vs width_split_reduce at every legal k.
    Every measured configuration is also checked for correctness. Perf is reported, never asserted.
    A measurement sweep: runs only with EXAMPLES_SWEEPS=1 or WS_SWEEPS=1 (the CLI's --reduce sets it)."""
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
        f"    dtype=bfloat16 Layout::TILE, TensorMemoryLayout::INTERLEAVED DRAM in/out  "
        f"grid={grid.x}x{grid.y}={grid.x*grid.y} Tensix cores  arch={device.arch()}  "
        f"iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        "    cores = active Tensix cores; tiles/slice = tiles per Tensix core per tile-row",
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
    return _sweep_cells(device, WSRM_ROWS, WSRM_WTS, WSRM_KS, row_major=True)


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


@sweep_gate("WS_SWEEPS")
def test_width_split_reduce_rm_device_perf(device):
    """Sweep R_t x W x k on Layout::ROW_MAJOR sticks: row_split_reduce (k=1) vs width_split_reduce.
    Every measured configuration is also checked for correctness. Perf is reported, never asserted.
    A measurement sweep: runs only with EXAMPLES_SWEEPS=1 or WS_SWEEPS=1 (the CLI's --reduce sets it)."""
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
        f"    dtype=bfloat16 Layout::ROW_MAJOR, TensorMemoryLayout::INTERLEAVED DRAM in/out  "
        f"grid={grid.x}x{grid.y}={grid.x*grid.y} Tensix cores  "
        f"arch={device.arch()}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        "    cores = active Tensix cores",
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
# When width-splitting helps / when it does not. `measured` holds the speedup (baseline ns / lever
# ns) measured per architecture: "wh" = WH B0 n150 (8x8 = 64 Tensix cores), "bh" = BH p150b (12x10 =
# 120 Tensix cores for the elementwise regimes, 13x10 = 130 for the reduce ones). Per-board tables:
# README.md / report.md.
#
#   helps          Wt=64 (W=2048): single_core serializes 64 tiles; width_split puts
#                  1 tile on each of 64 Tensix cores (min(Wt, grid) Tensix cores).
#   measure        Wt=2  (W=64):   the whole tensor is 2 tiles — one 8-tile read block
#                  on one Tensix core costs ~2.1 us, dominated by the ~1.8 us fixed
#                  per-launch cost; a 2nd Tensix core only adds its own startup (WH B0); BH p150b gains.
#
# Measurement: perf_test_utils.interleaved_ns - N_WARMUP warm-up rounds, then _N_TRIALS
# trials per variant, a trial = mean device kernel duration over _LAUNCHES_PER_TRIAL launches of one
# variant (one profiler window), variants interleaved with a rotating start; medians compared.
#
# Assertion (perf_test_utils.check_regime): the bound is derived from the number measured on THIS
# architecture by perf_test_utils.regime_bound (helps: speedup >= min(half the measured gain,
# measured / 1.10), not asserted when that is below 1.02x; does_not_help: speedup <= max(1.0,
# measured x 1.10)). An architecture with no measured number is logged, not asserted. A missed bound
# is re-measured once and judged on the pooled samples. `measure` regimes only log.
#
# The R_t = 64 reduce regimes contrast 64 -> 128 Tensix cores (measured on a 130-Tensix-core grid).
# On a grid of <= 65 Tensix cores (WH B0) both variants land on the same Tensix-core count, so the
# contrast does not exist there and they skip.
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5

# (regime id, op, params, kind, measured speedup per arch)
_REGIMES = [
    ("helps_wt64", "eltwise", (2048,), "helps", {"wh": 6.19, "bh": 7.22}),
    # verdict differs between boards: WH B0 runs 0.89-0.99x, BH p150b 1.18x
    ("measure_wt2", "eltwise", (64,), "measure", {"wh": (0.89, 0.99), "bh": 1.18}),
    # one Tensix core either way
    ("does_not_help_wt1", "eltwise", (32,), "does_not_help", {"wh": 1.00, "bh": 1.02}),
    # reduce case (x - mean_W(x)): params = (R_t tile-rows, Wt tiles, split factor k).
    # helps: 1 tile-row on 1 -> 8 Tensix cores, 4 tiles per slice.            BH p150b 2.34-2.36x
    ("reduce_helps_rt1_wt32_k8", "reduce", (1, 32, 8), "helps", {"bh": (2.34, 2.36)}),
    # does not help: 64 tile-rows already on 64 of 130 Tensix cores (DRAM ~60% of its ceiling);
    # doubling to 128 Tensix cores (8 tiles per slice) buys nothing.               BH p150b 0.98x
    ("reduce_does_not_help_rt64_wt16_k2", "reduce", (64, 16, 2), "does_not_help", {"bh": 0.98}),
    # the same with 4-tile slices sits on the edge.                       BH p150b 1.03-1.04x
    ("reduce_measure_rt64_wt8_k2", "reduce", (64, 8, 2), "measure", {"bh": (1.03, 1.04)}),
    # reduce case on Layout::ROW_MAJOR sticks: params = (R_t tile-rows, Wt tiles, k); stick = Wt * 64 B,
    # each Tensix core reads / writes 32 pieces of stick / k B per tile-row whatever k is.
    # helps: 1 tile-row of 4096 B sticks on 1 -> 4 Tensix cores, 1024 B pieces.   BH p150b 1.86-1.88x
    ("reduce_rm_helps_rt1_w2048_k4", "reduce_rm", (1, 64, 4), "helps", {"bh": (1.86, 1.88)}),
    # does not help: 64 tile-rows of 512 B sticks (2048 x 256) on 64 -> 128 Tensix cores, 256 B
    # pieces: 2x the DRAM transfers, ~1 ns each across the grid.                  BH p150b 0.65x
    ("reduce_rm_does_not_help_rt64_w256_k2", "reduce_rm", (64, 8, 2), "does_not_help", {"bh": 0.65}),
    # 4096 B sticks with the tile-rows on half the grid: 2048 B pieces, at the edge. BH p150b 1.04-1.08x
    ("reduce_rm_measure_rt64_w2048_k2", "reduce_rm", (64, 64, 2), "measure", {"bh": (1.04, 1.08)}),
]


@pytest.mark.parametrize("regime,op,params,kind,measured", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_width_split_regimes(device, regime, op, params, kind, measured):
    """Baseline vs width split where splitting pays and where it does not. Checks correctness of
    both, then checks the speedup against the bound derived from the measured number (check_regime).
      eltwise: single_core vs width_split (relu, H=32, W=params[0]).
      reduce:  row_split_reduce vs width_split_reduce (x - mean_W(x), R_t x Wt, split factor k).
      reduce_rm: the same on Layout::ROW_MAJOR sticks (stick pieces read, tilize, untilize, pieces written)."""
    if op == "eltwise":
        (w,) = params
        tt_input = _make_input(device, w, dtype=ttnn.bfloat16)
        expected = torch.relu(ttnn.to_torch(tt_input).to(torch.float32))
        for variant in VARIANTS:
            out = ttnn.to_torch(width_split(tt_input, variant=variant)).to(torch.float32)
            assert_with_pcc(expected, out, 0.9999)
        run_base = lambda: width_split(tt_input, variant="single_core")
        run_lever = lambda: width_split(tt_input, variant="width_split")
        lever_cores = _active_cores(device, "width_split", w)
        detail = f"W={w} Wt={w // TILE} | single_core -> width_split on {lever_cores} Tensix cores"
    else:
        rows_t, wt, k = params
        base_cores = reduce_active_cores(device, rows_t, wt, "row_split_reduce", 1)
        lever_cores = reduce_active_cores(device, rows_t, wt, "width_split_reduce", k)
        if lever_cores <= base_cores:
            x, y = grid_xy(device)
            pytest.skip(
                f"{regime}: on this {x}x{y} grid both variants run on {base_cores} Tensix cores, so the "
                f"baseline-vs-width-split contrast this regime measures (more Tensix cores per tile-row) does not exist"
            )
        layout = ttnn.ROW_MAJOR_LAYOUT if op == "reduce_rm" else ttnn.TILE_LAYOUT
        tt_input, expected = _make_reduce_input(device, rows_t, wt, layout=layout)
        for variant in REDUCE_VARIANTS:
            _check_reduce(expected, ttnn.to_torch(width_split_reduce(tt_input, variant=variant, k=k)).to(torch.float32))
        run_base = lambda: width_split_reduce(tt_input, variant="row_split_reduce")
        run_lever = lambda: width_split_reduce(tt_input, variant="width_split_reduce", k=k)
        stick = f"Layout::ROW_MAJOR stick={wt * 64}B piece={wt * 64 // k}B " if op == "reduce_rm" else ""
        detail = (
            f"{stick}"
            f"R_t={rows_t} Wt={wt} k={k} tiles/slice={wt // k} | row_split_reduce -> width_split_reduce on "
            f"{base_cores} -> {lever_cores} Tensix cores"
        )

    def measure():
        samples = interleaved_ns(
            device,
            {"base": run_base, "lever": run_lever},
            trials=_N_TRIALS,
            launches=_LAUNCHES_PER_TRIAL,
            warmup=N_WARMUP,
        )
        return samples["base"], samples["lever"]

    check_regime(device, f"width_split {regime}", kind, measured, measure, detail=detail)


def test_width_split_reduce_rejects_oversized_tile_row(device, expect_error):
    """A `Layout::TILE` row whose circular buffers do not fit the per-Tensix-core L1 budget is rejected with
    a ValueError by validate_reduce(), before the output tensor or the program is built. Wt = 220 tiles at
    k = 1 needs 2 * 220 + 2 * 4 + 2 + 3 = 453 tiles = 927 744 B, over the 921 600 B budget."""
    wt = 220
    assert not reduce_fits_l1(wt, 1, False)
    assert reduce_cb_bytes(wt, 1, False) == 453 * 2048
    tt_input, _ = _make_reduce_input(device, 1, wt)
    with expect_error(ValueError, "L1 budget"):
        width_split_reduce(tt_input, variant="row_split_reduce")
