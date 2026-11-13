# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `noc_placement` data-movement example (placement × NoC × operation).

See ttnn/ttnn/operations/examples/noc_placement/README.md for what this illustrates.

    # every copy placement/NoC produces the identical, correct identity copy
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_correctness

    # measured device kernel duration for the 12-cell read/write × NoC × placement matrix
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_device_perf

    # when placement / NoC choice helps and when it does not interleaved A/B medians
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes

    # one op launch per matrix cell (NoC-trace driver for tt-npe; used by noc_report.py)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_matrix

    # write split: every k-th output page (one 2 KB bf16 tile) written on the read NoC
    # (correctness, then the measured sweep; the sweep is opt-in: NP_SWEEPS=1 or EXAMPLES_SWEEPS=1)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_correctness
    NP_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_perf

    # swap_columns: reads NoC1 / writes NoC0 on the N DRAM-nearest worker columns only
    # (correctness, then the measured sweep; the sweep is opt-in like the write-split one)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_correctness
    NP_SWEEPS=1 scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_perf
"""

import json
import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")  # silence the per-read profiler histograms

import pytest
import torch

import ttnn
from ttnn.operations.examples.noc_placement import noc_placement, PLACEMENTS, NOCS, GRID

from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    check_regime,
    chip_freq_mhz,
    grid_cores,
    grid_xy,
    interleaved_ns,
    read_kernel_ns,
    require_cores,
    require_grid,
    sweep_gate,
)

TILE = 32
# Overridable via env so the CLI (python -m ...noc_placement) measures the caller's params.
SHAPE = tuple(int(x) for x in os.environ.get("NP_SHAPE", "1024,2048").split(","))
NUM_CORES = int(os.environ.get("NP_CORES", "8"))
BLOCK = int(os.environ.get("NP_BLOCK", "16"))
KERNEL_ITERS = int(os.environ.get("NP_ITERS", "8"))  # in-kernel repeat for steady-state throughput
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("NP_PROFILE_ITERS", "20"))
_NS_OUT = os.environ.get("NP_NS_OUT")  # if set, device_perf dumps {cell: ns} JSON here (for noc_report.py)

# The read/write matrix, in a FIXED order so the NoC-trace op-id sequence maps to cells.
BENCH_MATRIX = [(op, noc, pl) for noc in NOCS for op in ("read", "write") for pl in PLACEMENTS]
BENCH_IDS = [f"{op}_{noc}_{pl}" for (op, noc, pl) in BENCH_MATRIX]
# The identity copy, canonical pairing (reads NoC0 / writes NoC1), for the placement story.
COPY_MATRIX = [("copy", "noc0", pl) for pl in PLACEMENTS]


def _require_placement(device, placement, num_cores, what):
    """Skip (never fail) when `num_cores` Tensix cores of `placement` do not fit this grid. noc_placement caps
    a line ("row", "column", "diagonal") at min(grid columns, grid rows) so a diagonal of that length fits:
    a line of N Tensix cores needs N columns AND N rows. "grid" needs N Tensix cores in total. None = the
    whole grid or the longest line, which always fits."""
    if num_cores is None:
        return
    if placement == GRID:
        require_cores(device, num_cores, what=what)
    else:
        require_grid(device, num_cores, num_cores, what=what)


def _make_input(device):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand(SHAPE, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _measure_ns(device, run_fn):
    """Average ns/op: warm up, flush, run N, read, divide by N."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # flush the warmup window
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    total_ns = read_kernel_ns(device)
    return total_ns / N_PROFILE_ITERS if total_ns is not None else None


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("noc", NOCS)
def test_noc_placement_correctness(device, noc, placement):
    """op="copy" is a pure identity copy: output must equal input for every placement/NoC."""
    _require_placement(device, placement, NUM_CORES, what=f"a {placement} line of NP_CORES={NUM_CORES} Tensix cores")
    tt_input = _make_input(device)
    expected = ttnn.to_torch(tt_input).to(torch.float32)
    out = ttnn.to_torch(noc_placement(tt_input, op="copy", noc=noc, placement=placement, num_cores=NUM_CORES)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)


def test_noc_placement_device_perf(device):
    """Measure device kernel duration for the 12-cell read/write × NoC × placement matrix."""
    _require_placement(device, "diagonal", NUM_CORES, what=f"lines of NP_CORES={NUM_CORES} Tensix cores")
    tt_input = _make_input(device)
    ns = {}
    for op, noc, pl in BENCH_MATRIX + COPY_MATRIX:
        run_fn = lambda o=op, n=noc, p=pl: noc_placement(
            tt_input, op=o, noc=n, placement=p, num_cores=NUM_CORES, kernel_iters=KERNEL_ITERS, block=BLOCK
        )
        value = _measure_ns(device, run_fn)
        assert value is not None, f"profiler produced no data for {op}_{noc}_{pl} (is the build profiler-enabled?)"
        ns[f"{op}_{noc}_{pl}"] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== noc_placement device perf (isolated read/write + copy; NoC × placement) ===",
        f"    shape={SHAPE}  cores={NUM_CORES}  arch={arch}  block={BLOCK}  kernel_iters={KERNEL_ITERS}",
        f"    {'op':<6} {'noc':<5} {'placement':<10} {'ns/op':>12}",
    ]
    for op, noc, pl in BENCH_MATRIX + COPY_MATRIX:
        lines.append(f"    {op:<6} {noc:<5} {pl:<10} {ns[f'{op}_{noc}_{pl}']:>12.1f}")
    logger.info("\n".join(lines))

    if _NS_OUT:
        with open(_NS_OUT, "w") as f:
            json.dump(ns, f)
        logger.info(f"noc_placement: wrote device-ns matrix to {_NS_OUT}")


@pytest.mark.parametrize("op,noc,placement", BENCH_MATRIX, ids=BENCH_IDS)
def test_noc_placement_matrix(device, op, noc, placement):
    """One op launch per matrix cell (kernel_iters=1) -> a small, comparable NoC trace per cell.
    Run under tools/tracy/profile_this.py --collect-noc-traces; consumed by noc_report.py."""
    _require_placement(device, placement, NUM_CORES, what=f"a {placement} line of NP_CORES={NUM_CORES} Tensix cores")
    tt_input = _make_input(device)
    noc_placement(tt_input, op=op, noc=noc, placement=placement, num_cores=NUM_CORES, kernel_iters=1, block=BLOCK)
    ttnn.synchronize_device(device)


# ---------------------------------------------------------------------------------------------
# When it helps / when it does not (numbers: WH B0 n150 and BH p150b in README.md and report.md). Each
# regime compares a BASELINE cell to a LEVER cell of the identity copy (op="copy": reads on `noc`, writes
# on the other NoC), measured with perf_test_utils.interleaved_ns (baseline and lever alternate, the
# starting one rotating per round, so drift hits both). One trial = mean of _REGIME_PER_TRIAL launches;
# the speedup is baseline median / lever median over _REGIME_TRIALS trials per variant.
#
#   placement lever  : column -> row (same NoC pairing, reads NoC0 / writes NoC1)
#   NoC lever        : reads NoC1 / writes NoC0 -> reads NoC0 / writes NoC1 (same placement)
#   write-split lever: all writes on NoC1 -> every k-th page's write on NoC0 (reads stay on NoC0;
#                      the lever runs NOC_MODE dynamic, the baseline the usual dedicated mode)
#   column-swap lever: swap_columns=3 (reads NoC1 / writes NoC0 on the 3 DRAM-nearest worker columns)
#
# Outstanding traffic on the line = Tensix cores x block (pages, each one 2 KB bf16 tile, in flight per
# barrier per Tensix core). Placement only pays once that traffic contends on shared NoC links;
# latency-bound lines (block=1, few Tensix cores) take the same time wherever they sit. The NoC lever needs
# a spread line: for one Tensix core at (0,0) the canonical pairing is slightly SLOWER than the reverse on
# WH B0. The write split only pays while the DRAM write stream is below the DRAM ceiling; at the ceiling
# it is flat, and wherever the NoC (not DRAM) is the bound it loses (both NoCs are already busy with one
# stream each).
#
# The assert: `measured` holds the speedup measured per architecture ("wh" Wormhole B0, "bh" Blackhole; a
# (low, high) pair where two boards or runs differ). perf_test_utils.check_regime derives the bound from the
# number measured on THIS architecture (perf_test_utils.regime_bound: "helps" asserts at least the lower of
# half the measured gain and the measured low less a 10% noise margin, and only logs a gain too small for
# that bound to reach 1.02x; "does_not_help" asserts at most the measured high plus the margin, never below
# 1.0x), re-measures once before failing, and only logs on an architecture with no measured number. The line regimes were measured on WH B0 n150 and BH p150b; the full-grid,
# write-split and swap_columns regimes on BH p150b only, so they log on Wormhole.
#
# Grid: a regime with a fixed Tensix-core count is skipped on a grid that lacks it (a line of N Tensix
# cores needs N columns and N rows; 65 Tensix cores is the first 5 rows of the 13x10 BH p150b grid). It is
# never shrunk to fit: the measured number belongs to that count.
# ---------------------------------------------------------------------------------------------
_REGIME_TRIALS = int(os.environ.get("NP_REGIME_TRIALS", "12"))
_REGIME_PER_TRIAL = 3
_REGIME_SHAPE = (1024, 2048)  # 2048 bf16 tiles = 4 MB
_REGIME_ITERS = 2  # in-kernel repeats of the page range
_LINE_SETUP = (_REGIME_SHAPE, _REGIME_ITERS, "dram", "dram")  # (shape, kernel_iters, input mem, output mem)


def _c(placement, noc, write_split=0, noc_mode=None, swap_columns=0):
    """A copy cell: placement, read NoC (writes on the other one), write split, NOC_MODE, and the number
    of DRAM-nearest worker columns (ranked from the DRAM column next to bank 0) that swap the pairing."""
    return (placement, noc, write_split, noc_mode or ("dynamic" if write_split else "dedicated"), swap_columns)


# id: (num_cores (None = whole grid), block, base cell, lever cell, kind, measured, setup)
#   measured: {arch: speedup} from the comment above each entry (README.md / report.md hold the stamped runs)
_NP_REGIMES = {
    # 8 Tensix cores x 16 pages (2 KB tiles) in flight = 128 outstanding reads: column saturates its shared links.
    # WH 2.91x, BH 2.24x (12x10 grid board) and 1.80x (13x10 grid board)
    "placement_helps_8cores_block16": (
        8,
        16,
        _c("column", "noc0"),
        _c("row", "noc0"),
        "helps",
        {"wh": 2.91, "bh": (1.80, 2.24)},
        _LINE_SETUP,
    ),
    # 2 Tensix cores x 1 page in flight = 2 outstanding reads: latency-bound, no link contention to remove.
    # WH 1.03x, BH 1.01x (12x10 grid board) and 1.00x (13x10 grid board)
    "placement_flat_2cores_block1": (
        2,
        1,
        _c("column", "noc0"),
        _c("row", "noc0"),
        "does_not_help",
        {"wh": 1.03, "bh": (1.00, 1.01)},
        _LINE_SETUP,
    ),
    # Spread row of 8 Tensix cores, bandwidth-bound: reads NoC0 disperse, reads NoC1 concentrate on DRAM columns.
    # WH 3.27x, BH 2.45x (12x10 grid board) and 1.97x (13x10 grid board)
    "noc_helps_row_8cores_block16": (
        8,
        16,
        _c("row", "noc1"),
        _c("row", "noc0"),
        "helps",
        {"wh": 3.27, "bh": (1.97, 2.45)},
        _LINE_SETUP,
    ),
    # A single Tensix core at (0,0): no line to disperse; the canonical pairing is within a few % either way.
    # WH 0.94x, BH p150b 1.02x (12x10 grid board) and 1.056x (13x10 grid board)
    "noc_flat_1core_block16": (
        1,
        16,
        _c("row", "noc1"),
        _c("row", "noc0"),
        "does_not_help",
        {"wh": 0.94, "bh": (1.02, 1.056)},
        _LINE_SETUP,
    ),
    # The whole worker grid, 32 MB DRAM -> DRAM copy: the swapped pairing still costs at the DRAM ceiling.
    # BH 1.21x
    "noc_helps_grid_dram": (
        None,
        16,
        _c(GRID, "noc1"),
        _c(GRID, "noc0"),
        "helps",
        {"bh": 1.21},
        ((4096, 4096), 1, "dram", "dram"),
    ),
    # Whole grid, input in L1, writes to DRAM: the NoC1 write stream (277 GB/s) is below the DRAM ceiling,
    # and a quarter of it on NoC0 lifts it. BH 1.10x
    "write_split_helps_grid_l1_to_dram_quarter": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", 4),
        "helps",
        {"bh": 1.10},
        ((2048, 4096), 1, "l1", "dram"),
    ),
    # Whole grid, 32 MB DRAM -> DRAM: the unsplit copy is already at the DRAM ceiling (~390 GB/s). BH 1.01-1.02x
    "write_split_flat_grid_dram_half": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", 2),
        "does_not_help",
        {"bh": (1.01, 1.02)},
        ((4096, 4096), 1, "dram", "dram"),
    ),
    # Whole grid, writes to L1: the NoC is the bound and both NoCs already carry one stream each. BH 0.88x
    "write_split_loses_grid_dram_to_l1_half": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", 2),
        "does_not_help",
        {"bh": 0.88},
        ((2048, 4096), 1, "dram", "l1"),
    ),
    # Row of 8 Tensix cores, DRAM -> DRAM: far from the DRAM ceiling, half the writes join the reads on NoC0.
    # BH 0.78-0.81x
    "write_split_loses_row8_half": (
        8,
        16,
        _c("row", "noc0"),
        _c("row", "noc0", 2),
        "does_not_help",
        {"bh": (0.78, 0.81)},
        _LINE_SETUP,
    ),
    # swap_columns=3: the 3 worker columns nearest the DRAM column next to bank 0 (queried from the device)
    # read on NoC1 / write on NoC0, the rest keep reads NoC0 / writes NoC1. Whole grid, 8 MB read+write stream,
    # 15 tiles per Tensix core: the usual pairing runs at 341 GB/s, below the DRAM ceiling. BH 1.15x
    "swap3_helps_grid_8mb": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        {"bh": 1.15},
        ((1024, 2048), 1, "dram", "dram"),
    ),
    # Same lever, partial grid: 65 Tensix cores (the first 5 rows of the 13x10 grid), 64 MB read+write,
    # 357 GB/s unswapped. BH 1.11x. Needs 65 Tensix cores: skipped on Wormhole (at most 8x8 = 64).
    "swap3_helps_grid65_64mb": (
        65,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        {"bh": 1.11},
        ((4096, 4096), 1, "dram", "dram"),
    ),
    # Same lever, whole grid, 34 MB read+write, 64 pages per barrier (the deep-issue form of the 34 MB copy;
    # at 16 pages per barrier the same copy only gains 1.03x). BH 1.07x
    "swap3_helps_grid_34mb_block64": (
        None,
        64,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        {"bh": 1.07},
        ((2048, 4096), 1, "dram", "dram"),
    ),
    # Same lever, whole grid, 176 MB read+write at 16 pages per barrier: the copy sits at the DRAM ceiling
    # (407 GB/s unswapped). BH 1.01x
    "swap3_flat_grid_176mb": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "does_not_help",
        {"bh": 1.01},
        ((10752, 4096), 1, "dram", "dram"),
    ),
    # Same lever, whole grid, 34 MB read+write, 1 page per barrier: latency-bound. BH 1.00x
    "swap3_flat_grid_34mb_block1": (
        None,
        1,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "does_not_help",
        {"bh": 1.00},
        ((2048, 4096), 1, "dram", "dram"),
    ),
}


def _cell_str(cell):
    pl, noc, split, mode, swap = cell
    return f"{pl}, reads {noc}, write_split={split}, {mode}, swap_columns={swap}"


@pytest.mark.parametrize("regime", list(_NP_REGIMES), ids=list(_NP_REGIMES))
def test_noc_placement_regimes(device, regime):
    """Placement / NoC / write-split / column-swap lever: regimes where each clearly helps, and where it does not."""
    num_cores, block, cell_a, cell_b, kind, measured, (shape, kernel_iters, in_mem, out_mem) = _NP_REGIMES[regime]
    for cell in (cell_a, cell_b):
        _require_placement(device, cell[0], num_cores, what=f"regime {regime} ({cell[0]} placement)")
    if num_cores is None:
        num_cores = grid_cores(device)
    tt_input = _make_tensor(device, shape, in_mem)

    def run(cell, iters=kernel_iters):
        pl, noc, split, mode, swap = cell
        return noc_placement(
            tt_input,
            op="copy",
            noc=noc,
            placement=pl,
            num_cores=num_cores,
            block=block,
            kernel_iters=iters,
            memory_config=_MEM[out_mem],
            write_split=split,
            noc_mode=mode,
            swap_columns=swap,
        )

    # Correctness: both cells are the identical identity copy.
    expected = ttnn.to_torch(tt_input)
    for cell in (cell_a, cell_b):
        out = ttnn.to_torch(run(cell, iters=1))
        assert torch.equal(out, expected), f"{regime}: copy ({_cell_str(cell)}) is not an identity copy"

    def measure():
        samples = interleaved_ns(
            device,
            {"base": lambda: run(cell_a), "lever": lambda: run(cell_b)},
            trials=_REGIME_TRIALS,
            launches=_REGIME_PER_TRIAL,
        )
        return samples["base"], samples["lever"]

    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    check_regime(
        device,
        regime,
        kind,
        measured,
        measure,
        detail=(
            f"shape={shape} ({tiles} bf16 tiles, {tiles // num_cores} tiles per Tensix core) on {num_cores} Tensix "
            f"cores, {in_mem}->{out_mem}, block={block} ({num_cores * block} outstanding 2 KB reads), "
            f"kernel_iters={kernel_iters}, {_REGIME_PER_TRIAL} launches per trial | baseline ({_cell_str(cell_a)}) "
            f"-> lever ({_cell_str(cell_b)})"
        ),
    )


# ---------------------------------------------------------------------------------------------
# Write split: on the identity copy (reads on `noc`, writes on the other NoC), move every k-th
# output page's write (one 2 KB bf16 tile per page) onto the READ NoC (write_split=k: 1/k of the writes
# change NoC; k=1 = all of them, i.e. reads and writes share one NoC). Reads never change.
#
# Methods, measured with perf_test_utils.interleaved_ns within each case (every method once per trial,
# the starting method rotating per trial, so clock drift hits all):
#   base       reads NoC0 / writes NoC1, NOC_MODE dedicated  (the usual pairing; the baseline)
#   dynamic    same pairing, NOC_MODE dynamic, no split      (the control: what the mode alone costs)
#   split4     dynamic, every 4th page written on NoC0       (1/4 of the writes on the other NoC)
#   split2     dynamic, every 2nd page written on NoC0       (1/2)
#   split1     dynamic, every page written on NoC0           (all writes share the read NoC)
#   swapped    reads NoC1 / writes NoC0, dedicated           (the reversed pairing)
#
# The two measurement sweeps (write split here, swap_columns below) only log a table and are opt-in:
# NP_SWEEPS=1 or EXAMPLES_SWEEPS=1 (the CLI's --write-split / --swap-columns set NP_SWEEPS=1). Their
# correctness tests run by default.
# ---------------------------------------------------------------------------------------------
_WS_METHODS = {
    "base": dict(noc="noc0", write_split=0, noc_mode="dedicated"),
    "dynamic": dict(noc="noc0", write_split=0, noc_mode="dynamic"),
    "split4": dict(noc="noc0", write_split=4, noc_mode="dynamic"),
    "split2": dict(noc="noc0", write_split=2, noc_mode="dynamic"),
    "split1": dict(noc="noc0", write_split=1, noc_mode="dynamic"),
    "swapped": dict(noc="noc1", write_split=0, noc_mode="dedicated"),
}
_WS_FRACTION = {"base": "0", "dynamic": "0", "split4": "1/4", "split2": "1/2", "split1": "1", "swapped": "all (swap)"}

_DRAM, _L1 = "dram", "l1"
_MEM = {_DRAM: ttnn.DRAM_MEMORY_CONFIG, _L1: ttnn.L1_MEMORY_CONFIG}

# Tensix-core counts of the parametrized cases: None = every Tensix core of the grid ("grid") or the
# longest line, an int = exactly that many (the case is skipped on a grid that lacks them, see
# _require_placement), or one of these two, derived from the attached grid so the case runs on every board:
_LINE8 = "line8"  # a line of 8 Tensix cores, or the longest line the grid fits when that is shorter (8x7: 7)
_PARTIAL = "partial"  # half the rows plus half a row of the grid: a partial "grid" that ends mid-row


def _case_cores(device, placement, spec, what):
    """Resolve a case's Tensix-core count on this grid (skips the test when a literal count does not fit)."""
    x, y = grid_xy(device)
    if spec is None:
        return x * y if placement == GRID else min(x, y)
    if spec == _LINE8:
        return min(8, x, y)
    if spec == _PARTIAL:
        return x * (y // 2) + x // 2
    _require_placement(device, placement, spec, what=what)
    return spec


# id: (placement, num_cores (see _case_cores), shape, block, kernel_iters, input mem, output mem)
_WS_CASES = {
    # the full worker grid, DRAM -> DRAM: DRAM is the ceiling. Three footprints, because the
    # unsplit copy sits AT the ceiling for 32 MB but below it for 64 / 128 MB.
    "grid_dram_to_dram_32mb": (GRID, None, (4096, 4096), 16, 1, _DRAM, _DRAM),
    "grid_dram_to_dram_64mb": (GRID, None, (8192, 4096), 16, 1, _DRAM, _DRAM),
    "grid_dram_to_dram_128mb": (GRID, None, (16384, 4096), 16, 1, _DRAM, _DRAM),
    # the first 64 Tensix cores of the grid: skipped on a grid with fewer (an 8x7 Wormhole grid has 56)
    "grid64_dram_to_dram_32mb": (GRID, 64, (4096, 4096), 16, 1, _DRAM, _DRAM),
    # full grid, only the writes touch DRAM (input already in L1)
    "grid_l1_to_dram": (GRID, None, (2048, 4096), 16, 1, _L1, _DRAM),
    # full grid, writes land in Tensix-core L1 (no DRAM on the write side)
    "grid_dram_to_l1": (GRID, None, (2048, 4096), 16, 1, _DRAM, _L1),
    # full grid, L1 -> L1: no DRAM anywhere, the NoC is the only shared resource
    "grid_l1_to_l1": (GRID, None, (2048, 4096), 16, 1, _L1, _L1),
    # few Tensix cores: per-core NoC issue / latency, not DRAM, bounds the copy (a row of 8 Tensix cores
    # needs 8 columns and 8 rows, so it is skipped on an 8x7 grid)
    "row8_dram_to_dram": ("row", 8, (1024, 2048), 16, 2, _DRAM, _DRAM),
    "row2_dram_to_dram": ("row", 2, (1024, 2048), 16, 2, _DRAM, _DRAM),
    "core1_dram_to_dram": ("row", 1, (1024, 2048), 16, 2, _DRAM, _DRAM),
    "core1_block64_dram_to_dram": ("row", 1, (1024, 2048), 64, 2, _DRAM, _DRAM),
}
if os.environ.get("NP_WS_CUSTOM"):  # the CLI's --write-split path: one case from the caller's params
    _WS_CASES = {
        "custom": (
            os.environ.get("NP_PLACEMENT", GRID),
            int(os.environ["NP_CORES"]) if os.environ.get("NP_CORES") else None,
            SHAPE,
            BLOCK,
            KERNEL_ITERS,
            os.environ.get("NP_IN_MEM", _DRAM),
            os.environ.get("NP_OUT_MEM", _DRAM),
        )
    }
_WS_TRIALS = int(os.environ.get("NP_WS_TRIALS", "5"))
_WS_PER_TRIAL = 3
_WS_REPORT = []  # filled per case, printed once by the last case


def _make_tensor(device, shape, mem):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_MEM[mem],
    )


@pytest.mark.parametrize("write_split", [1, 2, 4])
@pytest.mark.parametrize("noc", NOCS)
@pytest.mark.parametrize(
    "placement,num_cores,out_mem", [("row", _LINE8, _DRAM), (GRID, None, _DRAM), (GRID, None, _L1)]
)
def test_noc_placement_write_split_correctness(device, placement, num_cores, out_mem, noc, write_split):
    """A split write stream is still the identical identity copy (bitwise), on either pairing."""
    cores = _case_cores(device, placement, num_cores, what=f"write split on a {placement} placement")
    tt_input = _make_tensor(device, (1024, 2048), _DRAM)
    expected = ttnn.to_torch(tt_input)
    for mode, split in (("dynamic", 0), ("dynamic", write_split)):
        out = noc_placement(
            tt_input,
            op="copy",
            noc=noc,
            placement=placement,
            num_cores=cores,
            memory_config=_MEM[out_mem],
            write_split=split,
            noc_mode=mode,
        )
        assert torch.equal(ttnn.to_torch(out), expected), f"copy write_split={split} ({mode}, reads {noc}) differs"


@sweep_gate("NP_SWEEPS")
@pytest.mark.parametrize("case", list(_WS_CASES), ids=list(_WS_CASES))
def test_noc_placement_write_split_perf(device, case):
    """Measure the write split (and the swapped pairing) against the usual pairing. Correctness of
    every method is asserted; the timings are reported, never asserted (see the regimes test)."""
    import socket
    import statistics

    placement, num_cores, shape, block, kernel_iters, in_mem, out_mem = _WS_CASES[case]
    cores = _case_cores(device, placement, num_cores, what=f"write-split case {case}")
    tt_input = _make_tensor(device, shape, in_mem)
    expected = ttnn.to_torch(tt_input)
    grid_x, grid_y = grid_xy(device)

    def runner(method):
        kw = _WS_METHODS[method]
        return lambda: noc_placement(
            tt_input,
            op="copy",
            placement=placement,
            num_cores=cores,
            block=block,
            kernel_iters=kernel_iters,
            memory_config=_MEM[out_mem],
            **kw,
        )

    runs = {m: runner(m) for m in _WS_METHODS}
    for m, run in runs.items():
        assert torch.equal(ttnn.to_torch(run()), expected), f"{case}: method {m} is not an identity copy"

    samples = interleaved_ns(device, runs, trials=_WS_TRIALS, launches=_WS_PER_TRIAL, warmup=N_WARMUP)
    med = {m: statistics.median(v) for m, v in samples.items()}
    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    nbytes = tiles * 2048 * kernel_iters * 2  # read + write bytes per launch
    head = (
        f"{case}: placement={placement} on {cores} Tensix cores (grid {grid_x}x{grid_y}) shape={shape} "
        f"({tiles} bf16 tiles, {tiles // cores} tiles per Tensix core) block={block} kernel_iters={kernel_iters} "
        f"{in_mem}->{out_mem}  N={_WS_TRIALS} rotated trials x {_WS_PER_TRIAL} launches (median)  "
        f"CHIP_FREQ={chip_freq_mhz()} MHz"
    )
    lines = [
        head,
        f"    {'method':<8} {'other-NoC writes':>16} {'median ns':>11} {'min-max ns':>21} {'GB/s':>7} {'vs base':>8}",
    ]
    for m in _WS_METHODS:
        lines.append(
            f"    {m:<8} {_WS_FRACTION[m]:>16} {med[m]:>11.0f} {min(samples[m]):>10.0f}-{max(samples[m]):<10.0f} "
            f"{nbytes / med[m]:>7.1f} {med['base'] / med[m]:>7.3f}x"
        )
    _WS_REPORT.append("\n".join(lines))
    logger.info(
        f"\n=== noc_placement write split  box={socket.gethostname()}  arch={device.arch()} ===\n"
        + "\n\n".join(_WS_REPORT)
    )


# ---------------------------------------------------------------------------------------------
# Swap the pairing on the DRAM-adjacent columns only (swap_columns=N): the Tensix cores in the N
# worker-grid columns nearest a DRAM column read on NoC1 / write on NoC0; every other column keeps
# reads on NoC0 / writes on NoC1. The columns come from the device (the logical worker columns it names
# as closest to each DRAM bank), ranked by distance from the DRAM column next to bank 0 ("first", f<N>)
# or from the nearest DRAM column ("every", e<N>). Same kernels, same per-core pages (2 KB bf16 tiles),
# NOC_MODE dedicated. The perf sweep is opt-in (NP_SWEEPS=1, see the write-split block above).
# ---------------------------------------------------------------------------------------------
def _swap_methods():
    methods = {"base": dict(swap_columns=0)}
    for n in (1, 2, 3, 4, 6):
        methods[f"f{n}"] = dict(swap_columns=n, swap_anchor="first")
    for n in (2, 3, 4, 6):  # e1 == f1 (both start at the column next to bank 0)
        methods[f"e{n}"] = dict(swap_columns=n, swap_anchor="every")
    methods["all"] = dict(swap_columns="all")
    return methods


_SW_METHODS = _swap_methods()

# id: (placement, num_cores (see _case_cores), shape, block, kernel_iters, input mem, output mem)
# The id's size is the whole stream: read + write bytes per launch.
_SW_CASES = {
    "grid_8mb": (GRID, None, (1024, 2048), 16, 1, _DRAM, _DRAM),
    "grid_32mb": (GRID, None, (2048, 4096), 16, 1, _DRAM, _DRAM),
    "grid_64mb": (GRID, None, (4096, 4096), 16, 1, _DRAM, _DRAM),
    "grid_168mb": (GRID, None, (10752, 4096), 16, 1, _DRAM, _DRAM),
    "grid_168mb_block4": (GRID, None, (10752, 4096), 4, 1, _DRAM, _DRAM),
    "grid_168mb_block32": (GRID, None, (10752, 4096), 32, 1, _DRAM, _DRAM),
    "grid_32mb_block1": (GRID, None, (2048, 4096), 1, 1, _DRAM, _DRAM),
    # the first 5 rows of the 13x10 BH p150b grid; skipped on a grid with fewer than 65 Tensix cores
    "grid65_64mb": (GRID, 65, (4096, 4096), 16, 1, _DRAM, _DRAM),
    "grid_dram_to_l1_32mb": (GRID, None, (2048, 4096), 16, 1, _DRAM, _L1),
}
if os.environ.get("NP_SWAP_CUSTOM"):  # the CLI's --swap-columns path: one case from the caller's params
    _SW_CASES = {
        "custom": (
            os.environ.get("NP_PLACEMENT", GRID),
            int(os.environ["NP_CORES"]) if os.environ.get("NP_CORES") else None,
            SHAPE,
            BLOCK,
            KERNEL_ITERS,
            os.environ.get("NP_IN_MEM", _DRAM),
            os.environ.get("NP_OUT_MEM", _DRAM),
        )
    }
_SW_TRIALS = int(os.environ.get("NP_SW_TRIALS", "5"))
_SW_REPORT = []


# The partial grid (_PARTIAL) ends mid-row on every board, so the swapped columns hold different numbers of
# Tensix cores; the row (_LINE8) covers only some columns on a wide grid, so a swapped group can be empty.
@pytest.mark.parametrize("swap_anchor", ["first", "every"])
@pytest.mark.parametrize("swap_columns", [1, 3, 6, "all"])
@pytest.mark.parametrize(
    "placement,num_cores,out_mem", [(GRID, None, _DRAM), (GRID, _PARTIAL, _DRAM), ("row", _LINE8, _L1)]
)
def test_noc_placement_swap_columns_correctness(device, placement, num_cores, out_mem, swap_columns, swap_anchor):
    """Swapping the pairing on some columns is still the identical identity copy (bitwise)."""
    cores = _case_cores(device, placement, num_cores, what=f"swap_columns on a {placement} placement")
    tt_input = _make_tensor(device, (1024, 2048), _DRAM)
    expected = ttnn.to_torch(tt_input)
    out = noc_placement(
        tt_input,
        op="copy",
        placement=placement,
        num_cores=cores,
        memory_config=_MEM[out_mem],
        swap_columns=swap_columns,
        swap_anchor=swap_anchor,
    )
    assert torch.equal(ttnn.to_torch(out), expected), f"copy swap_columns={swap_columns} ({swap_anchor}) differs"


@sweep_gate("NP_SWEEPS")
@pytest.mark.parametrize("case", list(_SW_CASES), ids=list(_SW_CASES))
def test_noc_placement_swap_columns_perf(device, case):
    """Measure swap_columns=N against the usual pairing. Correctness of every method is asserted; the
    timings are reported, never asserted (see the regimes test)."""
    import socket
    import statistics

    from ttnn.operations.examples.noc_placement import dram_adjacent_columns

    placement, num_cores, shape, block, kernel_iters, in_mem, out_mem = _SW_CASES[case]
    cores = _case_cores(device, placement, num_cores, what=f"swap_columns case {case}")
    tt_input = _make_tensor(device, shape, in_mem)
    expected = ttnn.to_torch(tt_input)
    grid_x, grid_y = grid_xy(device)

    def runner(method):
        kw = _SW_METHODS[method]
        return lambda: noc_placement(
            tt_input,
            op="copy",
            noc="noc0",
            placement=placement,
            num_cores=cores,
            block=block,
            kernel_iters=kernel_iters,
            memory_config=_MEM[out_mem],
            **kw,
        )

    runs = {m: runner(m) for m in _SW_METHODS}
    for m, run in runs.items():
        assert torch.equal(ttnn.to_torch(run()), expected), f"{case}: method {m} is not an identity copy"

    samples = interleaved_ns(device, runs, trials=_SW_TRIALS, launches=_WS_PER_TRIAL, warmup=N_WARMUP)
    med = {m: statistics.median(v) for m, v in samples.items()}
    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    nbytes = tiles * 2048 * kernel_iters * 2  # read + write bytes per launch
    order = {a: dram_adjacent_columns(device, a) for a in ("first", "every")}
    head = (
        f"{case}: placement={placement} on {cores} Tensix cores (grid {grid_x}x{grid_y}) shape={shape} "
        f"({tiles} bf16 tiles, {tiles // cores} tiles per Tensix core, {nbytes / 1e6:.0f} MB read+write) "
        f"block={block} kernel_iters={kernel_iters} {in_mem}->{out_mem}  N={_SW_TRIALS} rotated trials x "
        f"{_WS_PER_TRIAL} launches (median)  CHIP_FREQ={chip_freq_mhz()} MHz\n"
        f"    column order nearest-DRAM first: first={order['first']} every={order['every']}"
    )
    lines = [
        head,
        f"    {'method':<6} {'swapped cols':<24} {'median ns':>11} {'min-max ns':>21} {'GB/s':>7} {'vs base':>8}",
    ]
    for m, kw in _SW_METHODS.items():
        n = kw["swap_columns"]
        cols = "-" if n == 0 else ("all" if n == "all" else str(sorted(order[kw.get("swap_anchor", "first")][:n])))
        lines.append(
            f"    {m:<6} {cols:<24} {med[m]:>11.0f} {min(samples[m]):>10.0f}-{max(samples[m]):<10.0f} "
            f"{nbytes / med[m]:>7.1f} {med['base'] / med[m]:>7.3f}x"
        )
    _SW_REPORT.append("\n".join(lines))
    logger.info(
        f"\n=== noc_placement swap_columns  box={socket.gethostname()}  arch={device.arch()} ===\n"
        + "\n\n".join(_SW_REPORT)
    )
