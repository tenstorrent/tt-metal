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

    # write split: every k-th output page written on the read NoC (correctness, then the measured sweep)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_write_split_perf

    # swap_columns: reads NoC1 / writes NoC0 on the N DRAM-nearest worker columns only (correctness, then the sweep)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_swap_columns_correctness
    scripts/run_safe_pytest.sh --run-all \\
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

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand(SHAPE, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _read_kernel_ns(device):
    """Sum of on-device kernel duration over programs since the last read (flush-consumes)."""
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


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("noc", NOCS)
def test_noc_placement_correctness(device, noc, placement):
    """op="copy" is a pure identity copy: output must equal input for every placement/NoC."""
    tt_input = _make_input(device)
    expected = ttnn.to_torch(tt_input).to(torch.float32)
    out = ttnn.to_torch(noc_placement(tt_input, op="copy", noc=noc, placement=placement, num_cores=NUM_CORES)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, 0.9999)


def test_noc_placement_device_perf(device):
    """Measure device kernel duration for the 12-cell read/write × NoC × placement matrix."""
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
    tt_input = _make_input(device)
    noc_placement(tt_input, op=op, noc=noc, placement=placement, num_cores=NUM_CORES, kernel_iters=1, block=BLOCK)
    ttnn.synchronize_device(device)


# ---------------------------------------------------------------------------------------------
# When it helps / when it does not (numbers: WH B0 n150 and BH p150b in README.md). Each regime compares a BASELINE cell to a
# LEVER cell of the identity copy (op="copy": reads on `noc`, writes on the other NoC), measured
# interleaved (A,B,A,B...) so drift hits both. One trial = mean of _REGIME_PER_TRIAL launches;
# the reported number is the median over _REGIME_TRIALS trials.
#
#   placement lever  : column -> row (same NoC pairing, reads NoC0 / writes NoC1)
#   NoC lever        : reads NoC1 / writes NoC0 -> reads NoC0 / writes NoC1 (same placement)
#   write-split lever: all writes on NoC1 -> every k-th page's write on NoC0 (reads stay on NoC0;
#                      the lever runs NOC_MODE dynamic, the baseline the usual dedicated mode)
#
# Outstanding traffic on the line = num_cores x block (2 KB pages in flight per barrier per core).
# Placement only pays once that traffic contends on shared NoC links; latency-bound lines
# (block=1, few cores) take the same time wherever they sit. The NoC lever needs a spread line:
# for one core at (0,0) the canonical pairing is slightly SLOWER than the reverse. The write split
# only pays while the DRAM write stream is below the DRAM ceiling; at the ceiling it is flat, and
# wherever the NoC (not DRAM) is the bound it loses (both NoCs are already busy with one stream each).
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


# id: (num_cores (None = whole grid), block, base cell, lever cell, kind, threshold, setup)
#   kind "helps"         -> assert speedup >= threshold  (T_help = 1 + 0.5 * (measured - 1), >= 1.05)
#   kind "does_not_help" -> assert speedup <= threshold  (1.05 for "flat", 1.0 for a measured loss)
# Thresholds hold on every measured board (the line regimes: WH B0 n150, BH p150b; the grid, write-split and
# swap_columns regimes: BH p150b only); numbers per board in README.md.
_NP_REGIMES = {
    # 8 cores x 16 pages in flight = 128 outstanding 2 KB reads: column saturates its shared links.
    # WH 2.91x, BH 2.24x
    "placement_helps_8cores_block16": (8, 16, _c("column", "noc0"), _c("row", "noc0"), "helps", 1.62, _LINE_SETUP),
    # 2 cores x 1 page in flight = 2 outstanding reads: latency-bound, no link contention to remove.
    # WH 1.03x, BH 1.01x
    "placement_flat_2cores_block1": (2, 1, _c("column", "noc0"), _c("row", "noc0"), "does_not_help", 1.05, _LINE_SETUP),
    # Spread 8-core row, bandwidth-bound: reads NoC0 disperse, reads NoC1 concentrate on DRAM columns.
    # WH 3.27x, BH 2.45x
    "noc_helps_row_8cores_block16": (8, 16, _c("row", "noc1"), _c("row", "noc0"), "helps", 1.72, _LINE_SETUP),
    # A single core at (0,0): no line to disperse; the canonical pairing is within a few % either way.
    # WH 0.94x, BH p150b 1.02x (12x10 grid board) and 1.056x (13x10 grid board) -> bound 1.08, not 1.05
    "noc_flat_1core_block16": (1, 16, _c("row", "noc1"), _c("row", "noc0"), "does_not_help", 1.08, _LINE_SETUP),
    # The whole worker grid, 32 MB DRAM -> DRAM copy: the swapped pairing still costs at the DRAM ceiling.
    # BH 1.21x
    "noc_helps_grid_dram": (
        None,
        16,
        _c(GRID, "noc1"),
        _c(GRID, "noc0"),
        "helps",
        1.10,
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
        1.05,
        ((2048, 4096), 1, "l1", "dram"),
    ),
    # Whole grid, 32 MB DRAM -> DRAM: the unsplit copy is already at the DRAM ceiling (~390 GB/s). BH 1.01-1.02x
    "write_split_flat_grid_dram_half": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", 2),
        "does_not_help",
        1.05,
        ((4096, 4096), 1, "dram", "dram"),
    ),
    # Whole grid, writes to L1: the NoC is the bound and both NoCs already carry one stream each. BH 0.88x
    "write_split_loses_grid_dram_to_l1_half": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", 2),
        "does_not_help",
        1.0,
        ((2048, 4096), 1, "dram", "l1"),
    ),
    # 8-core row, DRAM -> DRAM: far from the DRAM ceiling, half the writes join the reads on NoC0. BH 0.78-0.81x
    "write_split_loses_row8_half": (8, 16, _c("row", "noc0"), _c("row", "noc0", 2), "does_not_help", 1.0, _LINE_SETUP),
    # swap_columns=3: the 3 worker columns nearest the DRAM column next to bank 0 (queried from the device)
    # read on NoC1 / write on NoC0, the rest keep reads NoC0 / writes NoC1. Whole grid, 8 MB read+write stream,
    # 15 tiles per Tensix core: the usual pairing runs at 341 GB/s, below the DRAM ceiling. BH 1.15x
    "swap3_helps_grid_8mb": (
        None,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        1.07,
        ((1024, 2048), 1, "dram", "dram"),
    ),
    # Same lever, partial grid: 65 Tensix cores (the first 5 rows), 64 MB read+write, 357 GB/s unswapped. BH 1.11x
    "swap3_helps_grid65_64mb": (
        65,
        16,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        1.05,
        ((4096, 4096), 1, "dram", "dram"),
    ),
    # Same lever, whole grid, 34 MB read+write, 64 pages per barrier (the deep-issue form of the 34 MB copy;
    # at 16 pages per barrier the same copy only gains 1.03x). BH 1.07x; bound at the 1.05 floor.
    "swap3_helps_grid_34mb_block64": (
        None,
        64,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "helps",
        1.05,
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
        1.05,
        ((10752, 4096), 1, "dram", "dram"),
    ),
    # Same lever, whole grid, 34 MB read+write, 1 page per barrier: latency-bound. BH 1.00x
    "swap3_flat_grid_34mb_block1": (
        None,
        1,
        _c(GRID, "noc0"),
        _c(GRID, "noc0", swap_columns=3),
        "does_not_help",
        1.05,
        ((2048, 4096), 1, "dram", "dram"),
    ),
}


def _interleaved_trials(device, run_a, run_b, trials, per_trial):
    """A,B,A,B... trials; each trial = mean device-kernel ns over `per_trial` launches."""
    for _ in range(N_WARMUP):
        run_a()
        run_b()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    a_ns, b_ns = [], []
    for _ in range(trials):
        for samples, run in ((a_ns, run_a), (b_ns, run_b)):
            for _ in range(per_trial):
                run()
            total = _read_kernel_ns(device)
            assert total is not None, "profiler produced no data (is the build profiler-enabled?)"
            samples.append(total / per_trial)
    return a_ns, b_ns


def _cell_str(cell):
    pl, noc, split, mode, swap = cell
    return f"{pl}, reads {noc}, write_split={split}, {mode}, swap_columns={swap}"


@pytest.mark.parametrize("regime", list(_NP_REGIMES), ids=list(_NP_REGIMES))
def test_noc_placement_regimes(device, regime):
    """Placement / NoC / write-split lever: regimes where each clearly helps, and where it does not."""
    import statistics

    num_cores, block, cell_a, cell_b, kind, threshold, (shape, kernel_iters, in_mem, out_mem) = _NP_REGIMES[regime]
    tt_input = _make_tensor(device, shape, in_mem)
    grid = device.compute_with_storage_grid_size()
    if num_cores is None:
        num_cores = grid.x * grid.y

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

    base, lever = _interleaved_trials(
        device, lambda: run(cell_a), lambda: run(cell_b), _REGIME_TRIALS, _REGIME_PER_TRIAL
    )
    base_med, lever_med = statistics.median(base), statistics.median(lever)
    speedup = base_med / lever_med
    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    logger.info(
        f"\n=== noc_placement regime {regime} ({device.arch()}, CHIP_FREQ={_chip_freq_mhz()} MHz) ==="
        f"\n    shape={shape} ({tiles} tiles, {tiles // num_cores} tiles/core)  cores={num_cores}  {in_mem}->{out_mem}  "
        f"block={block} (outstanding 2 KB reads = {num_cores * block})  kernel_iters={kernel_iters}"
        f"\n    baseline ({_cell_str(cell_a)}): median {base_med:10.0f} ns  [min {min(base):.0f} - max {max(base):.0f}]"
        f"\n    lever    ({_cell_str(cell_b)}): median {lever_med:10.0f} ns  [min {min(lever):.0f} - max {max(lever):.0f}]"
        f"\n    speedup = {speedup:.3f}x  ({_REGIME_TRIALS} trials x {_REGIME_PER_TRIAL} launches)  kind={kind}  threshold={threshold}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.3f}x"


# ---------------------------------------------------------------------------------------------
# Write split: on the identity copy (reads on `noc`, writes on the other NoC), move every k-th
# output page's write onto the READ NoC (write_split=k: 1/k of the writes change NoC; k=1 = all of
# them, i.e. reads and writes share one NoC). Reads never change.
#
# Methods, measured interleaved within each case (rotated order per trial, so clock drift hits all):
#   base       reads NoC0 / writes NoC1, NOC_MODE dedicated  (the usual pairing; the baseline)
#   dynamic    same pairing, NOC_MODE dynamic, no split      (the control: what the mode alone costs)
#   split4     dynamic, every 4th page written on NoC0       (1/4 of the writes on the other NoC)
#   split2     dynamic, every 2nd page written on NoC0       (1/2)
#   split1     dynamic, every page written on NoC0           (all writes share the read NoC)
#   swapped    reads NoC1 / writes NoC0, dedicated           (the reversed pairing)
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

# id: (placement, num_cores (None = every Tensix core of the grid), shape, block, kernel_iters, input mem, output mem)
_WS_CASES = {
    # the full worker grid, DRAM -> DRAM: DRAM is the ceiling. Three footprints, because the
    # unsplit copy sits AT the ceiling for 32 MB but below it for 64 / 128 MB.
    "grid_dram_to_dram_32mb": (GRID, None, (4096, 4096), 16, 1, _DRAM, _DRAM),
    "grid_dram_to_dram_64mb": (GRID, None, (8192, 4096), 16, 1, _DRAM, _DRAM),
    "grid_dram_to_dram_128mb": (GRID, None, (16384, 4096), 16, 1, _DRAM, _DRAM),
    "grid64_dram_to_dram_32mb": (GRID, 64, (4096, 4096), 16, 1, _DRAM, _DRAM),
    # full grid, only the writes touch DRAM (input already in L1)
    "grid_l1_to_dram": (GRID, None, (2048, 4096), 16, 1, _L1, _DRAM),
    # full grid, writes land in Tensix-core L1 (no DRAM on the write side)
    "grid_dram_to_l1": (GRID, None, (2048, 4096), 16, 1, _DRAM, _L1),
    # full grid, L1 -> L1: no DRAM anywhere, the NoC is the only shared resource
    "grid_l1_to_l1": (GRID, None, (2048, 4096), 16, 1, _L1, _L1),
    # few Tensix cores: per-core NoC issue / latency, not DRAM, bounds the copy
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


def _chip_freq_mhz():
    """AICLK the profiler stamped in its device log header (the log this process just wrote)."""
    home = os.environ.get("TT_METAL_HOME", os.getcwd())
    path = os.path.join(home, "generated", "profiler", ".logs", "profile_log_device.csv")
    try:
        with open(path) as f:
            header = f.readline()
    except OSError:
        return None
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            return int(float(part.split(":")[1]))
    return None


def _make_tensor(device, shape, mem):
    torch.manual_seed(0)
    return ttnn.from_torch(
        torch.rand(shape, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_MEM[mem],
    )


def _rotated_trials(device, runs, trials, per_trial):
    """Every method once per trial, the starting method rotating each trial; one sample = mean
    device-kernel ns over `per_trial` launches. Returns {method: [ns, ...]}."""
    names = list(runs)
    for _ in range(N_WARMUP):
        for name in names:
            runs[name]()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush warmup window
    samples = {name: [] for name in names}
    for t in range(trials):
        for name in names[t % len(names) :] + names[: t % len(names)]:
            for _ in range(per_trial):
                runs[name]()
            total = _read_kernel_ns(device)
            assert total is not None, "profiler produced no data (is the build profiler-enabled?)"
            samples[name].append(total / per_trial)
    return samples


@pytest.mark.parametrize("write_split", [1, 2, 4])
@pytest.mark.parametrize("noc", NOCS)
@pytest.mark.parametrize("placement,num_cores,out_mem", [("row", 8, _DRAM), (GRID, None, _DRAM), (GRID, None, _L1)])
def test_noc_placement_write_split_correctness(device, placement, num_cores, out_mem, noc, write_split):
    """A split write stream is still the identical identity copy (bitwise), on either pairing."""
    tt_input = _make_tensor(device, (1024, 2048), _DRAM)
    expected = ttnn.to_torch(tt_input)
    for mode, split in (("dynamic", 0), ("dynamic", write_split)):
        out = noc_placement(
            tt_input,
            op="copy",
            noc=noc,
            placement=placement,
            num_cores=num_cores,
            memory_config=_MEM[out_mem],
            write_split=split,
            noc_mode=mode,
        )
        assert torch.equal(ttnn.to_torch(out), expected), f"copy write_split={split} ({mode}, reads {noc}) differs"


@pytest.mark.parametrize("case", list(_WS_CASES), ids=list(_WS_CASES))
def test_noc_placement_write_split_perf(device, case):
    """Measure the write split (and the swapped pairing) against the usual pairing. Correctness of
    every method is asserted; the timings are reported, never asserted (see the regimes test)."""
    import socket
    import statistics

    placement, num_cores, shape, block, kernel_iters, in_mem, out_mem = _WS_CASES[case]
    tt_input = _make_tensor(device, shape, in_mem)
    expected = ttnn.to_torch(tt_input)
    grid = device.compute_with_storage_grid_size()
    cores = num_cores if num_cores is not None else (grid.x * grid.y if placement == GRID else min(grid.x, grid.y))

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

    samples = _rotated_trials(device, runs, _WS_TRIALS, _WS_PER_TRIAL)
    med = {m: statistics.median(v) for m, v in samples.items()}
    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    nbytes = tiles * 2048 * kernel_iters * 2  # read + write bytes per launch
    freq = _chip_freq_mhz()
    head = (
        f"{case}: placement={placement} cores={cores} (grid {grid.x}x{grid.y}) shape={shape} ({tiles} tiles, "
        f"{tiles // cores} tiles/core) block={block} kernel_iters={kernel_iters} {in_mem}->{out_mem}  "
        f"N={_WS_TRIALS} rotated trials x {_WS_PER_TRIAL} launches (median)  CHIP_FREQ={freq} MHz"
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
# or from the nearest DRAM column ("every", e<N>). Same kernels, same per-core pages, NOC_MODE dedicated.
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

# id: (placement, num_cores (None = whole grid), shape, block, kernel_iters, input mem, output mem)
# The id's size is the whole stream: read + write bytes per launch.
_SW_CASES = {
    "grid_8mb": (GRID, None, (1024, 2048), 16, 1, _DRAM, _DRAM),
    "grid_32mb": (GRID, None, (2048, 4096), 16, 1, _DRAM, _DRAM),
    "grid_64mb": (GRID, None, (4096, 4096), 16, 1, _DRAM, _DRAM),
    "grid_168mb": (GRID, None, (10752, 4096), 16, 1, _DRAM, _DRAM),
    "grid_168mb_block4": (GRID, None, (10752, 4096), 4, 1, _DRAM, _DRAM),
    "grid_168mb_block32": (GRID, None, (10752, 4096), 32, 1, _DRAM, _DRAM),
    "grid_32mb_block1": (GRID, None, (2048, 4096), 1, 1, _DRAM, _DRAM),
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


@pytest.mark.parametrize("swap_anchor", ["first", "every"])
@pytest.mark.parametrize("swap_columns", [1, 3, 6, "all"])
@pytest.mark.parametrize("placement,num_cores,out_mem", [(GRID, None, _DRAM), (GRID, 65, _DRAM), ("row", 8, _L1)])
def test_noc_placement_swap_columns_correctness(device, placement, num_cores, out_mem, swap_columns, swap_anchor):
    """Swapping the pairing on some columns is still the identical identity copy (bitwise)."""
    tt_input = _make_tensor(device, (1024, 2048), _DRAM)
    expected = ttnn.to_torch(tt_input)
    out = noc_placement(
        tt_input,
        op="copy",
        placement=placement,
        num_cores=num_cores,
        memory_config=_MEM[out_mem],
        swap_columns=swap_columns,
        swap_anchor=swap_anchor,
    )
    assert torch.equal(ttnn.to_torch(out), expected), f"copy swap_columns={swap_columns} ({swap_anchor}) differs"


@pytest.mark.parametrize("case", list(_SW_CASES), ids=list(_SW_CASES))
def test_noc_placement_swap_columns_perf(device, case):
    """Measure swap_columns=N against the usual pairing. Correctness of every method is asserted; the
    timings are reported, never asserted (see the regimes test)."""
    import socket
    import statistics

    from ttnn.operations.examples.noc_placement import dram_adjacent_columns

    placement, num_cores, shape, block, kernel_iters, in_mem, out_mem = _SW_CASES[case]
    tt_input = _make_tensor(device, shape, in_mem)
    expected = ttnn.to_torch(tt_input)
    grid = device.compute_with_storage_grid_size()
    cores = num_cores if num_cores is not None else (grid.x * grid.y if placement == GRID else min(grid.x, grid.y))

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

    samples = _rotated_trials(device, runs, _SW_TRIALS, _WS_PER_TRIAL)
    med = {m: statistics.median(v) for m, v in samples.items()}
    tiles = (shape[0] // TILE) * (shape[1] // TILE)
    nbytes = tiles * 2048 * kernel_iters * 2  # read + write bytes per launch
    order = {a: dram_adjacent_columns(device, a) for a in ("first", "every")}
    head = (
        f"{case}: placement={placement} cores={cores} (grid {grid.x}x{grid.y}) shape={shape} ({tiles} tiles, "
        f"{tiles // cores} tiles/core, {nbytes / 1e6:.0f} MB read+write) block={block} kernel_iters={kernel_iters} "
        f"{in_mem}->{out_mem}  N={_SW_TRIALS} rotated trials x {_WS_PER_TRIAL} launches (median)  "
        f"CHIP_FREQ={_chip_freq_mhz()} MHz\n    column order nearest-DRAM first: first={order['first']} every={order['every']}"
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
