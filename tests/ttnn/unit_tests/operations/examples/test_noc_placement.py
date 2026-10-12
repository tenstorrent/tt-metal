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

    # when placement / NoC choice helps and when it does not (WH B0 n150), interleaved A/B medians
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_regimes

    # one op launch per matrix cell (NoC-trace driver for tt-npe; used by noc_report.py)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_noc_placement.py::test_noc_placement_matrix
"""

import json
import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device opens).
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch

import ttnn
from ttnn.operations.examples.noc_placement import noc_placement, PLACEMENTS, NOCS

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
# When it helps / when it does not (Wormhole B0 n150). Each regime compares a BASELINE cell to a
# LEVER cell of the identity copy (op="copy": reads on `noc`, writes on the other NoC), measured
# interleaved (A,B,A,B...) so drift hits both. One trial = mean of _REGIME_PER_TRIAL launches;
# the reported number is the median over _REGIME_TRIALS trials.
#
#   placement lever: column -> row (same NoC pairing, reads NoC0 / writes NoC1)
#   NoC lever      : reads NoC1 / writes NoC0 -> reads NoC0 / writes NoC1 (same row placement)
#
# Outstanding traffic on the line = num_cores x block (2 KB pages in flight per barrier per core).
# Placement only pays once that traffic contends on shared NoC links; latency-bound lines
# (block=1, few cores) take the same time wherever they sit. The NoC lever needs a spread line:
# for one core at (0,0) the canonical pairing is slightly SLOWER than the reverse.
# ---------------------------------------------------------------------------------------------
_REGIME_TRIALS = int(os.environ.get("NP_REGIME_TRIALS", "12"))
_REGIME_PER_TRIAL = 3
_REGIME_SHAPE = (1024, 2048)  # 2048 bf16 tiles = 4 MB
_REGIME_ITERS = 2  # in-kernel repeats of the page range

# id: (num_cores, block, (base_placement, base_noc), (lever_placement, lever_noc), kind, threshold)
#   kind "helps"         -> assert speedup >= threshold  (T_help = 1 + 0.5 * (measured - 1), >= 1.05)
#   kind "does_not_help" -> assert speedup <= threshold  (1.05 for "flat", 1.0 for a measured loss)
_NP_REGIMES = {
    # 8 cores x 16 pages in flight = 128 outstanding 2 KB reads: column saturates its shared links.
    "placement_helps_8cores_block16": (8, 16, ("column", "noc0"), ("row", "noc0"), "helps", 1.95),
    # 2 cores x 1 page in flight = 2 outstanding reads: latency-bound, no link contention to remove.
    "placement_flat_2cores_block1": (2, 1, ("column", "noc0"), ("row", "noc0"), "does_not_help", 1.05),
    # Spread 8-core row, bandwidth-bound: reads NoC0 disperse, reads NoC1 concentrate on DRAM columns.
    "noc_helps_row_8cores_block16": (8, 16, ("row", "noc1"), ("row", "noc0"), "helps", 2.13),
    # A single core at (0,0): no line to disperse; the canonical pairing is a few % slower.
    "noc_loses_1core_block16": (1, 16, ("row", "noc1"), ("row", "noc0"), "does_not_help", 1.0),
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


@pytest.mark.parametrize("regime", list(_NP_REGIMES), ids=list(_NP_REGIMES))
def test_noc_placement_regimes(device, regime):
    """Placement / NoC lever: one regime where it clearly helps, one where it does not (WH B0 n150)."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    import statistics

    num_cores, block, (pl_a, noc_a), (pl_b, noc_b), kind, threshold = _NP_REGIMES[regime]
    torch.manual_seed(0)
    tt_input = ttnn.from_torch(
        torch.rand(_REGIME_SHAPE, dtype=torch.float32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    def run(pl, noc, kernel_iters=_REGIME_ITERS):
        return noc_placement(
            tt_input, op="copy", noc=noc, placement=pl, num_cores=num_cores, block=block, kernel_iters=kernel_iters
        )

    # Correctness: both cells are the identical identity copy.
    expected = ttnn.to_torch(tt_input).to(torch.float32)
    for pl, noc in ((pl_a, noc_a), (pl_b, noc_b)):
        out = ttnn.to_torch(run(pl, noc, kernel_iters=1)).to(torch.float32)
        assert torch.equal(out, expected), f"{regime}: copy ({pl}, {noc}) is not an identity copy"

    base, lever = _interleaved_trials(
        device, lambda: run(pl_a, noc_a), lambda: run(pl_b, noc_b), _REGIME_TRIALS, _REGIME_PER_TRIAL
    )
    base_med, lever_med = statistics.median(base), statistics.median(lever)
    speedup = base_med / lever_med
    tiles = (_REGIME_SHAPE[0] // TILE) * (_REGIME_SHAPE[1] // TILE)
    logger.info(
        f"\n=== noc_placement regime {regime} (WH B0) ==="
        f"\n    shape={_REGIME_SHAPE} ({tiles} tiles, {tiles // num_cores} tiles/core)  cores={num_cores}  "
        f"block={block} (outstanding 2 KB reads on the line = {num_cores * block})  kernel_iters={_REGIME_ITERS}"
        f"\n    baseline ({pl_a}, reads {noc_a}): median {base_med:10.0f} ns  [min {min(base):.0f} - max {max(base):.0f}]"
        f"\n    lever    ({pl_b}, reads {noc_b}): median {lever_med:10.0f} ns  [min {min(lever):.0f} - max {max(lever):.0f}]"
        f"\n    speedup = {speedup:.2f}x  ({_REGIME_TRIALS} trials x {_REGIME_PER_TRIAL} launches)  kind={kind}  threshold={threshold}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
