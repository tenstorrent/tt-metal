# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `trid_double_issue` barrier-granularity example.

A DRAM reader that issues a block of async reads and then calls
`noc_async_read_barrier()` drains the NoC once per block: when the barrier
returns nothing is in flight, and the next block is not issued until after the
wait. Tagging each block with a transaction id and waiting only on the PREVIOUS
id (`noc_async_read_barrier_with_trid`) keeps >= 1 request on the wire at all
times. See ttnn/ttnn/operations/examples/trid_double_issue/README.md.

    # every variant/block/trid-depth is a bitwise-exact copy
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_trid_double_issue.py::test_trid_double_issue_correctness

    # device kernel duration + achieved DRAM GB/s, block x trid-depth (in-process profiler)
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_trid_double_issue.py::test_trid_double_issue_device_perf
"""

import os

# Enable the on-device profiler IN-PROCESS (needs all three, set before the device
# opens). Scoped to this module (not a dir conftest) so it doesn't perturb other
# examples' measurement. setdefault -> respects an outer tracy run if present.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
# The profiler logs a duration histogram per ReadDeviceProfiler at C++ INFO, which
# would bury the report across a variant x block x trid sweep. Gags only the C++
# logger; the loguru report below still prints and the numbers are unaffected.
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import pytest
import torch

import ttnn
from ttnn.operations.examples.trid_double_issue import trid_double_issue, VARIANTS

from loguru import logger


TILE = 32
_DTYPES = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16, "float32": ttnn.float32}

# Defaults are overridable via env so the CLI (python -m ...trid_double_issue) can
# measure the caller's own shape/params through the same in-process path.
SHAPE = tuple(int(x) for x in os.environ.get("TDI_SHAPE", "512,512").split(","))  # 256 tiles
NUM_CORES = int(os.environ.get("TDI_CORES", "1"))  # cores running the copy (each independent)
BLOCK_SWEEP = tuple(int(x) for x in os.environ.get("TDI_BLOCKS", "1,2,4,8,16").split(","))  # reads per barrier
TRID_SWEEP = tuple(int(x) for x in os.environ.get("TDI_TRIDS", "2,3,4").split(","))  # blocks in flight
# Baseline strength: blocks the full_barrier reader issues before its ONE barrier.
# 1 = the idiomatic loop; >1 spends the spare CB and is the strongest a non-trid
# reader can be. Both are reported, so the win is quoted against each.
AHEAD_SWEEP = tuple(int(x) for x in os.environ.get("TDI_AHEAD", "1,2,3,4").split(","))
CB_BLOCKS = int(os.environ.get("TDI_CB_BLOCKS", "6"))  # CB depth in blocks; FIXED across the whole table
KERNEL_ITERS = int(os.environ.get("TDI_ITERS", "1"))  # in-kernel repeat of the page range
DTYPE_NAME = os.environ.get("TDI_DTYPE", "bfloat16")  # transfer size: bfloat8_b | bfloat16 | float32
DTYPE = _DTYPES[DTYPE_NAME]
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("TDI_TRIALS", "10"))

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, dtype=DTYPE):
    torch.manual_seed(0)
    torch_input = torch.rand(SHAPE, dtype=torch.float32) * 2.0 - 1.0
    return ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


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
    measured time. bytes / ns == GB/s. page_bytes is the real per-tile DRAM page."""
    return 2 * num_pages * page_bytes * KERNEL_ITERS / ns_per_op


# (dtype, variant, block, num_trids) — all three transfer sizes x both variants,
# the sub-block remainder path (block=6 does not divide 256), block=1 (one read
# per barrier), and a trid depth deeper than the block count.
_CORRECTNESS_CASES = [
    (name, variant, block, trids)
    for name in _DTYPES
    for variant in VARIANTS
    for (block, trids) in [(1, 2), (6, 3), (8, 2), (8, 5)]
]
# The strong baseline needs its own coverage: issuing `ahead` blocks before one
# barrier walks the CB ring by hand, so a wrap bug shows up as a corrupt copy.
_AHEAD_CASES = [(name, block, ahead) for name in _DTYPES for (block, ahead) in [(1, 4), (4, 2), (6, 4), (8, 6)]]


@pytest.mark.parametrize("dtype_name,block,ahead", _AHEAD_CASES)
def test_trid_double_issue_strong_baseline_correctness(device, dtype_name, block, ahead):
    """The `ahead>1` baseline is still a bitwise-exact copy (exercises its CB wrap)."""
    tt_input = _make_input(device, dtype=_DTYPES[dtype_name])
    expected = ttnn.to_torch(tt_input)
    out = ttnn.to_torch(
        trid_double_issue(
            tt_input,
            variant="full_barrier",
            block=block,
            ahead=ahead,
            cb_blocks=CB_BLOCKS,
            num_cores=NUM_CORES,
            kernel_iters=KERNEL_ITERS,
        )
    )
    assert torch.equal(out, expected), f"copy is not bitwise exact (max |diff| {(out - expected).abs().max()})"


@pytest.mark.parametrize("dtype_name,variant,block,num_trids", _CORRECTNESS_CASES)
def test_trid_double_issue_correctness(device, dtype_name, variant, block, num_trids):
    """Every (dtype, variant, block, trid depth) is the same identity copy, bitwise.

    Two things here can silently corrupt rather than fail loudly, which is why the
    gate is bitwise equality and not a tolerance: the reader's landing-slot
    arithmetic (overwriting a block still in flight) and the writer releasing a CB
    slot on "the write left L1" rather than "the write was acked".
    """
    tt_input = _make_input(device, dtype=_DTYPES[dtype_name])
    expected = ttnn.to_torch(tt_input)

    out = ttnn.to_torch(
        trid_double_issue(
            tt_input,
            variant=variant,
            block=block,
            num_trids=num_trids,
            cb_blocks=CB_BLOCKS,
            num_cores=NUM_CORES,
            kernel_iters=KERNEL_ITERS,
        )
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    # A copy moves bytes; anything short of bitwise equality is a dropped or
    # overwritten tile, which is exactly the bug class the slot arithmetic risks.
    assert torch.equal(out, expected), f"copy is not bitwise exact (max |diff| {(out - expected).abs().max()})"


def test_trid_double_issue_device_perf(device):
    """Measure device kernel duration + achieved DRAM GB/s over block x trid depth.

    Correctness lives in test_trid_double_issue_correctness; this test only
    measures and reports (perf is evidence, never a pass/fail — the only assertion
    here is that the profiler produced a number).

    The `full_barrier` column is the CONTROL: CB depth, block size, page order,
    cores and the writer are all identical to the trid columns, so it should be
    flat across trid depths.
    """
    tt_input = _make_input(device)
    num_pages = (SHAPE[0] // TILE) * (SHAPE[1] // TILE)
    page_bytes = tt_input.buffer_aligned_page_size()

    def run(variant, block, trids=2, ahead=1):
        return _measure_ns(
            device,
            lambda: trid_double_issue(
                tt_input,
                variant=variant,
                block=block,
                num_trids=trids,
                ahead=ahead,
                cb_blocks=CB_BLOCKS,
                num_cores=NUM_CORES,
                kernel_iters=KERNEL_ITERS,
            ),
        )

    base, trid = {}, {}
    for block in BLOCK_SWEEP:
        for ahead in AHEAD_SWEEP:
            value = run("full_barrier", block, ahead=ahead)
            assert value is not None, f"profiler produced no data for full_barrier block={block} ahead={ahead}"
            base[(block, ahead)] = value
        for trids in TRID_SWEEP:
            value = run("trid_double_issue", block, trids=trids)
            assert value is not None, f"profiler produced no data for trids={trids} block={block}"
            trid[(block, trids)] = value

    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== trid_double_issue device perf (DRAM identity copy) — barrier granularity ===",
        f"    shape={SHAPE}  tiles={num_pages}  dtype={DTYPE_NAME}  tile_bytes={page_bytes}  cores={NUM_CORES}",
        f"    cb_blocks={CB_BLOCKS} (FIXED: same CB for every cell)  arch={arch}  iters={KERNEL_ITERS}  trials={N_PROFILE_ITERS}",
        f"    DRAM traffic = read+write = {2 * num_pages * page_bytes / 1e6:.2f} MB/launch; GB/s = traffic / kernel_ns",
        "",
        "    'base ahead=N' issues N blocks then ONE barrier, pushing them individually — ahead=1 is the",
        "    idiomatic loop, ahead>1 spends the spare CB and is the STRONGEST a non-trid reader can be.",
        "    in-flight = reads outstanding at the issue peak.  slots = CB slots the reader reserves at",
        f"    that peak, out of the {CB_BLOCKS} ALLOCATED in every cell — equal slots means equal L1 occupancy,",
        "    so a row-to-row ratio is only iso-L1 when the two rows show the same slot count.",
        "",
        f"    {'block':>5}  {'method':<16}  {'slots':>5}  {'in-flight':>9}  {'ns/op':>10}  {'GB/s':>7}  {'vs naive':>9}  {'vs best base':>12}",
    ]
    for block in BLOCK_SWEEP:
        naive = base[(block, AHEAD_SWEEP[0])]
        best_base = min(base[(block, a)] for a in AHEAD_SWEEP)
        best_ahead = min(AHEAD_SWEEP, key=lambda a: base[(block, a)])
        for ahead in AHEAD_SWEEP:
            v = base[(block, ahead)]
            tag = "  <- best base" if ahead == best_ahead else ""
            lines.append(
                f"    {block:>5}  {'base ahead=' + str(ahead):<16}  {ahead:>5}  {block * ahead:>9}  {v:>10.1f}  "
                f"{_gbps(v, num_pages, page_bytes):>7.1f}  {naive / v:>8.2f}x  {best_base / v:>11.2f}x{tag}"
            )
        for trids in TRID_SWEEP:
            v = trid[(block, trids)]
            iso = "  = iso-L1 vs base" if trids in AHEAD_SWEEP else ""
            lines.append(
                f"    {block:>5}  {'trid x' + str(trids):<16}  {trids:>5}  {block * trids:>9}  {v:>10.1f}  "
                f"{_gbps(v, num_pages, page_bytes):>7.1f}  {naive / v:>8.2f}x  {best_base / v:>11.2f}x{iso}"
            )
        lines.append("")
    logger.info("\n".join(lines))
