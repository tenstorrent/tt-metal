# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and on-device performance tests for the split-reader example."""

import os

# The profiler must be enabled before the device fixture opens the device.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

from loguru import logger
import pytest
import torch

import ttnn
from ttnn.operations.examples.split_reader import make_row_sharded_memory_config, row_gather_copy

TILE = 32
TILE_BYTES = TILE * TILE * 2
NUM_CORES = int(os.environ.get("SR_CORES", "8"))
TILES_PER_CORE = int(os.environ.get("SR_TILES_PER_CORE", "8"))
BLOCK_TILES = int(os.environ.get("SR_BLOCK_TILES", "8"))
TRANSACTION_BYTES = int(os.environ.get("SR_TRANSACTION_BYTES", "64"))
PERF_TRANSACTION_BYTES = [
    int(value) for value in os.environ.get("SR_TRANSACTION_BYTES_LIST", str(TRANSACTION_BYTES)).split(",")
]
N_WARMUP = 5
N_PROFILE_ITERS = int(os.environ.get("SR_ITERS", "20"))
_DURATION_KEYS = {
    "device": "DEVICE KERNEL DURATION [ns]",
    "brisc": "DEVICE BRISC KERNEL DURATION [ns]",
    "ncrisc": "DEVICE NCRISC KERNEL DURATION [ns]",
}


def _make_input(device):
    torch.manual_seed(0)
    shape = (NUM_CORES * TILES_PER_CORE * TILE, TILE)
    torch_input = torch.rand(shape, dtype=torch.float32)
    memory_config = make_row_sharded_memory_config(shape, NUM_CORES)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return torch_input, tt_input


def _expected(torch_input):
    return torch_input.to(torch.bfloat16).to(torch.float32)


def _read_kernel_metrics(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    totals = {name: 0.0 for name in _DURATION_KEYS}
    found = {name: False for name in _DURATION_KEYS}
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            for name, profiler_key in _DURATION_KEYS.items():
                entry = results.get(profiler_key)
                if entry is not None:
                    totals[name] += float(entry.duration)
                    found[name] = True
    return {name: totals[name] if found[name] else None for name in _DURATION_KEYS}


def _measure_kernel_metrics(device, run_fn):
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_metrics(device)
    for _ in range(N_PROFILE_ITERS):
        run_fn()
    totals = _read_kernel_metrics(device)
    return {name: value / N_PROFILE_ITERS if value is not None else None for name, value in totals.items()}


@pytest.mark.parametrize("transaction_bytes", [32, 64, 256, 2048], ids=lambda value: f"tx{value}")
@pytest.mark.parametrize("enabled", [False, True], ids=["off", "on"])
def test_split_reader_correctness(device, enabled, transaction_bytes):
    torch_input, tt_input = _make_input(device)
    expected = _expected(torch_input)
    tt_output = row_gather_copy(
        tt_input,
        split_reader=enabled,
        num_cores=NUM_CORES,
        block_tiles=BLOCK_TILES,
        transaction_bytes=transaction_bytes,
    )
    assert tt_output.memory_config().buffer_type == ttnn.BufferType.DRAM
    output = ttnn.to_torch(tt_output).to(torch.float32)

    assert list(output.shape) == list(expected.shape)
    assert torch.equal(expected, output)


def test_split_reader_device_perf(device):
    """Report RISC-V kernel lifetimes and end-to-end time; correctness is tested above."""
    _, tt_input = _make_input(device)

    results = {}
    for transaction_bytes in PERF_TRANSACTION_BYTES:
        metrics = {}
        for enabled in (False, True):
            run_fn = lambda value=enabled, tx=transaction_bytes: row_gather_copy(
                tt_input,
                split_reader=value,
                num_cores=NUM_CORES,
                block_tiles=BLOCK_TILES,
                transaction_bytes=tx,
            )
            value = _measure_kernel_metrics(device, run_fn)
            assert all(
                metric is not None for metric in value.values()
            ), "profiler produced incomplete data (is the build profiler-enabled?)"
            metrics[enabled] = value
        results[transaction_bytes] = metrics

    num_blocks = NUM_CORES * TILES_PER_CORE // BLOCK_TILES
    arch = os.environ.get("ARCH_NAME", "unknown")
    lines = [
        "",
        "=== split_reader RISC-V issue scaling (row-sharded L1 gather -> DRAM copy) ===",
        f"    cores={NUM_CORES}  tiles/source={TILES_PER_CORE}  block={BLOCK_TILES} tiles  "
        f"blocks={num_blocks}  arch={arch}",
        "    Per-RISC columns are kernel lifetimes, including NoC barriers and CB waits.",
        f"    {'transaction':>12}  {'NoC reads':>9}  {'off NCRISC':>11}  {'on NCRISC':>10}  "
        f"{'on BRISC':>9}  {'off device':>10}  {'on device':>9}  {'speedup':>8}",
    ]
    total_tiles = NUM_CORES * TILES_PER_CORE
    for transaction_bytes, metrics in results.items():
        reads_per_tile = TILE_BYTES // transaction_bytes
        reads_per_launch = total_tiles * reads_per_tile
        off = metrics[False]
        on = metrics[True]
        speedup = off["device"] / on["device"]
        lines.append(
            f"    {transaction_bytes:>9} B  {reads_per_launch:>9}  {off['ncrisc']:>11.1f}  "
            f"{on['ncrisc']:>10.1f}  {on['brisc']:>9.1f}  {off['device']:>10.1f}  "
            f"{on['device']:>9.1f}  {speedup:>7.2f}x"
        )
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# When the split reader helps / when its gain shrinks (Wormhole B0 n150).
#
# Baseline = split_reader=False (NCRISC issues every read); lever = split_reader=True
# (BRISC also issues the second half of every block). 8 source Tensix cores x 8 tiles,
# 8-tile blocks, 128 KiB payload in both regimes; only the NoC read size changes.
#
#   helps_issue_bound_32B   32 B reads -> 4096 reads/launch, 64 reads per 2 KiB tile.
#                           The single NCRISC is issue-bound (its lifetime scales with
#                           the read count at fixed bytes)                     -> ~1.8x
#   diminished_2048B        2048 B reads -> 64 reads/launch, 1 read per tile. NCRISC is
#                           no longer issue-bound (lifetime flat from 1024 B up); the
#                           remaining gain is only 2 tiles in flight instead of 1 under
#                           the per-tile read barrier, and BRISC (half the reads + all
#                           DRAM writes) becomes the critical RISC-V            -> ~1.35x
#
# No configuration reachable through row_gather_copy's arguments is flat (<= 1.05x) on
# WH B0: because the reader barriers after every tile, a second issuing RISC-V always
# doubles the reads in flight. The diminished case asserts the gain stays well below the
# issue-bound one instead.
#
# A trial = mean per-launch durations over _LAUNCHES_PER_TRIAL launches of one variant
# (one profiler window). Baseline/lever trials are interleaved A,B,A,B and the median
# of _N_TRIALS trials per variant is compared.
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5
_REGIME_CORES, _REGIME_TILES_PER_CORE, _REGIME_BLOCK_TILES = 8, 8, 8

# (regime id, transaction_bytes, kind, threshold). helps -> speedup >= 1 + 0.5*(measured-1);
# diminished -> speedup <= threshold (well below the issue-bound gain).
_REGIMES = [
    ("helps_issue_bound_32B", 32, "helps", 1.4),  # measured ~1.84x
    ("diminished_2048B", 2048, "diminished", 1.50),
]


def _regime_input(device):
    torch.manual_seed(0)
    shape = (_REGIME_CORES * _REGIME_TILES_PER_CORE * TILE, TILE)
    torch_input = torch.rand(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=make_row_sharded_memory_config(shape, _REGIME_CORES),
    )
    return torch_input, tt_input


def _trial_metrics(device, run_fn):
    for _ in range(_LAUNCHES_PER_TRIAL):
        run_fn()
    totals = _read_kernel_metrics(device)
    assert all(v is not None for v in totals.values()), "profiler produced incomplete data"
    return {name: value / _LAUNCHES_PER_TRIAL for name, value in totals.items()}


def _median(trials, key):
    return statistics.median(t[key] for t in trials)


@pytest.mark.parametrize("regime,transaction_bytes,kind,threshold", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_split_reader_regimes(device, regime, transaction_bytes, kind, threshold):
    """split off (baseline) vs on (lever) where NCRISC is issue-bound and where it is not."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    torch_input, tt_input = _regime_input(device)
    expected = _expected(torch_input)
    runs = {
        enabled: (
            lambda e=enabled: row_gather_copy(
                tt_input,
                split_reader=e,
                num_cores=_REGIME_CORES,
                block_tiles=_REGIME_BLOCK_TILES,
                transaction_bytes=transaction_bytes,
            )
        )
        for enabled in (False, True)
    }
    for run_fn in runs.values():
        assert torch.equal(expected, ttnn.to_torch(run_fn()).to(torch.float32))

    for _ in range(N_WARMUP):
        runs[False]()
        runs[True]()
    ttnn.synchronize_device(device)
    _read_kernel_metrics(device)  # flush warmup window
    off, on = [], []
    for _ in range(_N_TRIALS):
        off.append(_trial_metrics(device, runs[False]))
        on.append(_trial_metrics(device, runs[True]))

    b_med, l_med = _median(off, "device"), _median(on, "device")
    speedup = b_med / l_med
    reads = _REGIME_CORES * _REGIME_TILES_PER_CORE * (TILE_BYTES // transaction_bytes)
    b_dev = [t["device"] for t in off]
    l_dev = [t["device"] for t in on]
    logger.info(
        f"[WH_B0] split_reader {regime}: tx={transaction_bytes} B reads/launch={reads} "
        f"(cores={_REGIME_CORES} tiles/core={_REGIME_TILES_PER_CORE} block={_REGIME_BLOCK_TILES}) | "
        f"off median {b_med:.1f} ns (min {min(b_dev):.1f} max {max(b_dev):.1f}; NCRISC {_median(off, 'ncrisc'):.1f}) | "
        f"on median {l_med:.1f} ns (min {min(l_dev):.1f} max {max(l_dev):.1f}; NCRISC {_median(on, 'ncrisc'):.1f} "
        f"BRISC {_median(on, 'brisc'):.1f}) | speedup {speedup:.2f}x | trials {_N_TRIALS}x{_LAUNCHES_PER_TRIAL} launches"
    )
    if kind == "helps":
        assert speedup >= threshold, f"{regime}: expected >= {threshold}x, got {speedup:.2f}x"
    else:
        assert speedup <= threshold, f"{regime}: expected <= {threshold}x, got {speedup:.2f}x"
