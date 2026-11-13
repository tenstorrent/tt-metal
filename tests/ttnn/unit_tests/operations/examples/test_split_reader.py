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

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, require_grid

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
# When the split reader helps / when its gain shrinks (numbers below: WH B0 n150 and BH p150b; README.md).
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
# WH B0 and BH: because the reader barriers after every tile, a second issuing RISC-V always
# doubles the reads in flight. The diminished case is therefore a "does_not_help" regime against
# its own measured gain: it asserts the gain does not grow past what was measured (plus the noise
# margin), i.e. stays well below the issue-bound one.
#
# The assert: `measured` is the speedup measured per architecture ("wh" Wormhole B0, "bh" Blackhole).
# perf_test_utils.check_regime derives the bound from the number measured on THIS architecture
# (perf_test_utils.regime_bound: "helps" asserts at least the lower of half the measured gain and the
# measured number less a 10% noise margin; "does_not_help" asserts at most the measured high plus that
# margin), re-measures once before failing, and only logs on an architecture with no measured number.
#
# Sampling stays local (not perf_test_utils.interleaved_ns) because one profiler window also yields the
# NCRISC and BRISC kernel lifetimes, which are logged next to the verdict. A trial = mean per-launch
# durations over _LAUNCHES_PER_TRIAL launches of one variant (one profiler window); baseline and lever
# trials alternate, the starting variant rotating per trial, and check_regime compares the medians of the
# DEVICE KERNEL DURATION samples over _N_TRIALS trials per variant.
#
# Grid: the source shards sit on logical Tensix cores (0,0)..(7,0) and the op needs a second row, so the
# regimes need a grid of at least 8 columns x 2 rows (every Wormhole and Blackhole board has it).
# ---------------------------------------------------------------------------

_N_TRIALS = 12
_LAUNCHES_PER_TRIAL = 5
_REGIME_CORES, _REGIME_TILES_PER_CORE, _REGIME_BLOCK_TILES = 8, 8, 8

# (regime id, transaction_bytes, kind, measured speedup per arch)
_REGIMES = [
    ("helps_issue_bound_32B", 32, "helps", {"wh": 1.84, "bh": 1.90}),
    # stays well below the issue-bound gain
    ("diminished_2048B", 2048, "does_not_help", {"wh": 1.38, "bh": 1.61}),
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


@pytest.mark.parametrize("regime,transaction_bytes,kind,measured", _REGIMES, ids=[r[0] for r in _REGIMES])
def test_split_reader_regimes(device, regime, transaction_bytes, kind, measured):
    """split off (baseline) vs on (lever) where NCRISC is issue-bound and where it is not."""
    require_grid(device, _REGIME_CORES, 2, what=f"split_reader regime {regime} (sources on row 0, a second row)")
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

    def measure():
        for _ in range(N_WARMUP):
            runs[False]()
            runs[True]()
        ttnn.synchronize_device(device)
        _read_kernel_metrics(device)  # flush warmup window
        trials = {False: [], True: []}
        for trial in range(_N_TRIALS):
            for enabled in (False, True) if trial % 2 == 0 else (True, False):
                trials[enabled].append(_trial_metrics(device, runs[enabled]))
        off, on = trials[False], trials[True]
        logger.info(
            f"split_reader {regime} per-RISC-V kernel lifetimes (medians): off NCRISC {_median(off, 'ncrisc'):.1f} ns | "
            f"on NCRISC {_median(on, 'ncrisc'):.1f} ns, BRISC {_median(on, 'brisc'):.1f} ns"
        )
        return [t["device"] for t in off], [t["device"] for t in on]

    reads = _REGIME_CORES * _REGIME_TILES_PER_CORE * (TILE_BYTES // transaction_bytes)
    check_regime(
        device,
        regime,
        kind,
        measured,
        measure,
        detail=(
            f"tx={transaction_bytes} B reads/launch={reads} ({_REGIME_CORES} source Tensix cores, "
            f"{_REGIME_TILES_PER_CORE} tiles per source, block={_REGIME_BLOCK_TILES} tiles), "
            f"{_LAUNCHES_PER_TRIAL} launches per trial | baseline split_reader=False -> lever split_reader=True"
        ),
    )
