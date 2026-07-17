# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and on-device performance tests for the split-reader example."""

import os

# The profiler must be enabled before the device fixture opens the device.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

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
