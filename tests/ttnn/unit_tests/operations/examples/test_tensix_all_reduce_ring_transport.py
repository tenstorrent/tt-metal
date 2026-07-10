# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and profiling for ring synchronization/payload ablations."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.tensix_all_reduce_ring_transport import (
    VARIANTS,
    build_group_layout,
    create_sharded_memory_config,
    ring_transport,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _selected_variants():
    selected = tuple(os.environ.get("AR_RING_VARIANTS", ",".join(VARIANTS)).split(","))
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown AR_RING_VARIANTS: {sorted(unknown)}")
    return selected


def _cases(device):
    if raw := os.environ.get("AR_RING_GROUP_SHAPE"):
        rows, cols = (int(part) for part in raw.split(","))
        return [(f"custom_{rows}x{cols}", (rows, cols), int(os.environ.get("AR_RING_NUM_GROUPS", "1")))]
    grid = device.compute_with_storage_grid_size()
    half_row_cols = max(2, grid.x // 2)
    candidates = [
        ("half_rows", (1, half_row_cols), (grid.x // half_row_cols) * grid.y),
        ("whole_rows", (1, grid.x), grid.y),
        ("whole_columns", (grid.y, 1), grid.x),
        ("two_rows", (min(2, grid.y), grid.x), grid.y // min(2, grid.y)),
    ]
    return [case for case in candidates if case[1][0] * case[1][1] >= 2 and case[2] >= 1]


def _make_input(device, group_shape, num_groups, num_tiles):
    layout = build_group_layout(device, group_shape, num_groups)
    shape = (len(layout.active_cores) * ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE)
    values = (torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape) % 37) / 64.0
    tt_input = ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(device, group_shape, num_groups, num_tiles),
    )
    return tt_input, values.to(torch.bfloat16).to(torch.float32)


def _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters):
    output = ring_transport(
        tt_input,
        variant=variant,
        group_shape=group_shape,
        num_groups=num_groups,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    torch.testing.assert_close(ttnn.to_torch(output).to(torch.float32), expected)
    return output


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total = 0.0
    found = False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {variant: [] for variant in runners}
    for trial in range(trials + 1):
        for variant, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None
            if trial:
                samples[variant].append(duration / kernel_iters)
    return samples


def _format_report(results, *, box, arch, trials, kernel_iters):
    lines = [
        "# All-reduce ring transport cost report",
        "",
        f"box={box}  arch={arch}  N={trials} (median)  kernel-iters={kernel_iters}",
        "",
        "| Placement | Group | Groups | Tiles | Payload | NoC0 sem ns | NoC0 payload ns | NoC1 sem ns | NoC1 payload ns | NoC1 / NoC0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, shape, groups, num_tiles, page_bytes, samples in results:
        sync = statistics.median(samples["semaphore_only"]) if "semaphore_only" in samples else None
        payload = statistics.median(samples["payload_ring"]) if "payload_ring" in samples else None
        sync_noc1 = statistics.median(samples["semaphore_only_noc1"]) if "semaphore_only_noc1" in samples else None
        payload_noc1 = statistics.median(samples["payload_ring_noc1"]) if "payload_ring_noc1" in samples else None
        ratio = payload_noc1 / payload if payload is not None and payload_noc1 is not None else None
        sync_text = f"{sync:.1f}" if sync is not None else "n/a"
        payload_text = f"{payload:.1f}" if payload is not None else "n/a"
        sync_noc1_text = f"{sync_noc1:.1f}" if sync_noc1 is not None else "n/a"
        payload_noc1_text = f"{payload_noc1:.1f}" if payload_noc1 is not None else "n/a"
        ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
        lines.append(
            f"| {name} | {shape[0]}x{shape[1]} | {groups} | {num_tiles} | {num_tiles * page_bytes} B | "
            f"{sync_text} | {payload_text} | {sync_noc1_text} | {payload_noc1_text} | {ratio_text} |"
        )
    return "\n".join(lines) + "\n"


def test_tensix_all_reduce_ring_transport_correctness(device):
    group_shape = (1, 3)
    num_groups = 1
    num_tiles = 2
    tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
    for variant in _selected_variants():
        _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, 3)


def test_tensix_all_reduce_ring_transport_device_perf(device):
    trials = int(os.environ.get("AR_RING_TRIALS", "5"))
    kernel_iters = int(os.environ.get("AR_RING_KERNEL_ITERS", "100"))
    num_tiles_values = tuple(int(value) for value in os.environ.get("AR_RING_TILES", "1,6,24").split(","))
    variants = _selected_variants()
    results = []

    for name, group_shape, num_groups in _cases(device):
        for num_tiles in num_tiles_values:
            tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
            for variant in variants:
                _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, 1)
            runners = {
                variant: lambda variant=variant: ring_transport(
                    tt_input,
                    variant=variant,
                    group_shape=group_shape,
                    num_groups=num_groups,
                    num_tiles=num_tiles,
                    kernel_iters=kernel_iters,
                )
                for variant in variants
            }
            results.append(
                (
                    name,
                    group_shape,
                    num_groups,
                    num_tiles,
                    tt_input.buffer_aligned_page_size(),
                    _measure(device, runners, trials, kernel_iters),
                )
            )

    report = _format_report(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("AR_RING_REPORT"):
        Path(report_path).write_text(report)
