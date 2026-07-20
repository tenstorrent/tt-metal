# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling for rectangular Tensix all-reduce."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.tensix_all_reduce import (
    VARIANTS,
    all_reduce,
    build_group_layout,
    create_sharded_memory_config,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _selected_variants():
    selected = tuple(part for part in os.environ.get("AR_VARIANTS", ",".join(VARIANTS)).split(",") if part)
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown AR_VARIANTS: {sorted(unknown)}")
    return selected


def _custom_case():
    raw = os.environ.get("AR_GROUP_SHAPE")
    if raw is None:
        return None
    rows, cols = (int(part) for part in raw.split(","))
    return ("custom", (rows, cols), int(os.environ.get("AR_NUM_GROUPS", "1")))


def _default_perf_cases(device):
    grid = device.compute_with_storage_grid_size()
    half_x = max(2, grid.x // 2)
    half_y = max(2, grid.y // 2)
    candidates = [
        ("whole_rows", (1, grid.x), grid.y),
        ("whole_columns", (grid.y, 1), grid.x),
        ("half_rows", (1, half_x), (grid.x // half_x) * grid.y),
        ("two_rows", (min(2, grid.y), grid.x), max(1, grid.y // min(2, grid.y))),
        ("two_columns", (grid.y, min(2, grid.x)), max(1, grid.x // min(2, grid.x))),
        ("quad", (half_y, half_x), (grid.x // half_x) * (grid.y // half_y)),
    ]
    unique = []
    seen = set()
    for case in candidates:
        _, shape, groups = case
        key = (shape, groups)
        if shape[0] * shape[1] >= 2 and groups >= 1 and key not in seen:
            unique.append(case)
            seen.add(key)
    return unique


def _cases(device):
    custom = _custom_case()
    return [custom] if custom is not None else _default_perf_cases(device)


def _make_input(device, group_shape, num_groups, num_tiles):
    layout = build_group_layout(device, group_shape, num_groups)
    shape = (layout.num_cores * ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE)
    torch_input = torch.empty(shape, dtype=torch.float32)
    shard_index = {core: index for index, core in enumerate(layout.active_cores)}

    element_pattern = (torch.arange(num_tiles * ttnn.TILE_SIZE, dtype=torch.float32) % 17).reshape(1, -1) / 256.0
    for group in layout.groups:
        for member, core in enumerate(group.cores):
            index = shard_index[core]
            value = (group.index + 1) * 0.5 + (member + 1) / 64.0
            torch_input[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] = value + element_pattern

    quantized = torch_input.to(torch.bfloat16).to(torch.float32)
    expected = torch.empty_like(quantized)
    for group in layout.groups:
        indices = [shard_index[core] for core in group.cores]
        reduced = sum(quantized[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] for index in indices)
        for index in indices:
            expected[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] = reduced

    memory_config = create_sharded_memory_config(device, group_shape, num_groups, num_tiles)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return tt_input, expected


def _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters=1):
    output = all_reduce(
        tt_input,
        variant=variant,
        group_shape=group_shape,
        num_groups=num_groups,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    actual = ttnn.to_torch(output).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.125)
    return output


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total = 0.0
    found = False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure_matrix(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)

    samples = {variant: [] for variant in runners}
    for trial in range(trials + 1):
        for variant, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"device profiler produced no duration for {variant}"
            if trial > 0:
                samples[variant].append(duration / kernel_iters)
    return samples


def _format_report(results, *, arch, box, num_tiles, trials, kernel_iters):
    lines = [
        "# Tensix rectangular all-reduce report",
        "",
        f"box={box}  arch={arch}  N={trials} (median)  kernel-iters={kernel_iters}  num-tiles={num_tiles}",
        "",
        "| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs ring push |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for case_name, group_shape, num_groups, case_results in results:
        baseline = case_results.get("ring_push")
        if baseline is None:
            baseline = next(iter(case_results.values()))
        for variant, values in case_results.items():
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            noise = std / median * 100.0 if median else float("nan")
            noise_text = f"{noise:.1f}%" + (" (noisy)" if noise >= 5.0 else "")
            ratio = baseline and statistics.median(baseline) / median
            rows, cols = group_shape
            lines.append(
                f"| {case_name} | {rows}x{cols} | {num_groups} | {rows * cols * num_groups} | "
                f"{variant} | {median:.1f} | {noise_text} | {ratio:.2f}x |"
            )
    return "\n".join(lines) + "\n"


def test_tensix_all_reduce_correctness(device):
    num_tiles = int(os.environ.get("AR_NUM_TILES", "6"))
    kernel_iters = int(os.environ.get("AR_KERNEL_ITERS", "1"))
    for case_name, group_shape, num_groups in _cases(device):
        tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
        for variant in _selected_variants():
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
            logger.info(
                f"PASS case={case_name} group={group_shape[0]}x{group_shape[1]} groups={num_groups} "
                f"tiles={num_tiles} variant={variant}"
            )


def test_tensix_all_reduce_device_perf(device):
    """Correctness-gate every variant, then measure kernel duration; perf never fails directionally."""
    num_tiles = int(os.environ.get("AR_NUM_TILES", "6"))
    kernel_iters = int(os.environ.get("AR_KERNEL_ITERS", "1"))
    trials = int(os.environ.get("AR_TRIALS", "5"))
    variants = _selected_variants()
    results = []

    for case_name, group_shape, num_groups in _cases(device):
        tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
        for variant in variants:
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)

        runners = {
            variant: (
                lambda variant=variant: all_reduce(
                    tt_input,
                    variant=variant,
                    group_shape=group_shape,
                    num_groups=num_groups,
                    num_tiles=num_tiles,
                    kernel_iters=kernel_iters,
                )
            )
            for variant in variants
        }
        samples = _measure_matrix(device, runners, trials, kernel_iters)
        results.append((case_name, group_shape, num_groups, samples))

    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    report = _format_report(
        results,
        arch=arch,
        box=socket.gethostname(),
        num_tiles=num_tiles,
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    report_path = os.environ.get("AR_REPORT")
    if report_path:
        Path(report_path).write_text(report)
