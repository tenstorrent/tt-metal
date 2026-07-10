# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and device profiling for all-reduce accumulation strategies."""

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

from ttnn.operations.examples.tensix_all_reduce_compute import (
    VARIANTS,
    create_sharded_memory_config,
    reduce_blocks,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _selected_variants():
    selected = tuple(os.environ.get("ARC_COMPUTE_VARIANTS", ",".join(VARIANTS)).split(","))
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown ARC_COMPUTE_VARIANTS: {sorted(unknown)}")
    return selected


def _make_input(device, num_blocks, num_tiles):
    width = num_blocks * num_tiles * ttnn.TILE_SIZE
    values = (torch.arange(ttnn.TILE_SIZE * width, dtype=torch.float32).reshape(ttnn.TILE_SIZE, width) % 29) / 128.0
    values += torch.arange(num_blocks, dtype=torch.float32).repeat_interleave(num_tiles * ttnn.TILE_SIZE).reshape(1, -1)
    quantized = values.to(torch.bfloat16).to(torch.float32)
    expected = sum(
        quantized[:, block * num_tiles * ttnn.TILE_SIZE : (block + 1) * num_tiles * ttnn.TILE_SIZE]
        for block in range(num_blocks)
    )
    tt_input = ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(num_blocks * num_tiles),
    )
    return tt_input, expected


def _run_checked(tt_input, expected, variant, num_blocks, num_tiles, kernel_iters):
    output = reduce_blocks(
        tt_input,
        variant=variant,
        num_blocks=num_blocks,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    actual = ttnn.to_torch(output).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.5)
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


def _format_report(results, *, box, arch, num_tiles, trials, kernel_iters):
    lines = [
        "# All-reduce compute accumulation report",
        "",
        f"box={box}  arch={arch}  N={trials} (median)  kernel-iters={kernel_iters}  num-tiles={num_tiles}",
        "",
        "| Blocks | Method | Median ns/reduction | Std / median | vs SFPU serial |",
        "|---:|---|---:|---:|---:|",
    ]
    for num_blocks, samples in results:
        baseline_values = samples.get("sfpu_serial_bf16") or next(iter(samples.values()))
        baseline = statistics.median(baseline_values)
        for variant, values in samples.items():
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            lines.append(
                f"| {num_blocks} | {variant} | {median:.1f} | {std / median * 100:.1f}% | {baseline / median:.2f}x |"
            )
    return "\n".join(lines) + "\n"


def test_tensix_all_reduce_compute_correctness(device):
    num_tiles = int(os.environ.get("ARC_COMPUTE_TILES", "6"))
    for num_blocks in (3, 8):
        tt_input, expected = _make_input(device, num_blocks, num_tiles)
        for variant in _selected_variants():
            _run_checked(tt_input, expected, variant, num_blocks, num_tiles, 2)


def test_tensix_all_reduce_compute_device_perf(device):
    num_tiles = int(os.environ.get("ARC_COMPUTE_TILES", "6"))
    num_blocks_values = tuple(int(value) for value in os.environ.get("ARC_COMPUTE_BLOCKS", "2,4,8,16").split(","))
    kernel_iters = int(os.environ.get("ARC_COMPUTE_KERNEL_ITERS", "100"))
    trials = int(os.environ.get("ARC_COMPUTE_TRIALS", "5"))
    variants = _selected_variants()
    results = []

    for num_blocks in num_blocks_values:
        tt_input, expected = _make_input(device, num_blocks, num_tiles)
        for variant in variants:
            _run_checked(tt_input, expected, variant, num_blocks, num_tiles, 1)
        runners = {
            variant: lambda variant=variant: reduce_blocks(
                tt_input,
                variant=variant,
                num_blocks=num_blocks,
                num_tiles=num_tiles,
                kernel_iters=kernel_iters,
            )
            for variant in variants
        }
        results.append((num_blocks, _measure(device, runners, trials, kernel_iters)))

    report = _format_report(
        results,
        box=socket.gethostname(),
        arch=os.environ.get("ARCH_NAME", str(device.arch())),
        num_tiles=num_tiles,
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if report_path := os.environ.get("ARC_COMPUTE_REPORT"):
        Path(report_path).write_text(report)
