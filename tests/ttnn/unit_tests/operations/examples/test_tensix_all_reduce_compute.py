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

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, interleaved_ns
from ttnn.operations.examples.tensix_all_reduce_compute import (
    VARIANTS,
    create_sharded_memory_config,
    reduce_blocks,
)


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


def _measure(device, runners, trials, kernel_iters):
    """{variant: [ns per kernel iteration, ...]}; warm-up and per-launch timing follow `interleaved_ns`."""
    return interleaved_ns(device, runners, trials=trials, per_launch_divisor=kernel_iters)


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


# (id, baseline variant, lever variant, num_blocks, num_tiles, kernel_iters, kind, measured)
# The bound is derived from the per-arch measured speedup by `perf_test_utils.regime_bound` (helps: half the
# measured gain, never closer than the noise margin; does_not_help: at most the measured value plus the noise
# margin, never below 1.0). An architecture without a measured number is logged, not asserted; a missed bound
# is re-measured once and the samples are pooled. `measure` regimes flip between boards and only log -- numbers
# per board in README.md. Every case runs on one Tensix core, so no grid guard is needed.
_REGIMES = [
    (
        "helps_8blocks_6tiles_steady",
        "sfpu_serial_bf16",
        "fpu_dest_reuse_fp32",
        8,
        6,
        100,
        "helps",
        {"wh": 5.89, "bh": 9.37},
    ),
    (
        "helps_weakly_2blocks_1tile_single_call",
        "sfpu_serial_bf16",
        "fpu_dest_reuse_fp32",
        2,
        1,
        1,
        "helps",
        {"wh": 1.38, "bh": 1.30},
    ),
    (
        "does_not_help_dst_precision_16blocks",
        "fpu_dest_reuse_bf16",
        "fpu_dest_reuse_fp32",
        16,
        6,
        100,
        "does_not_help",
        {"wh": 1.03, "bh": 0.94},
    ),
]


@pytest.mark.parametrize(
    "baseline, lever, num_blocks, num_tiles, kernel_iters, kind, measured",
    [r[1:] for r in _REGIMES],
    ids=[r[0] for r in _REGIMES],
)
def test_tensix_all_reduce_compute_regimes(
    device, baseline, lever, num_blocks, num_tiles, kernel_iters, kind, measured, request
):
    """Interleaved A/B (baseline, lever, baseline, ...) of one lever."""
    trials = 11
    tt_input, expected = _make_input(device, num_blocks, num_tiles)
    for variant in (baseline, lever):
        _run_checked(tt_input, expected, variant, num_blocks, num_tiles, 1)
    runners = {
        variant: lambda variant=variant: reduce_blocks(
            tt_input, variant=variant, num_blocks=num_blocks, num_tiles=num_tiles, kernel_iters=kernel_iters
        )
        for variant in (baseline, lever)
    }

    def measure():
        samples = _measure(device, runners, trials, kernel_iters)
        return samples[baseline], samples[lever]

    check_regime(
        device,
        request.node.callspec.id,
        kind,
        measured,
        measure,
        detail=f"{num_blocks} blocks x {num_tiles} tiles, {kernel_iters} kernel iterations per launch",
    )
