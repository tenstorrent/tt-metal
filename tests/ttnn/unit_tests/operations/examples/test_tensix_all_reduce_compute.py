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


# (id, baseline variant, lever variant, num_blocks, num_tiles, kernel_iters, kind, threshold)
# Thresholds: helps -> T_help = max(1.05, 1 + 0.5 * (measured - 1)); does_not_help -> <= 1.05.
# Measured on Wormhole B0 n150: 8x6 @ ki=100 -> ~5.9x (T=3.9); 2 blocks x 1 tile @ ki=1 -> ~1.38x (T=1.19);
# bf16-DST -> fp32-DST at 16 blocks x 6 tiles -> ~1.03x.
_REGIMES = [
    ("helps_8blocks_6tiles_steady", "sfpu_serial_bf16", "fpu_dest_reuse_fp32", 8, 6, 100, "helps", 3.9),
    ("helps_weakly_2blocks_1tile_single_call", "sfpu_serial_bf16", "fpu_dest_reuse_fp32", 2, 1, 1, "helps", 1.19),
    (
        "does_not_help_dst_precision_16blocks",
        "fpu_dest_reuse_bf16",
        "fpu_dest_reuse_fp32",
        16,
        6,
        100,
        "does_not_help",
        1.05,
    ),
]


@pytest.mark.parametrize(
    "baseline, lever, num_blocks, num_tiles, kernel_iters, kind, threshold",
    [r[1:] for r in _REGIMES],
    ids=[r[0] for r in _REGIMES],
)
def test_tensix_all_reduce_compute_regimes(
    device, baseline, lever, num_blocks, num_tiles, kernel_iters, kind, threshold
):
    """Interleaved A/B (baseline, lever, baseline, ...) of one lever; median of 11 retained trials."""
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
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
    samples = _measure(device, runners, trials, kernel_iters)
    base_ns = statistics.median(samples[baseline])
    lever_ns = statistics.median(samples[lever])
    speedup = base_ns / lever_ns
    logger.info(
        f"REGIME {kind} blocks={num_blocks} tiles/block={num_tiles} kernel_iters={kernel_iters} "
        f"{baseline}={base_ns:.1f}ns [{min(samples[baseline]):.1f}-{max(samples[baseline]):.1f}] "
        f"{lever}={lever_ns:.1f}ns [{min(samples[lever]):.1f}-{max(samples[lever]):.1f}] "
        f"speedup={speedup:.3f}x trials={trials} arch={device.arch()}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"expected <= {threshold}x, got {speedup:.3f}x"
