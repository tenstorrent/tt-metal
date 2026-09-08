# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled program-cache benchmark over distinct matmul programs.

This is intentionally one pytest test so the standard ``device`` fixture owns device setup and
teardown.  The timed region contains only matmul calls and device synchronization; inputs are
created beforehand.  Run each condition in a fresh process and select it with
``L1_BENCH_PROGRAM_CACHE={on,off}``.
"""

import json
import os
import statistics
import time

import pytest
import torch
import ttnn


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


@pytest.mark.parametrize("num_cases", [int(os.environ.get("L1_BENCH_NUM_CASES", "50"))])
def test_distinct_program_cache_benchmark(device, num_cases):
    cache_mode = os.environ.get("L1_BENCH_PROGRAM_CACHE")
    assert cache_mode in {"on", "off"}, "set L1_BENCH_PROGRAM_CACHE=on or off"
    assert "TT_METAL_CCACHE_KERNEL_SUPPORT" not in os.environ, (
        "Kernel ccache must be disabled by unsetting TT_METAL_CCACHE_KERNEL_SUPPORT; " "its value is ignored"
    )

    if cache_mode == "on":
        device.enable_program_cache()
        device.clear_program_cache()
    else:
        device.disable_and_clear_program_cache()

    # Matmul's program key includes both input tensor specs, so unique K dimensions guarantee
    # unique program hashes. Alternating formats adds a second source of variation and resembles
    # a heterogeneous golden-test matrix.
    cases = [
        {
            "index": index,
            "shape_a": (1, 1, 32, 32 * (index + 1)),
            "shape_b": (1, 1, 32 * (index + 1), 32),
            "dtype": ttnn.bfloat16 if index % 2 == 0 else ttnn.bfloat8_b,
            "dtype_name": "bfloat16" if index % 2 == 0 else "bfloat8_b",
        }
        for index in range(num_cases)
    ]

    inputs = []
    setup_started = time.perf_counter()
    for case in cases:
        inputs.append(
            (
                ttnn.from_torch(
                    torch.randn(case["shape_a"], dtype=torch.bfloat16),
                    dtype=case["dtype"],
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    torch.randn(case["shape_b"], dtype=torch.bfloat16),
                    dtype=case["dtype"],
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            )
        )
    ttnn.synchronize_device(device)
    setup_seconds = time.perf_counter() - setup_started

    entries_before = device.num_program_cache_entries()
    elapsed = []
    suite_started = time.perf_counter()
    for case, (input_a, input_b) in zip(cases, inputs):
        started = time.perf_counter()
        output = ttnn.matmul(input_a, input_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        seconds = time.perf_counter() - started
        elapsed.append(seconds)
        entries = device.num_program_cache_entries()
        print(
            "L1_BENCH_CASE "
            + json.dumps(
                {
                    "index": case["index"],
                    "shape_a": case["shape_a"],
                    "shape_b": case["shape_b"],
                    "dtype": case["dtype_name"],
                    "seconds": seconds,
                    "program_cache_entries": entries,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        ttnn.deallocate(output)
    ttnn.synchronize_device(device)
    suite_seconds = time.perf_counter() - suite_started
    entries_after = device.num_program_cache_entries()

    if cache_mode == "on":
        assert entries_after - entries_before == num_cases, (
            f"expected {num_cases} distinct matmul program hashes, got "
            f"{entries_after - entries_before} new cache entries"
        )
    else:
        assert entries_after == 0, f"program cache disabled but contains {entries_after} entries"

    summary = {
        "program_cache": cache_mode,
        "num_cases": num_cases,
        "kernel_ccache_env_present": "TT_METAL_CCACHE_KERNEL_SUPPORT" in os.environ,
        "tt_metal_cache": os.environ.get("TT_METAL_CACHE"),
        "setup_seconds": setup_seconds,
        "suite_seconds": suite_seconds,
        "mean_case_seconds": statistics.mean(elapsed),
        "median_case_seconds": statistics.median(elapsed),
        "p90_case_seconds": _percentile(elapsed, 0.90),
        "min_case_seconds": min(elapsed),
        "max_case_seconds": max(elapsed),
        "program_cache_entries_before": entries_before,
        "program_cache_entries_after": entries_after,
        "new_program_cache_entries": entries_after - entries_before,
    }
    print("L1_BENCH_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
