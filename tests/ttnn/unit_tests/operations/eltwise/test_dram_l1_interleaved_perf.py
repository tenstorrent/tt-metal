# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for GitHub issue #14421: DRAM interleaved and L1 interleaved inputs
in the relu op should have different kernel duration times.

Tests all 4 memory layout combinations (input × output):
  1. DRAM → DRAM
  2. DRAM → L1
  3. L1   → DRAM
  4. L1   → L1

Expected ordering: L1→L1 is fastest, DRAM→DRAM is slowest.
"""

import pytest
import torch
import ttnn

MEMORY_CONFIGS = {
    "DRAM": ttnn.DRAM_MEMORY_CONFIG,
    "L1": ttnn.L1_MEMORY_CONFIG,
}


def _enable_profiler(monkeypatch):
    """Enable device profiler via environment variables."""
    for var, value in {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
    }.items():
        monkeypatch.setenv(var, value)


def _check_profiler_available():
    """Skip test if profiler bindings are not available."""
    profiler_mod = getattr(ttnn, "_ttnn", None)
    if profiler_mod is None or not hasattr(profiler_mod, "profiler"):
        pytest.skip("Profiler bindings not available in this build")


def _get_max_kernel_duration(device) -> int:
    """
    Read device profiler and return the max kernel duration (ns) from the
    latest program execution data.
    """
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    latest_data = ttnn.get_latest_programs_perf_data()
    if not latest_data:
        pytest.skip("No profiler data returned")

    device_id = next(iter(latest_data))
    programs = list(latest_data[device_id])
    if not programs:
        pytest.skip("No program entries in profiler data")

    max_duration = 0
    for program in programs:
        for _name, result in program.program_analyses_results.items():
            if result.duration > max_duration:
                max_duration = result.duration

    return max_duration


def _measure_relu(device, input_tensor, output_memory_config, warmup_iterations=2):
    """Run relu with warmup, then return measured kernel duration (ns)."""
    # Warmup
    for _ in range(warmup_iterations):
        out = ttnn.relu(input_tensor, memory_config=output_memory_config)
        ttnn.deallocate(out)

    # Flush profiler state
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)

    # Measured run
    out = ttnn.relu(input_tensor, memory_config=output_memory_config)
    duration = _get_max_kernel_duration(device)
    ttnn.deallocate(out)
    return duration


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1024, 1024),
        (1, 4, 2048, 2048),
    ],
    ids=["1x1x1024x1024", "1x4x2048x2048"],
)
def test_relu_memory_layout_kernel_duration(shape, monkeypatch):
    """
    Measure relu kernel duration for all 4 input/output memory layout
    combinations and verify expected ordering:
      - L1→L1 < DRAM→L1  (L1 input should be faster than DRAM input)
      - L1→L1 < L1→DRAM  (L1 output should be faster than DRAM output)
      - L1→L1 < DRAM→DRAM (fully L1 should be fastest)

    See https://github.com/tenstorrent/tt-metal/issues/14421
    """
    _check_profiler_available()
    _enable_profiler(monkeypatch)

    if ttnn.GetNumAvailableDevices() < 1:
        pytest.skip("No devices available")

    device = ttnn.open_device(device_id=0)
    try:
        try:
            ttnn.get_all_programs_perf_data()
        except RuntimeError as exc:
            if "profiler_state_manager is nullptr" in str(exc):
                pytest.skip("Profiler state manager not initialized")
            raise

        torch.manual_seed(0)
        torch_input = torch.randn(shape, dtype=torch.bfloat16)

        durations = {}
        for in_name, in_mem_config in MEMORY_CONFIGS.items():
            input_tensor = ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=in_mem_config,
            )
            for out_name, out_mem_config in MEMORY_CONFIGS.items():
                label = f"{in_name}→{out_name}"
                durations[label] = _measure_relu(device, input_tensor, out_mem_config)

            ttnn.deallocate(input_tensor)
    finally:
        ttnn.close_device(device)

    # Print results
    print(f"\nShape: {shape}")
    for label, duration in durations.items():
        print(f"  {label:12s} kernel duration: {duration} ns")
    if durations["DRAM→DRAM"] > 0:
        baseline = durations["DRAM→DRAM"]
        for label, duration in durations.items():
            print(f"  {label:12s} relative to DRAM→DRAM: {duration / baseline:.2f}x")

    # Assertions: L1→L1 should be strictly faster than the other 3 combinations
    assert durations["L1→L1"] < durations["DRAM→L1"], (
        f"Expected L1→L1 ({durations['L1→L1']} ns) < DRAM→L1 ({durations['DRAM→L1']} ns). "
        f"See https://github.com/tenstorrent/tt-metal/issues/14421"
    )
    assert durations["L1→L1"] < durations["L1→DRAM"], (
        f"Expected L1→L1 ({durations['L1→L1']} ns) < L1→DRAM ({durations['L1→DRAM']} ns). "
        f"See https://github.com/tenstorrent/tt-metal/issues/14421"
    )
    assert durations["L1→L1"] < durations["DRAM→DRAM"], (
        f"Expected L1→L1 ({durations['L1→L1']} ns) < DRAM→DRAM ({durations['DRAM→DRAM']} ns). "
        f"See https://github.com/tenstorrent/tt-metal/issues/14421"
    )
