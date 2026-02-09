# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test for GitHub issue #14421: DRAM interleaved and L1 interleaved inputs
in the relu op should have different kernel duration times.

L1 interleaved input → L1 interleaved output should be faster than
DRAM interleaved input → L1 interleaved output.
"""

import pytest
import torch
import ttnn


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

    Returns the maximum 'duration' value found across all analysis results
    for the latest profiler read.
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


def _run_relu(device, input_tensor):
    """Run relu and return the result tensor (stays on device)."""
    return ttnn.relu(input_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1024, 1024),
        (1, 4, 2048, 2048),
    ],
    ids=["1x1x1024x1024", "1x4x2048x2048"],
)
def test_relu_l1_input_faster_than_dram_input(shape, monkeypatch):
    """
    Verify that relu with L1 interleaved input has lower kernel duration
    than relu with DRAM interleaved input (both with L1 interleaved output).
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

        warmup_iterations = 2

        # --- Measure DRAM interleaved input → L1 interleaved output ---
        dram_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Warmup
        for _ in range(warmup_iterations):
            out = _run_relu(device, dram_input)
            ttnn.deallocate(out)

        # Measured run — flush profiler state first
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        out = _run_relu(device, dram_input)
        dram_duration = _get_max_kernel_duration(device)
        ttnn.deallocate(out)
        ttnn.deallocate(dram_input)

        # --- Measure L1 interleaved input → L1 interleaved output ---
        l1_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        # Warmup
        for _ in range(warmup_iterations):
            out = _run_relu(device, l1_input)
            ttnn.deallocate(out)

        # Measured run — flush profiler state first
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        out = _run_relu(device, l1_input)
        l1_duration = _get_max_kernel_duration(device)
        ttnn.deallocate(out)
        ttnn.deallocate(l1_input)

    finally:
        ttnn.close_device(device)

    print(f"\nShape: {shape}")
    print(f"  DRAM→L1 kernel duration: {dram_duration} ns")
    print(f"  L1→L1   kernel duration: {l1_duration} ns")
    if dram_duration > 0:
        speedup = dram_duration / l1_duration
        print(f"  Speedup (DRAM/L1): {speedup:.2f}x")

    assert l1_duration < dram_duration, (
        f"Expected L1 interleaved input to be faster than DRAM interleaved input, "
        f"but L1 duration ({l1_duration} ns) >= DRAM duration ({dram_duration} ns). "
        f"See https://github.com/tenstorrent/tt-metal/issues/14421"
    )
