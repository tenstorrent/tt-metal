# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


@pytest.mark.parametrize(
    "fn_name",
    [
        "GetLatestProgramsPerfData",
        "GetAllProgramsPerfData",
    ],
)
def test_profiler_perf_data_bindings_exist_and_return_mapping(fn_name):
    """
    Sanity check that the Python bindings for program perf data are present and callable.
    The underlying C++ implementation may return an empty mapping if profiling is disabled
    (e.g., TRACY not enabled or mid-run dump not requested), but the call itself should
    succeed and produce a Python mapping.
    """
    profiler_mod = getattr(ttnn, "_ttnn", None)
    if profiler_mod is None or not hasattr(profiler_mod, "profiler"):
        pytest.skip("Profiler bindings not available in this build")

    # Binding exists on the profiler submodule; test through the public re-export on ttnn
    fn = getattr(ttnn, fn_name, None)
    if fn is None:
        pytest.skip(f"{fn_name} binding not available")

    result = fn()
    assert isinstance(result, dict)


def test_profiler_perf_data_with_workload(monkeypatch):
    """
    Run a trivial workload, force a profiler read, and verify the program perf
    data APIs return non-empty results when device profiling is enabled with
    mid-run dumps and C++ post-processing.
    """
    profiler_mod = getattr(ttnn, "_ttnn", None)
    if profiler_mod is None or not hasattr(profiler_mod, "profiler"):
        pytest.skip("Profiler bindings not available in this build")

    for var, value in {
        "TT_METAL_DEVICE_PROFILER": "1",
        "TT_METAL_PROFILER_MID_RUN_DUMP": "1",
        "TT_METAL_PROFILER_CPP_POST_PROCESS": "1",
        # Avoid writing profiler artifacts to disk during the test.
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES": "1",
    }.items():
        monkeypatch.setenv(var, value)

    if ttnn.GetNumAvailableDevices() < 1:
        pytest.skip("No devices available for profiling")

    device = ttnn.open_device(device_id=0)
    try:
        try:
            baseline_all_data = ttnn.GetAllProgramsPerfData()
        except RuntimeError as exc:
            if "profiler_state_manager is nullptr" in str(exc):
                pytest.skip("Profiler state manager not initialized (profiling disabled?)")
            raise

        shape = (1, 1, 32, 32)
        lhs = ttnn.ones(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        rhs = ttnn.ones(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.add(lhs, rhs)

        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        latest_data = ttnn.GetLatestProgramsPerfData()
        all_data = ttnn.GetAllProgramsPerfData()
        print("\nFirst call to Latest programs perf data:\n" + _format_perf_data(latest_data))
        print("\nFirst call to All programs perf data:\n" + _format_perf_data(all_data))

        # Run a heavier/different op (elementwise mul on a larger tensor) to create more profiler records.
        heavy_shape = (1, 32, 256, 256)
        heavy_a = ttnn.ones(heavy_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        heavy_b = ttnn.ones(heavy_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _ = ttnn.mul(heavy_a, heavy_b)

        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)

        latest_data = ttnn.GetLatestProgramsPerfData()
        all_data = ttnn.GetAllProgramsPerfData()
        print("\nSecond call to Latest programs perf data (after mul workload):\n" + _format_perf_data(latest_data))
        print("\nSecond call to All programs perf data (after mul workload):\n" + _format_perf_data(all_data))
    finally:
        ttnn.close_device(device)

    if not latest_data or not all_data:
        pytest.skip("Program perf data unavailable (profiler mid-run dump/post-process disabled?)")

    device_id = next(iter(latest_data))
    initial_count = len(baseline_all_data.get(device_id, [])) if baseline_all_data else 0

    assert device_id in all_data
    assert len(latest_data[device_id]) >= 1
    assert len(all_data[device_id]) >= len(latest_data[device_id])
    assert len(all_data[device_id]) > initial_count


def _format_perf_data(perf_data: dict) -> str:
    """Pretty-print program perf data mapping: chip_id -> set[ProgramAnalysisData]."""
    if not perf_data:
        return "(empty)"

    def _program_sort_key(p):
        uid = p.program_execution_uid
        return (uid.runtime_id, uid.trace_id, uid.trace_id_counter)

    lines = []
    for device_id in sorted(perf_data.keys()):
        lines.append(f"device {device_id}:")
        programs = sorted(list(perf_data[device_id]), key=_program_sort_key)
        if not programs:
            lines.append("  (no programs)")
            continue
        for program in programs:
            uid = program.program_execution_uid
            analyses_items = sorted(program.program_analyses_results.items())
            lines.append(
                f"  uid(runtime={uid.runtime_id}, trace={uid.trace_id}, ctr={uid.trace_id_counter}), "
                f"analyses={list(k for k, _ in analyses_items)}"
            )
            for name, res in analyses_items:
                lines.append(
                    f"    {name}: start={res.start_timestamp}, end={res.end_timestamp}, duration={res.duration}"
                )
    return "\n".join(lines)
