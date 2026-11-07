# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import statistics
import time

from loguru import logger


# Device program timing for perf tests, from the streaming profiler (TT_METAL_STREAMING_PROFILER=1). The streaming
# profiler drains every kernel's firmware "*-KERNEL" zone to the host while it runs; a C++ collector
# (ttnn.device.StreamingProgramCollector) folds them into one record per program launch per chip: first kernel start
# to last kernel end over every core and RISC, as ELAPSED time on the host timeline: when the chip lowers AICLK under
# load (DVFS), the duration grows with it. ("clock_ghz" is the average device clock the program ran at.) The real-time
# profiler this replaced converted cycles at a clock fitted over seconds (~nominal), so it did not see throttling;
# baselines calibrated under it read low for ops that throttle. Kernels run instrumented in this mode, so durations
# include the zone-marker cost (measured negligible for ms-scale ops).
#
# Turning it on: the device fixtures in the root conftest.py do it per device open
# (ttnn.device.SetStreamingProfilerEnabled), for every test that reaches this helper (a module-scoped device: any
# selected test of the module), so no environment variable is needed and other tests stay uninstrumented. A test that
# opens its own device calls ttnn.device.SetStreamingProfilerEnabled(True) before opening it. Setting
# TT_METAL_STREAMING_PROFILER in the environment instead turns it on for the whole process.

DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS = 5.0

# A program's "*-KERNEL" zones arrive per core, so settle on the zone count, not the program count. Measured on
# Blackhole, every zone of the measured region lands within 4 ms of synchronize_device returning (spool and direct
# push alike, 1-core and 110-core programs); 50 ms keeps >10x headroom.
STREAMING_RECORD_SETTLE_SECONDS = 0.05


def is_streaming_profiler_active() -> bool:
    """True when TT_METAL_STREAMING_PROFILER=1 is running. Needs an open device."""
    import ttnn

    return bool(ttnn.device.IsStreamingProfilerActive())


def is_perf_profiler_active() -> bool:
    """True when the profiler that times perf-test programs is running. Needs an open device."""
    return is_streaming_profiler_active()


def profile_realtime_program(
    device,
    run_fn,
    *,
    collect_all=False,
    record_timeout_seconds=DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS,
    record_settle_seconds=STREAMING_RECORD_SETTLE_SECONDS,
) -> tuple:
    """Run measured device work and return (result, record or records), records ordered by start time.

    Each record is {"runtime_id", "chip_id", "duration_ns" (elapsed), "clock_ghz", "kernel_sources", "num_cores",
    "processor_duration_ns", "execution"}; without ``collect_all`` it is the first program that started in the measured region. Programs
    still in flight to the host from before the region are excluded by host time."""
    import ttnn

    if not is_streaming_profiler_active():
        raise RuntimeError(
            "Streaming profiler is not active: open the device through the conftest device fixtures (they turn it on "
            "for tests that use this helper) or call ttnn.device.SetStreamingProfilerEnabled(True) before opening it."
        )

    # Everything dispatched before the measured region finishes here, so its programs start before region_start_ns
    # (host steady_clock == CLOCK_MONOTONIC == time.monotonic_ns).
    ttnn.synchronize_device(device)
    collector = ttnn.device.StreamingProgramCollector()
    collector.start()
    try:
        region_start_ns = time.monotonic_ns()
        result = run_fn()
        ttnn.synchronize_device(device)

        # On a multi-chip mesh, a chip's records are held on the host until the cross-chip clock sync has solved its
        # links, which right after a mesh opens takes ~0.25 s longer for the non-root chips than for the root chip, so
        # a quiet window alone would end the wait with only the root chip's records. Wait for every chip of the mesh
        # that ran something; a program that runs on fewer chips falls back to the timeout.
        expected_chips = set(device.get_device_ids()) if hasattr(device, "get_device_ids") else set()
        deadline = time.monotonic() + record_timeout_seconds
        last_count = 0
        last_change = time.monotonic()
        seen_chips = set()
        while time.monotonic() < deadline:
            count = collector.num_kernel_zones
            if count != last_count:
                last_count = count
                last_change = time.monotonic()
                if not expected_chips <= seen_chips:
                    seen_chips = {
                        int(record.chip_id) for record in collector.records() if record.start_ns >= region_start_ns
                    }
            elif count and (time.monotonic() - last_change) >= record_settle_seconds and expected_chips <= seen_chips:
                break
            time.sleep(0.005)
        records = collector.records()
        dropped_bytes = collector.dropped_bytes
        stall_count = collector.stall_count
    finally:
        collector.stop()

    # Dropped data makes the set partial: on a mesh, losing the slowest chip's zones silently under-reports a
    # program's critical path, so a perf gate could read green off incomplete data.
    if dropped_bytes:
        raise RuntimeError(
            f"Streaming profiler dropped {dropped_bytes} bytes — the record set is incomplete and durations may "
            "under-report. Raise TT_METAL_STREAMING_PROFILER_FIFO_MB."
        )
    if stall_count:
        logger.warning(
            f"Streaming profiler: cores stalled {stall_count} time(s) waiting for the profiler to drain; "
            "measured durations include those stalls."
        )

    profile_records = [
        {
            "runtime_id": int(record.runtime_id),
            "chip_id": int(record.chip_id),
            "duration_ns": float(record.duration_ns),
            "clock_ghz": float(record.clock_ghz),
            "kernel_sources": tuple(str(source) for source in record.kernel_sources),
            "num_cores": int(record.num_cores),
            "processor_duration_ns": dict(record.processor_duration_ns),
            "execution": int(record.execution),
        }
        for record in records
        if record.start_ns >= region_start_ns and record.duration_ns > 0
    ]
    if not profile_records:
        raise RuntimeError(
            "Streaming profiler returned no program records for the measured region. "
            "Ensure the measured op dispatched a device program."
        )
    return result, profile_records if collect_all else profile_records[0]


def require_realtime_profiler(what: str) -> None:
    """Fail (not skip) if the profiler is inactive — an unmeasured perf run must not report green.
    Needs an open device."""
    import pytest

    if not is_perf_profiler_active():
        pytest.fail(
            f"Streaming profiler must be active for {what}: open the device through the conftest device fixtures or "
            "call ttnn.device.SetStreamingProfilerEnabled(True) before opening it"
        )


def profile_realtime_program_merged(
    device, run_fn, *, record_timeout_seconds=DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS
) -> tuple:
    """profile_realtime_program with the per-chip records merged per program: returns (result,
    {runtime_id -> {"duration_ns" (max across chips = critical path), "kernel_sources"}}) in dispatch
    order. Callers identify their own program, by kernel path or as the only entry."""
    result, records = profile_realtime_program(
        device, run_fn, collect_all=True, record_timeout_seconds=record_timeout_seconds
    )

    per_program: dict = {}
    for record in records:
        runtime_id = record["runtime_id"]
        if not runtime_id:
            continue
        entry = per_program.setdefault(runtime_id, {"duration_ns": 0.0, "kernel_sources": record["kernel_sources"]})
        entry["duration_ns"] = max(entry["duration_ns"], record["duration_ns"])

    assert per_program, "streaming profiler returned no valid program records for the measured region"
    return result, per_program


def assert_op_duration_merged(
    device, run_fn, kernel_path, *, expected_ns, margin, label, iters=1, verbose=False
) -> float:
    """Median device duration of the one program whose kernel sources contain ``kernel_path``, over
    ``iters`` runs of ``run_fn`` in a single profiler window, asserted within +/-``margin`` of
    ``expected_ns``. Requires one match per run so the number is attributable to that op; ``verbose``
    logs every program in the window, and a wrong match count dumps them and fails."""

    def run_all():
        for _ in range(iters):
            run_fn()

    _, per_program = profile_realtime_program_merged(device, run_all)

    def dump(log):
        for seq, (runtime_id, entry) in enumerate(per_program.items()):  # arrival = dispatch order
            log(
                f"  [{seq}] runtime_id={runtime_id} duration_ns={entry['duration_ns']:.0f} "
                f"kernels={sorted({source.rsplit('/', 1)[-1] for source in entry['kernel_sources']})}"
            )

    if verbose:
        dump(logger.info)

    matched = [
        entry["duration_ns"]
        for entry in per_program.values()
        if any(kernel_path in source.replace("\\", "/") for source in entry["kernel_sources"])
    ]
    if len(matched) != iters:
        if not verbose:
            dump(logger.error)
        raise AssertionError(f"expected {iters} programs matching {kernel_path}, got {len(matched)}")

    median_ns = statistics.median(matched)
    lower, upper = expected_ns * (1 - margin), expected_ns * (1 + margin)
    logger.info(
        f"RT-CAL {label}: {round(median_ns):_} ns  # median of {iters}, expected {expected_ns:_}, "
        f"band [{lower:.0f}, {upper:.0f}]"
    )
    assert lower <= median_ns <= upper, (
        f"{label} device time {median_ns:.0f} ns outside band [{lower:.0f}, {upper:.0f}] ns "
        f"(expected {expected_ns} ns, margin +/- {margin * 100:.0f}%)"
    )
    return median_ns
