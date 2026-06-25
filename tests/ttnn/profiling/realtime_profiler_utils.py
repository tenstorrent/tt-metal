# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time


DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS = 1.0

_RT_PROFILER_PROCESS_LIFETIME_HANDLES = []


def create_realtime_profiler_publish_sentinel(device, shape=(1, 1, 32, 32)):
    """Create a tiny post-measure program that publishes the previous RT record."""
    import ttnn

    sentinel = ttnn.zeros(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    def publish_realtime_profile_record():
        return ttnn.add(sentinel, sentinel)

    return publish_realtime_profile_record


def profile_realtime_program(
    device,
    run_fn,
    *,
    flush_fn=None,
    collect_all=False,
    record_timeout_seconds=DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS,
) -> tuple:
    """Run measured device work and return (result, rt_record or rt_records)."""
    import ttnn

    profile_records = []

    def collect(record):
        if profile_records and not collect_all:
            return

        start_timestamp = int(record.start_timestamp)
        end_timestamp = int(record.end_timestamp)
        frequency = float(record.frequency)
        if frequency <= 0 or end_timestamp <= start_timestamp:
            return

        profile_records.append(
            {
                "runtime_id": int(record.runtime_id),
                "chip_id": int(record.chip_id),
                "duration_ns": (end_timestamp - start_timestamp) / frequency,
                "kernel_sources": tuple(str(source) for source in record.kernel_sources),
            }
        )

    callback_state = {"active": True}

    def dispatch_if_active(record):
        if callback_state["active"]:
            collect(record)

    # Keep the Python callback handle alive for process lifetime. Unregistering
    # currently releases the GIL before dropping the Python callback ref.
    _RT_PROFILER_PROCESS_LIFETIME_HANDLES.append(
        ttnn.device.RegisterProgramRealtimeProfilerCallback(dispatch_if_active)
    )

    try:
        result = run_fn()
        if flush_fn is not None:
            flush_fn()
        ttnn.synchronize_device(device)

        deadline = time.monotonic() + record_timeout_seconds
        while not profile_records and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        callback_state["active"] = False

    if not profile_records:
        raise RuntimeError(
            "Real-time profiler returned no valid program records. "
            "Ensure the profiler is active and the measured op dispatched a device program."
        )

    return result, profile_records if collect_all else profile_records[0]
