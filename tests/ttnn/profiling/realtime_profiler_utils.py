# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time


DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS = 1.0

# collect_all: records are delivered asynchronously from the RT profiler receiver thread, and
# UnregisterProgramRealtimeProfilerCallback only waits for in-flight callbacks — not for records that
# have not yet been delivered. For a multi-chip mesh run, breaking on the first record would capture
# only a subset (possibly a single chip's), making the downstream max(duration_ns) non-deterministic.
# Once the first record arrives, keep polling until no new record has landed for this settle window.
RT_PROFILER_RECORD_SETTLE_SECONDS = 0.1


def profile_realtime_program(
    device,
    run_fn,
    *,
    collect_all=False,
    record_timeout_seconds=DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS,
) -> tuple:
    """Run measured device work and return (result, rt_record or rt_records)."""
    import ttnn

    profile_records = []

    def collect_records(batch):
        for record in batch.records:
            if profile_records and not collect_all:
                return

            start_timestamp = int(record.start_timestamp)
            end_timestamp = int(record.end_timestamp)
            frequency = float(record.frequency)
            if frequency <= 0 or end_timestamp <= start_timestamp:
                continue

            profile_records.append(
                {
                    "runtime_id": int(record.runtime_id),
                    "chip_id": int(record.chip_id),
                    "duration_ns": (end_timestamp - start_timestamp) / frequency,
                    "kernel_sources": tuple(str(source) for source in record.kernel_sources),
                }
            )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_records)

    try:
        result = run_fn()
        ttnn.synchronize_device(device)

        deadline = time.monotonic() + record_timeout_seconds
        last_count = 0
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            count = len(profile_records)
            if count and not collect_all:
                break
            if count != last_count:
                last_count = count
                last_change = time.monotonic()
            elif count and (time.monotonic() - last_change) >= RT_PROFILER_RECORD_SETTLE_SECONDS:
                # collect_all: records have stopped arriving for the settle window.
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    if not profile_records:
        raise RuntimeError(
            "Real-time profiler returned no valid program records. "
            "Ensure the profiler is active and the measured op dispatched a device program."
        )

    return result, profile_records if collect_all else profile_records[0]
