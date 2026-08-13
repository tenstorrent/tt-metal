# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import time
import threading


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
    drain_before_run=False,
    record_timeout_seconds=DEFAULT_RT_PROFILER_RECORD_TIMEOUT_SECONDS,
) -> tuple:
    """Run measured device work and return (result, rt_record or rt_records)."""
    import ttnn

    profile_records = []
    invalid_profile_records = []
    dropped_profile_records = 0
    records_lock = threading.Lock()

    def collect_records(batch):
        nonlocal dropped_profile_records
        with records_lock:
            dropped_profile_records += int(batch.dropped)
            for record in batch.records:
                if profile_records and not collect_all:
                    return

                start_timestamp = int(record.start_timestamp)
                end_timestamp = int(record.end_timestamp)
                frequency = float(record.frequency)
                if frequency <= 0 or end_timestamp <= start_timestamp:
                    invalid_profile_records.append(
                        (int(record.runtime_id), int(record.chip_id), start_timestamp, end_timestamp, frequency)
                    )
                    continue

                profile_records.append(
                    {
                        "runtime_id": int(record.runtime_id),
                        "chip_id": int(record.chip_id),
                        "start_timestamp": start_timestamp,
                        "end_timestamp": end_timestamp,
                        "frequency": frequency,
                        "duration_ns": (end_timestamp - start_timestamp) / frequency,
                        "kernel_sources": tuple(str(source) for source in record.kernel_sources),
                    }
                )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_records)

    try:
        if drain_before_run:
            # A device synchronize drains dispatch-side records, but their receiver-thread callbacks
            # can arrive after this callback is registered. Wait for a quiet window, then discard that
            # pre-existing traffic so the measured device span starts with run_fn's first program.
            ttnn.synchronize_device(device)
            drain_deadline = time.monotonic() + record_timeout_seconds
            with records_lock:
                last_count = len(profile_records)
            last_change = time.monotonic()
            while time.monotonic() < drain_deadline:
                with records_lock:
                    count = len(profile_records)
                if count != last_count:
                    last_count = count
                    last_change = time.monotonic()
                elif (time.monotonic() - last_change) >= RT_PROFILER_RECORD_SETTLE_SECONDS:
                    break
                time.sleep(0.01)
            with records_lock:
                profile_records.clear()
                invalid_profile_records.clear()
                dropped_profile_records = 0

        result = run_fn()
        ttnn.synchronize_device(device)

        deadline = time.monotonic() + record_timeout_seconds
        last_count = 0
        last_change = time.monotonic()
        while time.monotonic() < deadline:
            with records_lock:
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
            "Ensure the profiler is active and the measured op dispatched a device program. "
            f"Invalid raw records: {invalid_profile_records}"
        )
    if dropped_profile_records:
        raise RuntimeError(f"Real-time profiler dropped {dropped_profile_records} measured program records")

    return result, profile_records if collect_all else profile_records[0]
