# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Real-time profiler support shared by KDA performance tests."""

from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import ttnn

T = TypeVar("T")


def profile_realtime_program(device: ttnn.Device, run_fn: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    """Profile one KDA program while retaining the device clock used by its record."""
    profile_record = None
    dropped = 0

    def collect_records(batch: Any) -> None:
        nonlocal dropped, profile_record
        dropped += int(batch.dropped)
        for record in batch.records:
            if profile_record is not None:
                return
            start_timestamp = int(record.start_timestamp)
            end_timestamp = int(record.end_timestamp)
            frequency = float(record.frequency)
            if frequency > 0 and end_timestamp > start_timestamp:
                profile_record = {
                    "runtime_id": int(record.runtime_id),
                    "chip_id": int(record.chip_id),
                    "duration_ns": (end_timestamp - start_timestamp) / frequency,
                    "frequency_ghz": frequency,
                    "kernel_sources": tuple(str(source) for source in record.kernel_sources),
                }

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect_records)
    try:
        result = run_fn()
        ttnn.synchronize_device(device)
        deadline = time.monotonic() + 1.0
        while profile_record is None and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)

    if dropped:
        raise RuntimeError(f"Real-time profiler dropped {dropped} record(s)")
    if profile_record is None:
        raise RuntimeError("Real-time profiler returned no valid KDA program record")
    return result, profile_record
