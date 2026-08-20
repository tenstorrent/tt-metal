# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import statistics
import time

from loguru import logger


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
    dropped = [0]

    def collect_records(batch):
        dropped[0] += int(batch.dropped)
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

    # Dropped records make the set partial: on a mesh, losing the slowest chip's record silently
    # under-reports a program's critical path, so a perf gate could read green off incomplete data.
    if dropped[0]:
        raise RuntimeError(
            f"Real-time profiler dropped {dropped[0]} record(s) — the receiver could not keep up, so the "
            "record set is incomplete and durations may under-report."
        )

    if not profile_records:
        raise RuntimeError(
            "Real-time profiler returned no valid program records. "
            "Ensure the profiler is active and the measured op dispatched a device program."
        )

    return result, profile_records if collect_all else profile_records[0]


def require_realtime_profiler(what: str) -> None:
    """Fail (not skip) if the profiler is inactive — an unmeasured perf run must not report green.
    Needs an open device: IsProgramRealtimeProfilerActive segfaults if called at collection time."""
    import pytest

    import ttnn

    if not ttnn.device.IsProgramRealtimeProfilerActive():
        pytest.fail(f"Real-time profiler must be active for {what}")


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

    assert per_program, "real-time profiler returned no valid program records for the measured region"
    return result, per_program


def collect_op_durations_merged(
    device, run_fn, kernel_path, *, iters=1, allow_stale_prefix=False, verbose=False
) -> list[float]:
    """Return one device-program duration per ``run_fn`` invocation.

    ``kernel_path`` identifies the program from the kernel sources attached to each real-time
    profiler record. Callers must warm up compilation before using this helper: only work inside
    its profiler window is measured. ``allow_stale_prefix`` is for callback consumers that start
    after earlier device work: it discards older matching records in dispatch order, but still
    requires exactly ``iters`` newest records from ``run_fn``.
    """

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

    matched = [
        entry["duration_ns"]
        for entry in per_program.values()
        if any(kernel_path in source.replace("\\", "/") for source in entry["kernel_sources"])
    ]
    if len(matched) < iters or (len(matched) != iters and not allow_stale_prefix):
        if not verbose:
            dump(logger.error)
        raise AssertionError(f"expected {iters} programs matching {kernel_path}, got {len(matched)}")
    if allow_stale_prefix:
        matched = matched[-iters:]
    if verbose:
        dump(logger.info)
    return matched


def assert_op_duration_merged(
    device, run_fn, kernel_path, *, expected_ns, margin, label, iters=1, verbose=False
) -> float:
    """Median device duration of the one program whose kernel sources contain ``kernel_path``, over
    ``iters`` runs of ``run_fn`` in a single profiler window, asserted within +/-``margin`` of
    ``expected_ns``. Requires one match per run so the number is attributable to that op; ``verbose``
    logs every program in the window, and a wrong match count dumps them and fails."""

    matched = collect_op_durations_merged(device, run_fn, kernel_path, iters=iters, verbose=verbose)
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
