# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared collection loops of the KDA perf tools: realtime-profiler program records, and sustained trace replays."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import ttnn
from models.demos.deepseek_v3_d_p.tests.kda.perf.telemetry import Telemetry


@dataclass(frozen=True)
class ProgramRecord:
    """One program run on one chip, as the realtime profiler reports it."""

    chip: int
    runtime_id: int
    start_ns: float
    end_ns: float
    kernel_sources: tuple[str, ...]


def collect_program_records(
    mesh_device: ttnn.MeshDevice, run, settle_s: float, timeout_s: float
) -> tuple[object, float, list[ProgramRecord]]:
    """Run ``run()`` under the realtime profiler and synchronize, then wait until no record has arrived for
    ``settle_s`` (at most ``timeout_s``). Returns run's result, the host seconds of run plus the synchronize, and the
    records; asserts that some arrived and none were dropped."""
    records: list[ProgramRecord] = []
    dropped = [0]

    def collect(batch):
        dropped[0] += int(batch.dropped)
        for record in batch.records:
            frequency = float(record.frequency)
            start, end = int(record.start_timestamp), int(record.end_timestamp)
            if record.runtime_id and frequency > 0 and end > start:
                records.append(
                    ProgramRecord(
                        chip=int(record.chip_id),
                        runtime_id=int(record.runtime_id),
                        start_ns=start / frequency,
                        end_ns=end / frequency,
                        kernel_sources=tuple(str(source) for source in record.kernel_sources),
                    )
                )

    handle = ttnn.device.RegisterProgramRealtimeProfilerCallback(collect)
    try:
        begin = time.perf_counter()
        result = run()
        ttnn.synchronize_device(mesh_device)
        seconds = time.perf_counter() - begin
        deadline, last, changed = time.monotonic() + timeout_s, -1, time.monotonic()
        while time.monotonic() < deadline:
            if len(records) != last:
                last, changed = len(records), time.monotonic()
            elif records and time.monotonic() - changed >= settle_s:
                break
            time.sleep(0.01)
    finally:
        ttnn.device.UnregisterProgramRealtimeProfilerCallback(handle)
    assert records and not dropped[0], f"realtime profiler: {len(records)} records, {dropped[0]} dropped"
    return result, seconds, records


def sustained_run(
    mesh_device: ttnn.MeshDevice, trace_id: int, telemetry: Telemetry, seconds: float, block: int, idle_s: float
) -> dict:
    """Idle for ``idle_s``, then replay the trace back to back in blocks of ``block`` calls (a host sync after each) for
    ``seconds``. The steady state is the run's second half: its median ms per call and the telemetry over it; the idle
    telemetry is the pause's."""
    ttnn.synchronize_device(mesh_device)
    time.sleep(idle_s)
    blocks = []
    run_start = time.perf_counter()
    while time.perf_counter() - run_start < seconds:
        start = time.perf_counter()
        for _ in range(block):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        end = time.perf_counter()
        blocks.append((start, end, (end - start) * 1e3 / block))
    half = run_start + seconds / 2
    steady = [ms for start, _, ms in blocks if start >= half]
    return {
        "block_calls": block,
        "seconds": seconds,
        "first_blocks_ms_per_call": [ms for _, _, ms in blocks[:3]],
        "steady_ms_per_call": statistics.median(steady),
        "steady_blocks": len(steady),
        "steady_telemetry": telemetry.window(half, blocks[-1][1]),
        "idle_telemetry": telemetry.window(run_start - idle_s + 0.05, run_start),
        "ms_per_call_by_block": [round(ms, 4) for _, _, ms in blocks],
    }
