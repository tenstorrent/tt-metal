# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic runner shim. Importing this file does not import TTNN or open devices.

Launch through QWEN_EXPERIMENT_ENTRY or QWEN_PROFILE_MODULE. Normal runner timer
endpoints exclude /proc collection, but collection can perturb thread wake state;
compare diagnostic runs with each other and confirm with untouched runner runs.
"""

import argparse
import inspect
import json
import os
import resource
import time
from pathlib import Path
from types import SimpleNamespace


def main_usage():
    value = resource.getrusage(resource.RUSAGE_THREAD)
    return {
        "thread_cpu_ns": time.thread_time_ns(),
        "user_s": value.ru_utime,
        "system_s": value.ru_stime,
        "voluntary_switches": value.ru_nvcsw,
        "involuntary_switches": value.ru_nivcsw,
        "minor_faults": value.ru_minflt,
        "major_faults": value.ru_majflt,
    }


def thread_snapshot():
    result = {}
    for entry in Path("/proc/self/task").iterdir():
        try:
            sched = [int(value) for value in (entry / "schedstat").read_text().split()]
            status = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            result[entry.name] = {
                "name": status["Name"].strip(),
                "allowed_cpus": status["Cpus_allowed_list"].strip(),
                "cpu_ns": sched[0],
                "runnable_wait_ns": sched[1],
                "timeslices": sched[2],
                "voluntary_switches": int(status["voluntary_ctxt_switches"]),
                "involuntary_switches": int(status["nonvoluntary_ctxt_switches"]),
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
    return result


def delta(before, after):
    return {name: after[name] - value for name, value in before.items()}


def thread_deltas(before, after):
    result = {}
    for tid in before.keys() & after.keys():
        old, new = before[tid], after[tid]
        numeric = {key: value for key, value in old.items() if isinstance(value, int)}
        result[tid] = {
            "name": new["name"],
            "allowed_cpus": new["allowed_cpus"],
            **delta(numeric, new),
        }
    return result


class Probe:
    def __init__(self, sync, execute):
        self.original_sync = sync
        self.original_execute = execute
        self.blocking_control = os.getenv("QWEN_GAP_BLOCKING_TRACE", "0") == "1"
        self.skip_next_sync = False
        self.active = None
        self.intervals = []
        self.sync_calls = []

    def perf_counter(self):
        # The inspected runner calls perf_counter in start/end pairs. A proxy
        # specific to that module avoids modifying time for TTNN or other code.
        if self.active is None:
            frame = inspect.currentframe().f_back
            location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            del frame
            threads = thread_snapshot()
            usage = main_usage()
            self.active = {
                "id": len(self.intervals),
                "runner_start": location,
                "sync_call_ids": [],
                "execute_calls": [],
                "threads_before": threads,
                "usage_before": usage,
            }
            started = time.perf_counter()
            self.active["started"] = started
            return started
        ended = time.perf_counter()
        usage = main_usage()
        threads = thread_snapshot()
        record, self.active = self.active, None
        record["wall_ms"] = (ended - record.pop("started")) * 1000
        record["main_thread_delta"] = delta(record.pop("usage_before"), usage)
        record["thread_deltas"] = thread_deltas(record.pop("threads_before"), threads)
        self.intervals.append(record)
        return ended

    def execute(self, *args, **kwargs):
        requested_blocking = kwargs.get("blocking", True)
        if self.blocking_control and not requested_blocking:
            kwargs = {**kwargs, "blocking": True}
            self.skip_next_sync = True
        started = time.perf_counter_ns()
        try:
            return self.original_execute(*args, **kwargs)
        finally:
            ended = time.perf_counter_ns()
            if self.active is not None:
                self.active["execute_calls"].append(
                    {
                        "wall_ms": (ended - started) / 1e6,
                        "requested_blocking": requested_blocking,
                        "effective_blocking": kwargs.get("blocking", True),
                    }
                )

    def synchronize(self, *args, **kwargs):
        # For timed runner windows, /proc snapshots belong to the enclosing
        # timer and cover execute plus its sync. Standalone syncs get their own
        # snapshots. Only cheap main-thread counters remain inside runner timing.
        standalone = self.active is None
        threads = thread_snapshot() if standalone else None
        before = main_usage()
        skipped = self.skip_next_sync
        self.skip_next_sync = False
        started = time.perf_counter_ns()
        try:
            if not skipped:
                return self.original_sync(*args, **kwargs)
        finally:
            ended = time.perf_counter_ns()
            after = main_usage()
            record = {
                "id": len(self.sync_calls),
                "interval_id": None if standalone else self.active["id"],
                "wall_ms": (ended - started) / 1e6,
                "main_thread_delta": delta(before, after),
                "skipped_after_blocking_trace": skipped,
                "thread_counter_scope": ("standalone_sync" if standalone else "enclosing_runner_interval"),
            }
            if standalone:
                record["thread_deltas"] = thread_deltas(threads, thread_snapshot())
            else:
                self.active["sync_call_ids"].append(record["id"])
            self.sync_calls.append(record)

    def write(self, path, error):
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pid": os.getpid(),
                    "main_tid": os.getpid(),
                    "error": error,
                    "environment": {
                        name: os.getenv(name)
                        for name in (
                            "TT_MESH_PASS_THROUGH_THREAD_POOL",
                            "TT_METAL_DEVICE_PROFILER",
                            "TT_METAL_PROFILER_SYNC",
                            "TT_METAL_DEVICE_PROFILER_DISPATCH",
                            "QWEN_GAP_BLOCKING_TRACE",
                        )
                    },
                    "limitations": [
                        "Diagnostic /proc snapshots perturb thread wake state despite occurring outside runner timer endpoints.",
                        "Per-thread snapshots are sequential and include snapshot collection; their deltas are attribution clues, not exact completion latency.",
                        "Main-thread CPU/rusage calls add a small cost inside runner timing; sync wall measures only the wrapped API.",
                        "Timed per-thread counters cover execute plus sync; standalone sync calls have independent snapshots.",
                        "Thread names and affinity do not prove completion-reader identity; use host scheduling traces or named C++ zones if needed.",
                        "Zero runnable_wait_ns may mean unavailable scheduler statistics; context switches alone do not prove contention.",
                        "Blocking control alters originally nonblocking replay and skips its following sync, including queued timing if requested.",
                    ],
                    "incomplete_interval": None if self.active is None else self.active["id"],
                    "intervals": self.intervals,
                    "sync_calls": self.sync_calls,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, required=True)
    options, _ = parser.parse_known_args()
    from models.autoports.qwen_qwen3_8_27b.tests import run_multichip_decoder as runner

    probe = Probe(runner.ttnn.synchronize_device, runner.ttnn.execute_trace)
    original_time = runner.time
    runner.time = SimpleNamespace(perf_counter=probe.perf_counter)
    runner.ttnn.synchronize_device = probe.synchronize
    runner.ttnn.execute_trace = probe.execute
    error = None
    try:
        runner.main()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        runner.time = original_time
        runner.ttnn.synchronize_device = probe.original_sync
        runner.ttnn.execute_trace = probe.original_execute
        probe.write(options.output.with_suffix(".host_gap.json"), error)


if __name__ == "__main__":
    main()
