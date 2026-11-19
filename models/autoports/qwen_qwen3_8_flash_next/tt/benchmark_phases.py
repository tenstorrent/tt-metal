# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Buffered host observations of ordered CQ0 completion events.

The observer only queries host completion counters. It never waits on a device,
reads a tensor, enables a profiler, or writes evidence during measured requests.
Create ``<log_path>.flush`` after requests finish to export an atomic snapshot.
The snapshot retains completion uncertainty and host postprocessing separately;
the collector, not this recorder, owns exclusive phase attribution.
"""

import copy
import json
import os
import sys
import threading
import time
from pathlib import Path


class PhaseRecorder:
    def __init__(self, mesh, log_path, *, poll_interval_seconds=0.001, query=None, record=None, start_thread=True):
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if query is None or record is None:
            import ttnn

            from . import _benchmark_event_query

            query = query or _benchmark_event_query.event_query
            record = record or (lambda: ttnn.record_event(mesh, 0))
            query_module = _benchmark_event_query.__file__
        else:
            query_module = "injected_host_test"
        self.mesh = mesh
        self.log_path = Path(log_path)
        self.flush_path = Path(str(self.log_path) + ".flush")
        self.query, self.record = query, record
        self.poll_interval_seconds = poll_interval_seconds
        self.started_ns = time.monotonic_ns()
        self.identity = {
            "pid": os.getpid(),
            "clock": "time.monotonic_ns",
            "cq_id": 0,
            "module": str(Path(__file__).resolve()),
            "query_module": query_module,
            "poll_interval_seconds": poll_interval_seconds,
            "completion_method": "nonblocking distributed::EventQuery on host completion counters",
            "added_device_waits": 0,
        }
        self._lock = threading.Lock()
        self._rows = []
        self._pending = []
        self._end_order = 0
        self._errors = []
        self._stop = threading.Event()
        self._thread = None
        self._previous_poll_ns = None
        self._max_poll_gap_ns = 0
        if start_thread:
            self._thread = threading.Thread(target=self._observe, name="qwen-phase-observer", daemon=True)
            self._thread.start()

    def begin(self, phase, metadata):
        started_ns = time.monotonic_ns()
        if phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        # Copy scheduler-owned lists before the next async submission mutates them.
        metadata = copy.deepcopy(metadata)
        with self._lock:
            ticket = len(self._rows)
            self._rows.append({"ticket": ticket, "phase": phase, "begin_ns": started_ns, "metadata": metadata})
        return ticket

    def end(self, ticket, event=None):
        # Only the serving thread enqueues events. Decode supplies its existing
        # readback event; non-final prefill needs a host-notifying marker on CQ0.
        end_enter_ns = time.monotonic_ns()
        added_event = event is None
        event = self.record() if added_event else event
        enqueued_ns = time.monotonic_ns()
        with self._lock:
            row = self._rows[ticket]
            if "end_order" in row:
                raise ValueError("ticket has already ended")
            row.update(
                end_order=self._end_order,
                end_enter_ns=end_enter_ns,
                event_enqueued_ns=enqueued_ns,
                added_completion_event=added_event,
                event_repr=repr(event),
                completion_lower_ns=row["begin_ns"],
            )
            self._end_order += 1
            self._pending.append((ticket, event))

    def annotate(self, ticket, **fields):
        """Retain metadata learned during submission before scheduler reuse."""
        with self._lock:
            self._rows[ticket]["metadata"].update(copy.deepcopy(fields))

    def host_complete(self, ticket):
        completed_ns = time.monotonic_ns()
        with self._lock:
            self._rows[ticket]["host_complete_ns"] = completed_ns

    def _poll_once(self):
        poll_ns = time.monotonic_ns()
        with self._lock:
            if self._previous_poll_ns is not None:
                self._max_poll_gap_ns = max(self._max_poll_gap_ns, poll_ns - self._previous_poll_ns)
            self._previous_poll_ns = poll_ns
            pending = list(self._pending)
        for index, (ticket, event) in enumerate(pending):
            query_begin_ns = time.monotonic_ns()
            completed = bool(self.query(event))
            query_end_ns = time.monotonic_ns()
            with self._lock:
                row = self._rows[ticket]
                if completed:
                    row.update(
                        completion_observed_ns=query_end_ns,
                        query_begin_ns=query_begin_ns,
                        completion_uncertainty_ns=query_end_ns - row["completion_lower_ns"],
                    )
                    self._pending = [(number, value) for number, value in self._pending if number != ticket]
                else:
                    # CQ ordering proves later events incomplete at this instant.
                    for number, _ in pending[index:]:
                        self._rows[number]["completion_lower_ns"] = query_begin_ns
                    break

    def _observe(self):
        next_control_ns = 0
        while not self._stop.is_set():
            try:
                self._poll_once()
                now = time.monotonic_ns()
                if now >= next_control_ns:
                    next_control_ns = now + 100_000_000
                    if self.flush_path.exists():
                        self.flush()
                        self.flush_path.unlink(missing_ok=True)
            except Exception as exc:
                with self._lock:
                    self._errors.append({"time_ns": time.monotonic_ns(), "error": repr(exc)})
                # A failed query must remain inspectable even though polling
                # stops. Otherwise the external flush request can only time out.
                print(f"QWEN_BENCHMARK_OBSERVER_ERROR {exc!r}", file=sys.stderr, flush=True)
                try:
                    self.flush()
                    self.flush_path.unlink(missing_ok=True)
                except Exception as export_error:
                    print(f"QWEN_BENCHMARK_EXPORT_ERROR {export_error!r}", file=sys.stderr, flush=True)
                # Do not spin on a broken device reference.
                self._stop.set()
            self._stop.wait(self.poll_interval_seconds)

    def snapshot(self):
        with self._lock:
            return {
                "schema_version": 1,
                "identity": dict(self.identity),
                "started_ns": self.started_ns,
                "snapshot_ns": time.monotonic_ns(),
                "max_observer_poll_gap_ns": self._max_poll_gap_ns,
                "pending_tickets": [ticket for ticket, _ in self._pending],
                "errors": copy.deepcopy(self._errors),
                "events": copy.deepcopy(self._rows),
            }

    def flush(self):
        """Export on explicit post-measurement request, never once per token."""
        snapshot = self.snapshot()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.log_path.with_name(self.log_path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(snapshot, separators=(",", ":")))
        os.replace(temporary, self.log_path)
        return snapshot

    def close(self):
        """Stop observation before mesh teardown; does not wait for device work."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return self.flush()
