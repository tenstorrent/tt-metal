# SPDX-License-Identifier: Apache-2.0
"""Host-only phase observations; never reads or synchronizes a device.

The operator-approved attribution uses consecutive observed completions, rather
than claiming these timestamps are exact device times or plugin completions.
"""

import hashlib
import json
import os
import socket
import threading
import time
from collections import deque
from pathlib import Path


def socket_path(path):
    """Short transient control address; evidence stays at the persistent path."""
    digest = hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]
    return f"/tmp/gemma4-phase-{digest}.sock"


class PhaseRecorder:
    def __init__(self, path, identity, max_events=1000000):
        self.path = Path(path)
        self.identity = identity
        self.max_events = max_events
        self.lock = threading.Lock()
        self.events = []
        self.pending = {}
        self.unbound = deque()
        self.readbacks = {}
        self.errors = []
        self.sequence = 0
        self.positions = None
        self.window_start_ns = time.perf_counter_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path = socket_path(path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Never remove a socket that may belong to another live server.
        self.server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self.server.listen(4)
        self.thread = threading.Thread(target=self._serve, daemon=True, name="gemma4-phase-export")
        self.thread.start()

    def _append(self, event):
        if len(self.events) >= self.max_events:
            if "event buffer overflow" not in self.errors:
                self.errors.append("event buffer overflow")
            return
        self.events.append(event)

    def begin(self, phase, *, batch_size, sampling_mode, **metadata):
        now = time.perf_counter_ns()
        with self.lock:
            self.sequence += 1
            record = dict(
                step_id=self.sequence,
                phase=phase,
                batch_size=batch_size,
                submit_ns=now,
                sampling_mode=sampling_mode,
                request_ids=None,
                **metadata,
            )
            self.pending[self.sequence] = record
            self._append(dict(record, event="submit", monotonic_ns=now))
            return self.sequence

    def decode_positions(self, supplied, reload_inputs):
        with self.lock:
            if reload_inputs:
                self.positions = list(map(int, supplied))
            if self.positions is None:
                raise RuntimeError("phase collector has no authoritative decode positions")
            result = self.positions[:]
            # The captured model advances every nonnegative resident position.
            self.positions = [p + 1 if p >= 0 else p for p in self.positions]
            return result

    def submitted_output(self, step_id, output):
        with self.lock:
            self.unbound.append((step_id, output))

    def bind_readback(self, output, host):
        with self.lock:
            if not self.unbound or self.unbound[0][1] is not output:
                self.errors.append("readback does not match oldest unbound submission")
                raise RuntimeError(self.errors[-1])
            step_id, _ = self.unbound.popleft()
            if id(host) in self.readbacks:
                self.errors.append("host readback identity reused while pending")
                raise RuntimeError(self.errors[-1])
            self.readbacks[id(host)] = (step_id, host)

    def complete_output(self, output):
        with self.lock:
            pair = self.readbacks.pop(id(output), None)
            if pair is None:
                # Synchronous no-read-hook path: the existing conversion waits.
                if self.unbound and self.unbound[0][1] is output:
                    pair = self.unbound.popleft()
                else:
                    self.errors.append("unknown output completion")
                    raise RuntimeError(self.errors[-1])
            step_id = pair[0]
        self.complete(step_id)

    def complete(self, step_id):
        now = time.perf_counter_ns()
        with self.lock:
            record = self.pending.pop(step_id)
            self._append(dict(record, event="complete", monotonic_ns=now))

    def command(self, command):
        with self.lock:
            now = time.perf_counter_ns()
            result = dict(
                schema_version=1,
                pid=os.getpid(),
                clock="perf_counter_ns",
                identity=self.identity,
                buffered_events=len(self.events),
                pending_steps=len(self.pending),
                errors=list(self.errors),
                boundary_ns=now,
                window_start_ns=self.window_start_ns,
            )
            if command == "reset":
                if self.pending or self.unbound or self.readbacks:
                    raise RuntimeError("cannot reset phase collector with pending work")
                if self.errors:
                    raise RuntimeError("cannot reset collector after accounting errors")
                self.events.clear()
                self.window_start_ns = now
                result.update(window_start_ns=now, buffered_events=0)
            elif command in ("drain", "flush"):
                result["events"] = self.events[:]
                self.events.clear()
            elif command != "check":
                raise ValueError("command must be check, reset, drain or flush")
        if command in ("drain", "flush"):
            with self.path.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
        return result

    def _serve(self):
        while True:
            conn, _ = self.server.accept()
            with conn:
                conn.settimeout(10)
                try:
                    data = b""
                    while b"\n" not in data:
                        part = conn.recv(4096)
                        if not part or len(data) > 65536:
                            raise ValueError("expected one bounded newline JSON command")
                        data += part
                    request = json.loads(data.split(b"\n", 1)[0])
                    result = self.command(request.get("command", request.get("action")))
                except Exception as exc:
                    result = {"error": str(exc)}
                try:
                    conn.sendall((json.dumps(result) + "\n").encode())
                except OSError:
                    # A disconnected control client must not disable later export.
                    pass
