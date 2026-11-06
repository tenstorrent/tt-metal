# SPDX-License-Identifier: Apache-2.0
"""Host-only phase observations; never reads or synchronizes a device.

The operator-approved attribution uses consecutive observed completions, rather
than claiming these timestamps are exact device times or plugin completions.
"""

import hashlib
import json
import logging
import math
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
    def __init__(
        self,
        path,
        identity,
        max_events=1000000,
        *,
        auto_export_interval=None,
        max_file_bytes=None,
        export_chunk_events=1024,
    ):
        if auto_export_interval is None:
            auto_export_interval = float(os.environ.get("GEMMA4_PHASE_AUTOEXPORT_SECONDS", "0"))
        if not math.isfinite(auto_export_interval) or auto_export_interval < 0:
            raise ValueError("auto export interval must be finite and nonnegative")
        if max_file_bytes is None:
            max_file_bytes = int(os.environ.get("GEMMA4_PHASE_MAX_FILE_BYTES", str(4 * 1024**3)))
        if max_file_bytes < 4096 or export_chunk_events < 1 or max_events < 1:
            raise ValueError("invalid phase export bounds")
        self.auto_export_interval = auto_export_interval
        self.max_file_bytes = max_file_bytes
        self.export_chunk_events = export_chunk_events
        self.export_lock = threading.Lock()
        self.export_stream = None
        self.export_failed = False
        self.file_exhausted = False
        self.truncated = False
        # Resolve only documented placeholders; ordinary paths stay literal.
        resolved_path = str(path).replace("{pid}", str(os.getpid())).replace("{time_ns}", str(time.time_ns()))
        self.path = Path(resolved_path)
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
        self.socket_path = socket_path(self.path)
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Never remove a socket that may belong to another live server.
        self.server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self.server.listen(4)
        if self.auto_export_interval:
            self.server.settimeout(self.auto_export_interval)
        self.thread = threading.Thread(target=self._serve, daemon=True, name="gemma4-phase-export")
        self.thread.start()

    def _append(self, event):
        if self.file_exhausted:
            return
        if len(self.events) >= self.max_events:
            self.truncated = True
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

    def _snapshot(self):
        return dict(
            schema_version=1,
            pid=os.getpid(),
            clock="perf_counter_ns",
            identity=self.identity,
            buffered_events=len(self.events),
            pending_steps=len(self.pending),
            errors=list(self.errors),
            boundary_ns=time.perf_counter_ns(),
            window_start_ns=self.window_start_ns,
            **({"truncated": self.truncated} if self.auto_export_interval else {}),
        )

    def _write_export(self, result):
        if not self.auto_export_interval:
            with self.path.open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            return
        if self.export_failed:
            raise RuntimeError("phase export previously failed; timing window is invalid")
        if self.export_stream is None:
            # Never append to a stale run or follow a symlink. Keep the descriptor.
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            self.export_stream = os.fdopen(fd, "wb", buffering=0)
        payload = (json.dumps(result) + "\n").encode()
        # Keep room for a compact, explicit terminal status even for a huge event.
        if self.export_stream.tell() + len(payload) > self.max_file_bytes - 2048:
            with self.lock:
                self.truncated = True
                self.file_exhausted = True
                self.errors.append("phase file cap reached; subsequent events omitted")
            payload = (
                json.dumps(
                    dict(
                        schema_version=1,
                        truncated=True,
                        errors=["phase file cap reached; subsequent events omitted"],
                        boundary_ns=result["boundary_ns"],
                        events=[],
                    )
                )
                + "\n"
            ).encode()
        remaining = memoryview(payload)
        while remaining:
            written = self.export_stream.write(remaining)
            if not written:
                raise OSError("phase export made no write progress")
            remaining = remaining[written:]

    def _export(self, *, automatic=False):
        # Export ownership serializes manual drains and periodic chunks. Device
        # submission/completion only takes self.lock, never this I/O lock.
        with self.export_lock:
            with self.lock:
                result = self._snapshot()
                count = min(len(self.events), self.export_chunk_events) if automatic else len(self.events)
                result["events"] = self.events[:count]
                exhausted = self.file_exhausted
            if exhausted:
                return result
            try:
                self._write_export(result)
            except Exception:
                with self.lock:
                    self.export_failed = True
                    self.truncated = True
                    if "phase export failed; timing window invalid" not in self.errors:
                        self.errors.append("phase export failed; timing window invalid")
                raise
            with self.lock:
                del self.events[:count]
                if self.file_exhausted:
                    self.events.clear()
                    result["truncated"] = True
                    result["errors"] = list(self.errors)
            return result

    def command(self, command):
        if command in ("drain", "flush"):
            return self._export()
        with self.export_lock, self.lock:
            result = self._snapshot()
            if command == "reset":
                if self.pending or self.unbound or self.readbacks:
                    raise RuntimeError("cannot reset phase collector with pending work")
                if self.errors:
                    raise RuntimeError("cannot reset collector after accounting errors")
                self.events.clear()
                self.window_start_ns = result["boundary_ns"]
                result.update(window_start_ns=self.window_start_ns, buffered_events=0)
            elif command != "check":
                raise ValueError("command must be check, reset, drain or flush")
            return result

    def _serve(self):
        next_export = time.monotonic() + self.auto_export_interval
        while True:
            if self.auto_export_interval and time.monotonic() >= next_export:
                if not self.export_failed and not self.file_exhausted:
                    try:
                        self._export(automatic=True)
                    except Exception:
                        logging.exception("Phase autoexport failed; timing window invalid")
                next_export = time.monotonic() + self.auto_export_interval
            try:
                conn, _ = self.server.accept()
            except socket.timeout:
                continue
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
