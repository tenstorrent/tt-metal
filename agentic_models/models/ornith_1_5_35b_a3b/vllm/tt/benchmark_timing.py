# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host phase records, with no device calls or per-step file I/O.

Decode completion is called after the plugin's existing event wait and host
conversion. An intermediate prefill return is explicitly NOT a completion.
The background writer also serves a flush socket for the host-only collector.
"""
import atexit
import hashlib
import json
import os
import socket
import threading
import time
from pathlib import Path


class HostPhaseRecorder:
    def __init__(self, adapter, module, module_file):
        self.root = Path(os.environ["ORNITH_BENCHMARK_TIMING_DIR"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.lock = threading.Lock()
        self.pending = {}
        self.requests = [None] * adapter.batch_size
        self.serial = 0
        self.prefill_pending = []
        self.current_prefill = None
        self.generator = adapter.generator
        self.path = self.root / "events.jsonl"
        # A profile launch must never append events from an earlier process.
        self.path.write_text("")
        self.socket_path = str(self.root / "flush.sock")
        if len(self.socket_path.encode()) >= 104:
            # Linux abstract namespace avoids pathname length limits; ready.json
            # records the actual address, and PID prevents profile collisions.
            self.socket_path = "@ornith-benchmark-" + str(os.getpid())
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        address = "\0" + self.socket_path[1:] if self.socket_path.startswith("@") else self.socket_path
        if not self.socket_path.startswith("@"):
            Path(self.socket_path).unlink(missing_ok=True)
        self.socket.bind(address)
        self.socket.listen(2)
        self.socket.settimeout(1)
        mesh = adapter.model.device
        self.identity = dict(
            schema=1,
            pid=os.getpid(),
            generator_module=module,
            generator_module_file=str(Path(module_file).resolve()),
            layer_count=len(adapter.model.layer_indices),
            layers=adapter.model.layer_indices,
            precision=adapter.model.precision,
            max_num_seqs=adapter.batch_size,
            max_model_len=adapter.context,
            mesh_shape=list(mesh.shape),
            device_ids=list(mesh.get_device_ids()),
            phase_event_path=str(self.path),
            ready_path=str(self.root / "ready.json"),
            flush_socket=self.socket_path,
            clock="time.perf_counter; same Linux host as vLLM client",
            completion="decode after existing plugin event wait and host conversion; sampled prefill after existing token readback",
        )
        (self.root / "ready.json").write_text(json.dumps(self.identity, indent=2) + "\n")
        self.writer = threading.Thread(target=self._writer, name="ornith-benchmark-writer", daemon=True)
        self.writer.start()
        atexit.register(self.flush)

    def append(self, event):
        with self.lock:
            self.events.append(event)

    def flush(self):
        with self.lock:
            events, self.events = self.events, []
        if events:
            with self.path.open("a") as stream:
                for event in events:
                    stream.write(json.dumps(event, separators=(",", ":")) + "\n")

    def _writer(self):
        while True:
            try:
                connection, _ = self.socket.accept()
            except socket.timeout:
                self.flush()
                continue
            with connection:
                connection.recv(16)
                self.flush()
                connection.sendall(b"flushed\n")

    def begin_prefill(self, tokens, starts, ends, slots, emit):
        started = time.perf_counter()
        rows = []
        for row, (start, end, slot, enabled) in enumerate(zip(starts, ends, slots, emit)):
            if start == 0:
                self.serial += 1
                self.requests[slot] = self.serial
            ids = tokens[row, :end].tolist()
            rows.append(
                dict(
                    request=self.requests[slot],
                    slot=int(slot),
                    start=int(start),
                    end=int(end),
                    emit=bool(enabled),
                    prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                )
            )
        self.current_prefill = dict(phase="prefill", started=started, rows=rows, chunks=[])
        return self.current_prefill

    def chunk(self, n, slot):
        if self.current_prefill is not None:
            self.current_prefill["chunks"].append(dict(tokens=int(n), slot=int(slot)))

    def end_prefill(self, event, sampled_completion):
        self.current_prefill = None
        event["returned"] = time.perf_counter()
        self.prefill_pending.append(event)
        if sampled_completion:
            for pending in self.prefill_pending:
                pending["completed"] = event["returned"]
                pending["completion_boundary"] = "existing synchronous prefill token/logit readback"
                self.append(pending)
            self.prefill_pending = []

    def begin_decode(self):
        return time.perf_counter()

    def submitted_decode(self, output, started, remap):
        old = list(self.requests)
        self.requests = [old[int(src)] for src in remap]
        g = self.generator
        event = dict(
            phase="decode",
            serial=int(output.serial),
            started=started,
            requests=list(self.requests),
            active=list(output.active),
            positions=[g._position(i) - 1 if output.active[i] else -1 for i in range(g.batch_size)],
            executed_slots=g.batch_size,
        )
        self.pending[output.serial] = event

    def completed_decode(self, output):
        event = self.pending.pop(output.serial, None)
        if event is None:
            return
        event["completed"] = time.perf_counter()
        event[
            "completion_boundary"
        ] = "existing async read event awaited by plugin, then host conversion and token validation"
        self.append(event)
        for pending in self.prefill_pending:
            pending["completed"] = event["completed"]
            pending[
                "completion_boundary"
            ] = "next existing decode readback (intermediate prefill remained asynchronous)"
            self.append(pending)
        self.prefill_pending = []
