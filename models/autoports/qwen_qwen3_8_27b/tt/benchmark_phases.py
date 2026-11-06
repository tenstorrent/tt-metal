# SPDX-License-Identifier: Apache-2.0
"""Opt-in host completion evidence; no TT operations or device synchronization."""
import copy
import hashlib
import json
import os
import struct
import threading
import time
from pathlib import Path


class PhaseRecorder:
    """Buffer immutable invocation snapshots; flush only on a collector request."""

    def __init__(self, path, batch, *, clock=time.perf_counter, background=True):
        self.path = Path(path)
        self.batch = batch
        self.clock = clock
        self.events = []
        self.sessions = [None] * batch
        self.hashes = [None] * batch
        self.next_session = 0
        self.next_submission = 0
        self.completed = set()
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.with_suffix(self.path.suffix + ".ready").write_text(
            json.dumps(
                dict(
                    schema_version=1,
                    pid=os.getpid(),
                    clock="time.perf_counter",
                    units="seconds",
                    batch=batch,
                    prompt_hash_scheme="sha256(tokens:i64le)",
                    completion="host output processing after existing readback fence",
                )
            )
            + "\n"
        )
        if background:
            threading.Thread(target=self._flush_loop, daemon=True, name="qwen-phase-flush").start()

    def _flush_loop(self):
        flag = Path(str(self.path) + ".flush")
        while True:
            if flag.exists():
                try:
                    self.flush()
                    flag.unlink()
                except Exception as exc:
                    Path(str(self.path) + ".error").write_text(repr(exc) + "\n")
            time.sleep(0.05)

    def flush(self):
        with self.lock:
            snapshot = list(self.events)
        tmp = Path(str(self.path) + ".tmp")
        tmp.write_text("".join(json.dumps(row) + "\n" for row in snapshot))
        tmp.replace(self.path)

    def remap(self, mapping):
        self.sessions = [self.sessions[source] for source in mapping]
        self.hashes = [self.hashes[source] for source in mapping]

    def release(self, slot):
        self.sessions[slot] = None
        self.hashes[slot] = None

    def prefill_sessions(self, slots, executed_starts, token_rows):
        for slot, start, tokens in zip(slots, executed_starts, token_rows):
            if start == 0 or self.sessions[slot] is None:
                self.next_session += 1
                self.sessions[slot] = self.next_session
            packed = struct.pack("<" + "q" * len(tokens), *tokens)
            self.hashes[slot] = hashlib.sha256(packed).hexdigest()

    def begin(self, phase, start, slots, **metadata):
        self.next_submission += 1
        return copy.deepcopy(
            dict(
                phase=phase,
                start=start,
                submission_id=self.next_submission,
                batch=self.batch,
                slots=list(slots),
                sessions=[self.sessions[slot] for slot in slots],
                prompt_hashes=[self.hashes[slot] for slot in slots],
                **metadata,
            )
        )

    def finish(self, pending, *, output_tokens=None, fenced=True, mode="device"):
        end = self.clock() if fenced else None
        row = dict(
            pending,
            end=end,
            completion="complete" if fenced else "unfenced",
            output_tokens=list(output_tokens) if output_tokens is not None else None,
            mode=mode,
        )
        with self.lock:
            if pending["submission_id"] in self.completed:
                raise RuntimeError("Phase submission completed twice")
            self.completed.add(pending["submission_id"])
            self.events.append(row)
