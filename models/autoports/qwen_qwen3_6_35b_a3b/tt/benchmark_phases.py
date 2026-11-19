# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in CPU metadata recorder at the adapter's existing completion boundaries.

No device calls, synchronization, token reads, or per-step file output. SIGUSR1
exports a snapshot only when the external collector has observed an idle server.
"""

import json
import os
import signal
import threading
import time
from pathlib import Path


class PhaseRecorder:
    def __init__(self, directory, capacity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.lock = threading.RLock()
        self.events = []
        self.pending = {}
        self.slots = [None] * capacity
        self.serial = 0
        self.call = 0
        self.instance = f"{os.getpid()}-{time.time_ns()}"
        self.record("ready", capacity=capacity, pid=os.getpid(), clock="CLOCK_MONOTONIC", instance=self.instance)
        if signal.getsignal(signal.SIGUSR1) != signal.SIG_DFL:
            raise RuntimeError("Cannot replace an existing SIGUSR1 handler")
        signal.signal(signal.SIGUSR1, self.flush)
        self.flush()
        print(
            "QWEN_BENCHMARK_READY "
            + json.dumps(dict(pid=os.getpid(), directory=str(self.directory), instance=self.instance)),
            flush=True,
        )

    def record(self, kind, **fields):
        with self.lock:
            self.events.append(dict(kind=kind, time=time.perf_counter(), seq=len(self.events), **fields))

    def flush(self, *_):
        with self.lock:
            snapshot = dict(
                instance=self.instance,
                pid=os.getpid(),
                captured=time.perf_counter(),
                pending=list(self.pending),
                events=list(self.events),
            )
        temp = self.directory / "snapshot.tmp"
        temp.write_text(json.dumps(snapshot))
        temp.replace(self.directory / "snapshot.json")

    def prefill_start(self, starts, ends, slots):
        with self.lock:
            self.call += 1
            call = self.call
            chunks = []
            for slot, start, end in zip(slots, starts, ends):
                if start == 0:
                    self.serial += 1
                    self.slots[slot] = dict(request=self.serial, position=0, prompt_end=0)
                state = self.slots[slot]
                if state is None or state["prompt_end"] != start:
                    raise RuntimeError("Phase recorder cannot prove contiguous prefill ownership")
                state["position"] = state["prompt_end"] = end
                chunks.append(dict(slot=slot, request=state["request"], start=start, end=end))
            self.record("prefill_start", call=call, chunks=chunks)
            return call

    def prefill_done(self, call):
        self.record("prefill_done", call=call)

    def decode_start(self, *, positions, rows, remap, reload_inputs):
        with self.lock:
            self.call += 1
            call = self.call
            old = self.slots
            order = list(range(self.capacity)) if remap is None else list(remap)
            self.slots = [old[index] for index in order]
            if reload_inputs:
                for slot, position in zip(rows, positions):
                    if position >= 0:
                        if self.slots[slot] is None:
                            raise RuntimeError("Decode has no observed prefill lifetime")
                        self.slots[slot]["position"] = position
            self.record("decode_start", call=call, reload_inputs=reload_inputs, remap=order)
            return call

    def decode_accepted(self, call, handle, active, trace_key):
        with self.lock:
            members = []
            for slot in active:
                state = self.slots[slot]
                if state is None:
                    raise RuntimeError("Active decode slot lacks request ownership")
                members.append(dict(slot=slot, request=state["request"], position=state["position"]))
                state["position"] += 1
            self.pending[handle.generation] = call
            self.record(
                "decode_accepted",
                call=call,
                generation=handle.generation,
                lease=handle.lease,
                rows=list(handle.rows),
                members=members,
                trace_key=trace_key,
                is_tokens=handle.is_tokens,
            )

    def decode_done(self, handle):
        with self.lock:
            call = self.pending.pop(handle.generation)
            self.record("decode_done", call=call, generation=handle.generation)

    def release(self, slot):
        with self.lock:
            state = self.slots[slot]
            self.record("release", slot=slot, request=None if state is None else state["request"])
            self.slots[slot] = None
