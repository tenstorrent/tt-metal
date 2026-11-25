# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Optional CPU-only phase markers, consumed at existing host readback boundaries.

No TTNN calls, synchronization, signal handlers or per-token file writes occur in
these hooks. A background writer drains immutable events and services flushes.
"""

import collections
import hashlib
import json
import os
import threading
import time
from pathlib import Path


class PhaseTiming:
    def __init__(self, path, slots, query_event=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise ValueError(f"Phase log must be fresh: {self.path}")
        self.queue = collections.deque()
        self.slots = [None] * slots
        self.positions = [-1] * slots
        self.request_serial = self.serial = 0
        self.query_event = query_event
        self.observations = {}
        self.next_completion = 1
        self.writer_error = None
        self.queue.append(dict(kind="startup", pid=os.getpid(), clock="perf_counter", time=time.perf_counter()))
        self.thread = threading.Thread(target=self._writer, daemon=True, name="qwen-phase-writer")
        self.thread.start()

    def _writer(self):
        try:
            with self.path.open("x") as out:
                last_flush = 0.0
                while True:
                    # FIFO CQ completion observations, never a blocking wait.
                    while self.next_completion in self.observations:
                        observation = self.observations[self.next_completion]
                        if observation["event"] is not None and not self.query_event(observation["event"]):
                            break
                        del self.observations[self.next_completion]
                        self.next_completion += 1
                        record = dict(
                            kind="complete",
                            phase=observation["phase"],
                            serials=[observation["serial"]],
                            end=time.perf_counter(),
                            host_completion=observation["host_completion"],
                            boundary=observation["boundary"],
                        )
                        self.queue.append(record)
                    now = time.perf_counter()
                    if now - last_flush < 0.2:
                        time.sleep(0.0005)
                        continue
                    last_flush = now
                    while self.queue:
                        out.write(json.dumps(self.queue.popleft(), separators=(",", ":")) + "\n")
                    out.flush()
                    request = Path(str(self.path) + ".flush")
                    if request.exists():
                        nonce = request.read_text()
                        # Drain once more before acknowledgement of the snapshot.
                        while self.queue:
                            out.write(json.dumps(self.queue.popleft(), separators=(",", ":")) + "\n")
                        out.flush()
                        Path(str(self.path) + ".flushed").write_text(nonce)
                    time.sleep(0.0005)
        except Exception as error:
            self.writer_error = repr(error)

    def ready(self, metadata):
        metadata = dict(
            metadata,
            pid=os.getpid(),
            phase_log=str(self.path),
            clock="perf_counter",
            instrumentation_version=1,
            server_slots=len(self.slots),
        )
        target = Path(str(self.path) + ".ready.json")
        temp = Path(str(target) + ".tmp")
        temp.write_text(json.dumps(metadata, indent=2, default=str) + "\n")
        temp.replace(target)

    def prefill(self, starts, ends, slots, original_lens, sampling_mask, token_prefixes, started):
        rows = []
        for start, end, slot, original, final, prefix in zip(
            starts, ends, slots, original_lens, sampling_mask, token_prefixes
        ):
            if start == 0:
                self.request_serial += 1
                self.slots[slot] = f"{os.getpid()}:{self.request_serial}"
            if self.slots[slot] is None:
                raise ValueError("Timing cannot identify resumed prefill without initial chunk")
            rows.append(
                dict(
                    request=self.slots[slot],
                    slot=slot,
                    start=start,
                    end=end,
                    original_prompt_len=original,
                    sampled=final,
                    prefix_sha256=(hashlib.sha256(json.dumps(prefix).encode()).hexdigest() if start == 0 else None),
                )
            )
        event = self._begin("prefill", rows, started)
        return event

    def decode(self, positions, remap, reload_inputs, execution_rows, started):
        if remap is not None:
            self.slots = [self.slots[i] for i in remap]
            self.positions = [self.positions[i] for i in remap]
        if reload_inputs:
            self.positions = positions + [-1] * (len(self.positions) - len(positions))
        rows = [
            dict(request=self.slots[i], slot=i, position=pos)
            for i, pos in enumerate(self.positions)
            if i < execution_rows and pos >= 0
        ]
        if any(row["request"] is None for row in rows):
            raise ValueError("Timing cannot identify active decode slot")
        for row in rows:
            self.positions[row["slot"]] += 1
        return self._begin("decode", rows, started, execution_rows=execution_rows)

    def _begin(self, phase, rows, started, **extra):
        if self.writer_error:
            raise RuntimeError(self.writer_error)
        self.serial += 1
        event = dict(kind="submit", phase=phase, serial=self.serial, start=started, rows=rows, **extra)
        self.queue.append(event)
        return event

    def complete_prefill(self, event, device_event):
        self.observations[event["serial"]] = dict(
            phase="prefill",
            serial=event["serial"],
            event=device_event,
            host_completion=time.perf_counter(),
            boundary="nonblocking_prefill_CQ_event_query",
        )

    def read_decode(self, event, device_event):
        if event is not None:
            self.observations[event["serial"]] = dict(
                phase="decode",
                serial=event["serial"],
                event=device_event,
                host_completion=time.perf_counter(),
                boundary="existing_decode_readback_event_query",
            )

    def complete_decode(self, event, started):
        if event is not None:
            self.queue.append(
                dict(kind="host_complete", serial=event["serial"], start=started, end=time.perf_counter())
            )

    def release(self, slot):
        self.queue.append(dict(kind="release", request=self.slots[slot], slot=slot, time=time.perf_counter()))
        self.slots[slot] = None
        self.positions[slot] = -1


def tensor_inventory(model, ttnn):
    """Read tensor metadata only, once during setup. No tensor readback."""
    result = []
    for layer in model.layers:
        weights = dict(layer.weights)
        # A_log is transformed once during setup; decode reads neg_a instead.
        weights.pop("linear_attn.A_log", None)
        if layer.kind == "linear_attention":
            weights["neg_a"] = layer.neg_a
            weights.update({f"conv_tap_{i}": t for i, t in enumerate(layer.taps_rm)})
        for name, tensor in weights.items():
            for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                result.append(
                    dict(
                        layer=layer.layer_idx,
                        name=name,
                        rank=rank,
                        shape=list(shard.shape),
                        padded_shape=list(shard.padded_shape),
                        dtype=str(shard.dtype),
                        layout=str(shard.layout),
                    )
                )
    for name, tensor in (("lm_head", model.lm_head), ("embedding", model.embedding)):
        for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
            result.append(
                dict(
                    layer=None,
                    name=name,
                    rank=rank,
                    shape=list(shard.shape),
                    padded_shape=list(shard.padded_shape),
                    dtype=str(shard.dtype),
                    layout=str(shard.layout),
                )
            )
    return result
