# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host markers, buffered until the benchmark requests a snapshot.

No TT synchronization, profiler, or device read is introduced. Decode completion
is recorded after the existing host materialization, never at trace submission.
The abstract Unix socket copies Python-only data when the client is idle.
"""

import hashlib
import inspect
import json
import os
import socket
import threading
import time
from pathlib import Path


def socket_address(path):
    return "\0ornith-phase-" + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()[:32]


def scheduler_rows():
    """Read existing immutable input metadata; never query device state.

    The optional observer lives in the autoport instead of changing a shared
    plugin. Fail closed on unsupported plugin call stacks during collection.
    """
    frame = inspect.currentframe()
    try:
        for _ in range(7):
            frame = frame.f_back
            if frame is None:
                break
            inp = frame.f_locals.get("model_input")
            if inp is None:
                continue
            runner = frame.f_locals.get("runner") or frame.f_locals.get("self")
            if not hasattr(runner, "input_batch"):
                runner = getattr(runner, "runner", None)
            ids = getattr(inp, "row_req_ids", None)
            if ids is None and runner is not None:
                ids = runner.input_batch.req_ids
            live = getattr(inp, "live_rows", None)
            if live is not None and hasattr(live, "tolist"):
                live = live.tolist()
            return {"request_ids": list(ids or []), "live_rows": list(live) if live is not None else None}
        return {"request_ids": [], "live_rows": None}
    finally:
        del frame


def tensor_record(tensor):
    import ttnn

    shards = ttnn.get_device_tensors(tensor)
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "shard_shapes": [list(t.shape) for t in shards],
        "stored_bytes_by_rank": [int(t.buffer_num_pages()) * int(t.buffer_page_size()) for t in shards],
        "memory": str(tensor.memory_config()),
    }


def model_accounting(adapter):
    model = adapter.model
    layers = []
    for index, layer, cache in zip(model.layer_indices, model.layers, adapter.generator.cache):
        rec = {
            "index": index,
            "kind": layer.kind,
            "weights": {name: tensor_record(t) for name, t in layer.dram_weights.items()},
            "norms": {name: tensor_record(t) for name, t in layer.weights.items() if "norm" in name},
        }
        if layer.kind == "linear_attention":
            rec["recurrent"] = tensor_record(cache[0].recurrent)
            rec["conv"] = tensor_record(cache[0].conv)
            rec["conv_constants"] = [tensor_record(t) for t in [*layer.float_taps, layer.dt_bias, layer.decay]]
        layers.append(rec)
    grid = model.device.compute_with_storage_grid_size()
    source = Path(inspect.getfile(type(adapter))).resolve()
    head = model.decode_head.output_weights[0] if model.decode_head else model.head
    return {
        "generator_module": type(adapter).__module__,
        "generator_module_file": str(source),
        "generator_module_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layer_count": len(model.layers),
        "configured_layer_count": model.config.num_hidden_layers,
        "layer_indices": list(model.layer_indices),
        "precision": model.precision_summary(),
        "checkpoint": str(model.checkpoint.path),
        "model_revision": model.checkpoint.path.name,
        "max_num_seqs": adapter.max_batch_size,
        "max_model_len": adapter.max_seq_len,
        "mesh_shape": list(model.device.shape),
        "compute_grid": [grid.x, grid.y],
        "participating_devices": list(model.device.get_device_ids()),
        "layers": layers,
        "head": tensor_record(head),
        "config": model.config.to_dict(),
        "clock_hz": 1.35e9,
        "dram_bytes_per_second_per_chip": 512e9,
        "peak_source": "https://docs.tenstorrent.com/aibs/blackhole/p300.html; https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/matrix_engine/matrix_engine.md",
    }


class PhaseRecorder:
    def __init__(self, adapter, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.events = []
        self.pending = {}
        self.next_id = 0
        self.metadata = model_accounting(adapter)
        self.metadata.update(pid=os.getpid(), monotonic=time.perf_counter(), wall_time=time.time(), schema=1)
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(socket_address(self.path))
        self.socket.listen(2)
        self.write_snapshot()
        threading.Thread(target=self._serve, daemon=True, name="ornith-phase-snapshot").start()

    def write_snapshot(self):
        # Invoked on explicit collector request, outside measured traffic.
        snapshot = {
            "metadata": self.metadata,
            "events": list(self.events),
            "pending": list(self.pending.values()),
            "snapshot_monotonic": time.perf_counter(),
        }
        temporary = self.path.with_suffix(self.path.suffix + ".new")
        temporary.write_text(json.dumps(snapshot, separators=(",", ":")))
        temporary.replace(self.path)

    def _serve(self):
        while True:
            connection, _ = self.socket.accept()
            with connection:
                if connection.recv(16).strip() == b"snapshot":
                    try:
                        self.write_snapshot()
                        connection.sendall(b"OK\n")
                    except Exception as exc:
                        connection.sendall(str(exc).encode()[:1024])

    def begin(self, phase):
        started = time.perf_counter()
        event = {"phase": phase, "start": started, "id": self.next_id, **scheduler_rows()}
        self.next_id += 1
        return event

    def prefill_end(self, event, starts, ends, originals, slots, discarded):
        completed = time.perf_counter()
        event.update(
            end=completed,
            starts=starts,
            ends=ends,
            original_prompt_lens=list(originals),
            slots=list(slots) if slots is not None else list(range(len(ends))),
            discarded=list(discarded),
            completed=not all(d or e == 0 for d, e in zip(discarded, ends)),
        )
        self.events.append(event)

    def decode_submitted(self, event, result, generator):
        submitted = time.perf_counter()
        state = generator.states[(id(generator.cache), tuple(range(generator.max_batch_size)))]
        event.update(
            submitted=submitted,
            positions=[p + state.position_advances - 1 if p >= 0 else -1 for p in state.host_positions],
            active=list(state.batch_state.active),
            slots=list(state.slots),
        )
        if hasattr(result, "submission_id"):
            event["submission_id"] = result.submission_id
            self.pending[result.submission_id] = event
        else:
            event.update(end=submitted, completed=True, synchronous=True)
            self.events.append(event)

    def decode_complete(self, handle):
        completed = time.perf_counter()
        if not hasattr(handle, "submission_id"):
            return
        event = self.pending.pop(handle.submission_id, None)
        if event is not None:
            event.update(end=completed, completed=True)
            self.events.append(event)
