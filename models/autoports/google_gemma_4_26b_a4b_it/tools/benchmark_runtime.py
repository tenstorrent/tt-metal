# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in host completion observer; never waits on or reads a device itself."""

import functools
import hashlib
import inspect
import json
import os
import socket
import threading
import time
from pathlib import Path


class Recorder:
    def __init__(self, identity):
        self.identity = identity
        self.events = []
        self.pending = {}
        self.requests = {}
        self.next_positions = {}
        self.errors = []
        self.lock = threading.Lock()
        self.sequence = 0
        self.last_completion = 0

    def begin(self, runner, model_input):
        ids = list(runner.input_batch.req_ids[: runner.input_batch.num_reqs])
        if not ids or not any("gemma4-" in rid for rid in ids):
            return None
        if not all("gemma4-" in rid for rid in ids):
            self.errors.append("mixed observed and unobserved request cohort")
        prefill = model_input.prompt_lens is not None
        lens = [int(x) for x in model_input.prompt_lens] if prefill else []
        supplied = [int(x) for x in model_input.input_positions.reshape(-1).tolist()]
        with self.lock:
            for rid in ids:
                if rid not in self.requests:
                    state = runner.requests[rid]
                    tokens = state.prompt_token_ids
                    self.requests[rid] = {
                        "prompt_tokens": len(tokens),
                        "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        "max_tokens": state.sampling_params.max_tokens,
                        "temperature": state.sampling_params.temperature,
                        "top_k": state.sampling_params.top_k,
                    }
            positions = []
            if prefill:
                if len(lens) != len(ids) or any(supplied):
                    self.errors.append("unsupported prefill rows or continuation")
                for rid, length in zip(ids, lens):
                    self.next_positions[rid] = length
            else:
                for row, rid in enumerate(ids):
                    position = self.next_positions.get(rid, supplied[row])
                    if model_input.reset_batch and supplied[row] != position:
                        self.errors.append(f"position reset mismatch {rid}: {supplied[row]} != {position}")
                    positions.append(position)
                    self.next_positions[rid] = position + 1
            self.sequence += 1
            sid = str(self.sequence)
            started = getattr(runner, "_gemma_benchmark_entry_ns", time.perf_counter_ns())
            # execute_model drains older asynchronous work before prefill. That
            # wait belongs to the older decode's completion interval.
            if prefill:
                started = max(started, self.last_completion)
            record = {
                "submission_id": sid,
                "phase": "prefill" if prefill else "decode",
                "request_ids": ids,
                "timestamp_ns": started,
                "event": "dispatch",
                "prompt_lens": lens,
                "positions": positions,
                "batch_slots": len(ids),
                "wire_rows": len(supplied),
                "device_sampling": bool(model_input.perform_device_sampling),
            }
            self.pending[sid] = record
            self.events.append(record)
            return sid

    def complete(self, sid):
        if sid is None:
            return
        completed = time.perf_counter_ns()
        with self.lock:
            record = self.pending.pop(sid)
            self.events.append({**record, "event": "completion", "timestamp_ns": completed})
            self.last_completion = max(self.last_completion, completed)

    def snapshot(self, full=False):
        with self.lock:
            result = {
                "schema_version": 1,
                "identity": self.identity,
                "pending": list(self.pending),
                "errors": list(self.errors),
                "clock": "time.perf_counter_ns",
                "export_time_ns": time.perf_counter_ns(),
                "requests": dict(self.requests),
                "event_count": len(self.events),
            }
            if full:
                result["events"] = list(self.events)
            return result


def serve_control(recorder, path):
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    os.chmod(path, 0o600)
    sock.listen(4)

    def respond():
        while True:
            conn, _ = sock.accept()
            with conn:
                conn.settimeout(10)
                command = conn.recv(1024).decode().strip()
                data = recorder.snapshot(full=command == "export")
                conn.sendall(json.dumps(data).encode())

    threading.Thread(target=respond, name="benchmark-export", daemon=True).start()


def install(model, control_path):
    """Wrap only the exact loaded model's normal dispatch/finalization paths."""
    from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    source = Path(inspect.getfile(type(model))).resolve()
    gen = model.generator
    identity = {
        "pid": os.getpid(),
        "generator_module": type(model).__module__,
        "generator_file": str(source),
        "generator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "layer_count": len(gen.model.layers),
        "configured_layer_count": gen.model.config.num_hidden_layers,
        "layer_indices": list(gen.model.layer_indices),
        "mesh": list(gen.mesh.shape),
        "max_model_len": gen.model.max_seq_len,
        "max_num_seqs": model.max_batch_size,
        "precision": gen.model.precision_summary(),
        "model_revision": "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        "tokenizer_revision": gen.tokenizer.init_kwargs.get("revision"),
        "plugin_file": inspect.getfile(TTModelRunner),
        "clock_anchor": {"monotonic_ns": time.perf_counter_ns(), "wall_ns": time.time_ns()},
        "hostname": socket.gethostname(),
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "profiling": False,
    }
    recorder = Recorder(identity)
    model._benchmark_recorder = recorder
    forward_ids = {}

    def wrap(cls, name, factory):
        original = getattr(cls, name)
        setattr(cls, name, functools.wraps(original)(factory(original)))

    def execute(original):
        def call(runner, *args, **kwargs):
            if runner.model is model:
                runner._gemma_benchmark_entry_ns = time.perf_counter_ns()
            return original(runner, *args, **kwargs)

        return call

    def forward(original):
        def call(runner, model_input, *args, **kwargs):
            sid = recorder.begin(runner, model_input) if runner.model is model else None
            result = original(runner, model_input, *args, **kwargs)
            if sid is not None:
                forward_ids[id(result)] = sid
            return result

        return call

    def finish_sync(original):
        def call(runner, *args, **kwargs):
            fwd = kwargs.get("fwd")
            sid = forward_ids.pop(id(fwd), None)
            result = original(runner, *args, **kwargs)
            recorder.complete(sid)
            return result

        return call

    def submit_async(original):
        def call(controller, model_input, *args, **kwargs):
            sid = recorder.begin(controller.runner, model_input) if controller.runner.model is model else None
            result = original(controller, model_input, *args, **kwargs)
            result._gemma_benchmark_sid = sid
            return result

        return call

    def finish_async(original):
        def call(output, *args, **kwargs):
            result = original(output, *args, **kwargs)
            if output._controller.runner.model is model:
                recorder.complete(getattr(output, "_gemma_benchmark_sid", None))
            return result

        return call

    wrap(TTModelRunner, "execute_model", execute)
    wrap(TTModelRunner, "_forward_with_model_input", forward)
    wrap(TTModelRunner, "_finish_nondp_sync", finish_sync)
    wrap(TTAsyncDecodeController, "submit_async_non_dp_decode", submit_async)
    wrap(AsyncTTModelRunnerOutput, "_get_output_impl", finish_async)
    serve_control(recorder, control_path)
    print("BENCHMARK_COMPLETION_OBSERVER_READY " + json.dumps(identity), flush=True)
    return recorder
