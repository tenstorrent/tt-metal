# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Passive completion audit: never invent a boundary for an unfinished prefill.

All callbacks observe existing host execution. No TT operation, event, readback,
wait, profiler, tensor retention, or synchronization is added. Mixed unfenced
prefill/decode epochs use the owner-authorized inclusive elapsed bound.
"""

import functools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_EVENTS = []
_PENDING_PREFILLS = []
_INSTALLED = False
_LOCK = threading.Lock()
_NEXT_INDEX = 0


def install(model, identity):
    global _INSTALLED
    path = os.environ.get("MUSE_PHASE_EVENTS")
    if not path or _INSTALLED:
        return
    from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    _INSTALLED = True
    active = threading.local()

    def plain(x):
        return x.tolist() if hasattr(x, "tolist") else x

    def meta(runner, mi):
        global _NEXT_INDEX
        with _LOCK:
            index = _NEXT_INDEX
            _NEXT_INDEX += 1
        return dict(
            observation_index=index,
            start=runner._muse_execute_start,
            phase="decode" if mi.prompt_lens is None else "prefill",
            requests=list(mi.row_req_ids),
            positions=plain(mi.input_positions),
            prompt_lens=plain(mi.prompt_lens),
            physical_slots=runner.model.batch_size,
            devices=4,
            intermediate=plain(mi.intermediate_prefill_mask),
        )

    def finish(data, *, intermediate=False):
        now = time.perf_counter()
        with _LOCK:
            if intermediate:
                data["host_output_time"] = now
                data["device_completion_observed"] = False
                _PENDING_PREFILLS.append(data["observation_index"])
            else:
                data["end"] = now
                data["device_completion_observed"] = True
                # A read completion only covers commands submitted earlier on
                # the shared CQ. A newer unfenced prefill may already exist.
                covered = [i for i in _PENDING_PREFILLS if i < data["observation_index"]]
                data["covers_pending_prefills"] = covered
                _PENDING_PREFILLS[:] = [i for i in _PENDING_PREFILLS if i not in covered]
            _EVENTS.append(data)

    original_execute = TTModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute(runner, *args, **kwargs):
        runner._muse_execute_start = time.perf_counter()
        return original_execute(runner, *args, **kwargs)

    TTModelRunner.execute_model = execute

    original_forward = TTModelRunner._forward_with_model_input

    @functools.wraps(original_forward)
    def forward(runner, model_input):
        data = meta(runner, model_input)
        active.data = data
        try:
            fwd = original_forward(runner, model_input)
        finally:
            active.data = None
        if fwd is not None:
            if not hasattr(runner, "_muse_sync_phases"):
                runner._muse_sync_phases = {}
            runner._muse_sync_phases[id(fwd)] = data
        return fwd

    TTModelRunner._forward_with_model_input = forward

    original_sync = TTModelRunner._finish_front_packed_sync

    @functools.wraps(original_sync)
    def sync(runner, *args, fwd=None, **kwargs):
        result = original_sync(runner, *args, fwd=fwd, **kwargs)
        if fwd is not None:
            data = runner._muse_sync_phases.pop(id(fwd))
            intermediate = data["phase"] == "prefill" and all(plain(fwd.model_input.intermediate_prefill_mask))
            finish(data, intermediate=intermediate)
        return result

    TTModelRunner._finish_front_packed_sync = sync

    original_submit = TTAsyncDecodeController.submit_async_decode

    @functools.wraps(original_submit)
    def submit(controller, model_input, *args, **kwargs):
        data = meta(controller.runner, model_input)
        with _LOCK:
            data["unfenced_prefill_at_decode_submission"] = list(_PENDING_PREFILLS)
        active.data = data
        try:
            result = original_submit(controller, model_input, *args, **kwargs)
        finally:
            active.data = None
        result._muse_phase = data
        return result

    TTAsyncDecodeController.submit_async_decode = submit

    original_complete = AsyncTTModelRunnerOutput._get_output_impl

    @functools.wraps(original_complete)
    def complete(wrapper):
        if not hasattr(wrapper, "_muse_phase"):
            raise RuntimeError("Missing immutable timing metadata for async completion")
        result = original_complete(wrapper)
        finish(wrapper._muse_phase)
        return result

    AsyncTTModelRunnerOutput._get_output_impl = complete

    original_decode = model.decode_forward

    @functools.wraps(original_decode)
    def decode(*args, **kwargs):
        data = getattr(active, "data", None)
        if data is not None:
            reload = kwargs.get("reload_inputs", True)
            data["scheduler_positions"] = data["positions"]
            # This is the generator's existing host ledger: authoritative
            # uploads on reload, incremented once by every resident replay.
            data["positions"] = plain(kwargs["start_pos"]) if reload else list(model.generator._host_positions)
            data["reload_inputs"] = reload
            data["reload_page_table"] = kwargs.get("reload_page_table", False)
            data["device_sampling"] = kwargs.get("sampling_params") is not None
        return original_decode(*args, **kwargs)

    model.decode_forward = decode

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with _LOCK:
                snapshot = list(_EVENTS)
            payload = dict(
                pid=os.getpid(),
                instrumentation=__file__,
                identity=identity,
                counters=dict(model.generator.counters) if model.generator else {},
                audit=dict(model.audit),
                events=snapshot,
                monotonic=time.perf_counter(),
                wall_time=time.time(),
            )
            body = json.dumps(payload).encode()
            Path(path).write_text("\n".join(json.dumps(e) for e in snapshot) + "\n")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8001), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"MUSE_PHASE_TIMING_READY {__file__} pid={os.getpid()}", flush=True)
