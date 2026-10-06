# SPDX-License-Identifier: Apache-2.0
"""Passive completion audit: never invent a boundary for an unfinished prefill.

All callbacks observe existing host execution. No TT operation, event, readback,
wait, profiler, tensor retention, or synchronization is added. Mixed unfenced
prefill/decode epochs are explicitly rejected by the benchmark collector.
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


def install():
    global _INSTALLED
    path = os.environ.get("KOLIBRI_PHASE_EVENTS")
    if not path or _INSTALLED:
        return
    from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    _INSTALLED = True

    def plain(x):
        return x.tolist() if hasattr(x, "tolist") else x

    def meta(runner, mi):
        return dict(
            start=runner._kolibri_execute_start,
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
            data["observation_index"] = len(_EVENTS)
            if intermediate:
                data["host_output_time"] = now
                data["device_completion_observed"] = False
                _PENDING_PREFILLS.append(data["observation_index"])
            else:
                data["end"] = now
                data["device_completion_observed"] = True
                data["covers_pending_prefills"] = list(_PENDING_PREFILLS)
                _PENDING_PREFILLS.clear()
            _EVENTS.append(data)

    original_execute = TTModelRunner.execute_model

    @functools.wraps(original_execute)
    def execute(runner, *args, **kwargs):
        runner._kolibri_execute_start = time.perf_counter()
        return original_execute(runner, *args, **kwargs)

    TTModelRunner.execute_model = execute

    original_forward = TTModelRunner._forward_with_model_input

    @functools.wraps(original_forward)
    def forward(runner, model_input):
        data = meta(runner, model_input)
        fwd = original_forward(runner, model_input)
        if fwd is not None:
            if not hasattr(runner, "_kolibri_sync_phases"):
                runner._kolibri_sync_phases = {}
            runner._kolibri_sync_phases[id(fwd)] = data
        return fwd

    TTModelRunner._forward_with_model_input = forward

    original_sync = TTModelRunner._finish_front_packed_sync

    @functools.wraps(original_sync)
    def sync(runner, *args, fwd=None, **kwargs):
        result = original_sync(runner, *args, fwd=fwd, **kwargs)
        if fwd is not None:
            data = runner._kolibri_sync_phases.pop(id(fwd))
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
        result = original_submit(controller, model_input, *args, **kwargs)
        result._kolibri_phase = data
        return result

    TTAsyncDecodeController.submit_async_decode = submit

    original_complete = AsyncTTModelRunnerOutput._get_output_impl

    @functools.wraps(original_complete)
    def complete(wrapper):
        result = original_complete(wrapper)
        finish(wrapper._kolibri_phase)
        return result

    AsyncTTModelRunnerOutput._get_output_impl = complete

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with _LOCK:
                snapshot = list(_EVENTS)
            payload = dict(
                pid=os.getpid(),
                instrumentation=__file__,
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
    print(f"KOLIBRI_PHASE_TIMING_READY {__file__} pid={os.getpid()}", flush=True)
