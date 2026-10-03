# SPDX-License-Identifier: Apache-2.0
"""Opt-in host events at existing vLLM async output completion boundaries.

No device calls, waits, or synchronization are added. A background thread writes
buffered events; the collector reconciles overlapping submissions on one timeline.
"""
import hashlib
import json
import os
import threading
import time
from collections import deque
from functools import wraps
from pathlib import Path

_installed = False


def install():
    global _installed
    path = os.environ.get("GRANITE_BENCHMARK_PHASES")
    if not path or _installed:
        return
    from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    from .generator import GraniteGenerator

    _installed = True
    queue = deque()
    local = threading.local()
    seen = set()
    counter = 0
    pending_prefills = []

    def writer():
        trigger = Path(path + ".flush")
        acknowledgement = Path(path + ".flushed")
        last = None
        with open(path, "a", buffering=1) as stream:
            stream.write(
                json.dumps(
                    dict(
                        event="collector_ready",
                        pid=os.getpid(),
                        time=time.perf_counter(),
                        clock="perf_counter",
                        synchronization_added=False,
                        export_policy="explicit_idle_flush",
                    )
                )
                + "\n"
            )
            while True:
                token = trigger.read_text() if trigger.exists() else None
                if token is not None and token != last:
                    # The client requests export only after workload completion.
                    # No JSON encoding or filesystem flush runs per model step.
                    count = len(queue)
                    for _ in range(count):
                        stream.write(json.dumps(queue.popleft(), default=str) + "\n")
                    stream.flush()
                    acknowledgement.write_text(token)
                    last = token
                time.sleep(0.1)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=writer, name="granite-phase-writer", daemon=True).start()

    def finish(record, end=None):
        if record is not None and "end" not in record:
            record["end"] = time.perf_counter() if end is None else end
            queue.append(record.copy())

    def observe_prefills(boundary):
        end = time.perf_counter()
        for record in pending_prefills:
            record["completion_boundary"] = boundary
            finish(record, end)
        pending_prefills.clear()

    setup_decode = GraniteGenerator._setup_decode

    @wraps(setup_decode)
    def setup(self, tokens, positions, table):
        state = setup_decode(self, tokens, positions, table)
        # _setup_decode already reads sampler seeds synchronously. Observe that
        # natural CQ completion, including earlier intermediate-only prefill.
        record = getattr(local, "record", None)
        if record is not None and record.get("phase") == "decode":
            observe_prefills("existing_decode_reload_seed_read")
        state._benchmark_positions = [int(x) for x in positions] + [-1] * (state.batch - len(positions))
        return state

    replay = GraniteGenerator.replay

    @wraps(replay)
    def replay_with_work(self, state, **kwargs):
        record = getattr(local, "record", None)
        positions = getattr(state, "_benchmark_positions", None)
        output = replay(self, state, **kwargs)
        if record is not None:
            if positions is None:
                raise RuntimeError("Phase collector has no resident-position shadow")
            record.setdefault("replays", []).append(
                dict(bucket=state.batch, positions=list(positions), mode=kwargs.get("sampling_mode", "device"))
            )
            if record["phase"] == "decode":
                record["positions"] = list(positions)
                record["wire_rows"] = state.batch
        if positions is not None:
            state._benchmark_positions = [p + 1 if p >= 0 else -1 for p in positions]
        return output

    execute = TTModelRunner.execute_model

    @wraps(execute)
    def execute_model(self, scheduler_output):
        nonlocal counter
        counter += 1
        record = dict(event="phase", step=counter, begin=time.perf_counter(), devices=4)
        local.record = record
        try:
            result = execute(self, scheduler_output)
            if "phase" in record:
                if not hasattr(self, "_granite_phase_pending"):
                    self._granite_phase_pending = deque()
                self._granite_phase_pending.append(record)
            return result
        finally:
            local.record = None

    build = TTModelRunner.build_model_input

    @wraps(build)
    def build_model_input(self, *args, **kwargs):
        result = build(self, *args, **kwargs)
        record = getattr(local, "record", None)
        if result is not None and record is not None:
            ids = list(result.row_req_ids or self.input_batch.req_ids)
            positions = result.input_positions.reshape(-1).tolist()
            phase = "decode" if result.prompt_lens is None else "prefill"
            record.update(
                phase=phase,
                req_ids=ids,
                positions=positions,
                ends=None if result.prompt_lens is None else [int(x) for x in result.prompt_lens],
                batch=len(ids),
                bucket=1 if len(ids) == 1 else 8 if len(ids) <= 8 else 16,
                device_sampling=result.perform_device_sampling,
                input_ready=time.perf_counter(),
                intermediate=None
                if result.intermediate_prefill_mask is None
                else result.intermediate_prefill_mask.tolist(),
            )
            for rid in ids:
                if rid not in seen:
                    seen.add(rid)
                    req = self.requests[rid]
                    tokens = req.prompt_token_ids
                    queue.append(
                        dict(
                            event="request",
                            req_id=rid,
                            time=record["begin"],
                            prompt_tokens=len(tokens),
                            prompt_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        )
                    )
        return result

    submit = TTAsyncDecodeController.submit_async_decode

    @wraps(submit)
    def submit_async_decode(self, *args, **kwargs):
        output = submit(self, *args, **kwargs)
        output._granite_phase_record = getattr(local, "record", None)
        return output

    complete = AsyncTTModelRunnerOutput._get_output_impl

    @wraps(complete)
    def get_output(self):
        result = complete(self)
        finish(getattr(self, "_granite_phase_record", None))
        return result

    sample = TTModelRunner.sample_tokens

    @wraps(sample)
    def sample_tokens(self, *args, **kwargs):
        pending = getattr(self, "_granite_phase_pending", None)
        record = pending.popleft() if pending else None
        result = sample(self, *args, **kwargs)
        if not isinstance(result, AsyncTTModelRunnerOutput) and record is not None:
            intermediate = record.get("intermediate")
            if record["phase"] == "prefill" and intermediate and all(intermediate) and record["device_sampling"]:
                pending_prefills.append(record)
            else:
                if record["phase"] == "prefill":
                    observe_prefills("existing_final_prefill_read_and_output")
                record["completion_boundary"] = "host_output_ready_after_existing_read"
                finish(record)
        return result

    TTModelRunner.execute_model = execute_model
    TTModelRunner.build_model_input = build_model_input
    TTAsyncDecodeController.submit_async_decode = submit_async_decode
    AsyncTTModelRunnerOutput._get_output_impl = get_output
    TTModelRunner.sample_tokens = sample_tokens
    GraniteGenerator._setup_decode = setup
    GraniteGenerator.replay = replay_with_work
