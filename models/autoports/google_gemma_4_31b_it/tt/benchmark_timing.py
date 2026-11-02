# SPDX-License-Identifier: Apache-2.0
"""Optional host completion ledger. No synchronization or device profiling."""
import collections
import functools
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def install(model):
    from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    if hasattr(TTModelRunner, "_benchmark_ledger"):
        raise RuntimeError("Benchmark recorder requires one engine per process")
    ledger = Ledger(model)
    TTModelRunner._benchmark_ledger = ledger
    prefill = TTModelRunner.submit_prefill
    decode = TTAsyncDecodeController.submit_decode
    complete = AsyncTTModelRunnerOutput._get_output_impl
    finish_prefill = TTModelRunner._finish_front_packed_sync

    @functools.wraps(prefill)
    def record_prefill(runner, model_input, *args, **kwargs):
        row = ledger.begin("prefill", runner, model_input)
        result = prefill(runner, model_input, *args, **kwargs)
        # Prefill edge traces can rewrite resident positions; the plugin
        # invalidates its decode chain after prefill and must upload next time.
        ledger.positions.invalidate()
        row["adapter_return"] = time.monotonic()
        intermediate = row["intermediate"]
        row["adapter_return_proves_device_complete"] = not (intermediate is not None and all(intermediate))
        if row["adapter_return_proves_device_complete"]:
            row["device_completion_proof_at"] = row["adapter_return"]
            row["device_completion_proof_source"] = "prefill adapter existing blocking output read"
        ledger.by_model_input[id(model_input)] = row
        ledger.submit(row)
        return result

    @functools.wraps(decode)
    def record_decode(controller, model_input, *args, **kwargs):
        row = ledger.begin("decode", controller.runner, model_input)
        result = decode(controller, model_input, *args, **kwargs)
        ledger.record_decode_positions(row, result)
        ledger.by_submission[id(result)] = row
        ledger.submit(row)
        return result

    @functools.wraps(complete)
    def record_complete(output):
        result = complete(output)
        row = ledger.by_submission.pop(id(output._submission), None)
        if row is not None:
            row["host_done"] = time.monotonic()
            row["host_done_proves_device_complete"] = True
            row["device_completion_proof_at"] = row["host_done"]
            row["device_completion_proof_source"] = "existing async decode readback and output finalization"
        return result

    @functools.wraps(finish_prefill)
    def record_finish_prefill(runner, grammar_output, *, fwd):
        row = None if fwd is None else ledger.by_model_input.get(id(fwd.model_input))
        result = finish_prefill(runner, grammar_output, fwd=fwd)
        if row is not None:
            row["host_done"] = time.monotonic()
            row["host_done_proves_device_complete"] = row["adapter_return_proves_device_complete"]
            ledger.by_model_input.pop(id(fwd.model_input), None)
        return result

    TTModelRunner._finish_front_packed_sync = record_finish_prefill
    TTModelRunner.submit_prefill = record_prefill
    TTAsyncDecodeController.submit_decode = record_decode
    AsyncTTModelRunnerOutput._get_output_impl = record_complete
    return ledger


class PositionMirror:
    """Replay-ordered mirror of device positions, independent of host commits."""

    def __init__(self, batch):
        self.batch = batch
        self.next_positions = None
        self.replay_count = 0
        self.refresh_replay = None

    def invalidate(self):
        self.next_positions = None
        self.refresh_replay = None

    def submitted(self, host_positions, reload_inputs):
        if len(host_positions) != self.batch:
            raise ValueError("Position accounting requires the full trace-slot vector")
        if reload_inputs:
            self.next_positions = [int(p) for p in host_positions]
            self.refresh_replay = self.replay_count
        if self.next_positions is None:
            raise RuntimeError("Resident decode has no known uploaded position vector")
        actual = list(self.next_positions)
        if any(p < -1 for p in actual):
            raise ValueError("Invalid resident decode position")
        self.next_positions = [p + 1 if p >= 0 else p for p in actual]
        ordinal = self.replay_count
        self.replay_count += 1
        return actual, ordinal, self.refresh_replay


class Ledger:
    def __init__(self, model):
        self.model = model
        self.positions = PositionMirror(model.batch)
        self.rows = []
        self.pending = collections.deque()
        self.by_submission = {}
        self.by_model_input = {}
        self.stopped = False
        self.error = None
        self.identity = dict(
            pid=os.getpid(),
            module=type(model).__module__,
            module_file=__import__("inspect").getfile(type(model)),
            layers=model.model.layer_indices,
            batch=model.batch,
            context=model.context,
            precision=model.model.precision_config,
            mesh=list(model.mesh.shape),
            method="ordered nonblocking all-device completion queries",
        )
        ledger = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                value = dict(
                    identity=ledger.identity,
                    pending=len(ledger.pending),
                    error=ledger.error,
                    events=ledger.rows if self.path == "/flush" else len(ledger.rows),
                )
                body = json.dumps(value).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", int(os.environ["GEMMA4_BENCHMARK_PORT"])), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self.poll, daemon=True).start()

    def begin(self, phase, runner, inp):
        slot_ids = getattr(inp, "grammar_row_req_ids", ()) if phase == "decode" else ()
        ids = list(slot_ids or (inp.row_req_ids if inp.row_req_ids is not None else runner.input_batch.req_ids))
        pos = [int(p) for p in inp.input_positions.flatten().tolist()]
        # Preserve holes and physical slot identity; request denominators may
        # exclude None later but work accounting must retain every trace slot.
        ids = ids[: len(pos)] + [None] * max(0, len(pos) - len(ids))
        row = dict(
            index=len(self.rows),
            phase=phase,
            start=time.monotonic(),
            requests=ids,
            host_positions=list(pos),
            positions=list(pos),
            lengths=None if inp.prompt_lens is None else [int(n) for n in inp.prompt_lens],
            intermediate=None
            if inp.intermediate_prefill_mask is None
            else [bool(n) for n in inp.intermediate_prefill_mask],
            sample_on_device=bool(inp.perform_device_sampling),
            speculative_mode=getattr(inp, "spec_mode", None),
            physical_batch=self.model.batch,
            host_done=None,
        )
        self.rows.append(row)
        return row

    def record_decode_positions(self, row, submission):
        if submission.tt_out is None:
            raise RuntimeError("Empty decode submission cannot supply benchmark work")
        if row["speculative_mode"] is not None:
            raise RuntimeError("Benchmark position mirror only supports plain decode")
        plan = submission.reload_plan
        if plan is None:
            raise RuntimeError("Benchmark requires explicit decode reload contract")
        actual, ordinal, refresh = self.positions.submitted(row["host_positions"], plan.reload_inputs)
        row.update(
            positions=actual,
            reload_inputs=bool(plan.reload_inputs),
            reload_page_table=bool(plan.reload_page_table),
            replay_ordinal=ordinal,
            position_refresh_replay=refresh,
            position_source="last uploaded vector advanced once per submitted model replay",
        )

    def submit(self, row, host_done=None):
        import ttnn

        row["host_done"] = host_done
        row["submitted"] = time.monotonic()
        event = ttnn.record_event(self.model.mesh, 0)
        self.pending.append((event, row))

    def poll(self):
        import ttnn

        try:
            while not self.stopped:
                if self.pending:
                    event, row = self.pending[0]
                    now = time.monotonic()
                    if ttnn.event_query(event):
                        row["device_completion_upper"] = time.monotonic()
                        row.setdefault("device_completion_lower", row["start"])
                        self.pending.popleft()
                    else:
                        row["device_completion_lower"] = now
                time.sleep(0.0002)
        except Exception as exc:
            self.error = repr(exc)

    def close(self):
        self.stopped = True
        self.server.shutdown()
