"""Buffered, nonblocking host phase observations for the benchmark stage."""
import collections
import inspect
import json
import os
import threading
import time
from pathlib import Path


def install(adapter, path):
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController
    from vllm_tt_plugin.model_runner import TTModelRunner

    import ttnn

    query = ttnn._ttnn.events.event_query
    entry_time = threading.local()
    pending = collections.deque()
    records = {}
    positions = {}
    lock = threading.Lock()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from ..tt.precision import runtime_summary

    runtime = runtime_summary(adapter.model)
    path.with_suffix(".runtime.json").write_text(json.dumps(runtime, indent=2) + "\n")
    identity = dict(
        event="identity",
        pid=os.getpid(),
        module_file=inspect.getfile(type(adapter)),
        layers=list(adapter.model.layer_indices),
        precision=adapter.model.precision,
        max_num_seqs=adapter.batch_size,
        max_model_len=adapter.context,
        mesh=[1, 4],
        instrumentation=__file__,
        monotonic=time.perf_counter(),
    )
    path.write_text(json.dumps(identity) + "\n")

    def begin(mi, phase):
        now = getattr(entry_time, "started", time.perf_counter())
        ids = list(mi.row_req_ids)
        raw = mi.input_positions.tolist() if hasattr(mi.input_positions, "tolist") else mi.input_positions
        row = dict(
            event="phase",
            phase=phase,
            start=now,
            request_ids=ids,
            raw_positions=raw,
            slots=adapter.batch_size,
            devices=[0, 1, 2, 3],
        )
        if phase == "prefill":
            starts = list(raw)
            ends = list(mi.prompt_lens)
            row["rows"] = [
                dict(
                    request_id=r,
                    start_pos=int(s),
                    token_count=int(e) - int(s),
                    end_pos=int(e),
                    head_token_count=int(not bool(mi.intermediate_prefill_mask[i])),
                )
                for i, (r, s, e) in enumerate(zip(ids, starts, ends))
            ]
            for r, end in zip(ids, ends):
                positions[r] = int(end)
        else:
            row["positions"] = [positions[r] for r in ids]
            for r in ids:
                positions[r] += 1
            row["batch"] = len(ids)
        state = dict(row=row, events=None, host_end=None)
        records[id(mi)] = state
        with lock:
            pending.append(state)
        return state

    original_execute = TTModelRunner.execute_model

    def execute(self, *args, **kwargs):
        entry_time.started = time.perf_counter()
        return original_execute(self, *args, **kwargs)

    TTModelRunner.execute_model = execute

    original_prefill = TTModelRunner.submit_prefill

    def prefill(self, model_input, batch_size_per_dp):
        state = begin(model_input, "prefill")
        result = original_prefill(self, model_input, batch_size_per_dp)
        state["events"] = [ttnn.record_event(adapter.model.mesh_device, 0)]
        return result

    TTModelRunner.submit_prefill = prefill

    original_decode = TTAsyncDecodeController.submit_decode

    def decode(self, model_input, *args, **kwargs):
        state = begin(model_input, "decode")
        result = original_decode(self, model_input, *args, **kwargs)
        state["events"] = result.read_events or [ttnn.record_event(adapter.model.mesh_device, 0)]
        return result

    TTAsyncDecodeController.submit_decode = decode

    original_output = TTModelRunner._get_output_tokens

    def output(self, *args, **kwargs):
        result = original_output(self, *args, **kwargs)
        state = records.pop(id(kwargs["model_input"]), None)
        if state is not None:
            state["host_end"] = time.perf_counter()
        return result

    TTModelRunner._get_output_tokens = output

    def observe():
        previous_end = None
        buffer = []
        flush_at = time.perf_counter() + 1
        while True:
            with lock:
                state = pending[0] if pending else None
            now = time.perf_counter()
            if state and state["events"] and state["host_end"] is not None and all(query(e) for e in state["events"]):
                with lock:
                    pending.popleft()
                row = state["row"]
                row.update(
                    completion=now,
                    host_end=state["host_end"],
                    previous_completion=previous_end,
                    observation_poll_seconds=0.0005,
                )
                previous_end = now
                buffer.append(row)
            else:
                time.sleep(0.0005)
            if now >= flush_at:
                if buffer:
                    with path.open("a") as stream:
                        stream.write("".join(json.dumps(row) + "\n" for row in buffer))
                    buffer.clear()
                flush_at = now + 1

    threading.Thread(target=observe, daemon=True, name="benchmark-phase-observer").start()
