# SPDX-License-Identifier: Apache-2.0
"""PipelineStageAdapter + measure_adapter profile WHATEVER emit-e2e emits.

Every PIPELINE_STAGES entry is traced and replayed on a single command queue (the tool measures
trace+1cq end to end); a decode-only pipeline still works via the single-stage fallback; a
repeat-prefill pipeline (no stage hooks, no decode_step) raises so the perf test falls back to
FORWARD_WALL_MS. Runs with a fake ttnn (no hardware).
"""

import sys
import types

import pytest

from agent.perf_adapter import PipelineStageAdapter


def _fake_ttnn():
    m = types.ModuleType("ttnn")
    m._tid = 0

    def begin_trace_capture(device, cq_id=0):
        m._tid += 1
        return m._tid

    m.begin_trace_capture = begin_trace_capture
    m.end_trace_capture = lambda device, tid, cq_id=0: None
    m.execute_trace = lambda device, tid, cq_id=0, blocking=False: None
    m.synchronize_device = lambda device: None
    m.release_trace = lambda device, tid: None
    m.record_event = lambda device, cq: ("ev", cq)
    m.wait_for_event = lambda cq, ev: None
    return m


@pytest.fixture
def trace_replay(monkeypatch):
    monkeypatch.setitem(sys.modules, "ttnn", _fake_ttnn())
    sys.modules.pop("agent.trace_replay", None)  # re-import so `import ttnn` binds the fake
    import agent.trace_replay as tr

    return tr


class _Dev:
    def num_command_queues(self):
        return 1


class _StagePipe:
    """A multi-stage pipeline as emit-e2e emits it: encode and decode each expose a host-op-free
    trace_step that is captured and replayed on a single command queue."""

    PIPELINE_STAGES = ["encode", "decode"]

    def __init__(self):
        self.calls = []

    def encode_trace_setup(self, inputs=None):
        self.calls.append("encode_setup")

    def encode_trace_step(self):
        self.calls.append("encode_step")

    def decode_trace_setup(self, inputs=None):
        self.calls.append("decode_setup")

    def decode_trace_step(self):
        self.calls.append("decode_step")


def test_stage_adapter_profiles_every_stage(trace_replay, capsys):
    pipe = _StagePipe()
    adapter = PipelineStageAdapter(lambda dev: pipe, batch=1)
    headline = trace_replay.measure_adapter(adapter, _Dev())
    out = capsys.readouterr().out

    assert "TRACE_STAGE_MS[encode]=" in out
    assert "TRACE_STAGE_MS[decode]=" in out
    assert "TRACE_STAGES=2" in out
    assert "TRACE_PER_TOKEN_MS=" in out
    # every stage is captured and replayed on a single command queue
    dec_line = [ln for ln in out.splitlines() if ln.startswith("TRACE_STAGE_MS[decode]")][0]
    enc_line = [ln for ln in out.splitlines() if ln.startswith("TRACE_STAGE_MS[encode]")][0]
    assert "trace+1cq" in dec_line
    assert "trace+1cq" in enc_line
    # host prep (trace_setup) ran for both stages, OUTSIDE the trace
    assert "encode_setup" in pipe.calls and "decode_setup" in pipe.calls
    assert isinstance(headline, float)


class _DecodeOnlyPipe:
    def decode_prefill(self, ids=None):
        return {"t": 0}

    def decode_step(self, state):
        return {"t": state["t"] + 1}


def test_decode_only_fallback_still_works(trace_replay, capsys):
    adapter = PipelineStageAdapter(lambda dev: _DecodeOnlyPipe(), prompt_ids=[1, 2], batch=1)
    trace_replay.measure_adapter(adapter, _Dev())
    out = capsys.readouterr().out
    assert "TRACE_STAGE_MS[decode]=" in out
    assert "TRACE_STAGES=1" in out
    assert "trace+1cq" in out


def test_repeat_prefill_raises(trace_replay):
    adapter = PipelineStageAdapter(lambda dev: object(), batch=1)
    with pytest.raises(AttributeError):  # allow-pytest.raises: no expect_error fixture
        trace_replay.measure_adapter(adapter, _Dev())


class _LegacyAdapter:
    """Old PipelineDecodeAdapter shape: setup/step, NO .stages."""

    def __init__(self):
        self.batch = 1

    def setup(self, device):
        pass

    def step(self):
        pass


def test_legacy_adapter_wrapped_as_decode_stage(trace_replay, capsys):
    trace_replay.measure_adapter(_LegacyAdapter(), _Dev())
    out = capsys.readouterr().out
    assert "TRACE_STAGE_MS[decode]=" in out
    assert "trace+1cq" in out
