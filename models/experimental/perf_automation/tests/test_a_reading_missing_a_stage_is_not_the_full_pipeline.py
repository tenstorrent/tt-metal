"""A run that timed only some of its stages is not a full-pipeline reading.

An image-edit port on a 32-chip mesh, 2026-10-13: an uncommitted edit crashed three of the pipeline's
five stages. The adapter and trace_replay give a stage that cannot be timed no row, so the others keep
theirs -- and a pipeline whose headline is the whole pass then sums only the stages that ran. The two
VAE stages (75.7 + 94.6 = 170.43 ms) were read as the full pipeline (12331.94 ms), reported `ok` at
-98.6%, stashed as the pending best, and promoted by the next commit. Every real reading after that was
refused as +7136% diverged, so nothing more could be recorded or committed.

The run now states each stage it could not time (stage_seams.UNMEASURED_MARK) and the gate refuses a
reading that lost a stage the committed best measured. A stage the bar never had does not refuse it.

Device steps are faked; the real adapter, trace_replay and gate functions run. Nothing touches a device.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from agent import probes as PR
from agent import stage_seams as S
from agent.perf_adapter import PipelineStageAdapter


@pytest.fixture(autouse=True)
def _no_env_leak():
    """_run_full_pipeline_ms writes os.environ directly; restore it whole (see the same fixture in
    test_a_stage_gets_the_same_filter_as_the_headline.py for what a leak once cost)."""
    import os

    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)


# --------------------------------------------------------------------------------------------------
# the marker: one spelling, printed and parsed by the same module
# --------------------------------------------------------------------------------------------------


def test_the_marker_round_trips():
    assert S.parse_unmeasured(S.unmeasured_line("enc", "RuntimeError: boom")) == ("enc", "RuntimeError: boom")


def test_a_multi_line_reason_stays_on_one_line():
    line = S.unmeasured_line("enc", "TT_FATAL\n  at x.cpp:1\n")
    assert "\n" not in line
    assert S.parse_unmeasured(line) == ("enc", "TT_FATAL at x.cpp:1")


def test_a_reason_holding_the_separator_is_kept_whole():
    assert S.parse_unmeasured(S.unmeasured_line("enc", "a[0]=1")) == ("enc", "a[0]=1")


@pytest.mark.parametrize("line", ["", "TRACE_STAGE_MS[enc]=1.0 path=trace", S.UNMEASURED_MARK + "[]=x"])
def test_other_lines_are_not_the_marker(line):
    assert S.parse_unmeasured(line) is None


_REGION_FATAL = (
    "TT_FATAL @ /x/tt_metal/distributed/mesh_trace.cpp:82: "
    "mesh_cq.device()->get_trace_buffers_size() <= trace_region_size\n"
    "info:\n"
    "Creating trace buffers of size 1073741824B on MeshDevice 0, but only 41943040B is allocated "
    "for trace region."
)


def test_a_shortened_reason_keeps_the_size_the_device_reported():
    """THE SIZING SURVIVES THE CLIP. The reason is shortened for the log, but the sentence naming the
    bytes sits several lines into a TT_FATAL -- clipped away, the loops that grow the region off these
    lines had nothing to size with and fell back to doubling (three re-runs of a 25-minute
    measurement, on Qwen-Image-Edit, to reach a size the first failure had already named)."""
    why = S.unmeasured_why(RuntimeError(_REGION_FATAL))
    assert "1073741824B" in why and "41943040B" in why
    assert len(why) < len(_REGION_FATAL)
    from agent.perf_test_gen import _needed_trace_region

    line = S.unmeasured_line("enc", "RuntimeError: %s" % why)
    assert _needed_trace_region(line) >= 1073741824, "the grow loop can size itself off the marker"


def test_an_ordinary_reason_is_just_shortened():
    why = S.unmeasured_why(RuntimeError("boom " * 200))
    assert len(why) == S.UNMEASURED_WHY_HEAD
    assert S.unmeasured_why(RuntimeError("short one")) == "short one"


# --------------------------------------------------------------------------------------------------
# the run states every stage that got no row
# --------------------------------------------------------------------------------------------------


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

    monkeypatch.setattr(PR, "thermal_yield", lambda label="": None)
    monkeypatch.delenv(PR.STAGES_KEPT_ENV, raising=False)
    return tr


class _Dev:
    def num_command_queues(self):
        return 1


class _Pipe:
    """Four stages: `ok` runs; `crashes` raises when stepped (the edit broke it); `no_inputs` cannot
    prepare its inputs; `no_step` is declared without a step hook. One pass, so the headline is the sum."""

    PIPELINE_STAGES = ["ok", "crashes", "no_inputs", "no_step"]
    PIPELINE_UNIT = "inference"

    def ok_trace_setup(self, inputs=None):
        pass

    def ok_trace_step(self):
        pass

    def crashes_trace_setup(self, inputs=None):
        pass

    def crashes_trace_step(self):
        raise RuntimeError("TT_FATAL: Coordinate MeshCoordinate([0, 4]) not found")

    def no_inputs_trace_inputs(self):
        raise FileNotFoundError("golden.pt")

    def no_inputs_trace_setup(self, inputs=None):
        pass

    def no_inputs_trace_step(self):
        pass

    def no_step_trace_setup(self, inputs=None):
        pass


def test_the_adapter_records_the_stages_it_drops():
    a = PipelineStageAdapter(lambda dev: _Pipe(), batch=1)
    a.setup(_Dev())
    assert [s.name for s in a.stages] == ["ok", "crashes"]
    assert sorted(a.unmeasured) == ["no_inputs", "no_step"]
    assert "golden.pt" in a.unmeasured["no_inputs"]
    assert "no_step" + S.STEP in a.unmeasured["no_step"]


def _measured(tr, capsys):
    tr.measure_adapter(PipelineStageAdapter(lambda dev: _Pipe(), batch=1), _Dev())
    return capsys.readouterr().out


def test_trace_replay_states_every_stage_without_a_row(trace_replay, capsys):
    out = _measured(trace_replay, capsys)
    got = dict(filter(None, (S.parse_unmeasured(ln) for ln in out.splitlines())))
    assert sorted(got) == ["crashes", "no_inputs", "no_step"]
    assert "MeshCoordinate" in got["crashes"]
    rows = [ln for ln in out.splitlines() if ln.startswith("TRACE_STAGE_MS[")]
    assert [r.split("[", 1)[1].split("]", 1)[0] for r in rows] == ["ok"], "the stage that ran keeps its row"


def test_a_run_where_every_stage_ran_states_nothing(trace_replay, capsys, monkeypatch):
    monkeypatch.setattr(_Pipe, "PIPELINE_STAGES", ["ok"])
    out = _measured(trace_replay, capsys)
    assert S.UNMEASURED_MARK not in out


# --------------------------------------------------------------------------------------------------
# the gate refuses a reading that lost a stage the committed best measured
# --------------------------------------------------------------------------------------------------

# The measured case, under neutral names: five stages in the bar, three crashed, the two left sum to 170.43.
_BAR = {"s1": 2821.7687, "s2": 3176.4454, "s3": 75.716, "s4": 6162.6238, "s5": 94.5709}
_BEST = 12331.9394


def _sample(stages: dict, lost: dict) -> str:
    lines = ["TRACE_STAGE_MS[%s]=%.4f path=trace+1cq" % kv for kv in stages.items()]
    lines += [S.unmeasured_line(k, v) for k, v in lost.items()]
    lines += [
        "TRACE_PER_TOKEN_MS=%.4f" % sum(stages.values()),
        "TRACE_HEADLINE_UNIT=inference",
        "TRACE_REPLAY_PATH=trace+1cq batch=1",
    ]
    return "\n".join(lines) + "\n"


_CRASH = "RuntimeError: TT_FATAL: Coordinate MeshCoordinate([0, 4]) not found in mesh MeshShape([8, 4])"
_PARTIAL = _sample({"s3": 75.7316, "s5": 94.6981}, {"s1": _CRASH, "s2": _CRASH, "s4": _CRASH})


@pytest.fixture()
def gate(monkeypatch, tmp_path):
    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "m"))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "test-run-1")
    monkeypatch.setattr(pm, "_FULLPIPE_SAMPLES", 1, raising=False)
    monkeypatch.setattr(pm, "_MODEL_ROOT", tmp_path / "m", raising=False)
    monkeypatch.setattr(pm, "_MANIFEST", {"perf_test_resolved": {"path": "t.py", "case": None}}, raising=False)
    monkeypatch.setattr(pm, "_FULLPIPE_BASELINE_1CQ_PATH", tmp_path / "base_1cq.json")
    monkeypatch.setattr(pm, "_FULLPIPE_TARGET_MS", 0.0)
    monkeypatch.setattr(pm, "_recover_if_board_is_dead", lambda *a, **k: False)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    out = {"text": _PARTIAL}
    monkeypatch.setattr(pm, "_adaptive_run", lambda *a, **k: pm._AdaptiveResult(0, out["text"]))
    return pm, out


def _write_bar(pm, stages=_BAR, best=_BEST):
    pm._FULLPIPE_BASELINE_1CQ_PATH.write_text(
        json.dumps({"full_pipeline_ms": best, "method": "trace", "mode": "trace", "stages": stages})
    )


def test_the_measured_case_is_no_reading(gate):
    pm, _ = gate
    _write_bar(pm)
    ms, _method, err, _path = pm._run_full_pipeline_ms()
    assert ms is None, "170.43 ms from two of five stages was taken for the whole pipeline"
    assert "s1, s2, s4" in err and "MeshCoordinate" in err


def test_the_gate_reports_a_crash_and_leaves_the_bar_alone(gate):
    pm, _ = gate
    _write_bar(pm)
    before = pm._FULLPIPE_BASELINE_1CQ_PATH.read_text()
    cfpl = getattr(pm.check_full_pipeline_latency, "fn", pm.check_full_pipeline_latency)
    v = cfpl()
    assert v["status"] == "crash", v
    assert not pm._fullpipe_pending_path().exists(), "a partial reading must never be stashed as a best"
    assert pm._FULLPIPE_BASELINE_1CQ_PATH.read_text() == before


def test_a_full_reading_is_measured_as_before(gate):
    pm, out = gate
    _write_bar(pm)
    out["text"] = _sample({k: v * 0.99 for k, v in _BAR.items()}, {})
    ms, method, err, _path = pm._run_full_pipeline_ms()
    assert err is None and method == "trace"
    assert ms == pytest.approx(sum(_BAR.values()) * 0.99, rel=1e-6)


def test_a_stage_the_bar_never_had_does_not_refuse_the_reading(gate):
    """Like with like: a model that has never timed a stage is compared on the stages it has timed."""
    pm, out = gate
    _write_bar(pm, stages={"s3": 75.716, "s5": 94.5709}, best=170.2869)
    out["text"] = _sample({"s3": 75.7316, "s5": 94.6981}, {"s6": "FileNotFoundError: golden.pt"})
    ms, _method, err, _path = pm._run_full_pipeline_ms()
    assert err is None and ms == pytest.approx(170.4297, rel=1e-6)


def test_with_no_bar_yet_the_reading_is_unchanged(gate):
    """The first reading has nothing to be compared with; refusing it would leave the run no baseline."""
    pm, _ = gate
    ms, _method, err, _path = pm._run_full_pipeline_ms()
    assert err is None and ms == pytest.approx(170.4297, rel=1e-6)


def test_the_run_output_feeds_the_gate_end_to_end(gate, trace_replay, capsys):
    """What trace_replay prints for a crashed stage is what the gate refuses: one marker, both ends."""
    pm, out = gate
    out["text"] = _measured(trace_replay, capsys)
    _write_bar(pm, stages={"ok": 1.0, "crashes": 1.0})
    ms, _method, err, _path = pm._run_full_pipeline_ms()
    assert ms is None and "crashes" in err
    assert "no_inputs" not in err and "no_step" not in err, "only stages the bar measured are counted"
