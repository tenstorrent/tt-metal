# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A HANG must not re-queue the same stub. Wall-kill used to return only
'WALL-CLOCK BUDGET EXHAUSTED', the gate kept naming the same unit/rung, and
the agent burned another full wall on the identical deadlock.

Pins: unchanged-stub re-run is gated; an edit lifts the gate and resets the
device (firmware is dirty after a wall-kill); hang evidence is folded into
the classified HANG text; nothing is keyed off a model/stage name."""
import importlib
import inspect
from pathlib import Path

import pytest

from scripts.tt_hw_planner import cli
from scripts.tt_hw_planner._cli_helpers.bringup_cc import _bringup_cc_prompt


@pytest.fixture()
def bmcp(tmp_path, monkeypatch):
    monkeypatch.setenv("BRINGUP_MCP_DEMO_DIR", str(tmp_path))
    monkeypatch.setenv("BRINGUP_MCP_MODEL_ID", "test/model")
    monkeypatch.setenv("BRINGUP_MCP_STATE", str(tmp_path / "state.json"))
    import scripts.tt_hw_planner.bringup_mcp as m

    importlib.reload(m)
    (tmp_path / "_stubs").mkdir(parents=True, exist_ok=True)
    return m, tmp_path


_HANG_SUMMARY = (
    "focused pytest WALL-CLOCK BUDGET EXHAUSTED at 1800s " "— killing process group (likely a hang)"
)
_HANG_EVIDENCE = "py-spy dump of hung pytest:\n  File stub.py, line 10, in __call__\n    ttnn.synchronize_device"


def _hang_pcc():
    return {
        "ran": True,
        "passed": False,
        "failed": True,
        "skipped": False,
        "summary": _HANG_SUMMARY,
        "details": _HANG_SUMMARY + "\n" + _HANG_EVIDENCE,
        "skip_reason": "",
    }


def _write_stub(tmp, name: str, body: str = "native body v1") -> None:
    (tmp / "_stubs" / f"{name}.py").write_text(body)


def test_unchanged_stub_after_hang_is_gated(bmcp, monkeypatch):
    """The second run_component on the same bytes must refuse the device,
    not start another wall-clock pytest."""
    m, tmp = bmcp
    _write_stub(tmp, "comp_a")
    calls = []
    monkeypatch.setattr(m, "_run_pcc", lambda c: calls.append(c) or _hang_pcc())
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: True)
    first = m.run_component("comp_a")
    assert first.get("gated") is not True
    assert first["failure_class"] == "HANG"
    assert len(calls) == 1
    second = m.run_component("comp_a")
    assert second.get("gated") is True
    assert second["failure_class"] == "HANG"
    assert "edit" in (second.get("reason") or "").lower()
    assert len(calls) == 1, "gated re-run must not touch the device"


def test_editing_the_stub_lifts_the_gate_and_resets_device(bmcp, monkeypatch):
    """After a hang-kill the firmware is dirty. An edit is the only thing
    that may re-run, and that re-run must tt-smi -r first."""
    m, tmp = bmcp
    _write_stub(tmp, "comp_a", "native body v1")
    calls = []
    resets = {"n": 0}
    monkeypatch.setattr(m, "_run_pcc", lambda c: calls.append(c) or _hang_pcc())
    monkeypatch.setattr(
        m._cli,
        "_run_tt_smi_reset",
        lambda **k: resets.__setitem__("n", resets["n"] + 1) or True,
    )
    m.run_component("comp_a")
    assert resets["n"] == 0, "first hang must not reset before the run"
    m.run_component("comp_a")
    assert resets["n"] == 0, "gated refusal must not reset"
    _write_stub(tmp, "comp_a", "native body v2 — fixed collective")
    third = m.run_component("comp_a")
    assert third.get("gated") is not True
    assert len(calls) == 2
    assert resets["n"] == 1, "lifting the hang gate must reset dirty firmware"


def test_hang_reset_is_not_paid_twice_by_the_shard_phase(bmcp, monkeypatch):
    """`_run_tt_smi_reset` is capped per process. A hang in single mode followed
    by the FIRST shard run must reset the card once, not once per reason."""
    m, tmp = bmcp
    _write_stub(tmp, "comp_a", "v1")
    resets = []
    monkeypatch.setattr(m, "_run_pcc", lambda c: _hang_pcc())
    monkeypatch.setattr(
        m._cli,
        "_run_tt_smi_reset",
        lambda **k: resets.append(k.get("context", "")) or True,
    )
    m.run_component("comp_a", mode="single")
    assert resets == [], "single mode must not reset before a run"
    _write_stub(tmp, "comp_a", "v2")
    m.run_component("comp_a", mode="shard")
    assert resets == ["hang:post-hang-device-reset"], "the shard phase must not reset a just-reset card"


def test_other_component_is_not_gated(bmcp, monkeypatch):
    m, tmp = bmcp
    _write_stub(tmp, "comp_a")
    _write_stub(tmp, "comp_b")
    monkeypatch.setattr(m, "_run_pcc", lambda c: _hang_pcc())
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: True)
    m.run_component("comp_a")
    assert m._hang_rerun_block_reason("comp_b") is None


def test_non_hang_class_does_not_gate(bmcp, monkeypatch):
    m, tmp = bmcp
    _write_stub(tmp, "comp_a")
    calls = []

    def shape_pcc(_c):
        calls.append("run")
        return {
            "ran": True,
            "passed": False,
            "failed": True,
            "skipped": False,
            "summary": "shape mismatch at dim 1",
            "details": "RuntimeError: shape",
            "skip_reason": "",
        }

    monkeypatch.setattr(m, "_run_pcc", shape_pcc)
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: True)
    m.run_component("comp_a")
    m.run_component("comp_a")
    assert len(calls) == 2
    assert m._hang_rerun_block_reason("comp_a") is None


def test_legacy_hang_without_sha_does_not_gate(bmcp):
    """State written before this gate has last_class=HANG but no sha —
    must not lock the agent out of a device run."""
    m, tmp = bmcp
    _write_stub(tmp, "comp_a")
    m._save_state({"last_failure_class": {"comp_a": "HANG"}})
    assert m._hang_rerun_block_reason("comp_a") is None


def test_termination_check_says_edit_before_rerun(bmcp, monkeypatch):
    m, tmp = bmcp
    _write_stub(tmp, "comp_a")
    monkeypatch.setattr(m, "_run_pcc", lambda c: _hang_pcc())
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: True)
    monkeypatch.setattr(m, "_components", lambda: ["comp_a"])
    monkeypatch.setattr(m, "_test_file_for", lambda _c: str(tmp / "test_comp_a.py"))
    m.run_component("comp_a")
    r = m.termination_check()
    nxt = r["next_target"]
    assert nxt["unit"] == "comp_a"
    assert nxt.get("hang_gated") is True
    reason = nxt["reason"].lower()
    assert "edit" in reason and "run_component" in reason
    assert "blind" in reason or "unchanged" in reason


def test_termination_check_keeps_shard_rung_when_hang_gated(bmcp, monkeypatch):
    """Same unit, same rung — the hang gate is a reason overlay, not a
    rung change, so shard work does not fall back to single-device repair."""
    m, tmp = bmcp
    monkeypatch.setenv("TT_HW_PLANNER_SHARD", "1")
    importlib.reload(m)
    (tmp / "_stubs").mkdir(parents=True, exist_ok=True)
    _write_stub(tmp, "self_attn")
    monkeypatch.setattr(m, "_run_pcc", lambda c: _hang_pcc())
    monkeypatch.setattr(m._cli, "_run_tt_smi_reset", lambda **k: True)
    monkeypatch.setattr(m, "_components", lambda: ["self_attn"])
    m.run_component("self_attn", mode="shard")
    r = m.termination_check()
    assert r["next_target"]["rung"] == "shard"
    assert r["next_target"].get("hang_gated") is True


def _stub_focused_pytest(m, monkeypatch, tmp, *, rc: int):
    captured: dict = {}
    monkeypatch.setattr(m, "_test_file_for", lambda _c: str(tmp / "test_comp.py"))
    monkeypatch.delenv("TT_HW_PLANNER_SHARD_RUN", raising=False)

    def fake_pytest(**kwargs):
        captured.update(kwargs)
        return rc

    monkeypatch.setattr(m._cli, "_run_focused_pytest", fake_pytest)
    monkeypatch.setattr(
        m._cli,
        "_parse_pytest_report",
        lambda: (_ for _ in ()).throw(AssertionError("JUnit must not be parsed after a wall kill")),
    )
    monkeypatch.setattr(m._cli, "_scope_report_to_demo", lambda report, _demo: report)
    return captured


def test_wall_kill_still_classifies_as_hang(bmcp, monkeypatch):
    m, tmp = bmcp
    _stub_focused_pytest(m, monkeypatch, tmp, rc=124)
    res = m._run_pcc("comp_a")
    assert res["failed"] is True
    assert cli._classify_failure(res["summary"], res["details"]) == "HANG"


def test_wall_kill_details_include_hang_evidence_once(bmcp, monkeypatch):
    """Evidence is the hang site. Stages live inside it; do not append them
    again or the agent sees a duplicated blank wall + dump."""
    m, tmp = bmcp
    _stub_focused_pytest(m, monkeypatch, tmp, rc=124)
    monkeypatch.setattr(
        m._cli,
        "_LAST_HANG_EVIDENCE",
        "Last reported stage(s) before kill:\n  t: stage=forward\n" + _HANG_EVIDENCE,
        raising=False,
    )
    monkeypatch.setattr(m._cli, "_LAST_PYTEST_STAGES", {"t": "forward"}, raising=False)
    res = m._run_pcc("comp_a")
    assert _HANG_EVIDENCE in res["details"]
    assert res["details"].count("Last reported stage") == 1
    assert cli._classify_failure(res["summary"], res["details"]) == "HANG"


def test_wall_kill_falls_back_to_stages_when_dump_empty(bmcp, monkeypatch):
    m, tmp = bmcp
    _stub_focused_pytest(m, monkeypatch, tmp, rc=124)
    monkeypatch.setattr(m._cli, "_LAST_HANG_EVIDENCE", "", raising=False)
    monkeypatch.setattr(m._cli, "_LAST_PYTEST_STAGES", {"t": "forward"}, raising=False)
    res = m._run_pcc("comp_a")
    assert "stage" in res["details"].lower()
    assert "forward" in res["details"]


def test_capture_hang_evidence_never_raises():
    class _Proc:
        pid = None

    assert isinstance(cli._capture_hang_evidence(_Proc()), str)


def test_prompt_forbids_blind_hang_rerun():
    p = _bringup_cc_prompt("some/model", Path("/tmp/demo"), 0.99)
    assert "hang_gated" in p
    assert "do NOT" in p or "Do NOT" in p
    assert "edit" in p.lower()


def test_hang_directive_leads_with_evidence_and_collectives():
    text = cli._strategy_directive_for_failure("HANG")
    lower = text.lower()
    assert "do not re-run" in lower or "refuses" in lower
    assert "collective" in lower or "all_reduce" in lower
    assert "1x1" not in lower, "carving a 1-device submesh is how the last hang was created"
    assert "head_dim" not in lower


def test_hang_helpers_have_no_model_or_stage_vocabulary(bmcp):
    """Constraint 3: the gate is keyed off stub bytes + failure class, never
    a component or inference-stage name."""
    m, _ = bmcp
    src = (
        inspect.getsource(m._stub_sha)
        + inspect.getsource(m._hang_rerun_block_reason)
        + inspect.getsource(cli._capture_hang_evidence)
    )
    for forbidden in (
        "hidden_size",
        "num_hidden_layers",
        "decode",
        "prefill",
        "encoder",
        "decoder",
        "language_model",
        "audio_tower",
        "attention",
        "lightning",
        "nemotron",
    ):
        assert forbidden not in src, f"hang gate must not reference {forbidden!r}"
