# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What the capture saw reaches the agent, and a round that edited nothing does not buy the gate twice.

Two halves of one failure, seen on a Qwen-Image-Edit bring-up:

1. DELIVERY. The trace gate produced real, model-specific evidence -- the stage markers the capture
   printed before it stopped, one line per attempt -- returned it in `capture_detail`, and wrote it to
   RUN_REPORT.md. `reasons` is the list the gate server turns into `next_target.reason` and
   `blocking[]`, i.e. the only thing the agent is handed, and `capture_detail` was not in it. So every
   round the agent received the verdict prose ("trace did not engage and the graduation state could
   not be read ..."), which is the same sentence whatever happened and names nothing.

2. NO-OP ROUNDS. Given that, the agent changed nothing, and the loop re-ran a ~3.5 h gate to
   re-derive the same verdict -- repeatedly, because nothing compared the code between rounds.

The gate now puts the capture's own words in the list that reaches the agent, and counts consecutive
driver-side failures over a byte-identical source fingerprint: the count is stated in the reason, and
at the limit the gate halts instead of paying for the same answer again.
"""

from __future__ import annotations

import json

import pytest

from scripts.tt_hw_planner import cc_harness as CH
from scripts.tt_hw_planner import trace_gate as TG
from scripts.tt_hw_planner.commands import emit_e2e as E

# --- 1. the capture's own evidence reaches the pipe the agent reads ------------------------------


def _fail_result(detail):
    return {
        "verdict": "FAIL",
        "reason": "trace did not engage and the graduation state could not be read",
        "capture_detail": detail,
        "repin_violation": None,
        "glue_violations": [],
    }


@pytest.fixture
def traceless_demo(tmp_path, monkeypatch):
    """A demo whose modules all graduated and whose capture fails: the FAIL branch, no device."""
    import scripts.tt_hw_planner.bringup_loop as bl

    monkeypatch.setattr(bl, "_stub_has_graduated_any", lambda p: True)
    monkeypatch.setattr(bl, "_safe_id", lambda n: n)
    demo = tmp_path / "d"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tt").mkdir()
    (demo / "_stubs" / "a.py").write_text("")
    (demo / "_stubs" / "a.py.last_good_native").write_text("")
    (demo / "bringup_status.json").write_text(json.dumps({"components": [{"name": "a"}]}))
    return demo


def _evaluate_with_capture(demo, monkeypatch, detail):
    monkeypatch.setattr(TG, "run_fresh_trace_capture", lambda d, **k: ({"trace_1cq": False}, detail))
    return TG.evaluate_trace_gate(demo, fresh=True)


def test_the_capture_detail_is_one_of_the_blockers(traceless_demo, monkeypatch):
    """THE DELIVERY BUG: this string was produced, stored, reported -- and never handed over."""
    detail = "attempt 1: invalid STAGE_BYTES[s]=15890066432 ops=68634 WEDGE: no forward progress"
    res = _evaluate_with_capture(traceless_demo, monkeypatch, detail)
    assert res["verdict"] == "FAIL"
    assert res["capture_detail"] == detail
    assert any(detail in r for r in res["reasons"]), "the evidence must be in the list the agent gets"


def test_every_attempt_survives_into_the_blockers(traceless_demo, monkeypatch):
    """Wedged in the same place twice, versus got further, need opposite fixes: keep the sequence."""
    detail = "attempt 1: invalid ops=68634 WEDGE\nattempt 2: invalid ops=68634 WEDGE"
    res = _evaluate_with_capture(traceless_demo, monkeypatch, detail)
    joined = " ".join(res["reasons"])
    assert "attempt 1" in joined and "attempt 2" in joined


def test_a_long_detail_cannot_crowd_out_the_other_blockers(traceless_demo, monkeypatch):
    res = _evaluate_with_capture(traceless_demo, monkeypatch, "x" * 5000)
    assert all(len(r) <= TG._CAPTURE_DETAIL_CHARS + 200 for r in res["reasons"])


def test_no_capture_no_extra_blocker(traceless_demo, monkeypatch):
    """A verdict read off existing caps has no capture to quote; it must not invent one."""
    monkeypatch.setattr(TG, "run_fresh_trace_capture", lambda d, **k: (_ for _ in ()).throw(AssertionError()))
    res = TG.evaluate_trace_gate(traceless_demo, {"trace_1cq": False})
    assert res["capture_detail"] is None
    assert res["reasons"] and all("what the capture itself reported" not in r for r in res["reasons"])


def test_a_pass_still_reports_nothing(traceless_demo, monkeypatch):
    res = TG.evaluate_trace_gate(traceless_demo, {"trace_1cq": True})
    assert res["verdict"] == "PASS" and res["reasons"] == []


def test_the_fix_directive_names_the_capture_instead_of_the_verdict_prose():
    """With nothing static to point at, the directive was the same sentence every round."""
    d = TG.build_fix_directive(_fail_result("attempt 1: invalid STAGE_BYTES[s]=1 ops=2 WEDGE"))
    assert "STAGE_BYTES[s]=1 ops=2" in d


def test_the_fix_directive_still_leads_with_the_actionable_fix():
    """A static violation is a better lead than a log line, and stays first."""
    res = _fail_result("attempt 1: whatever")
    res["glue_violations"] = ["glue host-op in `f`: torch.full"]
    d = TG.build_fix_directive(res)
    assert d.startswith("Port to on-device ttnn")


# --- 2. the fingerprint the no-edit check is built on --------------------------------------------


@pytest.fixture
def demo_sources(tmp_path):
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")
    return demo


def test_the_fingerprint_is_stable_and_notices_an_edit(demo_sources):
    before = E._source_fingerprint(demo_sources)
    assert before and before == E._source_fingerprint(demo_sources)
    (demo_sources / "tt" / "pipeline.py").write_text("X = 2\n")
    assert E._source_fingerprint(demo_sources) != before


def test_the_fingerprint_needs_no_run_identity(demo_sources, monkeypatch):
    """The cache key must not be cached without a run stamp; the fingerprint itself still exists."""
    monkeypatch.setenv("PERF_MCP_RUN_ID", "")
    assert E._correctness_key(demo_sources, 0.99, 32) is None
    assert E._source_fingerprint(demo_sources) is not None


# --- 3. consecutive rounds with no edit ----------------------------------------------------------


@pytest.fixture
def gate(monkeypatch, demo_sources):
    """The gate server, with the run identity pinned and nothing driver-side stubbed out."""
    pytest.importorskip("mcp")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-1")
    import scripts.tt_hw_planner.e2e_mcp as M

    monkeypatch.setattr(M, "_SERVING", False)
    monkeypatch.delenv(M._NOOP_LIMIT_ENV, raising=False)
    return M, demo_sources


def test_the_first_failure_accuses_nobody(gate):
    M, demo = gate
    out = M._failed(["G6 trace-gate: x"], {"unit": "u", "rung": "r"}, demo)
    assert out["halt"] is False and "NO CODE CHANGED" not in out["next_target"]["reason"]
    assert out["blocking"] == ["G6 trace-gate: x"]


def test_a_repeat_over_unchanged_sources_says_so_first(gate):
    M, demo = gate
    M._failed(["G6 trace-gate: x"], {"unit": "u"}, demo)
    out = M._failed(["G6 trace-gate: x"], {"unit": "u"}, demo)
    assert out["next_target"]["reason"].startswith("NO CODE CHANGED")
    assert "G6 trace-gate: x" in out["next_target"]["reason"]  # the report still follows
    assert out["halt"] is False  # one such round is tolerated


def test_the_limit_halts_instead_of_buying_the_same_answer_again(gate):
    M, demo = gate
    for _ in range(3):
        out = M._failed(["G6 trace-gate: x"], {"unit": "u"}, demo)
    assert out["halt"] is True
    assert "NO CODE CHANGED" in out["halt_reason"]


def test_an_edit_in_between_clears_the_count(gate):
    M, demo = gate
    M._failed(["x"], {"unit": "u"}, demo)
    M._failed(["x"], {"unit": "u"}, demo)
    (demo / "tt" / "pipeline.py").write_text("X = 99\n")
    out = M._failed(["x"], {"unit": "u"}, demo)
    assert out["halt"] is False and "NO CODE CHANGED" not in out["next_target"]["reason"]


def test_a_new_run_starts_the_count_over(gate, monkeypatch):
    """A restart has reasons that are not the agent sitting still; it must not inherit the count."""
    M, demo = gate
    for _ in range(2):
        M._failed(["x"], {"unit": "u"}, demo)
    monkeypatch.setenv("PERF_MCP_RUN_ID", "run-2")
    out = M._failed(["x"], {"unit": "u"}, demo)
    assert out["halt"] is False


def test_the_agents_own_mid_round_calls_are_not_counted(gate, monkeypatch):
    """It calls this same tool BEFORE editing; counting that would halt a working round."""
    M, demo = gate
    monkeypatch.setattr(M, "_SERVING", True)
    for _ in range(5):
        out = M._failed(["x"], {"unit": "u"}, demo)
    assert out["halt"] is False and not (demo / M._NOOP_STATE_FILE).exists()


def test_the_halt_can_be_switched_off(gate, monkeypatch):
    M, demo = gate
    monkeypatch.setenv(M._NOOP_LIMIT_ENV, "0")
    for _ in range(5):
        out = M._failed(["x"], {"unit": "u"}, demo)
    assert out["halt"] is False
    assert out["next_target"]["reason"].startswith("NO CODE CHANGED")  # still TOLD, just not halted


def test_a_corrupt_state_file_is_ignored_not_raised(gate):
    M, demo = gate
    (demo / M._NOOP_STATE_FILE).write_text("{not json")
    assert M._failed(["x"], {"unit": "u"}, demo)["halt"] is False


def test_an_unfingerprintable_demo_never_accuses(gate, tmp_path):
    """No sources found is not evidence of "unchanged"; an empty walk must hash to nothing."""
    M, demo = gate
    assert E._source_fingerprint(demo / "does-not-exist") is None
    assert E._source_fingerprint(tmp_path / "empty") is None
    for _ in range(5):
        out = M._failed(["x"], {"unit": "u"}, demo / "does-not-exist")
    assert out["halt"] is False


def test_the_reason_stays_within_the_field_budget(gate):
    M, demo = gate
    M._failed(["y" * 4000], {"unit": "u"}, demo)
    out = M._failed(["y" * 4000], {"unit": "u"}, demo)
    assert len(out["next_target"]["reason"]) <= 2000
    assert out["next_target"]["reason"].startswith("NO CODE CHANGED")


def test_both_failing_branches_go_through_the_same_builder():
    """A check wired into one branch and forgotten in the other is how this was missed once already."""
    import inspect

    import scripts.tt_hw_planner.e2e_mcp as M

    src = inspect.getsource(M.termination_check)
    assert src.count("_failed(") == 2
    assert "_count_unchanged_round" not in src  # it belongs to the builder, not to a branch


# --- 4. the halt is visible, and costs nothing extra ---------------------------------------------


def test_the_loop_reports_why_it_halted(capsys):
    """A silent halt is indistinguishable from running out of rounds."""
    res = CH.run_cc_loop(
        prompt="p",
        mcp_config_path="c",
        allowed_tools=[],
        cwd=".",
        env={},
        gate_fn=lambda: {"halt": True, "reason": "NO CODE CHANGED: ...", "can_stop": False},
        max_rounds=3,
    )
    assert res["halted"] is True and res["rounds"] == 0
    assert "NO CODE CHANGED" in capsys.readouterr().out


def test_the_loop_hands_back_the_verdict_that_halted_it():
    st = {"halt": True, "reason": "r", "can_stop": False}
    res = CH.run_cc_loop(
        prompt="p",
        mcp_config_path="c",
        allowed_tools=[],
        cwd=".",
        env={},
        gate_fn=lambda: st,
        max_rounds=3,
    )
    assert res["state"] is st


def test_a_halted_run_does_not_re_run_the_gate_to_be_told_the_same_thing():
    """The whole point of halting is not to spend the gate again."""
    import inspect

    src = inspect.getsource(E._run_emit_e2e_cc)
    assert 'res.get("state") if res.get("halted") else gate_fn()' in src


# --- 5. the constraints --------------------------------------------------------------------------


def test_it_names_no_model_or_stage():
    """Nothing here may assume what the model calls its stages or where its code lives.

    The EXECUTABLE body is scanned, not the prose, matching this suite's existing convention."""
    import ast
    import inspect
    import textwrap

    import scripts.tt_hw_planner.e2e_mcp as M

    for fn in (
        M._count_unchanged_round,
        M._unchanged_round_note,
        M._noop_round_limit,
        M._failed,
        E._source_fingerprint,
        E.run_stamp,
    ):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "encoder", "decoder", "vae", "demos/"):
            assert name not in lowered, f"{name!r} in {fn.__name__} assumes the model's own vocabulary"
