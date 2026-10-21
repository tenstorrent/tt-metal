# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F69: overseer notes reach the agents' briefs and are answered; an abort stops an attempt without spending it."""

import json
import os

import pytest

from models.demos.common.bringup import overseer as OV
from models.demos.common.bringup.orchestrator import DONE
from models.demos.common.bringup.selftest.test_orchestrator import impl_task, orch  # noqa: F401 (fixture)


def report(tid, role, n, answers):
    return {f"bringup/results/knowledge_use/{tid}.{role}.{n}.json": json.dumps({"overseer": answers})}


def test_notes_match_tasks_and_roles(tmp_path):
    a = OV.add(tmp_path, "shorten DS41-12", ["C.*"])
    b = OV.add(tmp_path, "only for the test role", ["C.1"], roles=["test"])
    OV.add(tmp_path, "every task")
    assert (a["id"], b["id"]) == ("OV-1", "OV-2")
    assert [x["id"] for x in OV.for_task(tmp_path, "C.1", "implement")] == ["OV-1", "OV-3"]
    assert [x["id"] for x in OV.for_task(tmp_path, "C.1", "test")] == ["OV-1", "OV-2", "OV-3"]
    assert [x["id"] for x in OV.for_task(tmp_path, "S.1", "implement")] == ["OV-3"]
    OV.close(tmp_path, "OV-3", "done")
    assert OV.for_task(tmp_path, "S.1", "implement") == [] and "OV-1" in OV.brief_text(OV.load(tmp_path)[:1])
    assert OV.brief_text([]) == ""


def test_answers_flag_missing_bad_and_disagreeing():
    shown = [{"id": "OV-1"}, {"id": "OV-2"}, {"id": "OV-3"}, {"id": "OV-4"}]
    rep = {
        "overseer": [
            {"id": "OV-1", "answer": "done"},
            {"id": "OV-2", "answer": "disagree", "evidence": "probe says otherwise"},
            {"id": "OV-3", "answer": "maybe"},
        ]
    }
    out = OV.answers(rep, shown)
    assert out == [
        "OVERSEER NOTE OV-2 DISAGREED: probe says otherwise",
        "OVERSEER NOTE OV-3 answer 'maybe' is not one of ('done', 'n/a', 'disagree')",
        "OVERSEER NOTE OV-4 unanswered",
    ]


def test_abort_records_a_note_and_ends_the_agent(tmp_path):
    bdir, rdir = tmp_path / "b", tmp_path / "r"
    bdir.mkdir()
    rdir.mkdir()
    with pytest.raises(SystemExit):
        OV.abort(bdir, rdir, "no agent")
    OV.agent_started(rdir, 4242, "SG.1", "scale", 2)
    killed = []
    r = OV.abort(bdir, rdir, "the spec is wrong: do not rebuild the prompt", kill=lambda pg, sig: killed.append(pg))
    assert killed == [4242] and r["note"] == "OV-1"
    note = OV.load(bdir)[0]
    assert note["tasks"] == ["SG.1"] and "attempt 2 (scale)" in note["text"] and "do not rebuild" in note["text"]
    rec = OV.take_abort(rdir)
    assert rec["reason"].startswith("the spec is wrong") and OV.take_abort(rdir) is None


def test_the_brief_shows_the_notes_and_the_answers_are_checked(orch):  # noqa: F811
    o = orch(
        [impl_task()],
        {
            "C.1.implement.1.md": {
                "write": {
                    "src/impl.txt": "0.999",
                    **report("C.1", "implement", 1, [{"id": "OV-1", "answer": "disagree", "evidence": "measured"}]),
                }
            }
        },
    )
    OV.add(o.spec.bringup_dir, "use the fp32 path for the sums", ["C.*"], roles=["implement"])
    assert o.run() == DONE
    brief = (o.run_dir / "briefs" / "C.1.implement.1.md").read_text()
    assert "## Overseer notes" in brief and "OV-1" in brief and "fp32 path" in brief
    assert "OV-1" not in (o.run_dir / "briefs" / "C.1.test.1.md").read_text()  # the note is for the implement role
    assert any("OVERSEER NOTE OV-1 DISAGREED: measured" in x for x in orch.lines)


def test_the_overseers_files_are_never_charged_to_an_agent(orch):  # noqa: F811
    o = orch([impl_task()], {"C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}}})
    real = o.run_agent

    def overseer_writes_meanwhile(task, role, attempt, brief, **kw):
        OV.add(o.spec.bringup_dir, "a note written while the agent works")
        (o.spec.bringup_dir / "supervision.md").write_text("- a log line\n")
        return real(task, role, attempt, brief, **kw)

    o.run_agent = overseer_writes_meanwhile
    assert o.run() == DONE and not o.led.state()["C.1"].get("review")


def test_an_aborted_attempt_does_not_spend_the_budget(orch, sandbox):  # noqa: F811
    o = orch([impl_task()], {})
    abort = json.dumps({"task": "C.1", "role": "implement", "attempt": 1, "reason": "stop: wrong approach"})
    script = {
        # attempt 1 is aborted by the overseer while it works (the ABORT file appears), attempts 2-4 fail, 5 passes
        "C.1.implement.1.md": {"write": {str(o.run_dir / OV.ABORT_FILE): abort, "src/impl.txt": "0.5"}},
        "C.1.implement.2.md": {"write": {"src/impl.txt": "0.5"}},
        "C.1.implement.3.md": {"write": {"src/impl.txt": "0.5"}},
        "C.1.implement.4.md": {"write": {"src/impl.txt": "0.999"}},
    }
    with open(os.environ["MOCK_AGENT_SCRIPT"], "w") as f:
        json.dump(script, f)
    OV.add(o.spec.bringup_dir, "stop: wrong approach", ["C.1"])
    assert o.run() == DONE
    st = o.led.state()["C.1"]
    assert [h["status"] for h in st["history"]].count("ABORTED") == 1
    assert any("ABORTED by the overseer (implement attempt 1)" in x for x in orch.lines)
    brief2 = (o.run_dir / "briefs" / "C.1.implement.2.md").read_text()
    assert "aborted by the overseer: stop: wrong approach" in brief2 and "OV-1" in brief2
    assert "outside the allowed paths" not in brief2  # F77: the abort's own state.json entry is not the agent's
    assert not (o.run_dir / OV.AGENT_FILE).exists() and not (o.run_dir / OV.ABORT_FILE).exists()


def test_watch_can_skip_passes():
    from models.demos.common.bringup import watch as W

    lines = [
        "  [C.1] gate PASS -> abc",
        "  [C.2] ABORTED by the overseer (implement attempt 1): x",
        "  [C.3] OVERSEER NOTE OV-1 unanswered",
    ]
    assert W.events(lines) == lines and W.events(lines, quiet_passes=True) == lines[1:]


def test_a_queued_fix_runs_next_with_an_agent_and_the_gates_it_names(orch):  # noqa: F811
    """F77: a fix the overseer queues runs at the next task boundary, its agent always starts (no precheck), its
    gate is the named tasks' gates, and it keeps the plan approval's ledger signature."""
    from models.demos.common.bringup.plan import approvals as A

    o = orch(
        [impl_task(), impl_task(id="C.2", deps=["C.1"])],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}},
            "FIX.1.overseer.1.md": {"write": {"src/impl.txt": "0.9995"}},
        },
    )
    sig = A.ledger_signature(o.spec)
    real = o.run_agent

    def overseer_queues_meanwhile(task, role, attempt, brief, **kw):
        if task["id"] == "C.1":
            led = {t: o.led.task(t) for t in o.led.tasks()}
            OV.queue_fix(o.spec.bringup_dir, "Store the weights at load. Not per chunk.", ["C.1"], tasks=led)
        return real(task, role, attempt, brief, **kw)

    o.run_agent = overseer_queues_meanwhile
    assert o.run() == DONE
    order = [x.split("]")[0].strip("[ ") for x in orch.lines if x.startswith("[")]
    assert order.index("FIX.1") < order.index("C.2"), order
    fix = o.led.task("FIX.1")
    c1 = o.led.task("C.1")["gate"]
    assert fix["step"] == OV.FIX_STEP and fix["precheck"] is False and fix["gate"]["metrics"] == c1["metrics"]
    # F79: the segment runs as C.1 (its tests see their own task id); the metrics still land in FIX.1 (it passed)
    assert fix["gate"]["cmd"] == f"BRINGUP_TASK=C.1 {c1['cmd']}"
    assert o.led.state()["FIX.1"]["status"] == "PASS"
    brief = (o.run_dir / "briefs" / "FIX.1.overseer.1.md").read_text()
    assert "Role: overseer fix" in brief and "Store the weights at load" in brief and "C.1" in brief
    assert (o.spec.repo / "src/impl.txt").read_text() == "0.9995"
    assert OV.fixes(o.spec.bringup_dir)[0]["status"] == "placed"
    assert A.ledger_signature(o.spec) == sig


def test_a_fix_needs_known_gate_tasks(tmp_path):
    with pytest.raises(SystemExit):
        OV.queue_fix(tmp_path, "x", [])
    with pytest.raises(SystemExit):
        OV.queue_fix(tmp_path, "x", ["C.9"], tasks={"C.1": {}})
    a = OV.queue_fix(tmp_path, "first", ["C.1"], tasks={"C.1": {}, "FIX.3": {}})
    OV.add(tmp_path, "a note keeps the fixes")
    assert a["id"] == "FIX.4" and OV.fixes(tmp_path)[0]["id"] == "FIX.4" and OV.load(tmp_path)[0]["id"] == "OV-1"
    assert [x["id"] for x in OV.take_fixes(tmp_path)] == ["FIX.4"] and OV.take_fixes(tmp_path) == []


def test_a_held_fix_is_on_record_but_waits_for_its_release(tmp_path):
    a = OV.queue_fix(tmp_path, "attention back to the reference", ["C.1"], hold="F76 installed")
    assert a["status"] == "held" and OV.take_fixes(tmp_path) == []
    with pytest.raises(SystemExit):
        OV.release(tmp_path, "FIX.9")
    OV.release(tmp_path, a["id"])
    assert [x["id"] for x in OV.take_fixes(tmp_path)] == [a["id"]]
    with pytest.raises(SystemExit):
        OV.release(tmp_path, a["id"])  # placed, not held


def test_metrics_land_in_the_record_task(monkeypatch, tmp_path):
    from models.demos.common.bringup.core import metrics as M

    monkeypatch.setenv(M.RESULTS_ENV, str(tmp_path))
    monkeypatch.setenv(M.TASK_ENV, "C.swa.attention")
    monkeypatch.setenv(M.RECORD_ENV, "FIX.2")
    M.record("pcc_attention_L00", 0.999)
    assert M.task_id() == "C.swa.attention" and M.load("FIX.2", tmp_path)["pcc_attention_L00"]["value"] == 0.999
    assert not (tmp_path / "C.swa.attention.json").exists()


def test_an_overseer_status_is_shown_until_the_run_starts(tmp_path):
    """F84: why the overseer has the run stopped; the dashboard shows it, the orchestrator clears it at start."""
    rec = OV.set_status(tmp_path, "re-verifying passed steps, 12/42")
    assert OV.status(tmp_path)["text"] == rec["text"]
    OV.clear_status(tmp_path)
    assert OV.status(tmp_path) is None


def test_a_resumed_fix_gates_before_an_agent(orch, sandbox):  # noqa: F811
    """F87: a fix job resumed after a stop runs its gate first; its work, finished meanwhile, passes without an agent."""
    from models.demos.common.bringup.orchestrator import now

    o = orch(
        [impl_task(), impl_task(id="C.2", deps=["C.1"])], {"C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}}}
    )
    assert o.run(until="C.1") == DONE
    led = {t: o.led.task(t) for t in o.led.tasks()}
    OV.queue_fix(o.spec.bringup_dir, "Fix it.", ["C.1"], tasks=led)
    (sandbox.repo / "src/impl.txt").write_text("0.5")  # the fix's agents never get it right: the fix stops
    assert o.run() != DONE and o.led.status("FIX.1") == "STOPPED"
    (sandbox.repo / "src/impl.txt").write_text("0.9995")  # the work is finished meanwhile
    o.led.update("FIX.1", status="TODO", history_add={"t": now(), "status": "RESUMED"})  # what `resume` does
    before = len(orch.calls())
    assert o.run() == DONE
    assert o.led.status("FIX.1") == "PASS" and not [c for c in orch.calls()[before:] if c.startswith("FIX.1")]


def test_a_fix_gated_on_an_unpassed_task_runs_before_it(orch):  # noqa: F811
    """F126: a fix whose gate task has not passed yet (DeepSeek-V4.1 X.3's timeline check) takes that task's deps and
    runs first; the task then follows the fix instead of failing its gate once more before the fix may start."""
    o = orch(
        [impl_task(), impl_task(id="C.2", deps=["C.1"])],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}},
            "FIX.1.overseer.1.md": {"write": {"src/impl.txt": "0.9995"}},
        },
    )
    real = o.run_agent

    def overseer_queues_meanwhile(task, role, attempt, brief, **kw):
        if task["id"] == "C.1":
            led = {t: o.led.task(t) for t in o.led.tasks()}
            OV.queue_fix(o.spec.bringup_dir, "Clear the memo at every chunk start.", ["C.2"], tasks=led)
        return real(task, role, attempt, brief, **kw)

    o.run_agent = overseer_queues_meanwhile
    assert o.run() == DONE
    order = [x.split("]")[0].strip("[ ") for x in orch.lines if x.startswith("[")]
    assert order.index("FIX.1") < order.index("C.2"), order
    assert o.led.task("FIX.1")["deps"] == ["C.1"] and "FIX.1" in o.led.task("C.2")["deps"]
    assert o.led.state()["FIX.1"]["status"] == "PASS" and o.led.state()["C.2"]["status"] == "PASS"
