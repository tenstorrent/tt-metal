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


def test_a_queued_fix_runs_before_the_next_task_under_until(orch):  # noqa: F811
    """F206: with --until, a fix whose gate tasks all passed was out of scope (nothing depended on it) and the next
    task ran first (DeepSeek-V4.1 FIX.6, 2026-10-28: X.1 profiled the old default on the new torus spec, where it
    hangs). The task a fix is placed before now depends on it."""
    o = orch(
        [impl_task(), impl_task(id="C.2", deps=["C.1"])],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}},
            "FIX.1.overseer.1.md": {"write": {"src/impl.txt": "0.9995"}},
            "C.2.implement.1.md": {"write": {"src/impl.txt": "0.9995"}},
        },
    )
    real = o.run_agent

    def overseer_queues_meanwhile(task, role, attempt, brief, **kw):
        if task["id"] == "C.1":
            led = {t: o.led.task(t) for t in o.led.tasks()}
            OV.queue_fix(o.spec.bringup_dir, "Change the default.", ["C.1"], tasks=led)
        return real(task, role, attempt, brief, **kw)

    o.run_agent = overseer_queues_meanwhile
    assert o.run(until="C.2") == DONE
    order = [x.split("]")[0].strip("[ ") for x in orch.lines if x.startswith("[")]
    assert "FIX.1" in order and order.index("FIX.1") < order.index("C.2"), order
    assert o.led.state()["FIX.1"]["status"] == "PASS" and "FIX.1" in o.led.task("C.2")["deps"]


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
    o.led.update(
        "FIX.1", status="TODO", history_add={"t": now(), "status": "RESUMED", "from": "STOPPED"}
    )  # what `resume` does
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


def test_a_fix_may_write_the_paths_it_names_even_when_its_gate_tasks_own_none(tmp_path):
    """F127: DeepSeek-V4.1 FIX.5 was gated on X.3 (a scripted task, no paths), so its agent could not edit tt/."""
    tasks = {"X.3": {"id": "X.3", "gate": {"cmd": "pytest x", "metrics": {}}, "deps": []}}
    job = OV.queue_fix(tmp_path, "Clear the memo per chunk.", ["X.3"], tasks=tasks, paths=["models/demos/m/tt"])
    assert OV.fix_task(job, tasks)["paths"] == ["models/demos/m/tt"]
    plain = OV.queue_fix(tmp_path, "Another.", ["X.3"], tasks=tasks)
    assert OV.fix_task(plain, tasks)["paths"] == [] and "paths" not in plain


def test_only_can_name_a_queued_fix(orch):  # noqa: F811
    """F143: `resume --only FIX.<n>` with the fix still queued said "nothing left to run" (DeepSeek-V4.1 FIX.8): the
    --only filter ran before the queued fixes were placed."""
    o = orch(
        [impl_task()],
        {
            "C.1.implement.1.md": {"write": {"src/impl.txt": "0.999"}},
            "FIX.1.overseer.1.md": {"write": {"src/impl.txt": "0.9995"}},
        },
    )
    assert o.run() == DONE
    led = {t: o.led.task(t) for t in o.led.tasks()}
    OV.queue_fix(o.spec.bringup_dir, "Store the weights at load.", ["C.1"], tasks=led)
    assert o.run(only="FIX.1") == DONE
    assert o.led.state()["FIX.1"]["status"] == "PASS"


def test_a_second_orchestrator_on_a_live_run_is_refused(tmp_path):
    """F147: DeepSeek-V4.1 FIX.9 ran two orchestrators at once (a paused one still in its attempt loop)."""
    import os
    import subprocess

    from models.demos.common.bringup.orchestrator import _live_orchestrator

    pid_file = tmp_path / "orchestrator.pid"
    assert _live_orchestrator(pid_file) is None
    p = subprocess.Popen(["sleep", "30"])
    try:
        pid_file.write_text(f"{p.pid} now\n")
        assert _live_orchestrator(pid_file) == p.pid
        pid_file.write_text(f"{os.getpid()} now\n")
        assert _live_orchestrator(pid_file) is None
    finally:
        p.kill()
        p.wait()
    pid_file.write_text(f"{p.pid} now\n")
    assert _live_orchestrator(pid_file) is None  # a dead process does not block


def test_a_fix_job_runs_its_gate_tasks_commands_exactly(sandbox):
    """F151: a fix job's segment of a full-model task (integrate, contract, perf) kept run_safe_pytest's precompile
    pass, which the task's own gate skips (gate_command adds --no-precompile by the task's step, and the fix's step is
    overseer_fix): DeepSeek-V4.1 FIX.8 / FIX.9 / FIX.5 compiled L.last and X.3 twice in 45-70 min gates."""
    from models.demos.common.bringup.core.gate import gate_command, gate_units, run_gate

    runner = sandbox.repo / "scripts/run_safe_pytest.sh"
    runner.parent.mkdir()
    runner.write_text(f"#!/bin/bash\necho \"$BRINGUP_TASK $*\" >> {sandbox.repo.parent / 'calls.txt'}\n")
    runner.chmod(0o755)
    safe = "scripts/run_safe_pytest.sh --run-all"
    tasks = {
        "C.1": {"id": "C.1", "title": "c", "step": "implement", "device": True, "gate": {"cmd": f"{safe} t_c.py"}},
        "L.1": {
            "id": "L.1",
            "title": "l",
            "step": "integrate",
            "device": True,
            "gate": {"cmd": f"BRINGUP_RUNG=s {safe} t_l.py"},
        },
        "X.3": {
            "id": "X.3",
            "title": "x",
            "step": "perf",
            "device": True,
            "gate": {"cmd": f"{safe} t_l.py && TT_METAL_DEVICE_PROFILER=1 {safe} test_profile.py"},
        },
    }
    sandbox.tasks(*tasks.values())
    job = OV.queue_fix(sandbox.spec.bringup_dir, "Fix.", ["C.1", "L.1", "X.3"], tasks=tasks)
    fix = OV.fix_task(job, tasks)
    units = gate_units(fix, tasks)
    # F170: the fix runs each task's units exactly as that task's own gate does (X.3: its ladder and its profile)
    assert units == [u for t in ("C.1", "L.1", "X.3") for u in gate_units(tasks[t], tasks)]
    assert "--no-precompile" not in units[0][1] and units[1][1].count("--no-precompile") == 1
    assert gate_command(fix, tasks).count("--no-precompile") == 4  # L.1, and X.3's ladder, clean wall and profile
    led = sandbox.tasks(*tasks.values(), fix)
    for t in tasks:
        led.update(t, status="PASS")
    assert run_gate(sandbox.spec, led, "FIX.1").verdict == "PASS"
    calls = (sandbox.repo.parent / "calls.txt").read_text().splitlines()
    assert calls[:2] == ["C.1 --run-all t_c.py", "L.1 --no-precompile --run-all t_l.py"]
    assert len(calls) == 5 and all(c.startswith("X.3 --no-precompile") for c in calls[2:])


def test_until_runs_the_task_and_what_it_needs_then_stops(orch):  # noqa: F811
    """F152: `--until X` stopped only X's downstream tasks, so every independent task still ran (DeepSeek-V4.1: FIX.5's
    70 min gate after `resume --until FIX.7`). It now runs X and its dependencies, then stops."""
    o = orch(
        [impl_task(), impl_task(id="C.2", deps=["C.1"]), impl_task(id="C.3"), impl_task(id="C.4", deps=["C.2"])],
        {f"C.{n}.implement.1.md": {"write": {"src/impl.txt": "0.999"}} for n in (1, 2, 3, 4)},
    )
    assert o.run(until="C.2") == DONE
    st = o.led.state()
    assert st["C.1"]["status"] == st["C.2"]["status"] == "PASS"
    assert "C.3" not in st or st["C.3"].get("status") in (None, "TODO")
    assert "C.4" not in st or st["C.4"].get("status") in (None, "TODO")
    assert any("reached C.2" in ln for ln in orch.lines)
    assert o.run(until="C.2") == DONE and not [c for c in orch.calls() if c.startswith(("C.3", "C.4"))]


def test_a_fix_is_not_placed_before_a_perf_pick_whose_ab_is_measured(orch):  # noqa: F811
    """F213: a perf pick waiting for (or holding) its A/B decision has its change in the tree; a fix released then was
    placed before it, worked on that tree and would sweep the pick into its own commit (DeepSeek-V4.1 FIX.9 before the
    accepted P.6). The fix now goes after the pick, before the next open task."""
    from models.demos.common.bringup.orchestrator import _place_fixes

    o = orch([impl_task(), dict(impl_task(id="P.1", deps=["C.1"]), step="perf"), impl_task(id="C.2", deps=["P.1"])], {})
    o.led.update("C.1", status="PASS")
    o.led.update("P.1", status="TODO", waiting="apply P.1 or not?")
    led = {t: o.led.task(t) for t in o.led.tasks()}
    OV.queue_fix(o.spec.bringup_dir, "Change the default.", ["C.1"], tasks=led)
    assert _place_fixes(o)
    ids = list(o.led.tasks())
    assert ids.index("P.1") < ids.index("FIX.1") < ids.index("C.2"), ids
    assert "FIX.1" not in (o.led.task("P.1").get("deps") or []) and "FIX.1" in o.led.task("C.2")["deps"]
