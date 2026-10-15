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
    assert not (o.run_dir / OV.AGENT_FILE).exists() and not (o.run_dir / OV.ABORT_FILE).exists()


def test_watch_can_skip_passes():
    from models.demos.common.bringup import watch as W

    lines = [
        "  [C.1] gate PASS -> abc",
        "  [C.2] ABORTED by the overseer (implement attempt 1): x",
        "  [C.3] OVERSEER NOTE OV-1 unanswered",
    ]
    assert W.events(lines) == lines and W.events(lines, quiet_passes=True) == lines[1:]
