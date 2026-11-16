# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import os
import pathlib
import shutil
import subprocess
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import merge_gate_audit as audit_mod
from merge_gate_audit import assess, landed, needs_gate, report, short_points

_ROOT = pathlib.Path(__file__).parents[5]
RUN = {"id": 1, "html_url": "https://x/1"}
G = "LLK perf merge gate"
PASS = {"blackhole": {"kind": "pass"}, "wormhole": {"kind": "pass"}}


def _jobs(measure="success", upload="success", bh="success", wh="success", legs=3):
    jobs = [{"name": f"{G} (changes)", "conclusion": "success"}]
    for arch, runner in (("blackhole", "bh"), ("wormhole", "wh")):
        jobs += [
            {
                "name": f"{G} (measure) / llk_perf_merge_gate_{arch} group {n}/3 [{runner}]",
                "conclusion": measure,
            }
            for n in range(1, legs + 1)
        ]
    jobs.append(
        {
            "name": f"{G} (measure) / Upload perf data to the warehouse",
            "conclusion": upload,
        }
    )
    jobs.append(
        {"name": f"{G} (blackhole) / Compare perf: current vs main", "conclusion": bh}
    )
    jobs.append(
        {"name": f"{G} (wormhole) / Compare perf: current vs main", "conclusion": wh}
    )
    return jobs


def test_measured_and_compared_is_no_finding():
    assert assess(RUN, _jobs(), PASS) is None


def test_no_queue_run():
    assert assess(None, [], {})[0] == "no-queue-run"


def test_skipped_when_the_changes_job_said_no():
    # #63474 on 2026-10-30: the changes job succeeded, and measure and both compares were skipped.
    jobs = _jobs(measure="skipped", upload="skipped", bh="skipped", wh="skipped")
    assert assess(RUN, jobs, {}) == ("skipped", ["changes job said no"])


def test_failed_compare():
    # #61006 on 2026-10-29: the Wormhole compare could not download the perf data.
    kind, detail = assess(RUN, _jobs(wh="failure"), {"blackhole": {"kind": "pass"}})
    assert kind == "failed"
    assert detail == ["wormhole compare failure"]


def test_failed_leg():
    # #59422 on 2026-10-30: a perf test failed in one leg of each arch.
    jobs = _jobs(bh="failure", wh="failure")
    jobs[1]["conclusion"] = "failure"
    kind, detail = assess(RUN, jobs, {})
    assert kind == "failed"
    assert "llk_perf_merge_gate_blackhole group 1/3 failure" in detail


def test_green_gate_that_compared_nothing():
    # 2026-11-14: the warehouse key was empty, the upload failed, and both compares
    # passed with the verdict "no-compare".
    nothing = {"blackhole": {"kind": "no-compare"}, "wormhole": {"kind": "no-compare"}}
    assert assess(RUN, _jobs(upload="failure"), nothing) == (
        "failed",
        [
            "Upload perf data to the warehouse failure",
            "blackhole: verdict no-compare",
            "wormhole: verdict no-compare",
        ],
    )
    assert assess(RUN, _jobs(), nothing)[0] == "incomplete"


def test_missing_leg_is_incomplete():
    kind, detail = assess(RUN, _jobs(legs=2), PASS)
    assert kind == "incomplete"
    assert detail == ["blackhole: legs [1, 2] of [3]", "wormhole: legs [1, 2] of [3]"]


def test_few_points_is_incomplete():
    # #62675 on 2026-10-30 compared 79,530 Wormhole points; the day's other runs 106,140.
    commits = [
        {"finding": None, "outputs": {"wormhole": {"points": p}}}
        for p in (79530, 106140, 106140, 106140)
    ]
    short_points(commits, 0.9)
    assert commits[0]["finding"] == (
        "incomplete",
        ["wormhole: 79,530 points compared, day median 106,140"],
    )
    assert all(c["finding"] is None for c in commits[1:])


def test_report_groups_the_same_finding():
    run = {"html_url": "https://x/1"}
    gated = [
        {"sha": "a" * 40, "pr": 1, "run": run, "finding": ("failed", ["x"])},
        {"sha": "b" * 40, "pr": 2, "run": run, "finding": ("failed", ["x"])},
        {"sha": "c" * 40, "pr": 3, "run": run, "finding": None},
    ]
    out = report("2026-11-14", [{}] * 5, gated)
    assert out["problems"] == ["merge gate: 2 landed failed (x): #1, #2"]
    assert out["lines"][0].startswith(
        "5 commits landed, 3 changed LLK perf inputs, 1 measured"
    )


def _git(repo, *args, date=None):
    env = dict(os.environ)
    if date:
        env.update(GIT_COMMITTER_DATE=date, GIT_AUTHOR_DATE=date)
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    ).stdout.strip()


def _commit(repo, path, text, subject, date):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "commit",
        "-qm",
        subject,
        date=date,
    )
    return _git(repo, "rev-parse", "HEAD")


def test_classify_with_the_commits_own_perf_tests(tmp_path):
    # #63474: a functional test that a perf test imports is a perf input. The CI
    # checkout had no perf_*.py and called it harmless.
    _git(tmp_path, "init", "-q")
    tests = "tt_metal/tt-llk/tests/python_tests"
    script = ".github/scripts/llk-perf-inputs-changed.sh"
    (tmp_path / script).parent.mkdir(parents=True)
    shutil.copy(_ROOT / script, tmp_path / script)
    _commit(
        tmp_path,
        f"{tests}/perf_x.py",
        "from test_x import run\n",
        "base",
        "2026-10-29T10:00:00Z",
    )
    imported = _commit(
        tmp_path, f"{tests}/test_x.py", "1\n", "x (#1)", "2026-10-30T10:00:00Z"
    )
    alone = _commit(
        tmp_path, f"{tests}/test_y.py", "1\n", "y (#2)", "2026-10-30T11:00:00Z"
    )
    _commit(tmp_path, "docs/z.md", "1\n", "z (#3)", "2026-10-31T00:00:01Z")
    assert needs_gate(str(tmp_path), imported)
    assert not needs_gate(str(tmp_path), alone)
    assert landed(str(tmp_path), "HEAD", dt.date(2026, 10, 30)) == [
        {"sha": imported, "pr": 1},
        {"sha": alone, "pr": 2},
    ]


def test_audit_asks_for_a_run_before_it_calls_a_commit_unqueued(monkeypatch):
    commits = [{"sha": "a", "pr": 1}, {"sha": "b", "pr": 2}]
    asked = []
    monkeypatch.setattr(audit_mod, "landed", lambda *a: [dict(c) for c in commits])
    monkeypatch.setattr(audit_mod, "needs_gate", lambda *a: True)
    monkeypatch.setattr(audit_mod, "queue_runs", lambda *a: {"a": RUN})
    monkeypatch.setattr(audit_mod, "queue_run", lambda repo, sha: asked.append(sha))
    monkeypatch.setattr(audit_mod, "run_jobs", lambda *a: _jobs())
    monkeypatch.setattr(audit_mod, "gate_outputs", lambda *a: PASS)
    _, gated = audit_mod.audit("r", ".", "HEAD", dt.date(2026, 11, 14), "cache")
    assert asked == ["b"]
    assert [c["finding"] and c["finding"][0] for c in gated] == [None, "no-queue-run"]
