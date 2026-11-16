# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import ci_trends
from ci_trends import cause, durations_report, leg_report, queue_report, read_day

DAY = dt.date(2026, 11, 16)
LEG = "llk_perf_merge_gate_blackhole group 1/4"
CAPS = {LEG: 15}


def _job(id, name, conclusion="success", minutes=9, steps=()):
    start = dt.datetime(2026, 11, 15, 3, 0, tzinfo=dt.timezone.utc)
    return {
        "id": id,
        "name": name,
        "conclusion": conclusion,
        "started_at": start.isoformat(),
        "completed_at": (start + dt.timedelta(minutes=minutes)).isoformat(),
        "steps": list(steps),
    }


def _days(values_by_offset):
    return {
        (DAY - dt.timedelta(days=n)).isoformat(): {LEG: v}
        for n, v in values_by_offset.items()
    }


def test_causes_come_from_annotations_and_the_unfinished_step():
    hang = _job(1, "Build Quasar Perf", "cancelled", 15, ["Checkout code"])
    test = _job(2, "x", "failure", steps=["Run tests"])
    assert (
        cause(hang, "The job has exceeded the maximum execution time of 15m0s")
        == "checkout hang"
    )
    assert cause(test, "The job has exceeded the maximum execution time") == "timeout"
    assert cause(test, "API rate limit exceeded for installation") == "API rate limit"
    assert cause(test, "Error: write ECONNRESET") == "network"
    assert cause(test, 'Load key "id_key": error in libcrypto') == "credentials"
    assert cause(test, "The runner has received a shutdown signal.") == "runner lost"
    assert cause(test, "Process completed with exit code 1.") == "test failure"
    assert cause(_job(3, "x", "failure", steps=["Upload"]), "") == "unknown"


def test_a_leg_near_its_cap_or_much_slower_than_last_week_is_a_problem():
    problems, lines = leg_report("merge gate", _days({1: [9, 9, 13]}), DAY, CAPS)
    assert problems == [f"merge gate {LEG}: p90 13.0 min = 87 % of the 15 min cap"]
    assert lines == ["merge gate BH (cap 15 min, 3 runs): 1/4 9.0/13.0/13.0"]
    problems, _ = leg_report(
        "merge gate", _days({1: [9, 10, 11], 8: [7, 7, 8]}), DAY, CAPS
    )
    assert problems == [
        f"merge gate {LEG}: p90 8.0 → 11.0 min against the 7 days before"
    ]
    assert (
        leg_report("merge gate", _days({1: [9, 10, 11], 8: [9, 9, 10]}), DAY, CAPS)[0]
        == []
    )


def test_no_leg_in_the_window_is_a_line_not_a_problem():
    assert leg_report("SoL nightly", _days({9: [30]}), DAY, CAPS) == (
        [],
        ["SoL nightly: no leg ran in the last 7 days"],
    )


def test_cap_line_shows_how_little_of_the_cap_the_slowest_leg_uses():
    legs = {"2026-11-08": {"llk_perf_wormhole group 1/5": [12, 4]}}
    caps = {"llk_perf_wormhole group 1/5": 150}
    assert (
        ci_trends.cap_line("non-SoL nightly", legs, DAY, caps)
        == "non-SoL nightly slowest leg in 14 days: WH 12 of 150 min (8 %)"
    )


def test_read_day_splits_queue_cancels_timeouts_and_failures(monkeypatch):
    runs = [
        {"id": 10, "conclusion": "cancelled", "created_at": ""},
        {"id": 11, "conclusion": "failure", "created_at": ""},
        {"id": 12, "conclusion": "startup_failure", "created_at": ""},
        {"id": 13, "conclusion": "failure", "created_at": ""},
    ]
    jobs = {
        10: [_job(1, f"LLK perf merge gate (measure) / {LEG} [bh]", "cancelled", 4)],
        11: [
            _job(2, f"LLK perf merge gate (measure) / {LEG} [bh]", "cancelled", 15),
            _job(3, "Upload perf data to the warehouse LLK", "failure", 1, ["Upload"]),
            _job(4, "ttnn tests", "failure"),
        ],
        13: [
            _job(5, "Upload perf data to the warehouse LLK", "failure", 1, ["Upload"])
        ],
    }
    monkeypatch.setattr(ci_trends, "runs", lambda *a: runs)
    monkeypatch.setattr(ci_trends, "jobs", lambda repo, run_id: jobs[run_id])
    monkeypatch.setattr(ci_trends, "annotations", lambda *a: "exit code 255")
    logs = []
    monkeypatch.setattr(
        ci_trends,
        "job_log",
        lambda repo, id: logs.append(id) or "Permission denied (publickey).",
    )
    day = read_day("r", "2026-11-15", CAPS, max_logs=10)
    assert day["runs"] == 4 and day["startup_failure"] == 1
    leg = day["jobs"][f"LLK perf merge gate (measure) / {LEG}"]
    assert (leg["runs"], leg["failures"], leg["cancelled"], leg["causes"]) == (
        1,
        1,
        1,
        {"timeout": 1},
    )
    upload = day["jobs"]["Upload perf data to the warehouse LLK"]
    assert upload["causes"] == {"credentials": 2} and upload["run"] == 13
    assert "ttnn tests" not in day["jobs"]
    assert day["legs"] == {LEG: [15.0]}
    assert logs == [3]


def test_queue_report_flags_a_job_that_fails_often_and_many_startup_failures():
    day = {
        "runs": 10,
        "startup_failure": 4,
        "jobs": {
            "Upload": {
                "runs": 5,
                "failures": 3,
                "cancelled": 1,
                "causes": {"credentials": 3},
                "run": 7,
            },
            "Leg": {
                "runs": 20,
                "failures": 1,
                "cancelled": 0,
                "causes": {"timeout": 1},
                "run": 8,
            },
        },
    }
    problems, lines = queue_report(day, "2026-11-15")
    assert problems == [
        "merge queue 2026-11-15: 4 of 10 runs startup_failure",
        "merge queue 2026-11-15: Upload: 3/5 failed (credentials 3)",
    ]
    assert lines[0] == (
        "merge queue 2026-11-15: 10 runs, 4 startup_failure; "
        "LLK jobs 25 ran, 4 failed, 1 cancelled by the queue"
    )
    assert lines[1:] == [
        "• Upload: 3/5 failed (credentials 3), e.g. run 7",
        "• Leg: 1/20 failed (timeout 1), e.g. run 8",
    ]
    assert queue_report(None, "2026-11-15")[0] == ["merge queue 2026-11-15: not read"]


def test_old_or_missing_durations_are_problems():
    problems, lines = durations_report(
        {"smoke": (0.4, 1), "SoL": (8.2, 2), "merge gate": (None, None)}, 7
    )
    assert problems == [
        "durations SoL: newest is 8.2 days old (run 2)",
        "durations merge gate: none in the last 60 green runs",
    ]
    assert lines == ["test durations age: smoke 0.4 d · SoL 8.2 d · merge gate none"]


def test_durations_age_takes_the_first_green_run_with_durations(monkeypatch):
    calls = []

    def api(*args):
        calls.append(args[0])
        if "/runs?" in args[0]:
            return '{"id": 1}\n{"id": 2}\n{"id": 3}\n'
        return '{"created_at": "2026-11-14T08:00:00Z"}\n' if "/2/" in args[0] else ""

    monkeypatch.setattr(ci_trends, "_api", api)
    now = dt.datetime(2026, 11, 16, 8, tzinfo=dt.timezone.utc)
    assert ci_trends.durations_age("r", "merge-gate.yaml", "", "merge_group", now) == (
        2.0,
        2,
    )
    assert "branch=" not in calls[0] and len(calls) == 3


def test_history_reads_only_missing_days_and_drops_old_ones(monkeypatch):
    read = []
    monkeypatch.setattr(
        ci_trends, "read_day", lambda repo, d, *a: read.append(d) or {"runs": 0}
    )
    history = {"2026-11-15": {"runs": 5}, "2026-10-01": {"runs": 1}}
    kept = ci_trends.update_history("r", history, DAY, CAPS, 2, 0)
    assert read == ["2026-11-14", "2026-11-13"]
    assert set(kept) == {"2026-11-15", "2026-11-14", "2026-11-13"}


def test_the_health_job_runs_the_check_before_check_and_send():
    workflow = (
        pathlib.Path(__file__).parents[5]
        / ".github"
        / "workflows"
        / "llk-perf-nightly-health.yaml"
    ).read_text()
    assert workflow.index("ci_trends.py") < workflow.index("name: Check and send")
    assert json.dumps("llk-ci-trends-history")[1:-1] in workflow
