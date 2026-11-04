# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import baseline_from_warehouse
import gate_slack_notify
from nightly_health import (
    assess,
    build_text,
    gate_result,
    last_to_finish,
    leg_limits,
    owners,
    payloads,
)

DAY = dt.date(2026, 10, 13)
RUN = {
    "id": 1,
    "html_url": "https://x/1",
    "head_sha": "3cdbe9ca17d0",
    "status": "completed",
    "conclusion": "success",
    "updated_at": "2026-10-13T07:00:00Z",
}
LIMITS = {"llk_perf_wormhole group 1/5": 150, "llk_perf_blackhole group 1/5": 240}
LOG = (
    "2026-10-13T05:30:00Z baseline baseline-20260929-1-blackhole (6d88aa4fe5b4a049, 2026-09-29 11:38:57+00:00): 9 rows\n"
    "2026-10-13T05:40:00Z verdict=clean regressions=0 should_post=True\n"
)


def _job(id, name, conclusion="success", minutes=60):
    start = dt.datetime(2026, 10, 13, 3, 0, tzinfo=dt.timezone.utc)
    return {
        "id": id,
        "name": name,
        "conclusion": conclusion,
        "started_at": start.isoformat(),
        "completed_at": (start + dt.timedelta(minutes=minutes)).isoformat(),
    }


def _jobs(wh_conclusion="success", upload="success"):
    return [
        _job(
            1,
            "LLK perf tests / llk_perf_wormhole group 1/5 [wh_n150_civ2]",
            wh_conclusion,
            150,
        ),
        _job(
            2,
            "LLK perf tests / llk_perf_blackhole group 1/5 [bh_p150b_civ2]",
            "success",
            120,
        ),
        _job(3, "LLK perf tests / Upload perf data to the warehouse", upload),
        _job(4, "Gate blackhole (speed of light) / Compare perf: current vs main"),
        _job(5, "Gate wormhole (speed of light) / Compare perf: current vs main"),
    ]


def test_a_clean_night_has_no_problem_and_reports_leg_use_and_baseline_age():
    report = assess("SoL", RUN, _jobs(), {4: LOG, 5: LOG}, LIMITS, DAY)
    assert report["problems"] == []
    text = build_text(DAY.isoformat(), [report])
    assert ":white_check_mark:" in text
    assert "BH: 1/5 50%" in text and "WH: 1/5 100%" in text
    assert "baseline `6d88aa4fe5b`, 14 days old" in text


def test_a_timeout_a_failed_upload_and_a_regression_are_problems():
    regressed = LOG.replace(
        "verdict=clean regressions=0", "verdict=regressed regressions=110671"
    )
    report = assess(
        "SoL", RUN, _jobs("cancelled", "failure"), {4: regressed, 5: LOG}, LIMITS, DAY
    )
    assert report["problems"] == [
        "SoL: llk_perf_wormhole group 1/5 timed out",
        "SoL: upload failure",
        "SoL blackhole: gate regressed, 110,671 points",
    ]
    assert ":red_circle:" in build_text(DAY.isoformat(), [report])


def test_a_missing_run_and_a_missing_verdict_are_problems():
    assert assess("non-SoL", None, [], {}, LIMITS, DAY)["problems"] == [
        "non-SoL: no scheduled run"
    ]
    report = assess("SoL", RUN, _jobs()[:3], {}, LIMITS, DAY)
    assert "SoL blackhole: no gate verdict" in report["problems"]


def test_gate_result_reads_the_verdict_and_the_baseline_date():
    assert gate_result(LOG, DAY) == {
        "verdict": "clean",
        "regressions": 0,
        "baseline": "6d88aa4fe5b",
        "baseline_age": 14,
    }


def test_leg_limits_take_the_timeout_of_each_entry(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "- name: llk_perf_wormhole group 1/5\n  skus:\n    wh_n150_civ2:\n      timeout: 150\n"
    )
    assert leg_limits(str(path)) == {"llk_perf_wormhole group 1/5": 150}


def test_a_run_in_progress_shows_unfinished_legs():
    run = {**RUN, "status": "in_progress", "conclusion": None}
    jobs = _jobs()
    jobs[0].update(conclusion=None, completed_at=None)
    report = assess("SoL", run, jobs, {4: LOG, 5: LOG}, LIMITS, DAY)
    assert report["problems"] == [
        "SoL: llk_perf_wormhole group 1/5 not finished",
        "SoL: run still in_progress",
    ]
    assert "WH: 1/5 ?% ✗" in build_text(DAY.isoformat(), [report])


def test_a_verdict_without_a_baseline_line_has_unknown_age():
    report = assess("SoL", RUN, _jobs(), {4: LOG.splitlines()[1], 5: LOG}, LIMITS, DAY)
    assert "gate blackhole: clean · baseline `?`, age unknown" in build_text(
        DAY.isoformat(), [report]
    )


def test_a_failed_job_outside_legs_upload_and_gates_is_a_problem():
    jobs = _jobs() + [_job(6, "LLK perf tests / Quasar perf compile", "failure")]
    run = {**RUN, "conclusion": "failure"}
    report = assess("SoL", run, jobs, {4: LOG, 5: LOG}, LIMITS, DAY)
    assert report["problems"] == ["SoL: LLK perf tests / Quasar perf compile failure"]


def test_only_the_last_nightly_to_finish_sends():
    sol = {**RUN, "id": 1, "updated_at": "2026-10-13T07:00:00Z"}
    non_sol = {**RUN, "id": 2, "updated_at": "2026-10-13T06:00:00Z"}
    assert last_to_finish([sol, non_sol], 1)
    assert not last_to_finish([sol, non_sol], 2)
    assert not last_to_finish([sol, {**non_sol, "status": "in_progress"}], 1)


def test_gate_result_reads_the_lines_the_gate_really_prints(
    tmp_path, capsys, monkeypatch
):
    report = tmp_path / "report.md"
    report.write_text("ok")
    have = tmp_path / "have_baseline.txt"
    have.write_text("true")
    gate_slack_notify.main(
        ["--channel", "C", "--report", str(report), "--have-baseline", str(have)]
        + ["--out", str(tmp_path / "payload.json")]
    )
    run = {"run_id": "baseline-20260929-1-blackhole", "commit_sha": "6d88aa4fe5b4a049"}
    run["run_ts"] = "2026-09-29 11:38:57+00:00"
    monkeypatch.setattr(baseline_from_warehouse, "connect", lambda: _Conn())
    monkeypatch.setattr(baseline_from_warehouse, "find_run", lambda *a: run)
    monkeypatch.setattr(baseline_from_warehouse, "fetch", lambda *a: [])
    monkeypatch.setattr(baseline_from_warehouse, "pivot", lambda rows: ([{}], [], 9))
    monkeypatch.setattr(baseline_from_warehouse, "write_csv", lambda *a: 1)
    monkeypatch.setattr(baseline_from_warehouse, "_write_meta", lambda *a, **k: None)
    baseline_from_warehouse.main(
        ["--arch", "blackhole", "--speed-of-light", "true"]
        + ["--out", str(tmp_path / "b.csv")]
    )
    log = "\n".join(
        f"2026-10-13T05:30:00Z {l}" for l in capsys.readouterr().out.splitlines()
    )
    assert gate_result(log, DAY) == {
        "verdict": "clean",
        "regressions": 0,
        "baseline": "6d88aa4fe5b",
        "baseline_age": 14,
    }


class _Conn:
    def cursor(self):
        return None

    def close(self):
        pass


def test_the_channel_copy_tags_the_owners_only_when_something_is_wrong(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(
        "- name: a\n  owner_id: U2\n  skus: {x: {timeout: 1}}\n"
        "- name: b\n  owner_id: U2\n  skus: {x: {timeout: 1}}\n"
    )
    assert owners(str(path)) == ["U2"]
    assert payloads("ok", False, "U1", "C1", ["U2"]) == [
        {"channel": "U1", "text": "ok"},
        {"channel": "C1", "text": "ok"},
    ]
    assert payloads("bad", True, "U1", "C1", ["U2"])[1]["text"] == "bad\n\ncc <@U2>"
    assert payloads("bad", True, "U1", "", ["U2"]) == [{"channel": "U1", "text": "bad"}]
