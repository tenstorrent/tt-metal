# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from nightly_health import assess, build_text, gate_result, leg_limits

DAY = dt.date(2026, 10, 13)
RUN = {
    "id": 1,
    "html_url": "https://x/1",
    "head_sha": "3cdbe9ca17d0",
    "status": "completed",
    "conclusion": "success",
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
    text = build_text("2026-10-13", [report])
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
        "SoL: llk_perf_wormhole group 1/5 cancelled",
        "SoL: upload failure",
        "SoL blackhole: gate regressed, 110,671 points",
    ]
    assert ":red_circle:" in build_text("2026-10-13", [report])


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
