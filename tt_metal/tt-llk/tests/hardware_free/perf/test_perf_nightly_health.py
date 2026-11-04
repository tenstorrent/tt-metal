# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
from nightly_health import NIGHTLIES, assess, build_text

DAY = dt.date(2026, 10, 13)
SOL, NON_SOL = NIGHTLIES
RUN = {
    "id": 1,
    "html_url": "https://x/1",
    "head_sha": "3cdbe9ca17d0",
    "status": "completed",
    "conclusion": "success",
}


def _verdict(arch, status="pass", regressions=0, complete=True, reason=""):
    return {
        "arch": arch,
        "status": status,
        "reason": reason,
        "counts": {"regressions": regressions},
        "baseline": {"commit_sha": "6d88aa4fe5b4a049", "run_ts": "2026-09-29 11:38:57"},
        "legs": [
            {
                "group": 1,
                "groups": 5,
                "minutes": 75,
                "timeout_minutes": 150,
                "complete": True,
            },
            {
                "group": 2,
                "groups": 5,
                "minutes": 150,
                "timeout_minutes": 150,
                "complete": complete,
            },
        ],
    }


def _verdicts(**wormhole):
    return {
        "blackhole": _verdict("blackhole"),
        "wormhole": _verdict("wormhole", **wormhole),
    }


def test_a_clean_night_has_no_problem_and_reports_leg_use_and_baseline_age():
    report = assess(SOL, RUN, _verdicts())
    assert report["problems"] == []
    text = build_text(DAY, [report])
    assert ":white_check_mark:" in text and "healthy" in text
    assert "legs, % of limit used: 1/5 50% 2/5 100%" in text
    assert "baseline `6d88aa4fe5b`, 14 days old" in text


def test_an_incomplete_leg_and_a_regression_are_problems():
    leg = "A wormhole measuring leg did not complete: leg 2/5 ran 10 of 40 tests."
    v = _verdicts(status="incomplete", complete=False, reason=leg)
    v["blackhole"] = _verdict("blackhole", "regression", 110671)
    report = assess(SOL, {**RUN, "conclusion": "failure"}, v)
    assert report["problems"] == [
        f"{SOL} blackhole: gate regressed, 110,671 points",
        f"{SOL} wormhole: gate incomplete. {leg}",
    ]
    text = build_text(DAY, [report])
    assert ":red_circle:" in text and "2/5 100% ✗" in text


def test_a_missing_run_and_a_missing_verdict_are_problems():
    assert assess(NON_SOL, None, {})["problems"] == [f"{NON_SOL}: no scheduled run"]
    report = assess(SOL, RUN, {"wormhole": _verdict("wormhole")})
    assert report["problems"] == [f"{SOL} blackhole: no gate verdict"]


def test_a_failed_run_with_clean_verdicts_is_still_a_problem():
    report = assess(SOL, {**RUN, "conclusion": "failure"}, _verdicts())
    assert report["problems"] == [f"{SOL}: run failure"]


def test_a_run_in_progress_is_a_problem_and_shows_what_it_has():
    run = {**RUN, "status": "in_progress", "conclusion": None}
    report = assess(SOL, run, {})
    assert report["problems"] == [f"{SOL}: run still in_progress"]
    assert "in_progress" in build_text(DAY, [report])


def test_a_verdict_without_a_baseline_date_has_unknown_age():
    v = _verdicts()
    v["blackhole"]["baseline"] = {}
    assert "baseline `?`, age unknown" in build_text(DAY, [assess(SOL, RUN, v)])
