# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's Slack notifier (lives in tt_metal/tt-llk/perf)."""

import json
import pathlib
import sys


_PERF = pathlib.Path(__file__).parents[2] / "perf"
sys.path.insert(0, str(_PERF))
from gate_slack_notify import (
    _describe,
    build_text,
    main,
    read_regressions,
    verdict,
)

_CTX = {
    "pr_number": "123",
    "pr_url": "https://github.com/tenstorrent/tt-metal/pull/123",
    "pr_title": "Speed up the matmul inner loop",
    "author": "someone",
    "arch": "blackhole",
    "run_types": "L1_TO_L1",
    "run_url": "https://github.com/tenstorrent/tt-metal/actions/runs/1",
    "baseline_sha": "abc1234",
    "reason": "",
}

_ROW = {
    "marker": "TILE_LOOP",
    "run_type": "L1_TO_L1",
    "current": "2130.0",
    "baseline": "2000.0",
    "delta_pct": "6.5",
    "delta_cycles": "130.0",
    "math_fidelity": "HiFi4",
}


def _write_regressions(tmp_path, rows):
    import csv

    path = tmp_path / "report.regressions.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_regression_is_reported():
    assert (
        verdict(exit_code=1, have_baseline=True, report_written=True)[0] == "regressed"
    )


def test_pass_needs_a_baseline_and_a_report():
    assert verdict(exit_code=0, have_baseline=True, report_written=True)[0] == "clean"


def test_no_baseline_is_skipped_not_clean():
    """A gate with nothing to compare against did not pass. It did not run."""
    status, reason = verdict(exit_code=0, have_baseline=False, report_written=True)
    assert status == "skipped"
    assert "baseline" in reason


def test_missing_report_is_skipped_not_clean():
    """The silent-pass trap."""
    status, reason = verdict(exit_code=0, have_baseline=True, report_written=False)
    assert status == "skipped"
    assert "did not run" in reason


def test_regression_message_names_the_author_and_the_pr():
    text = build_text("regressed", [_ROW], _CTX)
    assert "someone" in text
    assert "/pull/123" in text
    assert "1 point(s) regressed" in text


def test_regression_message_carries_a_reproduce_command():
    text = build_text("regressed", [_ROW], _CTX)
    assert "perf_compare_commits.sh blackhole '<perf_test_module>'" in text
    assert "--baseline abc1234" in text


def test_skipped_message_denies_the_green_check():
    ctx = dict(_CTX, reason="No baseline was found, so the gate compared nothing.")
    text = build_text("skipped", [], ctx)
    assert "does not mean the PR is clean" in text


def test_bullet_leads_with_the_marker():
    line = _describe(_ROW)
    assert line.startswith("• TILE_LOOP L1_TO_L1")


def test_a_broadcast_in_the_pr_title_cannot_ping_the_channel():
    text = build_text("regressed", [_ROW], dict(_CTX, pr_title="<!channel> ship it"))
    assert "<!channel>" not in text
    assert "@channel" in text


def test_payload_escapes_a_config_value_that_holds_a_quote(tmp_path, monkeypatch):
    """A sweep value must never be able to break the request body."""
    row = dict(_ROW, math_fidelity='HiFi4 "quoted", and\nnewline')
    regressions = _write_regressions(tmp_path, [row])
    have_baseline = tmp_path / "have_baseline.txt"
    have_baseline.write_text("true")
    out = tmp_path / "payload.json"
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))

    main(
        [
            "--regressions",
            str(regressions),
            "--report",
            str(regressions),
            "--have-baseline",
            str(have_baseline),
            "--exit-code",
            "1",
            "--channel",
            "C0TEST",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text())
    assert payload["channel"] == "C0TEST"
    assert "newline" in payload["text"]


def test_have_baseline_absent_reads_as_no_baseline(tmp_path, monkeypatch):
    out = tmp_path / "payload.json"
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    main(
        [
            "--have-baseline",
            str(tmp_path / "does_not_exist.txt"),
            "--exit-code",
            "0",
            "--channel",
            "C0TEST",
            "--out",
            str(out),
        ]
    )

    assert "status=skipped" in output.read_text()
    assert "should_post=true" in output.read_text()


def test_clean_run_posts_too(tmp_path, monkeypatch):
    have_baseline = tmp_path / "have_baseline.txt"
    have_baseline.write_text("true")
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    main(
        [
            "--have-baseline",
            str(have_baseline),
            "--report",
            str(have_baseline),
            "--exit-code",
            "0",
            "--channel",
            "C0TEST",
            "--out",
            str(tmp_path / "payload.json"),
        ]
    )

    assert "status=clean" in output.read_text()
    assert "should_post=true" in output.read_text()


def test_missing_regressions_file_reads_as_no_rows(tmp_path):
    assert read_regressions(str(tmp_path / "nope.csv")) == []
