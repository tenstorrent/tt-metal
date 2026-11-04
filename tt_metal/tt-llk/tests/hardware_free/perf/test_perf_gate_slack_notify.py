# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the gate's Slack notifier (lives in tt_metal/tt-llk/perf)."""

import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
from gate_slack_notify import (
    _describe,
    build_text,
    from_verdict,
    main,
    read_regressions,
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


def _verdict(status, reason="", **baseline):
    """A verdict-<arch>.json as gate.py writes it."""
    return {
        "schema": 1,
        "gate": "merge",
        "arch": "blackhole",
        "status": status,
        "reason": reason,
        "policy": {
            "title": "LLK perf merge gate",
            "speed_of_light": False,
            "threshold": 0.02,
            "min_cycles": 30,
            "modules": {"perf_matmul": 0.25},
            "run_types": ["L1_TO_L1"],
        },
        "baseline": {"commit_sha": "abc1234", **baseline},
    }


def _write_verdict(tmp_path, v):
    path = tmp_path / "verdict.json"
    path.write_text(json.dumps(v))
    return path


def test_regression_is_reported():
    assert from_verdict(_verdict("regression"))[0] == "regressed"


def test_pass_is_clean():
    assert from_verdict(_verdict("pass"))[0] == "clean"


def test_every_other_status_is_skipped_not_clean():
    """A gate that compared nothing did not pass. It did not run."""
    for status in ("incomplete", "no-baseline", "no-compare"):
        got, ctx = from_verdict(_verdict(status, "No baseline was found."))
        assert got == "skipped" and ctx["reason"] == "No baseline was found."


def test_no_verdict_is_skipped_not_clean():
    """The silent-pass trap."""
    status, ctx = from_verdict(None)
    assert status == "skipped" and "no verdict" in ctx["reason"]


def test_the_context_comes_from_the_policy():
    _, ctx = from_verdict(_verdict("pass"))
    assert ctx["gate_name"] == "LLK perf merge gate"
    assert ctx["module_thresholds"] == "perf_matmul=0.25"
    assert ctx["run_types"] == "L1_TO_L1" and ctx["baseline_sha"] == "abc1234"


def test_a_stale_baseline_is_a_note():
    _, ctx = from_verdict(
        _verdict("pass", stale=True, expected_commit="def5678", run_ts="2026-11-04")
    )
    assert ctx["note"].startswith("Stale baseline: none at def5678 yet")


def test_regression_message_names_the_author_and_the_pr():
    text = build_text("regressed", [_ROW], _CTX)
    assert "someone" in text
    assert "/pull/123" in text
    assert "1 point(s) regressed" in text


def test_regression_message_carries_a_reproduce_command():
    text = build_text("regressed", [_ROW], _CTX)
    assert "perf_compare_commits.sh blackhole '<perf_test_module>'" in text
    assert "--baseline abc1234" in text


def test_regression_message_links_the_report():
    ctx = dict(
        _CTX,
        report_url="https://github.com/tenstorrent/tt-metal/actions/runs/1#summary-7",
        artifact_url="https://github.com/tenstorrent/tt-metal/actions/runs/1/artifacts/9",
    )
    text = build_text("regressed", [_ROW], ctx)
    assert (
        "<https://github.com/tenstorrent/tt-metal/actions/runs/1#summary-7|full table>"
        in text
    )
    assert "/artifacts/9|download the report and CSVs>" in text


def test_without_report_links_the_message_points_at_the_run():
    text = build_text("regressed", [_ROW], _CTX)
    assert (
        "attached to the <https://github.com/tenstorrent/tt-metal/actions/runs/1|gate run>"
        in text
    )


def test_skipped_message_denies_the_green_check():
    ctx = dict(_CTX, reason="No baseline was found, so the gate compared nothing.")
    text = build_text("skipped", [], ctx)
    assert "does not mean anything was checked" in text


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
    out = tmp_path / "payload.json"
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))

    main(
        [
            "--regressions",
            str(regressions),
            "--verdict",
            str(_write_verdict(tmp_path, _verdict("regression"))),
            "--channel",
            "C0TEST",
            "--out",
            str(out),
        ]
    )

    payload = json.loads(out.read_text())
    assert payload["channel"] == "C0TEST"
    assert "newline" in payload["text"]


def test_a_missing_verdict_posts_a_skip(tmp_path, monkeypatch):
    out = tmp_path / "payload.json"
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    main(
        [
            "--verdict",
            str(tmp_path / "does_not_exist.json"),
            "--channel",
            "C0TEST",
            "--out",
            str(out),
        ]
    )

    assert "status=skipped" in output.read_text()
    assert "should_post=true" in output.read_text()


def test_clean_run_posts_too(tmp_path, monkeypatch):
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))

    main(
        [
            "--verdict",
            str(_write_verdict(tmp_path, _verdict("pass"))),
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


def test_a_skip_names_the_arch():
    for ctx in (_CTX, {**_CTX, "arch": "wormhole"}):
        text = build_text("skipped", [], ctx)
        assert f"{ctx['arch']} · L1_TO_L1" in text.splitlines()[1]


def test_one_change_on_many_points_is_one_finding():
    rows = [
        dict(_ROW, marker=m, run_type=rt)
        for m in ("TILE_LOOP", "KERNEL")
        for rt in ("L1_TO_L1", "MATH_ISOLATE")
    ]
    text = build_text("regressed", rows, _CTX)
    assert "4 point(s) regressed, in 1 finding(s)" in text
    assert "×4 points" in text


def test_the_merge_gate_names_itself_and_carries_the_note():
    ctx = {**_CTX, "gate_name": "LLK perf merge gate", "note": "Stale baseline."}
    for status, rows in (("clean", []), ("skipped", []), ("regressed", [_ROW])):
        text = build_text(status, rows, ctx)
        assert "LLK perf merge gate" in text.splitlines()[0]
        assert ":hourglass: Stale baseline." in text


def test_no_note_adds_no_line():
    text = build_text("clean", [], _CTX)
    assert "*LLK perf gate passed*" in text and ":hourglass:" not in text


def test_a_run_without_a_pr_names_what_started_it():
    ctx = {**_CTX, "pr_number": "", "trigger": "manual run on some/branch"}
    assert (
        "(manual run on some/branch)" in build_text("skipped", [], ctx).splitlines()[0]
    )
    assert "(nightly)" in build_text("skipped", [], {**_CTX, "pr_number": ""})


def _post(tmp_path, monkeypatch, status, *extra):
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    args = ["--verdict", str(_write_verdict(tmp_path, _verdict(status)))]
    main(
        args + ["--channel", "C0TEST", "--out", str(tmp_path / "payload.json"), *extra]
    )
    return output.read_text()


def test_post_pass_false_keeps_a_pass_out_of_slack(tmp_path, monkeypatch):
    out = _post(tmp_path, monkeypatch, "pass", "--post-pass", "false")
    assert "status=clean" in out and "should_post=false" in out


def test_post_pass_false_still_posts_a_regression_and_a_skip(tmp_path, monkeypatch):
    assert "should_post=true" in _post(
        tmp_path, monkeypatch, "regression", "--post-pass", "false"
    )
    assert "should_post=true" in _post(
        tmp_path, monkeypatch, "no-baseline", "--post-pass", "false"
    )
