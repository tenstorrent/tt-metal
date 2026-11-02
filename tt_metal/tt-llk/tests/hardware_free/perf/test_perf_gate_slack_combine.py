# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the one-post-for-all-arches Slack combiner (lives in tt_metal/tt-llk/perf)."""

import csv
import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
from gate_slack_combine import build_text, main, threshold_for

_ROW = {
    "marker": "TILE_LOOP",
    "run_type": "L1_TO_L1",
    "current": "57392.0",
    "baseline": "49315.0",
    "delta_pct": "16.38",
    "delta_cycles": "8077.0",
    "test_module": "perf_eltwise_unary_sfpu",
    "mathop": "MathOperation.Hardmish",
}


def _ctx(arch, **extra):
    return {
        "pr_number": "59612",
        "pr_url": "https://github.com/tenstorrent/tt-metal/pull/59612",
        "author": "someone",
        "arch": arch,
        "run_types": "L1_TO_L1",
        "gate_name": "LLK perf merge gate",
        "threshold": "0.02" if arch == "blackhole" else "0.13",
        "module_thresholds": "" if arch == "blackhole" else "perf_matmul=0.25",
        **extra,
    }


def _part(arch, status, rows=(), **extra):
    return {
        "status": status,
        "should_post": status != "clean",
        "rows": list(rows),
        "ctx": _ctx(arch, **extra),
    }


def _points(tmp_path, arch, rows):
    path = tmp_path / f"{arch}.points.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


def test_a_regression_shows_the_same_point_on_the_clean_arch(tmp_path):
    wh = dict(_ROW, delta_pct="7.59", delta_cycles="4153.0")
    other = dict(_ROW, mathop="MathOperation.Gelu", delta_pct="99")
    points = {"wormhole": _points(tmp_path, "wormhole", [other, wh])}
    parts = [_part("wormhole", "clean"), _part("blackhole", "regressed", [_ROW])]
    text = build_text(parts, points)
    lines = text.splitlines()
    assert "regression on blackhole" in lines[0]
    assert "same point on wormhole: +7.59% (+4,153 cy), under its 13% threshold" in text
    assert "+99" not in text
    assert text.index("*Blackhole*") < text.index("*Wormhole*")
    assert "*Wormhole* · L1_TO_L1: :white_check_mark:" in text


def test_both_arches_regressed_names_both(tmp_path):
    parts = [
        _part("blackhole", "regressed", [_ROW]),
        _part("wormhole", "regressed", [dict(_ROW, delta_pct="14.0")]),
    ]
    text = build_text(parts, {})
    assert "regression on blackhole, wormhole" in text.splitlines()[0]
    assert text.count("Reproduce the worst one locally") == 2


def test_a_point_absent_on_the_other_arch_adds_no_line(tmp_path):
    points = {"wormhole": _points(tmp_path, "wormhole", [dict(_ROW, mathop="X")])}
    parts = [_part("blackhole", "regressed", [_ROW]), _part("wormhole", "clean")]
    assert "same point on" not in build_text(parts, points)


def test_a_module_threshold_overrides_the_arch_threshold():
    ctx = _ctx("wormhole")
    assert threshold_for(ctx, "perf_matmul") == 0.25
    assert threshold_for(ctx, "perf_eltwise_unary_sfpu") == 0.13


def _run(tmp_path, monkeypatch, parts, *extra):
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    args = []
    for i, part in enumerate(parts):
        path = tmp_path / f"part{i}.json"
        path.write_text(json.dumps(part))
        args += ["--part", str(path)]
    out = tmp_path / "payload.json"
    main([*args, *extra, "--channel", "C0TEST", "--out", str(out)])
    return json.loads(out.read_text())["text"], output.read_text()


def test_two_passes_do_not_post(tmp_path, monkeypatch):
    parts = [_part("blackhole", "clean"), _part("wormhole", "clean")]
    _, output = _run(tmp_path, monkeypatch, parts, "--expect", "blackhole,wormhole")
    assert "should_post=false" in output


def test_an_arch_with_no_part_is_a_skip(tmp_path, monkeypatch):
    parts = [_part("blackhole", "clean")]
    text, output = _run(tmp_path, monkeypatch, parts, "--expect", "blackhole,wormhole")
    assert "should_post=true" in output
    assert "SKIPPED" in text.splitlines()[0]
    assert "*Wormhole* · L1_TO_L1: :warning: *SKIPPED.*" in text


def test_no_part_at_all_still_posts_a_skip(tmp_path, monkeypatch):
    out = tmp_path / "payload.json"
    output = tmp_path / "out.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    main(["--expect", "blackhole", "--channel", "C", "--out", str(out)])
    assert "should_post=true" in output.read_text()
    assert "*Blackhole*" in json.loads(out.read_text())["text"]
