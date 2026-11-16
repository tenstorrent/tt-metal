# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
import pathlib
import sys

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import nightly_stability as ns

_SCRIPT = pathlib.Path(__file__).parents[5] / ns.INPUTS_SCRIPT
DAY = dt.date(2026, 10, 19)
PREV = {"id": 1, "head_sha": "9f98dee9c5a0", "created_at": "2026-10-18T05:00:00Z"}
TODAY = {"id": 2, "head_sha": "01dfff6ac230", "created_at": "2026-10-19T05:00:00Z"}
REF = {"run_id": 3, "sha": "36dc937428d0", "date": "2026-10-08"}


def _run(root, arch, rows, module="perf_bcast"):
    path = root / f"perf-data-{arch}-1" / "leg" / module / f"{module}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["tile_cnt,marker,mean(L1_TO_L1),TEXT_SIZE(L1_TO_L1)"]
    lines += [f"{n},{m},{v},{t}" for n, m, v, t in rows]
    path.write_text("\n".join(lines) + "\n")
    path.with_name(f"{module}.post.csv").write_text("marker,mean(x)\nKERNEL,1\n")


ROWS = [(1, "KERNEL", 1000.0, 10), (2, "KERNEL", 2000.0, 10), (1, "INIT", 50.0, 10)]


def _stats(tmp_path, a_rows, b_rows, arch="blackhole"):
    _run(tmp_path / "a", arch, a_rows)
    _run(tmp_path / "b", arch, b_rows)
    return ns.same_code(ns.load(tmp_path / "a", arch), ns.load(tmp_path / "b", arch))


def test_load_reads_points_and_skips_post_csvs(tmp_path):
    _run(tmp_path, "wormhole", ROWS)
    points = ns.load(tmp_path, "wormhole")
    assert points[("perf_bcast", "KERNEL", "2")] == (2000.0, "10")
    assert len(points) == 3
    assert ns.load(tmp_path, "blackhole") == {}


def test_same_code_counts_moves_and_the_gate_rule(tmp_path):
    b = [(1, "KERNEL", 1000.0, 10), (2, "KERNEL", 2100.0, 10), (1, "INIT", 51.0, 10)]
    s = _stats(tmp_path, ROWS, b)
    assert (s["points"], s["off"], s["over"], s["text"]) == (3, 2, 1, 0)
    assert s["largest"][0] == 5.0
    assert s["modules"] == [("perf_bcast", 2)]


def test_a_quiet_day_with_bit_exact_data_has_no_problem(tmp_path):
    s = _stats(tmp_path, ROWS, ROWS)
    problems, lines = ns.quiet_day(
        "non-SoL", PREV, TODAY, (0, 3), {"blackhole": s, "wormhole": s}
    )
    assert problems == []
    assert "quiet day (3 commits" in lines[0]
    assert lines[1] == "BH: 0 of 3 points not bit-exact, 0 over the gate rule"


def test_bh_over_the_rule_and_wh_over_the_floor_are_problems(tmp_path):
    bh = _stats(tmp_path, ROWS, [(1, "KERNEL", 1100.0, 10), *ROWS[1:]])
    wh = dict(bh, off=ns.WH_FLOOR + 1, over=0)
    problems, _ = ns.quiet_day(
        "non-SoL", PREV, TODAY, (0, 3), {"blackhole": bh, "wormhole": wh}
    )
    assert problems == [
        "same code non-SoL BH: 1 points over the gate rule",
        f"same code non-SoL WH: {ns.WH_FLOOR + 1} points not bit-exact (floor {ns.WH_FLOOR})",
    ]
    wh["off"] = ns.WH_FLOOR
    problems, _ = ns.quiet_day(
        "non-SoL", PREV, TODAY, (0, 3), {"blackhole": wh, "wormhole": wh}
    )
    assert problems == []


def test_another_text_size_is_a_problem(tmp_path):
    s = _stats(tmp_path, ROWS, [(1, "KERNEL", 1000.0, 12), *ROWS[1:]])
    problems, _ = ns.quiet_day(
        "SoL", PREV, TODAY, (0, 3), {"blackhole": s, "wormhole": s}
    )
    assert any("another TEXT_SIZE" in p for p in problems)


def test_changed_inputs_are_not_a_quiet_day_and_no_problem():
    problems, lines = ns.quiet_day("SoL", PREV, TODAY, (2, 40), {})
    assert problems == []
    assert lines == [
        "SoL 2026-10-18 → 2026-10-19: not a quiet day, 2 of 40 commits change perf inputs"
    ]


def test_perf_input_commits_uses_the_gate_rules(monkeypatch):
    calls = []

    def fake_git(*args):
        calls.append(args)
        if args[0] == "log":
            return (
                "\0aaa\n\ntt_metal/tt-llk/tt_llk_blackhole/llk_lib/x.h\n"
                "\0bbb\n\ndocs/readme.md\nttnn/x.cpp\n"
                "\0ccc\n\ntt_metal/sfpi-version\n"
            )
        return ""

    monkeypatch.setattr(ns, "_git", fake_git)
    assert ns.perf_input_commits("base", "head", "2026-10-17", _SCRIPT) == (2, 3)
    assert "--shallow-since=2026-10-17" in calls[0]
    assert calls[1][-1] == "base..head"


def test_drift_takes_the_median_ratio_per_module(tmp_path):
    _run(tmp_path / "ref", "blackhole", ROWS)
    _run(tmp_path / "ref", "blackhole", ROWS, module="perf_matmul")
    _run(tmp_path / "cur", "blackhole", [(n, m, v * 1.05, t) for n, m, v, t in ROWS])
    _run(
        tmp_path / "cur",
        "blackhole",
        [(1, "KERNEL", 1010.0, 10), *ROWS[1:]],
        module="perf_matmul",
    )
    m = ns.drift(
        ns.load(tmp_path / "ref", "blackhole"), ns.load(tmp_path / "cur", "blackhole")
    )
    assert round(m["perf_bcast"][0], 6) == 1.05
    assert m["perf_matmul"] == (1.0, 3)
    problems, lines = ns.drift_report(
        "non-SoL", REF, DAY, {"blackhole": m, "wormhole": m}
    )
    assert problems == [
        "drift non-SoL BH: 1 module(s) slower than 3% by median: perf_bcast +5.0 %",
        "drift non-SoL WH: 1 module(s) slower than 3% by median: perf_bcast +5.0 %",
    ]
    assert (
        lines[0]
        == "non-SoL: today vs reference 2026-10-08 (`36dc937428d`, 11 days old)"
    )
    assert lines[1].startswith(
        "BH: 2 modules, 6 points matched; top movers perf_bcast +5.0 %"
    )


def test_an_old_reference_asks_for_a_re_pin():
    m = {"perf_bcast": (1.0, 3)}
    old = dict(REF, date=(DAY - dt.timedelta(days=ns.REPIN_DAYS)).isoformat())
    problems, _ = ns.drift_report("SoL", old, DAY, {"blackhole": m, "wormhole": m})
    assert len(problems) == 1 and "re-pin drift_reference.json" in problems[0]


def test_main_writes_both_sections(tmp_path, monkeypatch):
    for run in ("1", "2", "3"):
        for arch in ("blackhole", "wormhole"):
            _run(tmp_path / run, arch, ROWS)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({w: REF for _, w in ns.NIGHTLIES}))
    monkeypatch.setattr(ns, "good_runs", lambda repo, wf, day: [TODAY, PREV])
    monkeypatch.setattr(ns, "fetch", lambda repo, run_id, cache: tmp_path / str(run_id))
    monkeypatch.setattr(ns, "perf_input_commits", lambda base, head, since: (0, 3))
    out = tmp_path / "checks"
    ns.main(
        [
            "--repo",
            "x/y",
            "--date",
            "2026-10-19",
            "--out-dir",
            str(out),
            "--reference",
            str(ref),
        ]
    )
    quiet = json.loads((out / "quiet_day.json").read_text())
    drift = json.loads((out / "drift.json").read_text())
    assert quiet["problems"] == [] and drift["problems"] == []
    assert len(quiet["lines"]) == 6 and len(drift["lines"]) == 6


def test_no_run_today_is_reported_without_a_problem(tmp_path, monkeypatch):
    ref = tmp_path / "ref.json"
    ref.write_text("{}")
    monkeypatch.setattr(ns, "good_runs", lambda repo, wf, day: [PREV])
    out = tmp_path / "checks"
    ns.main(
        [
            "--repo",
            "x/y",
            "--date",
            "2026-10-19",
            "--out-dir",
            str(out),
            "--reference",
            str(ref),
        ]
    )
    quiet = json.loads((out / "quiet_day.json").read_text())
    assert quiet["problems"] == [] and quiet["lines"][0] == "SoL: no good nightly today"
