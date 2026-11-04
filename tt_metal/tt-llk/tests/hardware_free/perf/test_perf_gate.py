# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for perf/gate.py: the policy, the compare and the verdict. No hardware, no warehouse."""

import json
import pathlib
import sys

import pytest

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import gate  # noqa: E402

ARCHES = ("blackhole", "wormhole")


def _plan(directory, arch="blackhole", group=1, groups=1, ran=None):
    """The plan.json and ran.txt a measuring leg leaves."""
    directory.mkdir(parents=True, exist_ok=True)
    nodes = ["perf_x.py::test_x[a]", "perf_x.py::test_x[b c]"]
    plan = {"arch": arch, "group": group, "groups": groups, "nodes": nodes}
    (directory / "plan.json").write_text(json.dumps(plan))
    (directory / "ran.txt").write_text(
        "".join(f"{n}\n" for n in (nodes if ran is None else ran))
    )


def _csv(path, cycles, sol="False", run_type="L1_TO_L1"):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["marker,math_fidelity,speed_of_light,mean(%s)" % run_type]
    rows += [f"TILE_LOOP,HiFi{i},{sol},{c}" for i, c in enumerate(cycles)]
    path.write_text("\n".join(rows) + "\n")


def _run(
    tmp_path,
    gate_name="merge",
    arch="blackhole",
    current=(1000, 2000),
    baseline=(1000, 2000),
    **kw,
):
    """One compare in tmp_path; returns the verdict it wrote."""
    _plan(tmp_path / "cur", arch=arch)
    if current is not None:
        _csv(tmp_path / "cur" / "perf_x.csv", current, sol=kw.pop("sol", "False"))
    if baseline is not None:
        _csv(tmp_path / "base" / "perf_x.csv", baseline)
    meta = tmp_path / "baseline_metadata.json"
    if kw.pop("meta", True):
        meta.write_text(json.dumps({"commit_sha": "abc", **kw.pop("base", {})}))
    argv = [
        "compare",
        "--gate",
        gate_name,
        "--arch",
        arch,
        "--current",
        str(tmp_path / "cur"),
        "--baseline",
        str(tmp_path / "base"),
        "--baseline-meta",
        str(meta),
        "--out-dir",
        str(tmp_path),
    ]
    if kw.pop("incomplete", False):
        (tmp_path / "cur" / "ran.txt").write_text("perf_x.py::test_x[a]\n")
    assert gate.main(argv) == 0
    return json.loads((tmp_path / f"verdict-{arch}.json").read_text())


@pytest.mark.parametrize("name", ["pr", "merge", "nightly", "nightly_sol"])
@pytest.mark.parametrize("arch", ARCHES)
def test_every_gate_names_its_rules_for_both_arches(name, arch):
    p = gate.policy(name, arch)
    assert p["title"] and p["run_types"] and 0 < p["threshold"] < 1
    assert p["min_cycles"] == 30 and p["block"] in (True, False)


def test_the_nightlies_name_a_pinned_run():
    for name in ("nightly", "nightly_sol"):
        assert gate.policy(name, "wormhole")["baseline"]["run"]


def test_a_clean_compare_passes(tmp_path):
    v = _run(tmp_path)
    assert v["status"] == "pass" and not v["fails"]
    assert v["counts"]["points"] == 2 and (tmp_path / "perf_gate_report.md").exists()


def test_a_regression_is_reported_and_fails_only_when_it_blocks(tmp_path):
    v = _run(tmp_path, current=(1100, 2000))
    assert v["status"] == "regression" and v["counts"]["regressions"] == 1
    assert v["fails"] == v["policy"]["block"]
    assert (tmp_path / "perf_gate_report.regressions.csv").exists()


def test_the_module_threshold_of_the_policy_applies(tmp_path):
    """Wormhole matmul is allowed 25 % in the merge gate; other modules 13 %."""
    assert (
        _run(tmp_path / "x", arch="wormhole", current=(1200, 2000))["status"]
        == "regression"
    )
    _csv(tmp_path / "cur" / "perf_matmul.csv", (1200, 2000))
    _csv(tmp_path / "base" / "perf_matmul.csv", (1000, 2000))
    assert (
        _run(tmp_path, arch="wormhole", current=None, baseline=None)["status"] == "pass"
    )


@pytest.mark.parametrize(
    "kw, status",
    [
        ({"incomplete": True}, "incomplete"),
        ({"baseline": None}, "no-baseline"),
        ({"meta": False}, "no-compare"),
        ({"current": None}, "no-compare"),
        ({"sol": "True"}, "no-compare"),
    ],
)
def test_a_gate_that_compared_nothing_fails(tmp_path, kw, status):
    v = _run(tmp_path, **kw)
    assert v["status"] == status and v["fails"] and v["reason"]
    assert v["counts"]["points"] == 0


def test_a_stale_baseline_is_labelled(tmp_path):
    _run(tmp_path, base={"stale": True, "expected_commit": "def"})
    assert "merge baseline, stale" in (tmp_path / "perf_gate_report.md").read_text()


def test_verdict_exits_on_what_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path = tmp_path / "verdict-blackhole.json"
    base = {
        "arch": "blackhole",
        "reason": "",
        "counts": {"points": 1, "regressions": 1},
    }
    for status, fails, code in (
        ("pass", False, 0),
        ("regression", False, 0),
        ("regression", True, 1),
        ("no-baseline", True, 1),
    ):
        path.write_text(json.dumps({**base, "status": status, "fails": fails}))
        assert gate.main(["verdict", str(path)]) == code
    assert gate.main(["verdict", str(tmp_path / "missing.json")]) == 1


def test_policy_prints_the_compare_arguments(capsys):
    gate.main(["policy", "--gate", "merge", "--arch", "wormhole"])
    assert capsys.readouterr().out.split() == [
        "--threshold",
        "0.13",
        "--min-cycles",
        "30",
        "--module-threshold",
        "perf_matmul=0.25,perf_math_matmul=0.25",
    ]


def test_arches_lists_the_stage2_arches(capsys):
    gate.main(["arches", "--gate", "merge", "--stage2"])
    assert capsys.readouterr().out.split() == ["blackhole"]


REPLAY = pathlib.Path(__file__).parent / "replay_merge_blackhole"


def _replay(tmp_path, *extra, current=None):
    """Merge gate run 41816768387, Blackhole, cut to six modules over its four legs."""
    argv = ["compare", "--gate", "merge", "--arch", "blackhole"]
    argv += ["--current", str(current or REPLAY / "current")]
    argv += ["--baseline", str(REPLAY / "baseline")]
    argv += ["--baseline-meta", str(REPLAY / "baseline_metadata.json")]
    argv += ["--stage2", f"{REPLAY}/stage2/**/stage2.json", "--out-dir", str(tmp_path)]
    assert gate.main(argv + list(extra)) == 0
    return json.loads((tmp_path / "verdict-blackhole.json").read_text())


def test_replay_of_a_real_merge_gate_run_passes(tmp_path):
    """The gate of that run passed; the old regression_compare.py gives the same 348 points."""
    v = _replay(tmp_path)
    assert v["status"] == "pass" and not v["fails"]
    assert v["counts"] == {
        "points": 348,
        "regressions": 0,
        "improvements": 0,
        "new": 0,
        "missing": 0,
    }
    assert v["baseline"]["commit_sha"].startswith("9406a58a8a9")
    assert v["stage2"] == {"legs": 4, "notes": [], "outcomes": {}}


def test_replay_with_a_slower_leg_is_a_regression_stage2_has_not_judged(tmp_path):
    import shutil

    current = tmp_path / "current"
    shutil.copytree(REPLAY / "current", current)
    (csv,) = current.glob("*-blackhole-4/perf_eltwise_unary_typecast/*.csv")
    lines = csv.read_text().splitlines()
    col = lines[0].split(",").index("mean(L1_TO_L1)")
    slow = []
    for line in lines[1:]:
        cells = line.split(",")
        if cells[col]:
            cells[col] = str(float(cells[col]) * 1.5)
        slow.append(",".join(cells))
    csv.write_text("\n".join([lines[0], *slow]) + "\n")

    v = _replay(tmp_path, current=current)
    assert v["status"] == "regression" and v["counts"]["regressions"] > 0
    assert v["stage2"]["outcomes"] == {"unmeasured": v["counts"]["regressions"]}


def test_replay_without_a_baseline_fails(tmp_path):
    argv = ["compare", "--gate", "merge", "--arch", "blackhole"]
    argv += [
        "--current",
        str(REPLAY / "current"),
        "--baseline",
        str(tmp_path / "none"),
    ]
    argv += [
        "--baseline-meta",
        str(REPLAY / "baseline_metadata.json"),
        "--out-dir",
        str(tmp_path),
    ]
    gate.main(argv)
    v = json.loads((tmp_path / "verdict-blackhole.json").read_text())
    assert v["status"] == "no-baseline" and v["fails"]


def test_replay_with_a_stage2_crash_still_decides(tmp_path):
    """Stage 2 is report-only: a corrupt stage2.json is a line in the report, not a failed gate."""
    import shutil

    stage2 = tmp_path / "s2"
    shutil.copytree(REPLAY / "stage2", stage2)
    next(stage2.rglob("stage2.json")).write_text("{not json")
    v = _replay(tmp_path, "--stage2", f"{stage2}/**/stage2.json")
    assert v["status"] == "pass" and "error" in v["stage2"]
    assert (
        "Stage 2 could not build its report"
        in (tmp_path / "perf_gate_report.md").read_text()
    )


def test_replay_with_a_leg_that_left_no_stage2_says_so(tmp_path):
    import shutil

    stage2 = tmp_path / "s2"
    shutil.copytree(REPLAY / "stage2", stage2)
    shutil.rmtree(next(stage2.iterdir()))
    v = _replay(tmp_path, "--stage2", f"{stage2}/**/stage2.json")
    assert v["stage2"]["notes"] == ["3 of 4 leg(s) left a stage-2 result."]


def _replay_copy(tmp_path, part):
    import shutil

    copy = tmp_path / part
    shutil.copytree(REPLAY / part, copy)
    return copy


def test_replay_with_one_leg_missing_is_incomplete(tmp_path):
    import shutil

    current = _replay_copy(tmp_path, "current")
    shutil.rmtree(current / "41816768387-blackhole-5")
    v = _replay(tmp_path, current=current)
    assert v["status"] == "incomplete" and v["fails"]
    assert "leg 3/4 left no data" in v["reason"]


def test_replay_with_a_leg_cut_short_is_incomplete(tmp_path):
    current = _replay_copy(tmp_path, "current")
    ran = current / "41816768387-blackhole-6" / "ran.txt"
    ran.write_text(ran.read_text().splitlines()[0] + "\n")
    v = _replay(tmp_path, current=current)
    assert v["status"] == "incomplete"
    assert (
        "leg 4/4 ran 1 of 2 tests (first not run: perf_sfpu_ema.py::test_perf_sfpu_ema)"
        in v["reason"]
    )


def test_replay_with_a_short_baseline_fails(tmp_path):
    """2 Nov: a partial upload left a third of a run in the warehouse, and it read as a baseline."""
    import pandas as pd

    base = tmp_path / "baseline"
    base.mkdir()
    frame = pd.read_csv(REPLAY / "baseline" / "baseline.csv", low_memory=False)
    frame[frame.test_module.isin(["perf_bcast", "perf_fast_untilize"])].to_csv(
        base / "b.csv", index=False
    )
    argv = [
        "compare",
        "--gate",
        "merge",
        "--arch",
        "blackhole",
        "--current",
        str(REPLAY / "current"),
    ]
    argv += [
        "--baseline",
        str(base),
        "--baseline-meta",
        str(REPLAY / "baseline_metadata.json"),
    ]
    gate.main(argv + ["--out-dir", str(tmp_path)])
    v = json.loads((tmp_path / "verdict-blackhole.json").read_text())
    assert v["status"] == "no-compare" and v["fails"]
    assert v["reason"].startswith("The baseline holds 2 of the 6 test modules")


def test_replay_names_the_points_that_stopped_running(tmp_path):
    current = _replay_copy(tmp_path, "current")
    (
        current / "41816768387-blackhole-6" / "perf_sfpu_ema" / "perf_sfpu_ema.csv"
    ).unlink()
    v = _replay(tmp_path, current=current)
    assert v["counts"]["missing"] > 0
    assert (
        "were not measured now, in perf_sfpu_ema"
        in (tmp_path / "perf_gate_report.md").read_text()
    )


def _render(tmp_path, *extra):
    argv = ["render", "--gate", "merge", "--reports", str(tmp_path / "reports")]
    assert gate.main(argv + ["--out-dir", str(tmp_path), *extra]) == 0
    return (tmp_path / "comment.md").read_text(), (tmp_path / "slack.txt").read_text()


def _report(tmp_path, arch, **kw):
    """One arch's compare, in the directory its artifact downloads to."""
    run = tmp_path / arch
    _run(run, arch=arch, **kw)
    reports = tmp_path / "reports" / f"llk-perf-gate-report-{arch}"
    reports.mkdir(parents=True)
    for f in run.glob("perf_gate_report*"):
        f.rename(reports / f.name)
    (run / f"verdict-{arch}.json").rename(reports / f"verdict-{arch}.json")


def test_render_one_comment_for_both_arches(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _report(tmp_path, "blackhole")
    _report(tmp_path, "wormhole", current=(1300, 2000))
    comment, slack = _render(tmp_path)
    assert comment.startswith(gate.COMMENT_MARKER)
    assert (
        "Blackhole: ✅ no regressions" in comment
        and "Wormhole: ❌ REGRESSIONS FOUND" in comment
    )
    assert "regression on wormhole" in slack.splitlines()[0]
    assert "_Threshold: a point regresses when it is slower by more than 13%" in slack


def test_render_an_arch_without_a_verdict_is_a_skip(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _report(tmp_path, "blackhole")
    comment, slack = _render(tmp_path)
    assert "Wormhole\n\n**no verdict**" in comment
    assert "SKIPPED" in slack.splitlines()[0] and "wrote no verdict" in slack


def test_render_adds_the_pr_gate_stage2_section(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _report(tmp_path, "blackhole")
    stage2 = tmp_path / "reports" / "llk-perf-gate-stage2-blackhole"
    stage2.mkdir()
    (stage2 / "stage2-blackhole.md").write_text("## Stage 2 — Blackhole\n")
    comment, _ = _render(tmp_path)
    assert comment.index("Blackhole: ✅") < comment.index("## Stage 2 — Blackhole")


def test_render_posts_only_what_needs_action(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    posted = []
    monkeypatch.setattr(gate, "_post_slack", lambda channel, text: posted.append(text))
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C0TEST")
    _report(tmp_path, "blackhole")
    _report(tmp_path, "wormhole")
    _render(tmp_path, "--slack")
    assert posted == []
    _render(tmp_path, "--slack", "--post-pass")
    assert len(posted) == 1 and "passed" in posted[0]


def test_render_escapes_a_broadcast_in_a_config_value(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    _report(tmp_path, "blackhole", current=(1300, 2000))
    regressions = next((tmp_path / "reports").rglob("perf_gate_report.regressions.csv"))
    regressions.write_text(regressions.read_text().replace("HiFi0", "<!channel> HiFi0"))
    _, slack = _render(tmp_path)
    assert "<!channel>" not in slack and "@channel" in slack


def test_verdict_prints_the_line_the_health_check_reads(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    path = tmp_path / "verdict-blackhole.json"
    v = {"arch": "blackhole", "status": "pass", "reason": "", "fails": False}
    path.write_text(json.dumps({**v, "counts": {"points": 3, "regressions": 0}}))
    gate.main(["verdict", str(path)])
    assert "verdict=clean regressions=0" in capsys.readouterr().out
