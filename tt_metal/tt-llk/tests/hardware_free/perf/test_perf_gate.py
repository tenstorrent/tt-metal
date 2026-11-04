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
        str(tmp_path / "cur/**/*.csv"),
        "--baseline",
        str(tmp_path / "base/**/*.csv"),
        "--baseline-meta",
        str(meta),
        "--out-dir",
        str(tmp_path),
        "--incomplete",
        kw.pop("incomplete", ""),
    ]
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
        ({"incomplete": "A blackhole leg was cancelled."}, "incomplete"),
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
    argv += ["--current", f"{current or REPLAY / 'current'}/**/*.csv"]
    argv += ["--baseline", f"{REPLAY}/baseline/*.csv"]
    argv += ["--baseline-meta", str(REPLAY / "baseline_metadata.json")]
    argv += ["--stage2", f"{REPLAY}/stage2/**/stage2.json", "--out-dir", str(tmp_path)]
    assert gate.main(argv + list(extra)) == 0
    return json.loads((tmp_path / "verdict-blackhole.json").read_text())


def test_replay_of_a_real_merge_gate_run_passes(tmp_path):
    """The gate of that run passed; the old regression_compare.py gives the same 348 points."""
    v = _replay(tmp_path)
    assert v["status"] == "pass" and not v["fails"]
    assert v["counts"] == {"points": 348, "regressions": 0, "improvements": 0, "new": 0}
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
        f"{REPLAY / 'current'}/**/*.csv",
        "--baseline",
        f"{tmp_path}/none/*.csv",
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
