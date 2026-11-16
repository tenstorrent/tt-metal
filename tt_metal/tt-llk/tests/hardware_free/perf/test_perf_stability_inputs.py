# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "perf"))
import stability_inputs
from stability_inputs import PATHS, main, report

_REPO = pathlib.Path(__file__).parents[5]
_TESTS = pathlib.Path(__file__).parents[2]

COMMITS = [
    {
        "sha": "a" * 40,
        "date": "2026-10-23T10:00:00Z",
        "title": "SFPI 7.90.0 1068 (#62327)",
    },
    {"sha": "b" * 40, "date": "2026-10-23T11:00:00Z", "title": "Docs only (#1)"},
    {"sha": "c" * 40, "date": "2026-10-23T12:00:00Z", "title": "Direct push"},
]


def test_paths_exist_and_cover_every_runner_and_cost_table():
    assert [p for p in PATHS if not (_REPO / p).exists()] == []
    listed = {pathlib.Path(p).name for p in PATHS}
    for pattern in ("run_llk_perf_*.sh", "perf_split_costs*.json"):
        assert {p.name for p in _TESTS.glob(pattern)} <= listed


def test_report_lists_each_pr_and_flags_only_a_bypass():
    hits = {"a" * 40: ["tt_metal/sfpi-version"], "c" * 40: [f"{PATHS[2]}"]}
    out = report(COMMITS, hits, {"a" * 40: True, "c" * 40: False}, "1" * 40, "2" * 40)
    assert out["lines"] == [
        "2 merge(s) in `11111111111..22222222222` can change stability; check S1/S2 today:",
        "• #62327 SFPI 7.90.0 1068 (sfpi-version)",
        "• ccccccccccc Direct push (pytest.ini)",
    ]
    assert out["problems"] == [
        "ccccccccccc changed pytest.ini and merged without the LLK perf merge gate"
    ]


def test_report_unknown_gate_is_a_line_not_a_problem():
    out = report(
        COMMITS, {"a" * 40: ["tt_metal/sfpi-version"]}, {"a" * 40: None}, "1", "2"
    )
    assert out["problems"] == [] and out["lines"][1].endswith("(gate not checked)")
    assert report(COMMITS, {}, {}, "1", "2")["lines"] == [
        "no stability-sensitive merges in `1..2`"
    ]


def test_main_writes_the_report(tmp_path, monkeypatch):
    monkeypatch.setattr(stability_inputs, "nightly_range", lambda repo, day: ("1", "2"))
    monkeypatch.setattr(stability_inputs, "range_commits", lambda repo, b, h: COMMITS)
    monkeypatch.setattr(
        stability_inputs,
        "touching",
        lambda repo, path, head, since: (
            {"a" * 40, "f" * 40} if path.endswith("sfpi-version") else set()
        ),
    )
    monkeypatch.setattr(stability_inputs, "gate_ran", lambda repo, sha: True)
    out = tmp_path / "checks" / "stability_inputs.json"
    assert main(["--repo", "r", "--date", "2026-10-24", "--out", str(out)]) == 0
    data = json.loads(out.read_text())
    assert (
        data["problems"] == []
        and data["lines"][1] == "• #62327 SFPI 7.90.0 1068 (sfpi-version)"
    )


def test_main_without_two_nightlies(tmp_path, monkeypatch):
    monkeypatch.setattr(stability_inputs, "nightly_range", lambda repo, day: None)
    out = tmp_path / "s.json"
    main(["--repo", "r", "--date", "2026-11-16", "--out", str(out)])
    assert json.loads(out.read_text())["lines"] == [
        "no two non-SoL nightlies up to 2026-11-16; nothing to check"
    ]
