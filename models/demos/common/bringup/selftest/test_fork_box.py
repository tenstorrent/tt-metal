# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F256: fork tests for another box are skipped with the box as the reason, never run.

On a Galaxy 8x4, 155 fork-suite failures were 2x2 / 1x4 / 4x1 / 2x4 / 4x2 tests opened with all 32 chips visible
(Fabric Router Sync: Timeout) and 3 were 1x4 cases on four cards (requested_size <= system_size). Owner (2026-11-06):
"the box cases should just check and run on a correct box"."""

import json

from models.demos.common.bringup.selftest.test_fork_cases import _fake_runner
from models.demos.common.bringup.testing import fork_cases
from models.demos.common.bringup.testing import fork_suite as FS
from models.demos.common.bringup.testing import fork_tests as FT


def test_a_test_runs_only_on_its_own_box():
    box = (8, 4)
    assert FT.box_reason((2, 2), box) == "box: needs mesh 2x2 (this box 8x4)"
    assert FT.box_reason((4, 2), box) == "box: needs mesh 4x2 (this box 8x4)"
    assert FT.box_reason((8, 8), box) == "box: needs mesh 8x8 (this box 8x4)"  # more devices than the box
    assert FT.box_reason((8, 4), box) is None and FT.box_reason((4, 8), box) is None  # (the transpose: rotated)
    assert FT.box_reason((2, 2, FT.SUBMESH), box) is None  # declared: runs on a submesh of a bigger box
    assert FT.box_reason((8, 8, FT.SUBMESH), box) == "box: needs mesh 8x8 (this box 8x4)"  # (a submesh must fit)
    assert FT.box_reason((1, 32, FT.COUNT), box) is None and FT.box_reason((1, 4, FT.COUNT), box)
    assert FT.box_reason((2, 2), None) is None and FT.box_reason(None, box) is None  # no box / no mesh: as before


def test_a_2x2_case_on_an_8x4_spec_is_a_box_skip_and_an_8x4_case_runs(fx, tmp_path, monkeypatch):
    box = FT.box_of(fx(box={"mesh": [8, 4]}))
    assert box == (8, 4)
    nodes = {(2, 2): ["p/test_a.py::test_x[m-2x2]"], (8, 4): ["p/test_a.py::test_x[m-8x4]"]}
    ran = []
    fake = _fake_runner({})

    def runner(cmd, env):
        ran.extend(c for c in cmd if "::" in c)
        return fake(cmd, env)

    (tmp_path / "forks" / "a" / "tests").mkdir(parents=True)
    (tmp_path / "forks" / "a" / "tests" / "test_a.py").write_text("")
    monkeypatch.setattr(fork_cases, "FORKS", tmp_path / "forks")
    monkeypatch.setattr("models.demos.common.bringup.testing.cards.visible_devices", lambda mesh: None)
    t = fork_cases.run_tests(["a"], runner=runner, grouper=lambda f: (nodes, []), box=box)
    assert ran == ["p/test_a.py::test_x[m-8x4]"]  # the 2x2 case never runs
    assert (t["a"]["passed"], t["a"]["skipped"], t["a"]["skipped_box"]) == (1, 1, 1)
    assert "skipped: p/test_a.py::test_x[m-2x2]: box: needs mesh 2x2 (this box 8x4)" in t["a"]["problems"]
    st = fork_cases.tally_stats(t)
    assert st["fork_tests_skipped_box"] == 1 and st["fork_tests_failed"] == 0

    # a case that declares submesh_ok runs, on the cards that form its mesh (as before F256)
    ran.clear()
    seen = []
    monkeypatch.setattr(
        "models.demos.common.bringup.testing.cards.visible_devices", lambda mesh: seen.append(mesh) or "0,1,2,3"
    )
    sub = {(2, 2, FT.SUBMESH): ["p/test_a.py::test_x[m-2x2]"]}
    t = fork_cases.run_tests(["a"], runner=runner, grouper=lambda f: (sub, []), box=box)
    assert ran == ["p/test_a.py::test_x[m-2x2]"] and seen == [(2, 2)] and t["a"]["skipped_box"] == 0


def test_the_suite_summary_tallies_the_box_skips(monkeypatch):
    nodes = {(2, 2): ["p/test_a.py::test_x[c1]", "p/test_a.py::test_x[c2]"], (8, 4): ["p/test_a.py::test_x[c3]"]}
    monkeypatch.setattr("models.demos.common.bringup.testing.cards.visible_devices", lambda mesh: None)
    parts = {"cases": None}
    res = FS.run_suite(["a"], {}, runner=_fake_runner({}), parts=parts, box=(8, 4), grouper=lambda f: (nodes, []))
    tot = FS.totals(res)
    assert tot["fork_suite_skipped_box"] == 2 and tot["fork_suite_skipped"] == 2 and tot["fork_suite_failed"] == 0
    summary = json.loads(json.dumps(res))  # (what metrics.record_json("fork_suite", res) writes)
    assert summary["a"]["cases"]["skipped_box"] == 2 and summary["a"]["cases"]["passed"] == 1
    # without a box nothing changes: every group runs
    res = FS.run_suite(["a"], {}, runner=_fake_runner({}), parts=parts, grouper=lambda f: (nodes, []))
    assert FS.totals(res)["fork_suite_passed"] == 3 and FS.totals(res)["fork_suite_skipped_box"] == 0


def test_a_unit_tests_mesh_is_read_from_its_mesh_device_param(tmp_path, monkeypatch):
    """Unit files run as one group; their mesh comes from the collected ``mesh_device`` param. A param that is not a
    tuple or an int is not guessed: the test runs as before and is listed."""
    (tmp_path / "conftest.py").write_text(
        "import pytest\n\n@pytest.fixture\ndef mesh_device(request):\n    return getattr(request, 'param', None)\n"
    )
    (tmp_path / "test_u.py").write_text(
        "import pytest\n\n"
        "@pytest.mark.parametrize('mesh_device', [(2, 2), (8, 4), 32, 4, 'odd'], indirect=True)\n"
        "def test_m(mesh_device):\n    pass\n\n"
        "def test_system(mesh_device):\n    pass\n\n"
        "def test_cpu():\n    pass\n"
    )
    (tmp_path / "test_sub.py").write_text(
        "import pytest\n\nBRINGUP_SUBMESH_OK = True\n\n"
        "@pytest.mark.parametrize('mesh_device', [(2, 2), (16, 4)], indirect=True)\n"
        "def test_s(mesh_device):\n    pass\n"
    )
    monkeypatch.setattr(FT, "REPO", tmp_path)
    nodes, errors, needs = FT.collect_needs(["test_u.py", "test_sub.py"])
    assert not errors and len(nodes) == 9
    assert needs["test_u.py::test_m[mesh_device0]"] == [2, 2]
    assert needs["test_u.py::test_m[32]"] == [1, 32, FT.COUNT]
    assert needs["test_u.py::test_m[odd]"].startswith("?")
    assert needs["test_u.py::test_system"] is None and needs["test_u.py::test_cpu"] is None
    assert needs["test_sub.py::test_s[mesh_device0]"] == [2, 2, FT.SUBMESH]
    t = FT.empty_tally()
    run = FT.box_filter(t, "f", nodes, needs, (8, 4), "unit")
    assert sorted(run) == sorted(
        [
            "test_u.py::test_m[mesh_device1]",
            "test_u.py::test_m[32]",
            "test_u.py::test_m[odd]",
            "test_u.py::test_system",
            "test_u.py::test_cpu",
            "test_sub.py::test_s[mesh_device0]",
        ]
    )
    assert t["skipped"] == t["skipped_box"] == 3
    assert "skipped: test_u.py::test_m[4]: box: needs mesh 1x4 (this box 8x4)" in t["problems"]
    assert "skipped: test_sub.py::test_s[mesh_device1]: box: needs mesh 16x4 (this box 8x4)" in t["problems"]
    assert any(p.startswith("box unknown: test_u.py::test_m[odd]") for p in t["problems"])
    t = FT.empty_tally()
    assert FT.box_filter(t, "f", nodes, needs, None, "unit") == nodes and t["problems"] == []  # no box: as before
