# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import collections
import datetime as dt
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

_PERF = pathlib.Path(__file__).parents[3] / "perf"
sys.path.insert(0, str(_PERF))
import stability_check as sc  # noqa: E402

_TESTS = pathlib.Path(__file__).parents[2] / "python_tests"
_SENTINELS = json.loads(sc.SENTINELS.read_text())
_SUNDAY, _MONDAY = dt.date(2026, 11, 15), dt.date(2026, 11, 16)
_LEGS = {"blackhole": 5, "wormhole": 5}
_HEAD = "formats.input_A,math_fidelity,marker,mean(L1_TO_L1),TEXT_SIZE(L1_TO_L1),mean(PACK_ISOLATE),L1_TO_L1_INSTRUCTIONS\n"


def _run(root, module, rows):
    d = root / module
    d.mkdir(parents=True)
    (d / f"{module}.csv").write_text(_HEAD + "".join(rows))
    (d / f"{module}.post.csv").write_text(_HEAD + "Bfp8_b,LoFi,KERNEL,1,1,1,1\n")
    return root


def _compare(tmp_path, a_rows, b_rows, module="perf_math_matmul"):
    a = _run(tmp_path / "a", module, a_rows)
    b = _run(tmp_path / "b", module, b_rows)
    return sc.compare(a, b)


def _result(moves=(), points=100, **kw):
    return {
        "points": points,
        "only_a": 0,
        "only_b": 0,
        "text_differs": 0,
        "not_bit_exact": len(moves),
        "moves": [list(m) for m in moves],
        **kw,
    }


def _leg(sentinel=True, control=None, ran=None):
    files = {}
    if sentinel:
        files = {
            "nodes.txt": "x\ny\n",
            "repeat.json": _result(),
            "reorder.json": _result(),
        }
        files.update({f"junit-{p}.xml": 2 if ran is None else ran for p in "abc"})
    if control is not None:
        files["control.json"] = control
    return files


def _quiet(control=False):
    return {
        a: {
            str(g): _leg(sentinel=g == 1, control=_result() if control else None)
            for g in range(1, 6)
        }
        for a in sc.RULES
    }


def test_the_sentinel_list_is_matmul_first_and_two_of_every_other_module():
    for arch, nodes in _SENTINELS.items():
        assert len(nodes) == len(set(nodes)), arch
        modules = collections.Counter(n.split(".py::")[0] for n in nodes)
        assert modules.pop("perf_math_matmul") == 150, arch
        assert set(modules.values()) == {2}, arch
        assert len(modules) >= 20, arch
        assert all("LoFi" in n for n in nodes[:100]), arch


def test_every_sentinel_names_a_test_function_that_exists():
    for arch, nodes in _SENTINELS.items():
        for node in nodes:
            module, rest = node.split("::")
            func = rest.split("[")[0]
            source = _TESTS / module
            assert source.is_file(), f"{arch}: {node}"
            assert f"def {func}(" in source.read_text(), f"{arch}: {node}"


@pytest.mark.parametrize("arch", sorted(_SENTINELS))
def test_every_sentinel_is_still_collected(arch):
    # Node ids of perf_math_matmul hold a config index, so a sweep change renames them.
    pytest.importorskip("ttexalens")
    nodes = _SENTINELS[arch]
    with tempfile.TemporaryDirectory() as tmp:
        node_list = pathlib.Path(tmp) / "nodes.txt"
        node_list.write_text("\n".join(nodes))
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith(("PYTEST_", "LLK_HARDWARE_FREE_RUN_ID"))
        }
        env.update(CHIP_ARCH=arch, LLK_HARDWARE_FREE="1", RUNNER_TEMP=tmp, TMPDIR=tmp)
        files = sorted({n.split("::")[0] for n in nodes})
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "--collect-only",
                "-q",
                "--override-ini=addopts=",
                "-p",
                "no:cacheprovider",
                "-m",
                "perf and not accuracy",
                f"--node-list={node_list}",
                *files,
            ],
            cwd=_TESTS,
            env=env,
            capture_output=True,
            text=True,
            timeout=100,
        )
    collected = {line.strip() for line in out.stdout.splitlines() if "::" in line}
    missing = sorted(set(nodes) - collected)
    assert not missing, (
        f"{len(missing)} {arch} sentinel tests are not collected any more; rebuild "
        f"tests/perf_stability_sentinels.json. First: {missing[:3]}\n{out.stderr[-2000:]}"
    )


def test_nodes_writes_the_list_and_another_order(tmp_path):
    nodes = sc.write_nodes("blackhole", tmp_path)
    assert (tmp_path / "nodes.txt").read_text().splitlines() == nodes
    order = [
        t["test"] for t in json.loads((tmp_path / "order.json").read_text())["sentinel"]
    ]
    assert sorted(order) == sorted(nodes)
    assert order != nodes
    sc.write_nodes("blackhole", tmp_path / "again")
    again = json.loads((tmp_path / "again" / "order.json").read_text())["sentinel"]
    assert [t["test"] for t in again] == order


def test_compare_counts_points_that_are_not_bit_exact(tmp_path):
    r = _compare(
        tmp_path,
        [
            "Bfp8_b,LoFi,TILE_LOOP,1000.0,100,500.0,7\n",
            "Bfp8_b,LoFi,KERNEL,2000.0,100,,7\n",
            "Float32,HiFi4,TILE_LOOP,300.0,100,300.0,7\n",
        ],
        [
            "Bfp8_b,LoFi,TILE_LOOP,1300.0,100,500.0,9\n",
            "Bfp8_b,LoFi,KERNEL,2000.0,100,,9\n",
            "Float32,HiFi4,TILE_LOOP,300.0,104,301.0,9\n",
        ],
    )
    # Counter columns (L1_TO_L1_INSTRUCTIONS) are values, not test parameters.
    assert r["points"] == 5
    assert r["not_bit_exact"] == 2
    assert r["text_differs"] == 1
    assert r["only_a"] == r["only_b"] == 0
    largest = r["moves"][0]
    assert largest[:3] == ["perf_math_matmul", "L1_TO_L1", "TILE_LOOP"]
    assert largest[4:] == [1000.0, 1300.0]


def test_compare_counts_points_measured_once(tmp_path):
    r = _compare(
        tmp_path,
        ["Bfp8_b,LoFi,TILE_LOOP,1000.0,100,,7\n", "Float32,LoFi,TILE_LOOP,5,1,,1\n"],
        ["Bfp8_b,LoFi,TILE_LOOP,1000.0,100,,7\n"],
    )
    assert (r["points"], r["only_a"], r["only_b"], r["not_bit_exact"]) == (1, 1, 0, 0)


def test_compare_lists_moves_over_the_cycle_floor_with_parameters_for_the_largest(
    tmp_path,
):
    n = sc.MAX_PARAMS + 5
    r = _compare(
        tmp_path,
        [f"F{i},LoFi,TILE_LOOP,1000.0,1,,1\n" for i in range(n)]
        + ["Small,LoFi,TILE_LOOP,1000.0,1,,1\n"],
        [f"F{i},LoFi,TILE_LOOP,{1031.0 + i},1,,1\n" for i in range(n)]
        + ["Small,LoFi,TILE_LOOP,1030.0,1,,1\n"],
    )
    assert r["not_bit_exact"] == n + 1
    assert len(r["moves"]) == n
    assert all(m[3] for m in r["moves"][: sc.MAX_PARAMS])
    assert not any(m[3] for m in r["moves"][sc.MAX_PARAMS :])


def test_the_gate_rule_is_per_arch_and_per_module():
    def move(module, a, b):
        return [module, "L1_TO_L1", "TILE_LOOP", "", a, b]

    moves = [
        move("perf_pack", 1000, 1040),  # 4 %, 40 cycles
        move("perf_pack", 100, 120),  # 20 %, 20 cycles
        move("perf_math_matmul", 1000, 1200),  # 20 %
        move("perf_math_matmul", 1000, 1300),  # 30 %
    ]
    assert len(sc.over_rule("blackhole", moves)) == 3
    assert sc.over_rule("wormhole", moves) == [moves[3]]


def test_a_quiet_night_has_no_problem():
    r = sc.assess(_quiet(), _MONDAY, _LEGS)
    assert r["problems"] == []
    assert len(r["lines"]) == 4
    assert r["lines"][0].startswith(
        "BH sentinel measured twice: 0 of 100 points not bit-exact, 0 over the gate rule"
    )


def test_one_blackhole_move_over_the_rule_is_a_problem():
    found = _quiet()
    found["blackhole"]["1"]["repeat.json"] = _result(
        [["perf_math_matmul", "L1_TO_L1", "TILE_LOOP", "p", 1000.0, 1100.0]]
    )
    found["wormhole"]["1"]["repeat.json"] = _result(
        [["perf_math_matmul", "L1_TO_L1", "TILE_LOOP", "p", 1000.0, 1100.0]]
    )
    r = sc.assess(found, _MONDAY, _LEGS)
    assert r["problems"] == [
        "stability BH: sentinel measured twice, 1 points over the gate rule"
    ]
    assert "largest 10.0 %, perf_math_matmul L1_TO_L1 TILE_LOOP" in r["lines"][0]


def test_wormhole_fires_over_its_floor(monkeypatch):
    found = _quiet()
    found["wormhole"]["1"]["reorder.json"] = _result(
        [["perf_pack", "L1_TO_L1", "TILE_LOOP", "", 1000.0, 1200.0]] * 3
    )
    assert len(sc.assess(found, _MONDAY, _LEGS)["problems"]) == 1
    monkeypatch.setitem(sc.FLOOR, "wormhole", 3)
    assert sc.assess(found, _MONDAY, _LEGS)["problems"] == []


def test_missing_or_partial_sentinel_results_are_problems():
    found = _quiet()
    del found["blackhole"]["1"]
    found["wormhole"]["1"] = _leg(ran=1)
    del found["wormhole"]["1"]["reorder.json"]
    assert sc.assess(found, _MONDAY, _LEGS)["problems"] == [
        "stability BH: no sentinel result",
        "stability WH: sentinel ran 1 of 2 listed tests",
        "stability WH: sentinel in another order, no result",
    ]


def test_sunday_expects_a_control_from_every_leg():
    found = _quiet(control=True)
    r = sc.assess(found, _SUNDAY, _LEGS)
    assert r["problems"] == []
    assert any(
        "full suite measured twice (5/5 legs): 0 of 500 points" in l for l in r["lines"]
    )
    del found["wormhole"]["3"]["control.json"]
    assert sc.assess(found, _SUNDAY, _LEGS)["problems"] == [
        "stability WH: weekly full-suite control from 4 of 5 legs"
    ]
    assert sc.assess(_quiet(), _SUNDAY, _LEGS)["problems"] == [
        "stability BH: weekly full-suite control from 0 of 5 legs",
        "stability WH: weekly full-suite control from 0 of 5 legs",
    ]


def test_control_moves_from_all_legs_add_up():
    found = _quiet(control=True)
    found["blackhole"]["2"]["control.json"] = _result(
        [["perf_pack", "PACK_ISOLATE", "TILE_LOOP", "", 1000.0, 1050.0]], only_a=3
    )
    r = sc.assess(found, _MONDAY, _LEGS)
    assert r["problems"] == [
        "stability BH: full suite measured twice (5/5 legs), 1 points over the gate rule",
        "stability BH: full suite measured twice (5/5 legs), 3 points measured in one pass only",
    ]


def test_collect_reads_the_artifacts(tmp_path):
    leg = tmp_path / "llk-perf-stability-wormhole-1"
    leg.mkdir()
    (leg / "nodes.txt").write_text("a\nb\n")
    (leg / "order.json").write_text('{"sentinel": []}')
    (leg / "repeat.json").write_text(json.dumps(_result()))
    (leg / "junit-a.xml").write_text(
        '<testsuites><testsuite><testcase name="a"/><testcase name="b"><skipped/></testcase>'
        '<testcase name="c"><failure/></testcase></testsuite></testsuites>'
    )
    (leg / "junit-b.xml").write_text("not xml")
    (tmp_path / "llk-perf-stability-blackhole-3").mkdir()
    (tmp_path / "llk-perf-stability-blackhole-3" / "control.json").write_text("{}")
    found = sc.collect(tmp_path)
    assert found["wormhole"]["1"]["junit-a.xml"] == 1
    assert found["wormhole"]["1"]["junit-b.xml"] == 0
    assert "order.json" not in found["wormhole"]["1"]
    assert found["blackhole"]["3"] == {"control.json": {}}


def test_report_with_no_finished_nightly_leaves_it_to_the_nightly_section(
    monkeypatch,
):
    health = sc._health()
    monkeypatch.setattr(health, "scheduled_run", lambda *a: None)
    r = sc.report("o/r", _MONDAY, "unused.yaml")
    assert r["problems"] == [] and r["lines"]


def test_report_downloads_the_run_artifacts(monkeypatch, tmp_path):
    health = sc._health()
    run = {"id": 7, "status": "completed"}
    monkeypatch.setattr(health, "scheduled_run", lambda repo, wf, day: run)
    tests_yaml = tmp_path / "t.yaml"
    tests_yaml.write_text(
        "".join(f"- name: llk_perf_{a} group {g}/2\n" for a in sc.RULES for g in (1, 2))
    )
    calls = []

    def fake_gh(*args):
        calls.append(args)
        dest = pathlib.Path(args[args.index("-D") + 1])
        for a in sc.RULES:
            leg = dest / f"llk-perf-stability-{a}-1"
            leg.mkdir(parents=True)
            (leg / "nodes.txt").write_text("x\n")
            for p in "abc":
                (leg / f"junit-{p}.xml").write_text(
                    '<testsuite><testcase name="x"/></testsuite>'
                )
            for f in ("repeat.json", "reorder.json"):
                (leg / f).write_text(json.dumps(_result()))
        return ""

    monkeypatch.setattr(health, "_gh", fake_gh)
    r = sc.report("o/r", _MONDAY, tests_yaml)
    assert calls[0][:4] == ("run", "download", "7", "-R")
    assert sc.ARTIFACTS in calls[0]
    assert r["problems"] == []
    assert len(r["lines"]) == 4


def test_main_report_writes_the_check_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sc, "report", lambda repo, day, y: {"name": "n", "problems": [], "lines": []}
    )
    out = tmp_path / "nightly_checks" / "stability.json"
    assert sc.main(["report", "--repo", "o/r", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["name"] == "n"
