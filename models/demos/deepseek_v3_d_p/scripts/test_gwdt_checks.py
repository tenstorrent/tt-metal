# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from pathlib import Path

import pytest
from gwdt_checks import (
    LogFollower,
    Result,
    Status,
    check_smi,
    classify_loop,
    classify_pytest,
    didt_bad_devices,
    fabric_results,
    log_root,
    run_name,
    stress_result,
)


def _chip(fw="19.16.0.0", tdp="0x4b", key="TDP_LIMIT_MAX"):
    return {"firmwares": {"fw_bundle_version": fw}, "smbus_telem": {key: tdp}}


def test_smi_uniform_low_tdp_warns():
    r = check_smi({"device_info": [_chip()] * 32})
    assert r.status == Status.WARN
    assert "fw_bundle_version 19.16.0.0 (32/32)" in r.details
    assert "TDP_LIMIT_MAX 75 W (32/32)" in r.details
    assert "TDP 75 W < 130 W -> DS_PERF_IGNORE_POWER=1" in r.reason


def test_smi_uniform_high_tdp_passes():
    assert check_smi({"device_info": [_chip(tdp="0x96")] * 32}).status == Status.PASS


def test_smi_wrong_chip_count_fails():
    r = check_smi({"device_info": [_chip(tdp="0x96")] * 28})
    assert r.status == Status.FAIL
    assert r.reason == "found 28 chips, need 32"


def test_smi_mixed_fw_warns_with_chip_ids():
    chips = [_chip(tdp="0x96")] * 32
    chips[16] = _chip(fw="19.15.0.0", tdp="0x96")
    r = check_smi({"device_info": chips})
    assert r.status == Status.WARN
    assert "fw_bundle_version 19.16.0.0 chips 0-15,17-31" in r.details
    assert "fw_bundle_version 19.15.0.0 chips 16" in r.details


def test_smi_tdp_fallback_key():
    r = check_smi({"device_info": [_chip(tdp="0x96", key="SMBUS_TX_TDP_LIMIT_MAX")] * 32})
    assert "TDP_LIMIT_MAX 150 W (32/32)" in r.details


def test_pytest_skip_is_fail():
    r = classify_pytest("SKIPPED [1] x.py:1: golden trace not found\n=== 1 skipped in 3s ===", 1)
    assert r.status == Status.FAIL
    assert "skipped" in r.reason


def test_pytest_expected_count():
    assert classify_pytest("=== 2 passed in 9s ===", 2).status == Status.PASS
    assert classify_pytest("=== 1 passed in 9s ===", 2).status == Status.FAIL


def test_loop_triage_is_hang(tmp_path):
    (tmp_path / "log_01").write_text("forward_layer_3_start\nTEST_DONE_EXIT=137\n")
    (tmp_path / "crash_triage_01.csv").write_text("dev,core\n")
    r = classify_loop(tmp_path / "log_01", tmp_path / "crash_triage_01.csv")
    assert r.status == Status.HANG
    assert str(tmp_path / "crash_triage_01.csv") in r.reason


def test_loop_pass(tmp_path):
    (tmp_path / "log_01").write_text("=== 1 passed in 9s ===\nTEST_DONE_EXIT=0\n")
    assert classify_loop(tmp_path / "log_01", tmp_path / "none.csv").status == Status.PASS


def test_loop_signal_is_fail(tmp_path):
    (tmp_path / "log_01").write_text("TEST_DONE_EXIT=135\n")
    r = classify_loop(tmp_path / "log_01", tmp_path / "none.csv")
    assert r.status == Status.FAIL
    assert r.reason.startswith("SIGBUS")


def test_loop_determinism_reason(tmp_path):
    (tmp_path / "log_01").write_text(
        "[determinism] iter 1 KV cache DIFFERS from iter 0: 112 element(s), max abs delta 3.1e-02\n"
        "=== 1 failed in 9s ===\nTEST_DONE_EXIT=1\n"
    )
    r = classify_loop(tmp_path / "log_01", tmp_path / "none.csv")
    assert r.status == Status.FAIL
    assert "iter 1 KV cache DIFFERS" in r.reason


def test_didt_bad_devices():
    log = (
        "Number of non-deterministic outputs on device 3 is 0\n"
        "Number of non-deterministic outputs on device 17 is 2\n"
    )
    assert didt_bad_devices(log) == {17: 2}


def test_fabric_results():
    log = "PASSED x.py::test_fabric_opens[FABRIC_2D]\n" "FAILED x.py::test_fabric_opens[FABRIC_2D_TORUS_XY] - E\n"
    assert fabric_results(log) == {"FABRIC_2D": True, "FABRIC_2D_TORUS_XY": False}


def test_log_root_falls_back_when_not_writable(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.is_dir", lambda self: True)
    monkeypatch.setattr(os, "access", lambda p, m: False)
    assert log_root("nobody", tmp_path) == tmp_path / "galaxy-workload-deployment-test"


def test_run_name():
    assert (
        run_name(datetime(2026, 10, 29, 3, 0), "bh-glx-110-c10u20", "70d7d6c")
        == "LOG_2026_10_29_03_00_bh-glx-110-c10u20_70d7d6c"
    )


def test_log_follower_loading(tmp_path):
    log = tmp_path / "log_01"
    log.write_text("2026-10-29 03:00:00.000 | INFO | Building TtPrefillTransformer\nBuilding layer 12/61\n")
    f = LogFollower(log)
    f.update()
    assert f.line(600, now=log.stat().st_mtime) == "iter 0/600  loading layer 12/61  idle 0s"


def test_screen_has_no_ansi_when_not_tty(tmp_path, monkeypatch, capsys):
    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    step = Step("didt", "didt", "gate", None, True, lambda ctx, state: Result(Status.PASS))
    step.state.result = Result(Status.FAIL, "device 17: 2 non-deterministic outputs")
    screen = Screen([step], ["hdr"], tmp_path / "status.json", {})
    screen.update(step)
    screen.done = True
    text = capsys.readouterr().out + "\n".join(screen.lines())
    assert "\x1b[" not in text
    assert "FAIL" in text


def test_loop_skip_reason(tmp_path):
    (tmp_path / "log_01").write_text(
        "SKIPPED [1] t.py:9: pretrained weights unavailable\n=== 1 skipped in 3s ===\nTEST_DONE_EXIT=0\n"
    )
    r = classify_loop(tmp_path / "log_01", tmp_path / "none.csv")
    assert r.status == Status.FAIL
    assert "pretrained weights unavailable" in r.reason


def test_loop_reads_only_tail(tmp_path):
    (tmp_path / "log_01").write_text("=== 1 passed in 9s ===\n" + "x" * (9 << 20) + "\nTEST_DONE_EXIT=1\n")
    assert classify_loop(tmp_path / "log_01", tmp_path / "none.csv").status == Status.FAIL


def test_stress_result_unstarted_loop_is_fail(tmp_path):
    r = stress_result([Result(Status.PASS), Result(Status.PENDING)], tmp_path / "stress.log")
    assert r.status == Status.FAIL
    assert str(tmp_path / "stress.log") in r.reason


def test_stress_result_all_pass(tmp_path):
    assert stress_result([Result(Status.PASS)] * 3, tmp_path / "stress.log").status == Status.PASS


def test_log_follower_counts_across_appends(tmp_path):
    log = tmp_path / "log_01"
    log.write_text("2026-10-29 03:00:00.000 | INFO | Building TtPrefillTransformer\niter 0 done (20 chunks) in 1 s\n")
    f = LogFollower(log)
    f.update()
    with open(log, "a") as fh:
        fh.write(
            "2026-10-29 03:48:10.000 | INFO | forward_layer_0_start\niter 1 done (20 chunks) in 1 s\nforward_layer_7_end\n"
        )
    f.update()
    assert f.line(600, now=log.stat().st_mtime) == "iter 2/600  forward_layer_7_end  load 48m10s  idle 0s"


def _steps(*specs):
    from gwdt_steps import Ctx, Step

    steps = [Step(k, k, g, m, True, lambda ctx, state, r=r: Result(r)) for k, g, m, r in specs]
    return steps, Ctx(Path("."), Path("."), False)


def test_run_all_gate_fail_blocks_everything():
    from gwdt_steps import run_all

    steps, ctx = _steps(
        ("a", "gate", None, Status.FAIL), ("b", "gate", None, Status.PASS), ("c", "smoke", "m", Status.PASS)
    )
    run_all(steps, ctx, set(), lambda s: None)
    assert [s.state.result.status for s in steps] == [Status.FAIL, Status.PENDING, Status.PENDING]


def test_run_all_model_fail_blocks_only_that_model():
    from gwdt_steps import run_all

    steps, ctx = _steps(
        ("b1", "block", "m1", Status.FAIL),
        ("b2", "block", "m2", Status.PASS),
        ("s1", "smoke", "m1", Status.PASS),
        ("s2", "smoke", "m2", Status.PASS),
    )
    run_all(steps, ctx, set(), lambda s: None)
    assert [s.state.result.status for s in steps] == [Status.FAIL, Status.PASS, Status.PENDING, Status.PASS]


def test_run_all_skip_by_user():
    from gwdt_steps import run_all

    steps, ctx = _steps(("a", "gate", None, Status.FAIL), ("b", "smoke", "m", Status.PASS))
    run_all(steps, ctx, {"a"}, lambda s: None)
    assert steps[0].state.result == Result(Status.SKIP, "by user")
    assert steps[1].state.result.status == Status.PASS


def _fake_sh(monkeypatch, rc, write=None):
    import gwdt_steps

    calls = []

    def sh(ctx, cmd, log, env=None, timeout=None):
        calls.append(cmd)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("")
        if write:
            write(cmd, log, env or {})
        return rc

    monkeypatch.setattr(gwdt_steps, "sh", sh)
    return calls


def test_fabric_hang_skips_config_sweep(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, fabric_step

    def write(cmd, log, env):
        cmd_env = env.get("TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE", "")
        if "--llm-output-path=" in cmd_env:
            Path(cmd_env.split("--llm-output-path=")[1].split(";")[0]).write_text("dev,core\n")

    calls = _fake_sh(monkeypatch, 137, write)
    r = fabric_step().run(Ctx(tmp_path, tmp_path, False), State())
    assert r.status == Status.HANG
    assert len(calls) == 1


def test_reset_failure_points_to_log(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, reset_gate

    _fake_sh(monkeypatch, 1)
    state = State()
    r = reset_gate().run(Ctx(tmp_path, tmp_path, False), state)
    assert r.status == Status.FAIL
    assert state.log == str(tmp_path / "reset" / "log_01")


def test_status_json_has_run_fields(tmp_path):
    import json

    from galaxy_workload_deployment_test import Screen

    meta = {"host": "h", "run": "LOG_x", "mode": "real", "commit": "abc"}
    Screen([], ["hdr"], tmp_path / "status.json", meta).write_status()
    assert json.loads((tmp_path / "status.json").read_text()).items() >= meta.items()


def test_models_grid_aligned_with_color(tmp_path, monkeypatch):
    import re

    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    steps = [Step(f"{g}:m", g, g, "kimi_k2_7", True, None) for g in ("block", "smoke", "soak")]
    steps[0].state.result = Result(Status.PASS)
    screen = Screen(steps, ["hdr"], tmp_path / "status.json", {})
    rows = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in screen.lines() if l.startswith((" K2.7", "MODELS"))]
    assert rows[0].index("smoke") == rows[1].index("·")


@pytest.mark.parametrize("tty", [True, False])
def test_screen_columns(tmp_path, monkeypatch, tty):
    import re
    import time

    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: tty)
    paths = Step("paths", "paths", "gate", None, True, None)
    paths.state.result = Result(Status.WARN, "5/6 exist", ["K2.7", "  A_VAR = /a", "  B_VAR = /b   MISSING"])
    smi = Step("tt-smi", "tt-smi", "gate", None, True, None)
    smi.state.result = Result(Status.WARN, "TDP low", ["fw_bundle_version 19.16.0.0 (32/32)"])
    models = [Step(f"{g}:m", g, g, "kimi_k2_7", True, None) for g in ("block", "smoke", "soak")]
    models += [Step("block:g", "block", "block", "glm53", True, None)]
    models[0].state.result = Result(Status.PASS)
    models[0].state.start, models[0].state.end = time.time() - 360000, time.time()
    models[1].state.result = Result(Status.FAIL)
    for s in (paths, smi, *models):
        s.state.start = s.state.start or time.time()
    lines = [
        re.sub(r"\x1b\[[0-9;]*m", "", l) for l in Screen([paths, smi, *models], ["hdr"], tmp_path / "s", {}).lines()
    ]
    gates = next(i for i, l in enumerate(lines) if l.startswith("GATES"))
    note = lines[gates].index("note")
    assert lines[gates + 1].index("5/6 exist") == note
    assert lines[gates + 2 : gates + 5] == ["   K2.7", "     A_VAR = /a", "     B_VAR = /b   MISSING"]
    assert lines[gates + 5].index("TDP low") == note
    assert lines[gates + 6].index("fw_bundle_version") == note
    assert lines[gates + 5].index("0s") + 2 == lines[gates].index("time") + 4
    assert lines[gates + 7] == ""
    header, k2, glm = lines[gates + 8 : gates + 11]
    starts = [header.index(c) for c in ("block", "smoke", "soak 1/3", "soak 2/3")]
    assert [k2[i - 1] for i in starts[1:]] == [" "] * 3
    assert [k2.index(t) for t in ("PASS", "FAIL")] == starts[:2]
    assert glm.index("-") == starts[1]


def test_only_runs_exact_keys():
    from galaxy_workload_deployment_test import skipped_keys

    assert skipped_keys(["paths", "reset", "block:kimi_k2_7"], skip="", only="block:kimi_k2_7") == {"paths", "reset"}


def test_skip_and_unknown_keys():
    import pytest
    from galaxy_workload_deployment_test import skipped_keys

    assert skipped_keys(["paths", "reset"], skip="reset", only="") == {"reset"}
    with pytest.raises(ValueError, match="unknown --only keys: nope"):
        skipped_keys(["paths"], skip="", only="nope")


def test_missing_node_ids_matches_tree_names():
    from gwdt_checks import missing_node_ids

    wanted = ["a/t.py::test_x[p1]", "b/u.py::test_y[q]"]
    assert missing_node_ids(wanted, ["test_x[p1]"]) == ["b/u.py::test_y[q]"]


def _fake_collect(monkeypatch, steps):
    import gwdt_steps

    calls = []

    def sh(ctx, cmd, log, env=None, timeout=None):
        calls.append(cmd)
        log.parent.mkdir(parents=True, exist_ok=True)
        names = [i.split("::")[-1] for s in steps if s.collect for i in s.collect(ctx)]
        log.write_text("".join(f"  <Function {n}>\n" for n in names))
        return 0

    monkeypatch.setattr(gwdt_steps, "sh", sh)
    return calls


def test_collect_gate_is_one_pytest_call(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    steps = build_steps(["kimi_k2_7", "glm53"])
    calls = _fake_collect(monkeypatch, steps)
    r = steps[1].run(Ctx(Path(__file__).resolve().parents[4], tmp_path, False), State())
    assert r.status == Status.PASS
    assert len(calls) == 1
    assert "-p no:benchmark" in calls[0]


def test_co_steps_reuse_gate_collection(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    steps = build_steps(["glm53"])
    calls = _fake_collect(monkeypatch, steps)
    ctx = Ctx(Path(__file__).resolve().parents[4], tmp_path, True)
    steps[1].run(ctx, State())
    block = next(s for s in steps if s.key == "block:glm53")
    assert block.run(ctx, State()).status == Status.PASS
    assert len(calls) == 1


def test_collect_gate_names_only_the_bad_step(tmp_path, monkeypatch):
    import gwdt_steps
    from gwdt_steps import Ctx, State, build_steps

    steps = build_steps(["glm53"])
    bad = next(s for s in steps if s.key == "block:glm53")

    def sh(ctx, cmd, log, env=None, timeout=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        if "test_glm_prefill_block" in cmd:
            log.write_text("ERROR: not found: x.py::test_glm_prefill_block\n")
        else:
            names = [a.strip("'").split("::")[-1] for a in cmd.split() if "::" in a]
            log.write_text("".join(f"  <Function {n}>\n" for n in names))
        return 0

    monkeypatch.setattr(gwdt_steps, "sh", sh)
    r = steps[1].run(Ctx(Path(__file__).resolve().parents[4], tmp_path, False), State())
    assert r.status == Status.FAIL
    assert [d.split(":")[0] + ":" + d.split(":")[1] for d in r.details] == [bad.key]


def test_help_lists_every_step_key(capsys):
    import pytest
    from galaxy_workload_deployment_test import main
    from gwdt_steps import build_steps

    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert all(s.key in out for s in build_steps(["kimi_k2_7", "glm53"]))


def test_mla_determinism_reps():
    from gwdt_checks import mla_determinism_reps

    log = "✓ Chunked prefill determinism: 50 reps of every forward bit-identical (1 iter(s) x 1 user(s))\n"
    assert mla_determinism_reps(log) == 50
    assert mla_determinism_reps("=== 1 passed in 9s ===") is None


def _dram_db(path, rows, errors=()):
    import sqlite3

    con = sqlite3.connect(path)
    con.execute('CREATE TABLE diagnostics ("Script" TEXT, "Type" TEXT, "Device" INTEGER, "Message" TEXT)')
    con.executemany("INSERT INTO diagnostics VALUES (?, 'error', ?, ?)", errors)
    con.execute(
        'CREATE TABLE check_dram_health ("Dev" TEXT, "GDDR Inst" INTEGER, '
        '"Corrected Reads" INTEGER, "Corrected Writes" INTEGER)'
    )
    con.executemany("INSERT INTO check_dram_health VALUES (?, ?, ?, ?)", rows)
    con.commit()
    con.close()
    return path


def test_dram_health_reports_corrected_counts(tmp_path):
    from gwdt_checks import dram_health

    r = dram_health(_dram_db(tmp_path / "dram.sqlite", [("0", 0, 0, 0), ("17", 3, 2, 255)]))
    assert r.status == Status.PASS
    assert r.details == ["dev 17 GDDR 3: corrected rd/wr 2/255"]


def test_dram_health_error_fails(tmp_path):
    from gwdt_checks import dram_health

    db = _dram_db(
        tmp_path / "dram.sqlite",
        [("5", 1, 0, 0)],
        [("check_dram_health", 5, "GDDR instance 1 (DDR bank 1): uncorrected EDC errors rd/wr 1/0")],
    )
    r = dram_health(db)
    assert r.status == Status.FAIL
    assert "dev 5: GDDR instance 1 (DDR bank 1): uncorrected EDC errors rd/wr 1/0" in r.details


def test_dram_health_without_rows_fails(tmp_path):
    from gwdt_checks import dram_health

    assert dram_health(tmp_path / "none.sqlite").status == Status.FAIL


def test_mla_step_fails_without_reps_line(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, mla_step

    def write(cmd, log, env):
        if "pytest" in cmd:
            log.write_text("✓ Chunked prefill determinism: 3 reps of every forward\n=== 1 passed in 9s ===\n")
        elif "--sqlite-output-path=" in cmd:
            _dram_db(Path(cmd.split("--sqlite-output-path=")[1].split()[0]), [("0", 0, 0, 0)])

    _fake_sh(monkeypatch, 0, write)
    r = mla_step("mla-smoke", 50).run(Ctx(tmp_path, tmp_path, False), State())
    assert r.status == Status.FAIL
    assert "50 reps" in r.reason
