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


LOGURU = "2026-10-31 03:12:44.123 | {} | {} - {}\n"


def test_run_to_run_bad_devices():
    from gwdt_checks import run_to_run_bad_devices

    where = "tests.didt.test_matmul_run_to_run_determinism:test_matmul_run_to_run_determinism:172"
    log = "".join(
        LOGURU.format("WARNING ", where, m)
        for m in (
            "  cores= 16 (4x4 at 5,7): shard 21 (device id 14), logical core (6, 8) -> physical (7,9): "
            "run-to-run NOT deterministic, 40 elements",
            "  cores=110 (11x10 at 0,0): shard 21 (device id 14), logical core (6, 8) -> physical (7,9): "
            "run-to-run NOT deterministic, 12 elements",
            "  cores=110 (11x10 at 0,0): shard 18 (device id 23), logical core (5, 8) -> physical (6,9): "
            "run-to-run NOT deterministic, 3 elements",
        )
    ) + LOGURU.format("INFO    ", where, "  cores=110: stable vs-shard0 offsets on shards [3] (not ND)")
    assert run_to_run_bad_devices(log) == {
        14: {"shard 21 core (6, 8) phys (7,9)": 52},
        23: {"shard 18 core (5, 8) phys (6,9)": 3},
    }


def test_sweep_bad_devices_and_skipped_windows():
    from gwdt_checks import sweep_bad_devices, sweep_skipped_windows

    where = "tests.didt.scratch_core_sweep:test_core_sweep:108"
    log = (
        LOGURU.format("WARNING ", where, "  window 11x10 at (0,0): device 21 NOT deterministic, 96 elements")
        + LOGURU.format("INFO    ", where, "skipped windows: 0 -> []")
        + LOGURU.format("SUCCESS ", where, "mode=grid: 1 (device,window) ND hits; devices=[21]")
    )
    assert sweep_bad_devices(log) == {21: 96}
    assert sweep_skipped_windows(log) == 0
    assert sweep_skipped_windows(LOGURU.format("INFO    ", where, "skipped windows: 1 -> [((0, 0), 'TT_THROW')]")) == 1
    assert sweep_skipped_windows("=== 1 failed in 9s ===") is None


def _write_ack_log(chips, hung):
    where = "__main__:scan_all_chips:274"
    lines = [LOGURU.format("INFO    ", where, "scanning 32 chips, dispatch watchdog=30s, logs under /tmp/x")]
    for c in chips:
        if c in hung:
            why = f"HUNG chip {c} (runtime id 0): BusDeviceFunction=0000:c3:00.0 /dev/tenstorrent/{c} T3:N3"
            lines.append(LOGURU.format("ERROR   ", where, f"chip {c} FAIL: {why}"))
            lines.append(LOGURU.format("ERROR   ", where, f"  cat /tmp/x/chip{c}/pytest.log"))
        else:
            lines.append(LOGURU.format("SUCCESS ", where, f"chip {c} PASS"))
    end = f"hung chips: {sorted(hung)} -- reset before re-running" if hung else f"all {len(chips)} chips passed"
    return "".join(lines) + LOGURU.format("INFO    ", where, end)


def test_write_ack_result():
    from gwdt_checks import write_ack_result

    assert write_ack_result(_write_ack_log(range(32), set())) == Result(Status.PASS, "no hang on 32/32 chips")
    r = write_ack_result(_write_ack_log([19, 3, 0], {19, 3}))
    assert r.status == Status.FAIL
    assert r.reason == "hang on 2/3 chips"
    assert r.details == [
        "chip 3: hang BusDeviceFunction=0000:c3:00.0 /dev/tenstorrent/3 T3:N3",
        "chip 19: hang BusDeviceFunction=0000:c3:00.0 /dev/tenstorrent/19 T3:N3",
    ]
    crash = "Traceback (most recent call last):\nFileNotFoundError: [Errno 2] No such file\n"
    assert write_ack_result(crash) == Result(Status.FAIL, "FileNotFoundError: [Errno 2] No such file")


MATMUL_KEYS = [
    "matmul:minimal",
    "matmul:ff1",
    "matmul:duty-cycle-bursty-5",
    "matmul:run-to-run",
    "matmul:core-sweep",
    "matmul:mcast-ack",
]


def test_matmul_runs_replace_didt_after_the_gates():
    from gwdt_steps import build_steps

    keys = [s.key for s in build_steps(["glm53"])]
    assert "didt" not in keys
    first = keys.index("matmul:minimal")
    assert keys[first - 1] == "mla-soak"
    assert keys[first : first + len(MATMUL_KEYS)] == MATMUL_KEYS
    assert keys[first + len(MATMUL_KEYS)] == "block:glm53"


def test_matmul_commands(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    seen = {}
    calls = _fake_sh(monkeypatch, 0, lambda cmd, log, env: seen.setdefault(cmd, env))
    steps = {s.key: s for s in build_steps(["glm53"])}
    cmds = {}
    for key in MATMUL_KEYS:
        calls.clear()
        steps[key].run(Ctx(tmp_path, tmp_path, False), State())
        assert calls[0] == "tt-smi -glx_reset_auto" and calls[2] == "tt-smi -s --snapshot_no_tty"
        cmds[key] = calls[1]
    assert cmds == {
        "matmul:minimal": "pytest -vs -rA 'tests/didt/test_minimal_matmul.py::test_minimal_matmul"
        "[blackhole-bf16_HiFi2-galaxy]' '--didt-workload-iterations=1000' '--determinism-check-interval=50' "
        "'--timeout=0'",
        "matmul:ff1": "pytest -vs -rA 'tests/didt/test_ff1_matmul.py::test_ff1_matmul[blackhole-galaxy-without_gelu]' "
        "'--didt-workload-iterations=1000' '--determinism-check-interval=50' '--timeout=0'",
        "matmul:duty-cycle-bursty-5": "pytest -vs -rA 'tests/didt/test_duty_cycle.py::test_duty_cycle"
        "[blackhole-duty-5-rep-1000x-all]' '--didt-workload-iterations=10' '--determinism-check-interval=1' "
        "'--timeout' '1800'",
        "matmul:run-to-run": "pytest -vs -rA 'tests/didt/test_matmul_run_to_run_determinism.py"
        "::test_matmul_run_to_run_determinism[blackhole-galaxy]' '-p' 'no:randomly' '--timeout=0'",
        "matmul:core-sweep": "pytest -vs -rA 'tests/didt/scratch_core_sweep.py::test_core_sweep[blackhole-galaxy]' "
        "'-p' 'no:randomly' '--timeout=0'",
        "matmul:mcast-ack": "python tests/didt/test_mm_mcast_write_ack_failure.py",
    }
    assert seen[cmds["matmul:duty-cycle-bursty-5"]]["TT_MESH_GRAPH_DESC_PATH"] == (
        "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"
    )
    assert seen[cmds["matmul:core-sweep"]].items() >= {
        "TT_SWEEP_MODE": "grid",
        "TT_SWEEP_GRID": "11x10",
        "TT_SWEEP_ITERS": "10",
    }.items()
    assert seen[cmds["matmul:run-to-run"]]["TT_DET_SWEEP_ITERS"] == "10"
    for key in MATMUL_KEYS[:-1]:
        assert "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" in seen[cmds[key]]
    assert seen[cmds["matmul:mcast-ack"]] == {"TMPDIR": str(tmp_path / "matmul" / "mcast-ack"), "MATMUL_ITERS": "10"}


def test_matmul_nd_details_per_device(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    def write(cmd, log, env):
        if "test_minimal_matmul" in cmd:
            log.write_text(
                "Number of non-deterministic outputs on device 17 is 2\n"
                "FAILED tests/didt/test_minimal_matmul.py::x - assert False\n=== 1 failed in 9s ===\n"
            )

    _fake_sh(monkeypatch, 0, write)
    step = next(s for s in build_steps(["glm53"]) if s.key == "matmul:minimal")
    r = step.run(Ctx(tmp_path, tmp_path, False), State())
    assert r.status == Status.FAIL
    assert r.details == ["device 17: 2 non-deterministic outputs"]


def test_core_sweep_skipped_window_fails(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    def write(cmd, log, env):
        if "scratch_core_sweep" in cmd:
            log.write_text("skipped windows: 1 -> [((0, 0), 'TT_THROW')]\n=== 1 passed in 9s ===\n")

    _fake_sh(monkeypatch, 0, write)
    step = next(s for s in build_steps(["glm53"]) if s.key == "matmul:core-sweep")
    r = step.run(Ctx(tmp_path, tmp_path, False), State())
    assert r == Result(Status.FAIL, "11x10 window did not run, skipped windows: 1")


def test_run_all_matmul_fail_blocks_nothing():
    from gwdt_steps import run_all

    steps, ctx = _steps(
        ("m1", "matmul", None, Status.FAIL), ("m2", "matmul", None, Status.HANG), ("b", "block", "m", Status.PASS)
    )
    run_all(steps, ctx, set(), lambda s: None)
    assert [s.state.result.status for s in steps] == [Status.FAIL, Status.HANG, Status.PASS]


def test_matmul_nd_layout(tmp_path, monkeypatch):
    import json
    import re

    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    ok = Step("matmul:minimal", "matmul minimal", "matmul", None, True, None)
    ok.state.result = Result(Status.PASS, "1 passed")
    bad = Step("matmul:duty-cycle-bursty-5", "matmul duty-cycle-bursty-5", "matmul", None, True, None)
    bad.state.result = Result(Status.FAIL, "assert False", ["device 17: 2 non-deterministic outputs"])
    block = Step("block:kimi_k2_7", "block K2.7", "block", "kimi_k2_7", True, None)
    screen = Screen([ok, bad, block], ["hdr"], tmp_path / "status.json", {})
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in screen.lines()]
    m = lines.index("MATMUL ND")
    run, failed, dev = lines[m + 1 : m + 4]
    assert run.startswith("   minimal ") and failed.startswith("   duty-cycle-bursty-5 ")
    assert dev.startswith("     device 17 ")
    assert run.index("PASS") == failed.index("FAIL") == dev.index("FAIL")
    assert dev.endswith("2 non-deterministic outputs")
    assert lines[m + 4 : m + 6] == ["", "MODELS"]
    screen.write_status()
    rows = {s["key"]: s for s in json.loads((tmp_path / "status.json").read_text())["steps"]}
    assert rows["matmul:duty-cycle-bursty-5"]["details"] == ["device 17: 2 non-deterministic outputs"]


def _nd_step(key):
    from gwdt_steps import build_steps

    return next(s for s in build_steps(["glm53"]) if s.key == key)


@pytest.mark.parametrize("key", MATMUL_KEYS)
def test_matmul_fail_reruns_throttled(tmp_path, monkeypatch, key):
    from gwdt_steps import Ctx, State

    seen = []
    calls = _fake_sh(monkeypatch, 0, lambda cmd, log, env: seen.append((log, env)))
    state = State()
    r = _nd_step(key).run(Ctx(tmp_path, tmp_path, False), state)
    d = tmp_path / key.replace(":", "/")
    assert r.status == Status.FAIL and state.log == str(d / "log_01")
    assert calls[3:6] == calls[:3] and calls[0] == "tt-smi -glx_reset_auto"
    assert [log for log, _ in seen[:6]] == [
        d / "reset.log",
        d / "log_01",
        d / "tt-smi.json",
        d / "reset_02.log",
        d / "log_02",
        d / "tt-smi_02.json",
    ]
    assert "TT_MM_THROTTLE_PERF" not in seen[1][1] and seen[4][1]["TT_MM_THROTTLE_PERF"] == "5"
    first = state.fallbacks[0]
    assert first.result.status == Status.FAIL and first.log == str(d / "log_02")
    assert first.env == {"TT_MM_THROTTLE_PERF": "5"} and first.opts == [] and first.end >= first.start
    assert len(state.fallbacks) == (2 if key == "matmul:minimal" else 1)
    if key == "matmul:mcast-ack":
        assert seen[4][1]["TMPDIR"] == str(d / "tmp_02")
    else:
        assert seen[4][1]["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"].endswith(
            f"--llm-output-path={d / 'crash_triage_02.csv'}; kill -9 $PPID"
        )


def test_matmul_hang_reruns_and_keeps_first_result(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State

    def write(cmd, log, env):
        if "TT_MM_THROTTLE_PERF" in env:
            log.write_text("=== 1 passed in 250s ===\n")
        elif cmd.startswith("pytest"):
            cmd_env = env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
            Path(cmd_env.split("--llm-output-path=")[1].split(";")[0]).write_text("dev,core\n")

    _fake_sh(monkeypatch, 0, write)
    state = State()
    r = _nd_step("matmul:minimal").run(Ctx(tmp_path, tmp_path, False), state)
    assert r.status == Status.HANG and state.log == str(tmp_path / "matmul" / "minimal" / "log_01")
    assert state.fallbacks[0].result == Result(Status.PASS, "1 passed")


def test_minimal_second_fallback_caps_grid_without_throttle(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State

    seen = []
    calls = _fake_sh(monkeypatch, 0, lambda cmd, log, env: seen.append((log, env)))
    state = State()
    r = _nd_step("matmul:minimal").run(Ctx(tmp_path, tmp_path, False), state)
    d = tmp_path / "matmul" / "minimal"
    assert r.status == Status.FAIL and len(calls) == 9
    assert [log for log, _ in seen[6:]] == [d / "reset_03.log", d / "log_03", d / "tt-smi_03.json"]
    assert calls[7] == calls[1] + " '--grid-size' '11x10'"
    assert "TT_MM_THROTTLE_PERF" not in seen[7][1]
    assert seen[7][1]["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"].endswith(
        f"--llm-output-path={d / 'crash_triage_03.csv'}; kill -9 $PPID"
    )
    second = state.fallbacks[1]
    assert second.env == {} and second.opts == ["--grid-size", "11x10"]
    assert second.result.status == Status.FAIL and second.log == str(d / "log_03")


def test_minimal_first_fallback_pass_skips_second(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State

    def write(cmd, log, env):
        if "TT_MM_THROTTLE_PERF" in env:
            log.write_text("=== 1 passed in 250s ===\n")

    calls = _fake_sh(monkeypatch, 0, write)
    state = State()
    assert _nd_step("matmul:minimal").run(Ctx(tmp_path, tmp_path, False), state).status == Status.FAIL
    assert len(calls) == 6 and [f.result.status for f in state.fallbacks] == [Status.PASS]


def test_matmul_pass_does_not_rerun(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State

    calls = _fake_sh(monkeypatch, 0, lambda cmd, log, env: log.write_text("=== 1 passed in 9s ===\n"))
    state = State()
    assert _nd_step("matmul:ff1").run(Ctx(tmp_path, tmp_path, False), state).status == Status.PASS
    assert len(calls) == 3 and state.fallbacks == []


def test_matmul_co_does_not_rerun(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State

    calls = _fake_sh(monkeypatch, 0)
    state = State()
    assert _nd_step("matmul:ff1").run(Ctx(tmp_path, tmp_path, True), state).status == Status.FAIL
    assert len(calls) == 1 and state.fallbacks == []


def test_matmul_nd_fallback_rows(tmp_path, monkeypatch):
    import json
    import re
    import time

    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import State, Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    bad = Step("matmul:minimal", "matmul minimal", "matmul", None, True, None)
    bad.state.result = Result(Status.FAIL, "assert False", ["device 17: 2 non-deterministic outputs"])
    first = Result(Status.FAIL, "assert False", ["device 3: 1 non-deterministic outputs"])
    second = Result(Status.FAIL, "assert False", ["device 5: 4 non-deterministic outputs"])
    now = time.time()
    bad.state.fallbacks = [
        State(first, "log_02", now - 250, now, attempt=2, env={"TT_MM_THROTTLE_PERF": "5"}),
        State(second, "log_03", now - 70, now, attempt=3, opts=["--grid-size", "11x10"]),
    ]
    screen = Screen([bad], ["hdr"], tmp_path / "status.json", {})
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in screen.lines()]
    m = lines.index("MATMUL ND")
    step, dev, fb1, fb1_dev, fb2, fb2_dev = lines[m + 1 : m + 7]
    assert step.startswith("   minimal ") and dev.startswith("     device 17 ")
    assert fb1.startswith("     fallback 1 ") and fb1_dev.startswith("       device 3 ")
    assert fb2.startswith("     fallback 2 ") and fb2_dev.startswith("       device 5 ")
    assert step.index("FAIL") == dev.index("FAIL") == fb1.index("FAIL") == fb2_dev.index("FAIL")
    assert fb1.endswith("4m10s    TT_MM_THROTTLE_PERF=5  assert False")
    assert fb2.endswith("1m10s    --grid-size 11x10  assert False")
    screen.write_status()
    row = json.loads((tmp_path / "status.json").read_text())["steps"][0]
    assert row["status"] == "FAIL" and [f["status"] for f in row["fallbacks"]] == ["FAIL", "FAIL"]
    assert row["fallbacks"][0]["env"] == {"TT_MM_THROTTLE_PERF": "5"} and row["fallbacks"][0]["opts"] == []
    assert row["fallbacks"][1]["env"] == {} and row["fallbacks"][1]["opts"] == ["--grid-size", "11x10"]
    assert row["fallbacks"][0]["log"] == "log_02" and row["fallbacks"][1]["log"] == "log_03"
    assert row["fallbacks"][1]["details"] == ["device 5: 4 non-deterministic outputs"]


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
    step = Step("matmul:minimal", "matmul minimal", "matmul", None, True, lambda ctx, state: Result(Status.PASS))
    step.state.result = Result(Status.FAIL, "assert False", ["device 17: 2 non-deterministic outputs"])
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


def test_run_all_missing_path_blocks_only_steps_that_need_it():
    from gwdt_steps import run_all

    steps, ctx = _steps(("r", "block-random", "m", Status.PASS), ("s", "smoke", "m", Status.PASS))
    steps[0].needs, steps[1].needs = ["HF"], ["HF", "TRACE"]
    ctx.missing_paths["m"] = ["TRACE"]
    run_all(steps, ctx, set(), lambda s: None)
    assert steps[0].state.result.status == Status.PASS
    assert steps[1].state.result == Result(Status.PENDING, "not run: TRACE missing")


def test_path_needs_block_random_only_needs_hf_config():
    from gwdt_steps import path_needs

    assert path_needs("block-random", "kimi_k2_7") == ["KIMI_K2_7_HF_MODEL"]
    assert path_needs("block-random", "glm53") == ["GLM53_HF_MODEL"]
    assert path_needs("block", "glm53") == ["GLM53_HF_MODEL", "TT_GLM53_PREFILL_TTNN_CACHE"]
    assert path_needs("block", "kimi_k2_7") == ["KIMI_K2_7_HF_MODEL", "TT_KIMI_PREFILL_TTNN_CACHE"]
    assert "PREFILL_TRACE_DIR" in path_needs("smoke", "glm53")


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
    assert len([c for c in calls if c.startswith("pytest")]) == 1


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


def test_only_runs_exact_keys():
    from galaxy_workload_deployment_test import skipped_keys

    assert skipped_keys(["paths", "reset", "block:kimi_k2_7"], skip="", only="block:kimi_k2_7") == {"paths", "reset"}


def test_skip_and_unknown_keys():
    import pytest
    from galaxy_workload_deployment_test import skipped_keys

    assert skipped_keys(["paths", "reset"], skip="reset", only="") == {"reset"}
    with pytest.raises(ValueError, match="unknown --only keys: nope"):
        skipped_keys(["paths"], skip="", only="nope")


def test_nd_group_picks_every_matmul_step():
    from galaxy_workload_deployment_test import skipped_keys

    keys = ["reset", "matmul:minimal", "matmul:ff1", "block:glm53"]
    assert skipped_keys(keys, skip="", only="nd") == {"reset", "block:glm53"}
    assert skipped_keys(keys, skip="nd", only="") == {"matmul:minimal", "matmul:ff1"}


def test_missing_node_ids_matches_tree_names():
    from gwdt_checks import missing_node_ids

    wanted = ["a/t.py::test_x[p1]", "b/u.py::test_y[q]"]
    assert missing_node_ids(wanted, ["test_x[p1]"]) == ["b/u.py::test_y[q]"]


def test_base_env_disables_pinned_memory_cache(tmp_path, monkeypatch):
    from gwdt_steps import base_env

    monkeypatch.setenv("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "4294967296")
    assert base_env(tmp_path)["TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES"] == "0"


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
        if "test_glm_prefill_block[" in cmd:
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


def test_block_random_runs_per_layer_type_random_rows():
    from gwdt_steps import build_steps

    ids = {
        s.key: [i.split("::")[-1] for i in s.collect(None)]
        for s in build_steps(["kimi_k2_7", "glm53"])
        if s.group == "block-random"
    }
    assert ids == {
        "block-random:kimi_k2_7": [
            "test_kimi_prefill_block_chunked_random[blackhole-kimi_k2_7-torus-xy-8x4-dense_L0-chunks2-iter100]",
            "test_kimi_prefill_block_chunked_random[blackhole-kimi_k2_7-torus-xy-8x4-moe_L1-chunks2-iter100]",
        ],
        "block-random:glm53": [
            "test_glm_prefill_block_chunked_random[blackhole-glm53-torus-xy-8x4-dense_L0-chunks2-iter100]",
            "test_glm_prefill_block_chunked_random[blackhole-glm53-torus-xy-8x4-moe_full_L6-chunks2-iter100]",
            "test_glm_prefill_block_chunked_random[blackhole-glm53-torus-xy-8x4-moe_shared_L3-chunks2-iter100]",
        ],
    }


R = "x.py::test_r[blackhole-kimi_k2_7-torus-xy-8x4-{}-chunks2-iter100]"


def test_layer_label():
    from gwdt_checks import layer_label

    assert layer_label(R.format("moe_shared_L3")) == "moe_shared_L3"
    assert (
        layer_label("x.py::test_glm_prefill_block[blackhole-glm53-with_determinism-iter2-moe-seq5120-torus-xy-8x4]")
        == "moe"
    )
    assert (
        layer_label("x.py::t[blackhole-kimi_k2_7-torus-xy-8x4-moe-gate_host-chunks2-iter2-with_determinism]") == "moe"
    )


def test_row_results_from_summary():
    from gwdt_checks import row_results

    log = f"PASSED {R.format('dense_L0')}\nFAILED {R.format('moe_L1')} - AssertionError: x\n"
    rows = row_results(log, [R.format("dense_L0"), R.format("moe_L1")])
    assert [r.status for r in rows] == [Status.PASS, Status.FAIL]


def test_row_results_live():
    from gwdt_checks import row_results

    log = f"{R.format('dense_L0')} \nstuff\nPASSED\n{R.format('moe_L1')} \n  pass 95/100 done\n  pass 96/100 done\n"
    rows = row_results(log, [R.format("dense_L0"), R.format("moe_L1"), R.format("moe_L6")])
    assert [r.status for r in rows] == [Status.PASS, Status.RUN, Status.PENDING]
    assert rows[1].reason == "pass 96/100"


def test_models_layout_a(tmp_path, monkeypatch):
    import re

    from galaxy_workload_deployment_test import Screen
    from gwdt_steps import Step

    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    ids = [R.format("dense_L0"), R.format("moe_L1")]
    rand = Step("block-random:kimi_k2_7", "block-random K2.7", "block-random", "kimi_k2_7", True, None, lambda ctx: ids)
    rand.state.result = Result(Status.PASS)
    rand.state.rows = [Result(Status.PASS), Result(Status.PASS)]
    smoke = Step("smoke:kimi_k2_7", "smoke K2.7", "smoke", "kimi_k2_7", True, None)
    smoke.state.result = Result(Status.FAIL, "iter 1 KV cache DIFFERS")
    soak = Step("soak:kimi_k2_7", "soak K2.7", "soak", "kimi_k2_7", True, None)
    soak.state.result = Result(Status.SKIP, "by user")
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in Screen([rand, smoke, soak], ["hdr"], tmp_path / "s", {}).lines()]
    m = lines.index("MODELS")
    assert lines[m + 1] == " K2.7 (L61)"
    step, l0, l1, sm, so = lines[m + 2 : m + 7]
    assert step.startswith("   block-random ") and l0.startswith("     dense_L0 ") and l1.startswith("     moe_L1 ")
    assert step.index("PASS") == l0.index("PASS") == l1.index("PASS") == sm.index("FAIL") == so.index("SKIP")
    assert sm.startswith("   smoke ") and "iter 1 KV cache DIFFERS" in sm


def test_device_step_saves_smi_snapshot_after_run(tmp_path, monkeypatch):
    from gwdt_steps import Ctx, State, build_steps

    seen = []

    def write(cmd, log, env):
        seen.append((cmd, log))

    _fake_sh(monkeypatch, 0, write)
    step = next(s for s in build_steps(["glm53"]) if s.key == "block-random:glm53")
    step.run(Ctx(tmp_path, tmp_path, False), State())
    assert seen[-1] == ("tt-smi -s --snapshot_no_tty", tmp_path / "block-random" / "glm53" / "tt-smi.json")
