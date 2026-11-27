# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path

from tests.didt.didt_runner import main
from tests.didt.runner.chips import Chip
from tests.didt.runner.config import load
from tests.didt.runner.steps import INSPECTOR_PORT_BASE, double_iterations, run, sh, sh_all

TESTDATA = Path(__file__).parent / "testdata"
SHIPPED = Path(__file__).parents[1] / "runner_configs" / "weekly.json"
TRAY_BUS_HIGH = {1: 0x0, 2: 0x4, 3: 0xC, 4: 0x8}
CFG = load(SHIPPED)
KEYS = [s.key for s in CFG.steps]
IDS = {s.key: s.ids[0] for s in CFG.steps}


def _chips() -> list[Chip]:
    rows = sorted((f"0000:{(TRAY_BUS_HIGH[t] << 4) | a:02x}:00.0", t, a) for t in range(1, 5) for a in range(1, 9))
    return [Chip(i, bdf, t, a, f"0x{t:02x}{a:014x}", (t - 1) * 8 + a - 1) for i, (bdf, t, a) in enumerate(rows)]


CHIPS = _chips()


def _loop(start: str, end: str, iterations: int) -> str:
    """op_test_base's own debug lines around the iteration loop."""
    where = "tt_py_test_utils_common.op_test_base:run_op_test"
    return (
        f"2026-11-26 {start} | DEBUG    | {where}:276 - Starting iterations\n"
        f"2026-11-26 {end} | DEBUG    | {where}:316 - Iteration = {iterations - 1}, done!\n"
    )


ORIGINAL_LOOP = _loop("12:01:00.000", "12:03:44.400", 1000)  # 164.4 s
RERUN_LOOP = _loop("12:10:00.000", "12:12:50.000", 2486)  # 170.0 s

# each_chip outcomes: chip -> pass | hung | nd | build, every chip not named passes. A each_chip attempt's outcome is
# one of these names, a {chip: state} dict, or one state for every chip it runs (a pinned rerun runs one chip).
PER_CHIP = {
    "fail": {3: "hung"},
    "build": {12: "build"},
    "hung2": {3: "hung", 7: "hung"},
    "mixed": {3: "hung", 19: "nd"},
}
CHIP_LOGS = {
    "pass": (RERUN_LOOP + "=== 1 passed in 9.00s ===\n", 0),
    # the triage hook kills the child mid-run: no summary line
    "hung": ("Starting iterations\n", -9),
    "nd": (ORIGINAL_LOOP + "{nd}=== 1 failed in 9.00s ===\n", 1),
    "build": (
        "E   RuntimeError: TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages:\n"
        "=== 1 failed in 9.00s ===\n",
        1,
    ),
}
# a per-chip child sees itself as device 0
ND_CHILD = (
    "2026-11-26 12:05:01.100 | info     | tt_py_test_utils_common.op_test_base:log_nd_core_summary:160 - "
    "ND summary device 0 [UBB4 ASIC4]: 3/20 checks; pattern single core (6,8); failing checks per core: (6,8)x3\n"
    "2026-11-26 12:05:01.100 | info     | tt_py_test_utils_common.op_test_base:log_nd_core_summary:167 - "
    "ND summary device 0 core (6,8) translated=(7,10): failed 3/20 checks; face column lane 3 in 3/3 checks\n"
)


def read_csv(name: str) -> str:
    return (TESTDATA / name).read_text()


class FakeShell:
    """Stands in for every command the runner launches. outcomes: attempt dir (relative to the run dir) ->
    pass | fail | hang | timeout | raise. Collect output mimics pytest: one missing id hides the whole listing."""

    def __init__(self, run_dir, outcomes=None, missing=(), reset_fail=(), smi=None):
        self.run_dir, self.outcomes, self.missing = run_dir, outcomes or {}, set(missing)
        self.reset_fail, self.smi, self.calls = set(reset_fail), smi or {}, []
        self.chip_jobs = []

    def __call__(self, cmd, log, env, cwd, timeout):
        self.calls.append({"cmd": cmd, "log": log, "env": env, "timeout": timeout})
        log.parent.mkdir(parents=True, exist_ok=True)
        rel = str(log.parent.relative_to(self.run_dir))
        if cmd.startswith("pytest --collect-only"):
            ids = [t for t in shlex.split(cmd) if "::" in t]
            gone = [i for i in ids if i in self.missing]
            if gone:
                log.write_text("".join(f"ERROR: not found: /repo/{i}\n" for i in gone) + "collected 1 item\n")
                return 4
            log.write_text("\n".join(ids) + f"\n\n{len(ids)} tests collected in 1.0s\n")
            return 0
        if "glx_reset" in cmd or cmd.startswith("tt-smi -r"):
            log.write_text("reset\n")
            return 1 if rel in self.reset_fail else 0
        if cmd.startswith("tt-smi -s"):
            log.write_text(json.dumps(self.smi.get(rel, {"device_info": []})))
            return 0
        outcome = self.outcomes.get(rel, "pass")
        if outcome == "raise":
            raise OSError("disk full")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        if outcome == "timeout":
            log.write_text(f"\nTIMEOUT after {timeout}s\n")
            return 124
        if outcome == "hang":
            csv = re.search(r"--llm-output-path=(\S+);", env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]).group(1)
            Path(csv).write_text((TESTDATA / "crash_triage_hang.csv").read_text())
            log.write_text("Device 27: Timeout waiting for dispatch\n")
            return 137
        if outcome == "fail":
            log.write_text(ORIGINAL_LOOP + (TESTDATA / "nd_report.log").read_text())
            return 1
        if outcome == "r2r":
            log.write_text(
                "shard 3 (device id 21), logical core (6, 8) -> physical (7,10): run-to-run NOT deterministic, 4 elements\n"
                "=== 1 failed in 9.00s ===\n"
            )
            return 1
        log.write_text(RERUN_LOOP + "skipped windows: 0 -> []\n=== 1 passed in 9.00s ===\n")
        return 0

    def all(self, jobs, cwd, timeout):
        """each_chip launches: chips/chip<dev>/pytest.log under the attempt dir, the chip in TT_VISIBLE_DEVICES."""
        rcs = []
        for cmd, log, env in jobs:
            self.chip_jobs.append({"cmd": cmd, "log": log, "env": env, "timeout": timeout})
            log.parent.mkdir(parents=True, exist_ok=True)
            Path(env["TT_METAL_CACHE"]).mkdir(parents=True, exist_ok=True)
            dev = int(env["TT_VISIBLE_DEVICES"])
            outcome = self.outcomes.get(str(log.parents[2].relative_to(self.run_dir)), "pass")
            if outcome == "timeout":
                log.write_text(f"\nTIMEOUT after {timeout:g} s\n")
                rcs.append(124)
                continue
            if isinstance(outcome, dict):
                state = outcome.get(dev, "pass")
            else:
                state = PER_CHIP[outcome].get(dev, "pass") if outcome in PER_CHIP else outcome
            text, rc = CHIP_LOGS[state]
            log.write_text(text.replace("{nd}", ND_CHILD))
            if state == "hung":
                csv = re.search(r"--llm-output-path=(\S+);", env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]).group(
                    1
                )
                chip = next(c for c in CHIPS if c.dev == dev)
                Path(csv).write_text(read_csv("crash_triage_chip5.csv").replace("T1:N6", f"T{chip.tray}:N{chip.asic}"))
            rcs.append(rc)
        return rcs

    def chip_calls(self, rel):
        return [c for c in self.chip_jobs if str(c["log"].parents[2].relative_to(self.run_dir)) == rel]

    def reset_cmds(self, rel):
        return [
            c["cmd"]
            for c in self.calls
            if str(c["log"].parent.relative_to(self.run_dir)) == rel and "reset" in c["log"].name
        ]

    def workload_calls(self, rel):
        return [
            c for c in self.calls if str(c["log"].parent.relative_to(self.run_dir)) == rel and c["log"].name == "log"
        ]


def _run(tmp_path, only=None, skip=(), co=False, snap=None, **fake):
    run_dir = tmp_path / "LOG_2026_11_26_12_00_host_abcdef12345"
    run_dir.mkdir()
    shell = FakeShell(run_dir, **fake)
    rc = run(
        CFG, run_dir, CHIPS, set(only) if only else None, set(skip), co, shell=shell, snap=snap, shell_all=shell.all
    )
    doc = json.loads((run_dir / "status.json").read_text())
    return rc, {s["key"]: s for s in doc["steps"]}, doc, shell, run_dir


def test_fail_runs_every_fallback_status_stays_first(tmp_path):
    rc, steps, _, shell, _ = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    s = steps["nd:minimal"]
    assert rc == 1 and s["status"] == "FAIL"
    assert [(f["name"], f["status"]) for f in s["fallbacks"]] == [
        ("throttle", "PASS"),
        ("pin_core", "WARN"),
        ("pin_core", "WARN"),
    ]
    assert s["fallbacks"][0]["env"] == {"TT_MM_THROTTLE_PERF": "5"} and s["fallbacks"][0]["devices"] == []
    assert shell.workload_calls("nd/minimal_throttle")[0]["env"]["TT_MM_THROTTLE_PERF"] == "5"
    assert "TT_MM_THROTTLE_PERF" not in shell.workload_calls("nd/minimal")[0]["env"]
    assert [d["dev"] for d in s["devices"]] == [19, 26]
    assert s["devices"][0]["bdf"] == "0000:84:00.0" and s["devices"][0]["asic_id"] == CHIPS[19].asic_id
    assert s["details"][0].startswith("device 19 bdf=0000:84:00.0 tray=4 asic=4: 3/20 checks")


def test_pass_runs_no_fallback(tmp_path):
    rc, steps, doc, _, run_dir = _run(tmp_path, only=["nd:minimal"])
    assert rc == 0 and steps["nd:minimal"]["status"] == "PASS" and steps["nd:minimal"]["fallbacks"] == []
    assert steps["nd:minimal"]["devices"] == []
    assert not (run_dir / "nd" / "minimal_throttle").exists()
    assert doc["current"] is None and doc["mode"] == "real" and len(doc["chips"]) == 32


def test_hang_reads_triage_csv(tmp_path):
    rc, steps, _, _, run_dir = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "hang"})
    s = steps["nd:minimal"]
    assert rc == 1 and s["status"] == "HANG"
    assert s["reason"] == f"triage {run_dir / 'nd/minimal/crash_triage.csv'}"
    assert s["devices"][0]["dev"] == 27 and s["devices"][0]["hung"] is True and s["devices"][0]["tray"] == 3
    assert s["fallbacks"][0]["status"] == "PASS"


def test_triage_env_on_pytest_attempts(tmp_path):
    _, _, _, shell, run_dir = _run(tmp_path, only=["nd:minimal"])
    env = shell.workload_calls("nd/minimal")[0]["env"]
    assert env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == "30"
    hook = env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
    assert "tools/tt-triage.py" in hook and "--run=dump_running_operations" in hook and "--run=dump_op_mesh" in hook
    assert f"--llm-output-path={run_dir / 'nd/minimal/crash_triage.csv'};" in hook and hook.endswith("kill -9 $PPID")


def test_each_chip_wall_timeout_fails_every_chip_still_running(tmp_path):
    rc, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack"], outcomes={"nd/mcast-ack": "timeout"})
    s = steps["nd:mcast-ack"]
    assert rc == 1 and s["status"] == "FAIL" and s["reason"] == "failed on 32/32 chips"
    assert s["details"][0] == "device 0 bdf=0000:01:00.0 tray=1 asic=1: timeout after 1800 s"


def test_missing_id_fails_only_its_step(tmp_path):
    rc, steps, _, shell, run_dir = _run(tmp_path, missing={IDS["nd:ff1"]})
    assert rc == 1
    assert steps["nd:ff1"]["status"] == "FAIL" and steps["nd:ff1"]["reason"].startswith("does not collect")
    assert IDS["nd:ff1"] in steps["nd:ff1"]["reason"]
    assert all(steps[k]["status"] == "PASS" for k in KEYS if k != "nd:ff1")
    assert not (run_dir / "nd" / "ff1").exists() and steps["nd:ff1"]["fallbacks"] == []


def test_reset_fails_three_times(tmp_path):
    rc, steps, _, _, run_dir = _run(
        tmp_path, only=["nd:minimal", "nd:ff1"], reset_fail={"nd/minimal", "nd/minimal_throttle"}
    )
    s = steps["nd:minimal"]
    assert rc == 1 and s["status"] == "FAIL" and s["reason"].startswith("reset failed 3x")
    assert (run_dir / "nd/minimal/reset.retry3.log").is_file() and not (run_dir / "nd/minimal/log").exists()
    assert s["fallbacks"][0]["status"] == "FAIL"
    assert steps["nd:ff1"]["status"] == "PASS"


def test_fallback_folder_names(tmp_path):
    _, _, _, _, run_dir = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    for d in ("nd/minimal", "nd/minimal_throttle"):
        assert {p.name for p in (run_dir / d).iterdir()} >= {"reset.log", "log", "tt-smi.json"}


MCAST_ACK = "tests/didt/test_mm_mcast_write_ack_failure.py::test_mm_mcast_write_ack_failure_scan[blackhole-1x1]"


def test_each_chip_runs_every_chip_with_its_own_triage(tmp_path):
    rc, steps, _, shell, run_dir = _run(tmp_path, only=["nd:mcast-ack"])
    jobs = shell.chip_calls("nd/mcast-ack")
    assert [j["env"]["TT_VISIBLE_DEVICES"] for j in jobs] == [str(c.dev) for c in CHIPS]
    assert {j["cmd"] for j in jobs} == {f"pytest {shlex.quote(MCAST_ACK)} -p no:cacheprovider --timeout=0"}
    assert {j["timeout"] for j in jobs} == {1800} and not shell.workload_calls("nd/mcast-ack")
    cache = Path(tempfile.gettempdir()) / f"didt_runner_{run_dir.name}"
    for j in jobs:
        dev, env = j["env"]["TT_VISIBLE_DEVICES"], j["env"]
        out = run_dir / f"nd/mcast-ack/chips/chip{dev}"
        port = INSPECTOR_PORT_BASE + int(dev)
        assert j["log"] == out / "pytest.log" and env["OUT_DIR"] == env["TT_METAL_LOGS_PATH"] == str(out)
        assert env["TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS"] == f"localhost:{port}"
        assert env["TT_METAL_CACHE"] == str(cache / f"nd/mcast-ack/chip{dev}")
        assert env["MATMUL_ITERS"] == "10" and env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == "30"
        hook = env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
        assert f"--inspector-rpc-port={port} --inspector-log-path={out / 'generated/inspector'} " in hook
        assert "--inspector-disable-rank" in hook and f"--llm-output-path={out / 'crash_triage.csv'};" in hook
        assert hook.endswith("; kill -9 $PPID")
    # kernel caches on local disk, gone after the attempt; logs stay in the run dir
    assert not cache.exists()
    assert (
        rc == 0
        and steps["nd:mcast-ack"]["status"] == "PASS"
        and steps["nd:mcast-ack"]["reason"] == "32/32 chips passed"
    )
    log = (run_dir / "nd/mcast-ack/log").read_text().splitlines()
    assert len(log) == 32 and log[0] == "chip 0 PASS"


def test_each_chip_hang_reads_the_chips_own_triage(tmp_path):
    _, steps, _, _, run_dir = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes={"nd/mcast-ack-duty": "fail"})
    s = steps["nd:mcast-ack-duty"]
    assert s["status"] == "HANG" and s["reason"] == "hung on 1/32 chips"
    (d,) = s["devices"]
    assert d == {**d, **CHIPS[3].as_dict(), "hung": True, "op": "MatmulDeviceOperation", "core_count": 120}
    assert d["cores"][0] == {"x": 6, "y": 8, "translated": [10, 10]}
    assert s["details"] == ["device 3 bdf=0000:04:00.0 tray=1 asic=4: hung in MatmulDeviceOperation, 120 cores"]
    csv = run_dir / "nd/mcast-ack-duty/chips/chip3/crash_triage.csv"
    assert f"chip 3 HANG triage {csv}" in (run_dir / "nd/mcast-ack-duty/log").read_text()


def test_each_chip_other_failure_keeps_chip_reason(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack"], outcomes={"nd/mcast-ack": "build"})
    s = steps["nd:mcast-ack"]
    assert s["status"] == "FAIL" and s["reason"] == "failed on 1/32 chips"
    assert s["devices"] == [CHIPS[12].as_dict()]
    assert s["details"] == [
        "device 12 bdf=0000:45:00.0 tray=2 asic=5: TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages:"
    ]


def test_each_chip_nd_joins_on_the_chip(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes={"nd/mcast-ack-duty": {19: "nd"}})
    s = steps["nd:mcast-ack-duty"]
    (d,) = s["devices"]
    assert s["status"] == "FAIL" and d["dev"] == 19 and d["bdf"] == CHIPS[19].bdf
    assert (d["failed_checks"], d["checks"], d["cores"][0]["x"]) == (3, 20, 6)


def test_core_sweep_without_window_line_fails(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()

    class NoWindows(FakeShell):
        def __call__(self, cmd, log, env, cwd, timeout):
            rc = super().__call__(cmd, log, env, cwd, timeout)
            if log.name == "log":
                log.write_text(log.read_text().replace("skipped windows: 0 -> []\n", "skipped windows: 1 -> []\n"))
            return rc

    run(CFG, run_dir, CHIPS, {"nd:core-sweep"}, set(), False, shell=NoWindows(run_dir))
    (s,) = [s for s in json.loads((run_dir / "status.json").read_text())["steps"] if s["key"] == "nd:core-sweep"]
    assert s["status"] == "FAIL" and "skipped windows: 1" in s["reason"]


def test_gddr_corrected_warns(tmp_path):
    bus = CHIPS[19].bdf
    telem = {"ENABLED_GDDR": "0xff", "GDDR_UNCORR_ERRS": "0x0", "GDDR_4_5_CORR_ERRS": "0x00000003"}
    smi = {"nd/minimal": {"device_info": [{"board_info": {"bus_id": bus}, "smbus_telem": telem}]}}
    rc, steps, _, _, _ = _run(tmp_path, only=["nd:minimal"], smi=smi)
    s = steps["nd:minimal"]
    assert rc == 0 and s["status"] == "WARN" and s["details"][0].startswith("dev 19 (84:00.0) GDDR 4")


def test_exception_in_attempt_does_not_stop_others(tmp_path):
    rc, steps, _, _, _ = _run(tmp_path, only=["nd:minimal", "nd:ff1"], outcomes={"nd/minimal": "raise"})
    assert steps["nd:minimal"]["status"] == "FAIL" and "OSError: disk full" in steps["nd:minimal"]["reason"]
    assert steps["nd:ff1"]["status"] == "PASS" and rc == 1


def test_skip_is_marked_by_user(tmp_path):
    rc, steps, _, shell, _ = _run(tmp_path, only=["nd:minimal", "nd:ff1"], skip=["nd:ff1"])
    assert steps["nd:ff1"]["status"] == "SKIP" and steps["nd:ff1"]["reason"] == "by user"
    assert steps["nd:lm-head"]["status"] == "SKIP" and rc == 0
    assert not shell.workload_calls("nd/ff1")


def test_co_collects_only(tmp_path):
    rc, steps, doc, shell, _ = _run(tmp_path, co=True)
    assert rc == 0 and doc["mode"] == "co"
    assert all(s["status"] == "PASS" for s in steps.values())
    assert [c["cmd"].split()[0] for c in shell.calls] == ["pytest"]


def test_status_written_while_running(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    seen = []

    class Watch(FakeShell):
        def __call__(self, cmd, log, env, cwd, timeout):
            if log.name == "log":
                seen.append(json.loads((run_dir / "status.json").read_text()))
            return super().__call__(cmd, log, env, cwd, timeout)

    run(CFG, run_dir, CHIPS, {"nd:minimal"}, set(), False, shell=Watch(run_dir))
    assert seen[0]["current"] == "nd:minimal" and seen[0]["steps"][0]["status"] == "RUN"
    assert (run_dir / "run.log").read_text().count("nd:minimal") >= 2


def test_sh_timeout_kills_the_process_group(tmp_path):
    log, pid_file = tmp_path / "log", tmp_path / "pid"
    start = time.time()
    rc = sh(f"sleep 30 & echo $! > {pid_file}; wait", log, dict(os.environ), tmp_path, 1)
    assert rc == 124 and time.time() - start < 10
    assert "TIMEOUT after 1 s" in log.read_text()
    pid = int(pid_file.read_text())
    time.sleep(0.2)
    try:
        os.kill(pid, 0)
        alive = Path(f"/proc/{pid}/status").is_file() and "zombie" not in Path(f"/proc/{pid}/status").read_text()
    except ProcessLookupError:
        alive = False
    assert not alive


def test_sh_all_runs_jobs_at_once(tmp_path):
    jobs = [(f"sleep 1; echo {n}; exit {n}", tmp_path / f"c{n}" / "log", dict(os.environ)) for n in (3, 4)]
    start = time.time()
    assert sh_all(jobs, tmp_path, None) == [3, 4] and time.time() - start < 1.9
    assert [j[1].read_text() for j in jobs] == ["3\n", "4\n"]


def test_sh_all_timeout_kills_only_what_still_runs(tmp_path):
    jobs = [("exit 0", tmp_path / "fast", dict(os.environ)), ("sleep 30", tmp_path / "slow", dict(os.environ))]
    start = time.time()
    assert sh_all(jobs, tmp_path, 1) == [0, 124] and time.time() - start < 10
    assert "TIMEOUT after 1 s" in (tmp_path / "slow").read_text()


def test_sh_returns_exit_code(tmp_path):
    assert sh("echo hi; exit 3", tmp_path / "log", dict(os.environ), tmp_path, None) == 3
    assert (tmp_path / "log").read_text() == "hi\n"


def test_cli_unknown_only_key_exit_2(capsys):
    assert main(["--only", "nd:minimal,nd:nope"]) == 2
    assert "nd:nope" in capsys.readouterr().err


def test_cli_list_prints_keys(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    assert all(k in out for k in KEYS) and IDS["nd:ff1"] in out and "throttle" in out


def test_cli_bad_config_exit_2(tmp_path, capsys):
    bad = tmp_path / "c.json"
    bad.write_text("{}")
    assert main(["--config", str(bad), "--list"]) == 2
    assert "missing field" in capsys.readouterr().err


def test_cli_empty_selection_exit_2(capsys):
    assert main(["--skip", ",".join(KEYS)]) == 2
    assert "no steps selected" in capsys.readouterr().err


def test_caller_mgd_reaches_only_steps_that_set_one(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_MESH_GRAPH_DESC_PATH", "/caller/galaxy.textproto")
    _, _, _, shell, _ = _run(tmp_path, only=["nd:minimal", "nd:duty-cycle-bursty-5", "nd:mcast-ack"])
    assert "TT_MESH_GRAPH_DESC_PATH" not in shell.workload_calls("nd/minimal")[0]["env"]
    duty = shell.workload_calls("nd/duty-cycle-bursty-5")[0]["env"]["TT_MESH_GRAPH_DESC_PATH"]
    assert duty == CFG.steps[[s.key for s in CFG.steps].index("nd:duty-cycle-bursty-5")].env["TT_MESH_GRAPH_DESC_PATH"]
    chips = shell.chip_jobs
    assert len(chips) == len(CHIPS) and all("TT_MESH_GRAPH_DESC_PATH" not in c["env"] for c in chips)


def test_cli_config_tier_name(capsys):
    assert main(["--config", "daily", "--list"]) == 0
    assert capsys.readouterr().out.split("\n")[0] == "nd:minimal"


def test_cli_unknown_config_tier_exit_2(capsys):
    try:
        main(["--config", "nightly", "--list"])
    except SystemExit as e:
        assert e.code == 2
    else:
        raise AssertionError("argparse accepted an unknown tier")
    assert "unknown config tier 'nightly'; shipped tiers: daily, install, weekly" in capsys.readouterr().err


def test_cli_help_has_sbatch_note(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--propagate=NONE" in out and "sg slurmusers" in out


def test_runner_env_drops_device_selection_and_throttle(tmp_path, monkeypatch):
    # dev = rank among ALL the host's chips, and the first attempt must run unthrottled.
    inherited = {
        "TT_VISIBLE_DEVICES": "3",
        "TT_METAL_VISIBLE_DEVICES": "3",
        "TT_MM_THROTTLE_PERF": "5",
        "DIDT_DEVICE_ID": "3",
        "DIDT_MIN_LOOP_SECONDS": "9",
    }
    for k, v in inherited.items():
        monkeypatch.setenv(k, v)
    _, _, _, shell, _ = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    first = shell.workload_calls("nd/minimal")[0]["env"]
    assert not set(inherited) & set(first)
    assert shell.workload_calls("nd/minimal_throttle")[0]["env"]["TT_MM_THROTTLE_PERF"] == "5"
    assert all("TT_VISIBLE_DEVICES" not in c["env"] for c in shell.calls)


def test_chip_lost_at_reset_joins_on_new_snapshot(tmp_path):
    # tray 4 asic 2 (0000:82:00.0) gone after the reset: chip 20 is now 0000:86:00.0, not 0000:85:00.0
    chips31 = [c for c in CHIPS if c.bdf != "0000:82:00.0"]
    chips31 = [Chip(i, c.bdf, c.tray, c.asic, c.asic_id, c.kmd) for i, c in enumerate(chips31)]
    _, steps, doc, shell, _ = _run(
        tmp_path, only=["nd:mcast-ack"], outcomes={"nd/mcast-ack": {20: "nd"}}, snap=lambda: chips31
    )
    s = steps["nd:mcast-ack"]
    assert [j["env"]["TT_VISIBLE_DEVICES"] for j in shell.chip_calls("nd/mcast-ack")] == [str(i) for i in range(31)]
    assert s["devices"][0]["dev"] == 20 and s["devices"][0]["bdf"] == "0000:86:00.0"
    assert "chip set changed after reset: 31 chips, 32 at run start; gone 0000:82:00.0" in s["details"]
    assert len(doc["chips"]) == 32


def test_interrupt_marks_attempt_and_reraises(tmp_path):
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    shell = FakeShell(run_dir, outcomes={"nd/minimal": "interrupt"})
    try:
        run(CFG, run_dir, CHIPS, {"nd:minimal", "nd:ff1"}, set(), False, shell=shell)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interrupt swallowed")
    doc = json.loads((run_dir / "status.json").read_text())
    steps = {s["key"]: s for s in doc["steps"]}
    assert doc["current"] is None
    assert steps["nd:minimal"]["status"] == "FAIL" and steps["nd:minimal"]["reason"] == "interrupted"
    assert steps["nd:ff1"]["status"] == "PENDING"


def test_sh_interrupt_kills_the_process_group(tmp_path, monkeypatch):
    import subprocess

    pid_file = tmp_path / "pid"
    real_wait, calls = subprocess.Popen.wait, []

    def wait(self, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            for _ in range(50):
                if pid_file.is_file() and pid_file.read_text().strip():
                    break
                time.sleep(0.1)
            raise KeyboardInterrupt
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    try:
        sh(f"sleep 30 & echo $! > {pid_file}; wait", tmp_path / "log", dict(os.environ), tmp_path, None)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("interrupt swallowed")
    pid = int(pid_file.read_text())
    time.sleep(0.2)
    status = Path(f"/proc/{pid}/status")
    assert not status.is_file() or "zombie" in status.read_text()


def test_triage_hook_is_time_bounded(tmp_path):
    _, _, _, shell, _ = _run(tmp_path, only=["nd:minimal"])
    hook = shell.workload_calls("nd/minimal")[0]["env"]["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
    assert re.match(r"timeout \d+ \S+/python \S+/tools/tt-triage.py ", hook) and hook.endswith("; kill -9 $PPID")


PIN_MINIMAL = (
    "tests/didt/test_minimal_matmul.py::test_specific_chip_minimal_matmul[blackhole-bf16_HiFi2-galaxy-galaxy_chip_0]"
)


def _pins(steps, key):
    return [f for f in steps[key]["fallbacks"] if f["name"] == "pin_core"]


def test_pin_core_nd_reruns_every_failing_device(tmp_path):
    rc, steps, _, shell, run_dir = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    assert rc == 1 and steps["nd:minimal"]["status"] == "FAIL"
    pins = _pins(steps, "nd:minimal")
    assert [p["target"]["dev"] for p in pins] == [19, 26]
    p = pins[0]
    assert p["target"] == CHIPS[19].as_dict() and p["trigger"] == "nd"
    assert p["status"] == "WARN" and p["reason"] == "not reproduced on chip alone"
    # as long as the original loop ran, and at least twice its iterations
    assert p["env"] == {"DIDT_DEVICE_ID": "19", "DIDT_MIN_LOOP_SECONDS": "164.4"}
    assert p["loop"] == {"original_s": 164.4, "rerun_s": 170.0, "rerun_iterations": 2486}
    assert p["opts"] == ["--didt-workload-iterations=2000", "--determinism-check-interval=50", "--timeout=0"]
    assert p["iterations"] == {"--didt-workload-iterations": 2000}
    assert p["cores"]["first"][0]["x"] == 6 and p["cores"]["first"][0]["y"] == 8 and p["cores"]["rerun"] == []
    assert p["log"] == str(run_dir / "nd/minimal_pin_core_dev19/log")
    (call,) = shell.workload_calls("nd/minimal_pin_core_dev19")
    assert call["cmd"].startswith("pytest " + shlex.quote(PIN_MINIMAL) + " --didt-workload-iterations=2000")
    assert call["env"]["DIDT_DEVICE_ID"] == "19" and "TT_MM_THROTTLE_PERF" not in call["env"]
    assert call["env"]["DIDT_MIN_LOOP_SECONDS"] == "164.4"
    assert shell.reset_cmds("nd/minimal_pin_core_dev19") == ["tt-smi -glx_reset_auto"]
    # run.log shows the pin verdict, not the raw pytest result it was derived from
    assert "nd:minimal pin_core dev 19 result WARN not reproduced on chip alone" in (run_dir / "run.log").read_text()


def test_pin_core_reproduced_nd_is_fail_with_rerun_cores(tmp_path):
    _, steps, _, _, _ = _run(
        tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail", "nd/minimal_pin_core_dev19": "fail"}
    )
    p = _pins(steps, "nd:minimal")[0]
    assert p["status"] == "FAIL" and p["reason"].startswith("ND reproduced on chip alone")
    assert [d["dev"] for d in p["devices"]] == [19]
    assert p["cores"]["rerun"][0]["x"] == 6 and p["cores"]["rerun"][0]["failed_checks"] == 3


def test_pin_core_after_hang_resets_with_tt_smi_r(tmp_path):
    _, steps, _, shell, _ = _run(
        tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "hang", "nd/minimal_pin_core_dev27": "hang"}
    )
    (p,) = _pins(steps, "nd:minimal")
    assert p["trigger"] == "hang" and p["target"]["dev"] == 27 and p["status"] == "HANG"
    assert p["cores"]["first"][0] == {"x": 0, "y": 8, "translated": [1, 10]}
    assert shell.reset_cmds("nd/minimal_pin_core_dev27") == ["tt-smi -r"]
    # the hung attempt logged no iteration loop: doubled iterations only, and say so
    assert "DIDT_MIN_LOOP_SECONDS" not in p["env"] and p["loop"] == {}
    assert any(d.startswith("WARNING original loop time unknown") for d in p["details"])


def test_pin_core_reset_failure_aborts_the_run(tmp_path):
    rc, steps, doc, shell, _ = _run(
        tmp_path,
        only=["nd:minimal", "nd:ff1"],
        outcomes={"nd/minimal": "hang"},
        reset_fail={"nd/minimal_pin_core_dev27"},
    )
    (p,) = _pins(steps, "nd:minimal")
    assert p["status"] == "FAIL" and "run aborted" in p["reason"] and "tt-smi -r" in p["reason"]
    assert steps["nd:ff1"]["status"] == "SKIP" and steps["nd:ff1"]["reason"].startswith("run aborted")
    assert not shell.workload_calls("nd/ff1") and rc == 1 and doc["current"] is None


def test_pin_core_unsupported_step_warns(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:run-to-run"], outcomes={"nd/run-to-run": "r2r"})
    (p,) = _pins(steps, "nd:run-to-run")
    assert p["status"] == "SKIP" and p["reason"] == "unsupported fallback: nd:run-to-run (no per-chip test)"
    assert p["details"] == ["WARNING unsupported fallback: nd:run-to-run (no per-chip test)"]
    assert p["target"]["dev"] == 21 and p["trigger"] == "nd"


def test_each_chip_nd_only_fails_without_fallback(tmp_path):
    rc, steps, _, shell, run_dir = _run(
        tmp_path, only=["nd:mcast-ack-duty"], outcomes={"nd/mcast-ack-duty": {19: "nd"}}
    )
    s = steps["nd:mcast-ack-duty"]
    assert rc == 1 and s["status"] == "FAIL" and s["fallbacks"] == []
    assert [d["dev"] for d in s["devices"]] == [19] and s["devices"][0]["failed_checks"] == 3
    assert not (run_dir / "nd/mcast-ack-duty_throttle_hang").exists() and len(shell.chip_jobs) == 32


def test_each_chip_pass_runs_no_fallback(tmp_path):
    _, steps, _, shell, _ = _run(tmp_path, only=["nd:mcast-ack"])
    assert steps["nd:mcast-ack"]["status"] == "PASS" and steps["nd:mcast-ack"]["fallbacks"] == []
    assert len(shell.chip_jobs) == 32


def test_each_chip_hang_runs_throttle_hang_on_every_chip(tmp_path):
    rc, steps, _, shell, _ = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes={"nd/mcast-ack-duty": "fail"})
    s = steps["nd:mcast-ack-duty"]
    assert rc == 1 and s["status"] == "HANG"
    (f,) = s["fallbacks"]
    assert (f["name"], f["status"], f["reason"]) == ("throttle_hang", "PASS", "32/32 chips passed")
    assert f["env"] == {"TT_MM_THROTTLE_PERF": "5", "MATMUL_ITERS": "100"} and f["devices"] == []
    first, throttled = shell.chip_calls("nd/mcast-ack-duty"), shell.chip_calls("nd/mcast-ack-duty_throttle_hang")
    assert len(first) == len(throttled) == 32
    assert {(j["env"]["MATMUL_ITERS"], j["env"].get("TT_MM_THROTTLE_PERF")) for j in first} == {("10", None)}
    assert {(j["env"]["MATMUL_ITERS"], j["env"]["TT_MM_THROTTLE_PERF"]) for j in throttled} == {("100", "5")}
    assert shell.reset_cmds("nd/mcast-ack-duty_throttle_hang") == ["tt-smi -glx_reset_auto"]


def test_throttle_hang_nd_is_fail_with_device_rows(tmp_path):
    outcomes = {"nd/mcast-ack-duty": "fail", "nd/mcast-ack-duty_throttle_hang": {19: "nd"}}
    _, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes=outcomes)
    (f,) = steps["nd:mcast-ack-duty"]["fallbacks"]
    assert (f["status"], f["reason"]) == ("FAIL", "failed on 1/32 chips")
    (d,) = f["devices"]
    assert (d["dev"], d["bdf"], d["failed_checks"], d["checks"]) == (19, CHIPS[19].bdf, 3, 20)


def test_throttle_hang_hang_persists_keeps_triage(tmp_path):
    outcomes = {"nd/mcast-ack-duty": "fail", "nd/mcast-ack-duty_throttle_hang": "hung2"}
    _, steps, _, _, run_dir = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes=outcomes)
    (f,) = steps["nd:mcast-ack-duty"]["fallbacks"]
    assert (f["status"], f["reason"]) == ("HANG", "hang persists at throttle 5")
    assert [(d["dev"], d["hung"], d["op"]) for d in f["devices"]] == [
        (3, True, "MatmulDeviceOperation"),
        (7, True, "MatmulDeviceOperation"),
    ]
    assert (run_dir / "nd/mcast-ack-duty_throttle_hang/chips/chip7/crash_triage.csv").is_file()
    run_log = (run_dir / "run.log").read_text()
    assert "nd:mcast-ack-duty throttle_hang HANG hung on 2/32 chips" in run_log
    assert "nd:mcast-ack-duty throttle_hang result HANG hang persists at throttle 5" in run_log


def test_pin_id_that_does_not_collect_skips_only_the_pin(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"}, missing={PIN_MINIMAL})
    assert steps["nd:minimal"]["reason"] == "AssertionError: non-deterministic outputs"
    pins = _pins(steps, "nd:minimal")
    assert {p["status"] for p in pins} == {"SKIP"}
    assert pins[0]["reason"] == f"unsupported fallback: nd:minimal (pin id does not collect: {PIN_MINIMAL})"


def test_pin_core_passing_step_runs_no_pin(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:minimal"])
    assert steps["nd:minimal"]["fallbacks"] == []


def test_double_iterations():
    opts, env, used = double_iterations(
        ["--didt-workload-iterations=10", "--determinism-check-interval=1", "--timeout", "1800"], {"X": "1"}
    )
    assert opts == ["--didt-workload-iterations=20", "--determinism-check-interval=1", "--timeout", "1800"]
    assert env == {} and used == {"--didt-workload-iterations": 20}
    opts, env, used = double_iterations(
        ["--didt-workload-iterations", "7"], {"TT_DET_SWEEP_ITERS": "10", "MATMUL_ITERS": "2"}
    )
    assert opts == ["--didt-workload-iterations", "14"]
    assert env == {"TT_DET_SWEEP_ITERS": "20", "MATMUL_ITERS": "4"}
    assert used == {"--didt-workload-iterations": 14, "TT_DET_SWEEP_ITERS": 20, "MATMUL_ITERS": 4}
    assert double_iterations(["--timeout=0"], {}) == (["--timeout=0"], {}, {})
