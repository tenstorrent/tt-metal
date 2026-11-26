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
from tests.didt.runner.steps import run, sh

TESTDATA = Path(__file__).parent / "testdata"
SHIPPED = Path(__file__).parents[1] / "didt_runner_config.json"
TRAY_BUS_HIGH = {1: 0x0, 2: 0x4, 3: 0xC, 4: 0x8}
CFG = load(SHIPPED)
KEYS = [s.key for s in CFG.steps]
IDS = {s.key: s.ids[0] for s in CFG.steps}


def _chips() -> list[Chip]:
    rows = sorted((f"0000:{(TRAY_BUS_HIGH[t] << 4) | a:02x}:00.0", t, a) for t in range(1, 5) for a in range(1, 9))
    return [Chip(i, bdf, t, a, f"0x{t:02x}{a:014x}", (t - 1) * 8 + a - 1) for i, (bdf, t, a) in enumerate(rows)]


CHIPS = _chips()


class FakeShell:
    """Stands in for every command the runner launches. outcomes: attempt dir (relative to the run dir) ->
    pass | fail | hang | timeout | raise. Collect output mimics pytest: one missing id hides the whole listing."""

    def __init__(self, run_dir, outcomes=None, missing=(), reset_fail=(), smi=None):
        self.run_dir, self.outcomes, self.missing = run_dir, outcomes or {}, set(missing)
        self.reset_fail, self.smi, self.calls = set(reset_fail), smi or {}, []

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
        if "glx_reset" in cmd:
            log.write_text("reset\n")
            return 1 if rel in self.reset_fail else 0
        if cmd.startswith("tt-smi -s"):
            log.write_text(json.dumps(self.smi.get(rel, {"device_info": []})))
            return 0
        outcome = self.outcomes.get(rel, "pass")
        script = "test_mm_mcast_write_ack_failure.py" in cmd and not cmd.startswith("pytest")
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
        if script:
            work = Path(env["MCAST_ACK_WORK_DIR"])
            (work / "chip3").mkdir(parents=True)
            (Path(env["MCAST_ACK_CACHE_DIR"]) / "kmd2").mkdir(parents=True)
            (work / "chip3" / "pytest.log").write_text("1 passed\n")
            if outcome == "build":
                log.write_text(
                    "chip 12 FAIL: TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages:\nchip 1 PASS\n"
                )
                return 1
            if outcome == "fail":
                log.write_text("chip 3 FAIL: HUNG chip 3 (runtime id 0): BusDeviceFunction=0000:04:00.0\nchip 1 PASS\n")
                return 1
            log.write_text("chip 3 PASS\nchip 1 PASS\nall 2 chips passed\n")
            return 0
        if outcome == "fail":
            log.write_text((TESTDATA / "nd_report.log").read_text())
            return 1
        log.write_text("skipped windows: 0 -> []\n=== 1 passed in 9.00s ===\n")
        return 0

    def workload_calls(self, rel):
        return [
            c for c in self.calls if str(c["log"].parent.relative_to(self.run_dir)) == rel and c["log"].name == "log"
        ]


def _run(tmp_path, only=None, skip=(), co=False, snap=None, **fake):
    run_dir = tmp_path / "LOG_2026_11_26_12_00_host_abcdef12345"
    run_dir.mkdir()
    shell = FakeShell(run_dir, **fake)
    rc = run(CFG, run_dir, CHIPS, set(only) if only else None, set(skip), co, shell=shell, snap=snap)
    doc = json.loads((run_dir / "status.json").read_text())
    return rc, {s["key"]: s for s in doc["steps"]}, doc, shell, run_dir


def test_fail_runs_every_fallback_status_stays_first(tmp_path):
    rc, steps, _, shell, _ = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    s = steps["nd:minimal"]
    assert rc == 1 and s["status"] == "FAIL"
    assert [(f["name"], f["status"]) for f in s["fallbacks"]] == [("throttle", "PASS")]
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
    assert env["TT_METAL_OPERATION_TIMEOUT_SECONDS"] == "120"
    hook = env["TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE"]
    assert "tools/tt-triage.py" in hook and "--run=dump_running_operations" in hook and "--run=dump_op_mesh" in hook
    assert f"--llm-output-path={run_dir / 'nd/minimal/crash_triage.csv'};" in hook and hook.endswith("kill -9 $PPID")


def test_wall_timeout_is_fail(tmp_path):
    rc, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack"], outcomes={"nd/mcast-ack": "timeout"})
    s = steps["nd:mcast-ack"]
    assert rc == 1 and s["status"] == "FAIL" and s["reason"] == "timeout after 1800 s"


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


def test_script_runner_work_dir(tmp_path):
    rc, steps, _, shell, run_dir = _run(tmp_path, only=["nd:mcast-ack"])
    (call,) = shell.workload_calls("nd/mcast-ack")
    assert call["cmd"] == "python tests/didt/test_mm_mcast_write_ack_failure.py"
    assert call["env"]["MCAST_ACK_WORK_DIR"] == str(run_dir / "nd/mcast-ack/chips")
    assert call["env"]["MATMUL_WORKLOAD"] == "wkv_b1" and call["timeout"] == 1800
    assert "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE" not in call["env"]
    assert (run_dir / "nd/mcast-ack/chips/chip3/pytest.log").is_file()
    # kernel caches on local disk, one dir per attempt, gone afterwards; logs stay in the run dir
    cache = Path(tempfile.gettempdir()) / f"didt_runner_{run_dir.name}"
    assert call["env"]["MCAST_ACK_CACHE_DIR"] == str(cache / "nd/mcast-ack")
    assert not cache.exists()
    assert rc == 0 and steps["nd:mcast-ack"]["status"] == "PASS"


def test_write_ack_hang_names_chip(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack-duty"], outcomes={"nd/mcast-ack-duty": "fail"})
    s = steps["nd:mcast-ack-duty"]
    assert s["status"] == "FAIL" and s["devices"] == [{**CHIPS[3].as_dict(), "hung": True}]


def test_write_ack_other_failure_keeps_chip_reason(tmp_path):
    _, steps, _, _, _ = _run(tmp_path, only=["nd:mcast-ack"], outcomes={"nd/mcast-ack": "build"})
    s = steps["nd:mcast-ack"]
    assert s["devices"] == [CHIPS[12].as_dict()]
    assert s["details"] == [
        "device 12 bdf=0000:45:00.0 tray=2 asic=5: TT_THROW: trisc0 build failed. Log: /tmp/x.s: Assembler messages:"
    ]


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


def test_cli_help_has_sbatch_note(capsys):
    try:
        main(["--help"])
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "--propagate=NONE" in out and "sg slurmusers" in out


def test_runner_env_drops_device_selection_and_throttle(tmp_path, monkeypatch):
    # dev = rank among ALL the host's chips, and the first attempt must run unthrottled.
    for k, v in {"TT_VISIBLE_DEVICES": "3", "TT_METAL_VISIBLE_DEVICES": "3", "TT_MM_THROTTLE_PERF": "5"}.items():
        monkeypatch.setenv(k, v)
    _, _, _, shell, _ = _run(tmp_path, only=["nd:minimal"], outcomes={"nd/minimal": "fail"})
    first = shell.workload_calls("nd/minimal")[0]["env"]
    assert not {"TT_VISIBLE_DEVICES", "TT_METAL_VISIBLE_DEVICES", "TT_MM_THROTTLE_PERF"} & set(first)
    assert shell.workload_calls("nd/minimal_throttle")[0]["env"]["TT_MM_THROTTLE_PERF"] == "5"
    assert all("TT_VISIBLE_DEVICES" not in c["env"] for c in shell.calls)


def test_chip_lost_at_reset_joins_on_new_snapshot(tmp_path):
    # tray 4 asic 2 (0000:82:00.0) gone after the reset: chip 20 is now 0000:86:00.0, not 0000:85:00.0
    chips31 = [c for c in CHIPS if c.bdf != "0000:82:00.0"]
    chips31 = [Chip(i, c.bdf, c.tray, c.asic, c.asic_id, c.kmd) for i, c in enumerate(chips31)]

    class Hung20(FakeShell):
        def __call__(self, cmd, log, env, cwd, timeout):
            rc = super().__call__(cmd, log, env, cwd, timeout)
            if log.name == "log":
                log.write_text("chip 20 FAIL: HUNG chip 20 (runtime id 0): BusDeviceFunction=0000:86:00.0\n")
                return 1
            return rc

    run_dir = tmp_path / "r"
    run_dir.mkdir()
    run(CFG, run_dir, CHIPS, {"nd:mcast-ack"}, set(), False, shell=Hung20(run_dir), snap=lambda: chips31)
    doc = json.loads((run_dir / "status.json").read_text())
    (s,) = [s for s in doc["steps"] if s["key"] == "nd:mcast-ack"]
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
