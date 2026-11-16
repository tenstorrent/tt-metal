# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_diagnose.py: one measure-only wave diagnosis of the kernel in the tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "perf_diagnose.py"

FAKE_LAUNCHER = textwrap.dedent(
    """
    import json, os, socket, subprocess, sys, time
    from pathlib import Path
    args = sys.argv[1:]
    Path(os.environ["FAKE_ARGV_LOG"]).write_text(json.dumps(args))
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "fail":
        print("perf-capture: run_test.sh exited 1 and dumped no profiler zones", file=sys.stderr)
        sys.exit(2)
    if mode == "hang":  # an emulator run that outlives the timeout, with its dump on disk
        zrun = Path(os.environ["FAKE_SPACE"]) / "verification/emu/emu_out/1x3/test_umd_remote_1x3_0/zrun.7"
        zrun.mkdir(parents=True)
        (zrun / "waves.ztdb").write_text("")
        host = socket.gethostname().split("-special-", 1)[0]
        (zrun / "env.log").write_text(f"NNG_SOCKET_ADDR=tcp://{host}:5555\\n")
        child = subprocess.Popen(["sleep", "600"])  # same process group, like run_test.sh
        Path(os.environ["FAKE_CHILD_PID"]).write_text(str(child.pid))
        time.sleep(600)
    out = Path(args[args.index("--out-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
    rt = args[args.index("--run-type") + 1]
    llk = Path(os.environ["FAKE_LLK"])
    # Like the perf harness: the report goes to perf_data/runs/$PERF_RUN_TAG, latest points at it.
    tag = os.environ["PERF_RUN_TAG"]
    csv_dir = llk / "perf_data" / "runs" / tag / "perf_x_quasar"; csv_dir.mkdir(parents=True)
    latest = llk / "perf_data" / "latest"
    if latest.is_symlink():
        latest.unlink()
    latest.symlink_to(Path("runs") / tag, target_is_directory=True)
    per_tile = os.environ.get("FAKE_CSV_PER_TILE", "267.3")
    (csv_dir / "perf_x_quasar.post.csv").write_text(
        "mathop,marker,loop_factor,tile_cnt,mean(MATH_ISOLATE),mean(L1_TO_L1)\\n"
        f"Sigmoid,TILE_LOOP,32,4,{per_tile},{per_tile}\\n"
        "Sigmoid,KERNEL,32,4,1,1\\n")
    (out / "waves.vcd.gz").write_bytes(b"x" * 1000)
    # 128 tiles x 267.3 cycles: 230 issue, 32 bubble after SFPNONLINEAR, 4 sfpu stall, rest idle
    t = 128
    report = {"status": "complete", "missing_signals": [], "missing_core_signals": [],
              "window": {"cycles": int(267.3 * t), "begin_fs": 0, "end_fs": 1, "source": "wallclock"},
              "threads": {"math": {"issue": 230 * t, "stall": {"sfpu": 4 * t}, "idle": int(33.3 * t),
                                   "unknown": 0, "instructions_issued": 230 * t,
                                   "instructions_by_opcode": {},
                                   "bubbles": [{"after": "SFPNONLINEAR", "before": "SFPSTORE",
                                                "count": 32 * t, "cycles": 32 * t}]}}}
    if mode == "pack_bound":  # pack_untilize-like L1_TO_L1: pack held by its busy packer
        report["window"]["cycles"] = int(267.3 * t)
        report["threads"] = {
            "math": {"issue": 58 * t, "stall": {}, "idle": int(209.3 * t), "unknown": 0,
                     "instructions_issued": 58 * t, "instructions_by_opcode": {}, "bubbles": []},
            "pack": {"issue": 40 * t, "stall": {"instissue": 222 * t}, "idle": int(5.3 * t), "unknown": 0,
                     "instructions_issued": 40 * t, "instructions_by_opcode": {}, "bubbles": []},
        }
    if mode == "null_field":  # a field the judge does arithmetic on is null
        report["threads"]["math"]["issue"] = None
    (out / f"perf_{rt}.json").write_text(json.dumps(report))
    (out / "perf_capture.json").write_text(json.dumps({"status": "ok"}))
    """
)


@pytest.fixture
def env(tmp_path):
    llk = tmp_path / "wt" / "tt_metal" / "tt-llk"
    llk.mkdir(parents=True)
    launcher = tmp_path / "launcher.py"
    launcher.write_text(FAKE_LAUNCHER)
    private = tmp_path / "private"
    entry = private / "tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("")
    return {
        "llk": llk,
        "launcher": launcher,
        "private": private,
        "out": tmp_path / "log" / "perf" / "diag",
        "argv_log": tmp_path / "argv.json",
        "space": tmp_path,
    }


def _run(env, *extra, mode="ok", per_tile="267.3"):
    full = dict(os.environ)
    full.update(
        FAKE_MODE=mode,
        FAKE_LLK=str(env["llk"]),
        FAKE_ARGV_LOG=str(env["argv_log"]),
        FAKE_CSV_PER_TILE=per_tile,
        FAKE_SPACE=str(env["space"]),
        FAKE_CHILD_PID=str(env["space"] / "child.pid"),
    )
    argv = [
        sys.executable,
        str(SCRIPT),
        "--llk",
        str(env["llk"]),
        "--module",
        "perf_x_quasar.py",
        "--test-id",
        "test_x[Sigmoid]",
        "--run-type",
        "MATH_ISOLATE",
        "--out-dir",
        str(env["out"]),
        "--launcher",
        str(env["launcher"]),
        "--private-root",
        str(env["private"]),
        "--space-path",
        str(env["space"]),
        *extra,
    ]
    done = subprocess.run(argv, env=full, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    lines = [l for l in done.stdout.splitlines() if l.startswith("PERF_DIAG ")]
    assert len(lines) == 1, done.stdout
    return lines[0]


def test_ok_reports_floor_headroom_and_top_cause(env):
    line = _run(env)
    assert "status=ok" in line
    assert "run_type=MATH_ISOLATE" in line
    assert "floor=230.0" in line
    assert "top=math.bubble.SFPNONLINEAR->SFPSTORE:32.0" in line
    diag = json.loads((env["out"] / "diag.json").read_text())
    assert diag["status"] == "ok"
    assert diag["top_causes"][0] == ["math.bubble.SFPNONLINEAR->SFPSTORE", 32.0]
    assert diag["reconcile"]["ok"] is True


def test_capture_is_one_variant_one_run_type(env):
    _run(env)
    argv = json.loads(env["argv_log"].read_text())
    assert argv[0] == "perf-capture"
    assert argv[argv.index("--test-id") + 1] == "test_x[Sigmoid]"
    assert argv[argv.index("--run-type") + 1] == "MATH_ISOLATE"
    assert argv[argv.index("--worktree") + 1] == str(env["llk"])


def test_space_is_kept_small(env):
    _run(env)
    assert not (
        env["out"] / "waves.vcd.gz"
    ).exists(), "the wave file is deleted after the verdict"
    assert not (
        env["llk"] / "perf_data"
    ).exists(), "perf_data in the worktree is cleaned"
    assert (env["out"] / "perf.post.csv").is_file()
    total = sum(p.stat().st_size for p in env["out"].rglob("*") if p.is_file())
    assert total < 64 * 1024


def test_keep_waves(env):
    _run(env, "--keep-waves")
    assert (env["out"] / "waves.vcd.gz").exists()


def test_low_disk_skips_without_capturing(env):
    line = _run(env, "--min-free-gb", str(10**9))
    assert "status=skipped" in line and "free" in line
    assert not env["argv_log"].exists()


def test_missing_private_tool_skips(env):
    (env["private"] / "tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py").unlink()
    line = _run(env)
    assert "status=skipped" in line and "private" in line
    assert not env["argv_log"].exists()


def test_capture_failure_is_reported_not_raised(env):
    line = _run(env, mode="fail")
    assert "status=failed" in line
    assert "no profiler zones" in line
    assert json.loads((env["out"] / "diag.json").read_text())["status"] == "failed"


def test_not_reconciled_is_flagged(env):
    line = _run(env, per_tile="200.0")
    assert "status=not_reconciled" in line


def test_rerun_replaces_the_previous_diagnosis(env):
    _run(env, mode="fail")
    (env["out"] / "stale.txt").write_text("old")
    _run(env)
    assert not (env["out"] / "stale.txt").exists()


def test_a_dir_it_did_not_write_is_never_deleted(env):
    env["out"].mkdir(parents=True)
    (env["out"] / "keep.txt").write_text("someone's data")
    line = _run(env)
    assert "status=failed" in line and "not deleting" in line
    assert (env["out"] / "keep.txt").read_text() == "someone's data"
    assert sorted(p.name for p in env["out"].iterdir()) == ["keep.txt"]
    assert not env["argv_log"].exists(), "no capture into a refused dir"


def test_other_perf_data_runs_and_latest_are_kept(env):
    perf_data = env["llk"] / "perf_data"
    baseline = perf_data / "runs" / "local-baseline" / "perf_x_quasar"
    baseline.mkdir(parents=True)
    (baseline / "perf_x_quasar.post.csv").write_text("kept\n")
    (perf_data / "latest").symlink_to(Path("runs") / "local-baseline")
    line = _run(env)
    assert "status=ok" in line
    assert (baseline / "perf_x_quasar.post.csv").read_text() == "kept\n"
    assert [p.name for p in (perf_data / "runs").iterdir()] == ["local-baseline"]
    assert os.readlink(perf_data / "latest") == str(Path("runs") / "local-baseline")


def test_timeout_stops_the_process_group_and_removes_the_dump(env):
    line = _run(env, "--timeout", "3", mode="hang")
    assert "status=failed" in line and "timed out" in line
    child = int((env["space"] / "child.pid").read_text())
    for _ in range(50):  # the orphaned child is reaped by init shortly after the kill
        try:
            state = Path(f"/proc/{child}/stat").read_text().split()[2]
        except FileNotFoundError:
            break
        if state == "Z":
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"capture child {child} still running after the timeout")
    zrun = env["space"] / "verification/emu/emu_out/1x3/test_umd_remote_1x3_0/zrun.7"
    assert not zrun.exists(), "the timed-out run's emulator dump is deleted"


def test_an_unexpected_error_is_still_one_failed_line(env):
    line = _run(env, mode="null_field")
    assert "status=failed" in line and "TypeError" in line
    assert json.loads((env["out"] / "diag.json").read_text())["status"] == "failed"


def test_a_unit_bound_kernel_is_reported_but_not_judged(env):
    line = _run(env, "--run-type", "L1_TO_L1", mode="pack_bound")
    assert "status=ok" in line
    assert "bound=unit(pack)" in line
    assert "stall_headroom=n/a" in line
    diag = json.loads((env["out"] / "diag.json").read_text())
    assert diag["bound"]["kind"] == "unit" and diag["stall_headroom_pct"] is None


def test_an_issue_bound_kernel_reports_its_bound(env):
    line = _run(env)
    assert "bound=issue(math)" in line
