# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""perf_diagnose.py: one measure-only wave diagnosis of the kernel in the tree."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent / "perf_diagnose.py"

FAKE_LAUNCHER = textwrap.dedent(
    """
    import json, os, sys
    from pathlib import Path
    args = sys.argv[1:]
    Path(os.environ["FAKE_ARGV_LOG"]).write_text(json.dumps(args))
    mode = os.environ.get("FAKE_MODE", "ok")
    if mode == "fail":
        print("perf-capture: run_test.sh exited 1 and dumped no profiler zones", file=sys.stderr)
        sys.exit(2)
    out = Path(args[args.index("--out-dir") + 1]); out.mkdir(parents=True, exist_ok=True)
    rt = args[args.index("--run-type") + 1]
    llk = Path(os.environ["FAKE_LLK"])
    csv_dir = llk / "perf_data" / "runs" / "local-1" / "perf_x_quasar"; csv_dir.mkdir(parents=True)
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
    env["out"].mkdir(parents=True)
    (env["out"] / "stale.txt").write_text("old")
    _run(env)
    assert not (env["out"] / "stale.txt").exists()
