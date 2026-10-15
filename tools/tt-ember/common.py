# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by auto.py and the sweep scripts (run_all.py, run_power_cases.py)."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# Seconds to wait after `tt-smi -r` before the next run, so the device has re-initialized and
# its current has settled.
RESET_COOLDOWN_S = 10

# Written by auto.py into the run directory only after the parser succeeded. A sweep treats a
# run as done only if this file exists and records the same settings it is about to use.
RUN_COMPLETE_FILE = "run_complete.json"

# Default --trim-ms. run_power_cases.py uses 1 ms instead: the published POWER_CASE results were
# made with it, and changing it would change them.
DEFAULT_TRIM_MS = 6.0
POWER_CASES_TRIM_MS = 1.0

# auto.py's own exit codes. They are kept apart from the parser's (2-11), which auto.py passes
# through unchanged, and from the app's, which it also passes through.
EXIT_TELEMETRY_START = 20
EXIT_TELEMETRY_FILE_MISSING = 21
EXIT_SUMMARY_MISSING = 22
EXIT_TELEMETRY_DIED = 23
# parser.py's exit codes that mean the outputs were written but the run can't be used.
EXIT_RUN_INCOMPLETE = 10
EXIT_MULTIPLE_DEVICES = 11

EXIT_REASONS = {
    EXIT_RUN_INCOMPLETE: "parser: incomplete run (a grid with no samples, no base current, or lost profiler markers)",
    EXIT_MULTIPLE_DEVICES: "parser: telemetry from several devices and no --device-id",
    EXIT_TELEMETRY_START: "telemetry exited right after it started",
    EXIT_TELEMETRY_FILE_MISSING: "telemetry wrote no output file",
    EXIT_SUMMARY_MISSING: "the app wrote no output",
    EXIT_TELEMETRY_DIED: "telemetry exited before the run ended",
}

SUBDIR_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")

LONG_MATMUL_APP = "metal_example_long_matmul"

# Environment variables that change what is measured: long_matmul's knobs (see its README), and
# the debug features that add work on the device (the profiler, watcher and DPRINT).
WORKLOAD_ENV_VARS = (
    "POWER_CASE",
    "LONG_MATMUL_MODE",
    "LONG_MATMUL_OP",
    "LONG_MATMUL_BLOCK_M",
    "LONG_MATMUL_BLOCK_N",
    "LONG_MATMUL_DISABLE_READER",
    "LONG_MATMUL_DISABLE_COMPUTE",
    "LONG_MATMUL_DISABLE_WRITER",
    "LONG_MATMUL_WRITE_AMPLIFICATION_PCT",
    "TT_METAL_DEVICE_PROFILER",
    "TT_METAL_WATCHER",
    "TT_METAL_DPRINT_CORES",
)


def exit_reason(rc: int) -> str:
    """The return code, plus what it means when it is one of the codes above."""
    reason = EXIT_REASONS.get(rc)
    return f"{rc} ({reason})" if reason else str(rc)


def subdir_name(text: str) -> str:
    """argparse type for --subdir: one plain directory name."""
    if not SUBDIR_PATTERN.fullmatch(text) or text in (".", ".."):
        raise argparse.ArgumentTypeError(
            f"invalid subdir {text!r}: use one directory name made of letters, digits, '_', '-' and '.'"
        )
    return text


def long_matmul_args(app_args: List[str]) -> Tuple[int, int, int, int, int]:
    """Parse long_matmul's app args (M N K iterations [fixed_blocks_per_core]) as integers;
    fixed_blocks_per_core is 0 (split mode) when not given. Raises ValueError otherwise."""
    if len(app_args) not in (4, 5):
        raise ValueError("long_matmul needs --app-args M N K iterations [fixed_blocks_per_core]")
    values = [int(a) for a in app_args] + [0] * (5 - len(app_args))
    if any(v < 0 for v in values):
        raise ValueError("long_matmul's --app-args must not be negative")
    m, n, k, iterations, fixed_blocks_per_core = values
    return m, n, k, iterations, fixed_blocks_per_core


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def reset_hardware(tt_smi_cmd: str, dry_run: bool = False) -> int:
    print(f"\n[RESET] Running hardware reset: {tt_smi_cmd} -r", flush=True)
    if dry_run:
        print("[RESET] (dry-run) skipped", flush=True)
        return 0
    result = subprocess.run([tt_smi_cmd, "-r"])
    if result.returncode != 0:
        print(f"[RESET] ERROR: Hardware reset failed with return code {result.returncode}.", file=sys.stderr)
        return result.returncode
    print(f"[RESET] Hardware reset completed. Waiting {RESET_COOLDOWN_S}s for device to re-initialize...", flush=True)
    for remaining in range(RESET_COOLDOWN_S, 0, -1):
        print(f"[RESET] Resuming in {remaining}s...", flush=True)
        time.sleep(1)
    print("[RESET] Device ready.", flush=True)
    return 0


def run_settings(
    app_args: List[str],
    env: Mapping[str, str],
    telemetry_freq_hz: int,
    trim_ms: float,
    slot_ms: int,
    device_id: Optional[int],
) -> Dict[str, object]:
    """Everything that decides what a run measured, as recorded in RUN_COMPLETE_FILE."""
    return {
        "app_args": [str(a) for a in app_args],
        "env": {k: env[k] for k in WORKLOAD_ENV_VARS if env.get(k)},
        "telemetry_freq_hz": int(telemetry_freq_hz),
        "trim_ms": float(trim_ms),
        "slot_ms": int(slot_ms),
        "device_id": device_id,
    }


def write_run_complete(run_dir: Path, settings: Dict[str, object]) -> None:
    (run_dir / RUN_COMPLETE_FILE).write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_status(run_dir: Path, settings: Dict[str, object]) -> str:
    """'done' if run_dir holds a completed run with these settings, 'missing' if it holds no
    completed run, or 'different' if it holds one made with other settings."""
    marker = run_dir / RUN_COMPLETE_FILE
    if not marker.is_file():
        return "missing"
    try:
        recorded = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "missing"
    return "done" if recorded == settings else "different"


def auto_py_command(
    auto_script: Path,
    telemetry_exe: Path,
    telemetry_freq: int,
    app_exe: Path,
    app_args: List[str],
    parser_script: Path,
    tt_venv_activate: Path,
    tt_metal_root: Path,
    output_root: Path,
    subdir: str,
    slot_ms: int,
    device_id: Optional[int],
    trim_ms: float,
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(auto_script),
        "--telemetry-exe",
        str(telemetry_exe),
        "--telemetry-freq",
        str(telemetry_freq),
        "--app-exe",
        str(app_exe),
        "--parser-script",
        str(parser_script),
        "--tt-venv-activate",
        str(tt_venv_activate),
        "--tt-metal-root",
        str(tt_metal_root),
        "--output-root",
        str(output_root),
        "--subdir",
        subdir,
        "--slot-ms",
        str(slot_ms),
        "--trim-ms",
        str(trim_ms),
    ]
    if device_id is not None:
        cmd += ["--device-id", str(device_id)]
    # --app-args must be last: auto.py passes everything after it to the app.
    return cmd + ["--app-args"] + [str(a) for a in app_args]


def environ_with(overrides: Mapping[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(overrides)
    return env
