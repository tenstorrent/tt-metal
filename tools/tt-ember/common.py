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

# Defaults the sweeps forward to auto.py, so all scripts measure with the same settings.
DEFAULT_TELEMETRY_FREQ_HZ = 50
DEFAULT_SLOT_MS = 1
DEFAULT_DEVICE_ID = 0
DEFAULT_DPI = 150

# parser.py's exit codes (2-11). auto.py passes them through unchanged, as it does the app's.
# 2 is also argparse's code for a bad flag; both mean the parser was not run on usable input.
EXIT_INPUT_NOT_FOUND = 2
EXIT_NO_TELEMETRY_LINES = 3
EXIT_NO_DEVICE_SAMPLES = 4
EXIT_NO_SLOTS = 5
EXIT_PROGRAM_LOG_NOT_FOUND = 6
EXIT_NO_PROGRAM_ROWS = 7
EXIT_PROFILER_CSV_NOT_FOUND = 8
EXIT_PROFILER_CSV_WITHOUT_PROGRAM_LOG = 9
# The outputs are written, but the run can't be used.
EXIT_RUN_INCOMPLETE = 10
# Stopped before writing results.
EXIT_MULTIPLE_DEVICES = 11

# auto.py's own exit codes, kept apart from the parser's.
EXIT_TELEMETRY_START = 20
EXIT_TELEMETRY_FILE_MISSING = 21
EXIT_SUMMARY_MISSING = 22
EXIT_TELEMETRY_DIED = 23
EXIT_PROFILER_CSV_MISSING = 24

EXIT_REASONS = {
    EXIT_INPUT_NOT_FOUND: "parser: telemetry file not found, or a bad flag",
    EXIT_NO_TELEMETRY_LINES: "parser: no telemetry lines matched",
    EXIT_NO_DEVICE_SAMPLES: "parser: no telemetry samples for the device",
    EXIT_NO_SLOTS: "parser: no time slots",
    EXIT_PROGRAM_LOG_NOT_FOUND: "parser: program log not found",
    EXIT_NO_PROGRAM_ROWS: "parser: no grid rows in the program log",
    EXIT_PROFILER_CSV_NOT_FOUND: "parser: device profiler CSV not found",
    EXIT_PROFILER_CSV_WITHOUT_PROGRAM_LOG: "parser: --device-profiler-csv without --program-log",
    EXIT_RUN_INCOMPLETE: "parser: incomplete run (a grid with no samples, no base current, or lost profiler markers)",
    EXIT_MULTIPLE_DEVICES: "parser: telemetry from several devices and no --device-id",
    EXIT_TELEMETRY_START: "telemetry exited right after it started",
    EXIT_TELEMETRY_FILE_MISSING: "telemetry wrote no output file",
    EXIT_SUMMARY_MISSING: "the app wrote no output",
    EXIT_TELEMETRY_DIED: "telemetry exited before the run ended",
    EXIT_PROFILER_CSV_MISSING: "TT_METAL_DEVICE_PROFILER is set, but the app wrote no profiler CSV",
}

SUBDIR_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")

LONG_MATMUL_APP = "metal_example_long_matmul"

# Environment variables that change what is measured: long_matmul's knobs (see its README), and
# the tt-metal options that change the device code or add work on the device (the profilers,
# watcher, DPRINT, kernel asserts and the null/early-return kernels).
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
    "TT_METAL_DEVICE_PROFILER_DISPATCH",
    "TT_METAL_DEVICE_PROFILER_NOC_EVENTS",
    "TT_METAL_STREAMING_PROFILER",
    "TT_METAL_WATCHER",
    "TT_METAL_DPRINT_CORES",
    "TT_METAL_LIGHTWEIGHT_KERNEL_ASSERTS",
    "TT_METAL_LLK_ASSERTS",
    "TT_METAL_NULL_KERNELS",
    "TT_METAL_KERNELS_EARLY_RETURN",
    "TT_METAL_SLOW_DISPATCH_MODE",
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


def misplaced_options(parser: argparse.ArgumentParser, app_args: List[str]) -> List[str]:
    """Options of this script written after --app-args, which takes everything after it and
    would otherwise pass them to the app silently."""
    return [a for a in app_args if a.split("=", 1)[0] in parser._option_string_actions]


def check_long_matmul_device(app_exe: Path, device_id: Optional[int]) -> None:
    """Raise ValueError if device_id names a chip long_matmul does not run on: it always opens
    device 0, so with another one the parser would analyse an idle chip and still exit 0."""
    if app_exe.name == LONG_MATMUL_APP and device_id not in (None, 0):
        raise ValueError(
            f"{LONG_MATMUL_APP} always runs on device 0; --device-id {device_id} would measure an idle chip"
        )


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
    app_exe: Path,
    app_args: List[str],
    env: Mapping[str, str],
    telemetry_freq_hz: int,
    trim_ms: float,
    slot_ms: int,
    device_id: Optional[int],
) -> Dict[str, object]:
    """Everything that decides what a run measured, as recorded in RUN_COMPLETE_FILE. The app is
    identified by its resolved path; a rebuild in place is not detected."""
    return {
        "app_exe": str(Path(app_exe).expanduser().resolve()),
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


def save_figure(fig, out_path: Path, dpi: int) -> None:
    """Lay out, write and close a matplotlib figure, creating the directory it goes into."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def legacy_mode() -> bool:
    """LONG_MATMUL_MODE=legacy: reproduce how the published results were measured and analysed."""
    return os.environ.get("LONG_MATMUL_MODE") == "legacy"


def whole_run_value(row: Mapping[str, str], column: str, legacy: bool) -> float:
    """A charge or energy column of program_intervals.csv, for the whole run.

    Those columns cover the trimmed telemetry window (window_s), while a FLOP count covers the
    whole run (algo_time_s_reported). Updated mode scales the value up to the whole run so
    pJ/FLOP is not biased low by the trim; legacy mode uses the window value, as the published
    analysis did. Returns NaN if anything needed is missing or not finite.
    """
    try:
        value = float(row[column])
        if legacy:
            return value
        window_s = float(row["window_s"])
        run_s = float(row["algo_time_s_reported"])
    except (KeyError, ValueError):
        return float("nan")
    return value * run_s / window_s if window_s > 0 else float("nan")


def environ_with(overrides: Mapping[str, str]) -> Dict[str, str]:
    env = dict(os.environ)
    env.update(overrides)
    return env
