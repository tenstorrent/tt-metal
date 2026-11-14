#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from common import (
    DEFAULT_DEVICE_ID,
    DEFAULT_SLOT_MS,
    DEFAULT_TELEMETRY_FREQ_HZ,
    DEFAULT_TRIM_MS,
    EXIT_PROFILER_CSV_MISSING,
    EXIT_SUMMARY_MISSING,
    EXIT_TELEMETRY_DIED,
    EXIT_TELEMETRY_FILE_MISSING,
    EXIT_TELEMETRY_START,
    LONG_MATMUL_APP,
    RUN_COMPLETE_FILE,
    check_long_matmul_device,
    long_matmul_args,
    misplaced_options,
    run_settings,
    shell_join,
    subdir_name,
    write_run_complete,
)

# Idle time after the app finishes and before telemetry stops, so the last grid's tail and the
# return to idle are captured.
POST_RUN_IDLE_S = 10

# How long telemetry gets to start before auto.py checks that it is still running.
TELEMETRY_START_GRACE_S = 1.0


@dataclass
class RunPaths:
    run_dir: Path
    telemetry_file: Path
    summary_file: Path
    launcher_log_file: Path


def validate_existing_file(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def validate_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path not found: {path}")
    return path


def build_run_paths(output_root: Path, subdir: str) -> RunPaths:
    run_dir = output_root.expanduser().resolve() / subdir
    run_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_dir=run_dir,
        telemetry_file=run_dir / "telemetry.txt",
        summary_file=run_dir / "summary.txt",
        launcher_log_file=run_dir / "launcher.log",
    )


def stop_processes(procs: List[Optional[subprocess.Popen]], timeout_s: float = 10.0) -> None:
    """SIGTERM every still-running process group at once, wait, then SIGKILL what is left.
    Signalling all of them first means a caller that kills auto.py shortly after (as a sweep
    script does on Ctrl+C) still finds telemetry already told to stop."""
    running = [p for p in procs if p is not None and p.poll() is None]
    for proc in running:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.time() + timeout_s
    while time.time() < deadline and any(p.poll() is None for p in running):
        time.sleep(0.1)

    for proc in running:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def wait_for_file_to_appear(path: Path, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


# Fixed bash script; every path and the command itself arrive as positional arguments, so nothing
# from the command line is ever parsed as shell code. The app resolves programming-example kernel
# paths through TT_METAL_HOME, which the venv activation script does not set. exec keeps the
# process group the one auto.py stops.
BASH_WRAPPER = 'source "$1" && export TT_METAL_RUNTIME_ROOT="$2" TT_METAL_HOME="$2" && cd "$3" && shift 3 && exec "$@"'


def build_bash_command(
    command: List[str],
    activate_script: Path,
    runtime_root: Path,
    workdir: Path,
) -> List[str]:
    return [
        "bash",
        "-c",
        BASH_WRAPPER,
        "bash",
        str(activate_script),
        str(runtime_root),
        str(workdir),
        *command,
    ]


def start_bash_wrapped_process(
    command: List[str],
    activate_script: Path,
    runtime_root: Path,
    workdir: Path,
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    bash_cmd = build_bash_command(
        command=command,
        activate_script=activate_script,
        runtime_root=runtime_root,
        workdir=workdir,
    )

    return subprocess.Popen(
        bash_cmd,
        stdout=stdout,
        stderr=stderr,
        text=True,
        start_new_session=True,
    )


def run_parser(
    parser_script: Path,
    python_exe: str,
    telemetry_file: Path,
    summary_file: Path,
    output_root: Path,
    subdir: str,
    slot_ms: int,
    device_id: Optional[int],
    trim_ms: float,
    parser_cwd: Optional[Path],
    device_profiler_csv: Optional[Path],
    log_f,
) -> int:
    parser_cmd = [
        python_exe,
        str(parser_script),
        "-i",
        str(telemetry_file),
        "-o",
        str(output_root),
        "--subdir",
        subdir,
        "--program-log",
        str(summary_file),
        "--slot-ms",
        str(slot_ms),
        "--trim-ms",
        str(trim_ms),
    ]

    if device_id is not None:
        parser_cmd.extend(["--device-id", str(device_id)])

    if device_profiler_csv is not None:
        parser_cmd.extend(["--device-profiler-csv", str(device_profiler_csv)])

    print(f"Parser cmd: {shell_join(parser_cmd)}")
    print(f"Parser cmd: {shell_join(parser_cmd)}", file=log_f)
    log_f.flush()

    result = subprocess.run(
        parser_cmd,
        cwd=str(parser_cwd) if parser_cwd else None,
        text=True,
        capture_output=True,
    )

    if result.stdout:
        print(result.stdout, end="")
        print(result.stdout, end="", file=log_f)

    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
        print(result.stderr, end="", file=log_f)

    log_f.flush()
    return result.returncode


def run_workflow(
    settings: Dict[str, object],
    telemetry_exe: Path,
    telemetry_freq_hz: int,
    app_exe: Path,
    app_args: List[str],
    parser_script: Path,
    output_root: Path,
    subdir: str,
    slot_ms: int,
    device_id: Optional[int],
    trim_ms: float,
    python_exe: str,
    tt_venv_activate: Path,
    tt_metal_root: Path,
    telemetry_workdir: Path,
    app_workdir: Path,
    parser_cwd: Optional[Path],
    device_profiler_csv: Path,
) -> int:
    paths = build_run_paths(output_root, subdir)
    # A previous attempt's results must not survive into this one; RUN_COMPLETE_FILE is written
    # again only if this run gets all the way through the parser. The parser clears its own
    # kernel outputs, but keeps the profiler CSV copy, since a re-parse may need it.
    for stale in (
        paths.run_dir / RUN_COMPLETE_FILE,
        paths.run_dir / "program_intervals.csv",
        paths.run_dir / device_profiler_csv.name,
    ):
        stale.unlink(missing_ok=True)

    # tt-umd's telemetry tool takes -f as a polling PERIOD in microseconds, not a frequency in
    # Hz (see `telemetry --help`). --telemetry-freq is in Hz, so convert: 50 Hz -> 20000 us.
    # Passing the Hz value through unconverted polls every 50 us (20 kHz) instead.
    telemetry_period_us = max(1, round(1_000_000 / telemetry_freq_hz))

    telemetry_cmd = [
        "stdbuf",
        "-oL",
        "-eL",
        str(telemetry_exe),
        "-o",
        str(paths.telemetry_file),
        "-f",
        str(telemetry_period_us),
    ]

    app_cmd = ["stdbuf", "-oL", "-eL", str(app_exe), *app_args]

    with paths.launcher_log_file.open("w", encoding="utf-8") as log_f:

        def log(msg: str) -> None:
            print(msg)
            print(msg, file=log_f)
            log_f.flush()

        log(f"Run directory: {paths.run_dir}")
        log(f"Telemetry file: {paths.telemetry_file}")
        log(f"Summary file:   {paths.summary_file}")
        log(f"Launcher log:   {paths.launcher_log_file}")
        log("")
        log(f"Tenstorrent activate script: {tt_venv_activate}")
        log(f"TT_METAL_RUNTIME_ROOT:       {tt_metal_root}")
        log(f"Telemetry workdir:           {telemetry_workdir}")
        log(f"App workdir:                 {app_workdir}")
        log("")
        log(f"Telemetry raw cmd: {shell_join(telemetry_cmd)}")
        log(f"App raw cmd:       {shell_join(app_cmd)}")
        log("")

        telemetry_proc: Optional[subprocess.Popen] = None
        app_proc: Optional[subprocess.Popen] = None

        try:
            log("Starting telemetry...")
            telemetry_proc = start_bash_wrapped_process(
                command=telemetry_cmd,
                activate_script=tt_venv_activate,
                runtime_root=tt_metal_root,
                workdir=telemetry_workdir,
                stdout=log_f,
                stderr=log_f,
            )

            time.sleep(TELEMETRY_START_GRACE_S)

            if telemetry_proc.poll() is not None:
                log("ERROR: Telemetry exited immediately.")
                return EXIT_TELEMETRY_START

            with paths.summary_file.open("w", encoding="utf-8") as summary_f:
                if device_profiler_csv.exists():
                    log(f"Removing stale device profiler csv: {device_profiler_csv}")
                    device_profiler_csv.unlink()

                log("Starting application...")
                app_proc = start_bash_wrapped_process(
                    command=app_cmd,
                    activate_script=tt_venv_activate,
                    runtime_root=tt_metal_root,
                    workdir=app_workdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )

                assert app_proc.stdout is not None

                for line in app_proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    summary_f.write(line)
                    summary_f.flush()

                app_rc = app_proc.wait()
                log(f"Application exited with return code: {app_rc}")

                if app_rc != 0:
                    log("ERROR: Application failed.")
                    return app_rc

            log("Application finished successfully.")
            log(f"Waiting {POST_RUN_IDLE_S} seconds before stopping telemetry...")
            time.sleep(POST_RUN_IDLE_S)

            if telemetry_proc.poll() is not None:
                log(f"ERROR: Telemetry exited during the run (return code {telemetry_proc.returncode}).")
                return EXIT_TELEMETRY_DIED

            log("Stopping telemetry...")
            stop_processes([telemetry_proc])

            if not wait_for_file_to_appear(paths.telemetry_file):
                log("ERROR: Telemetry output file was not created.")
                return EXIT_TELEMETRY_FILE_MISSING

            if not paths.summary_file.exists() or paths.summary_file.stat().st_size == 0:
                log("ERROR: Summary output file is missing or empty.")
                return EXIT_SUMMARY_MISSING

            if device_profiler_csv.exists():
                log(f"Device profiler csv found: {device_profiler_csv}")
                parser_device_profiler_csv: Optional[Path] = device_profiler_csv
            elif os.environ.get("TT_METAL_DEVICE_PROFILER", "0") not in ("", "0"):
                # The run would otherwise be recorded as a profiler run without its kernel data.
                log(f"ERROR: TT_METAL_DEVICE_PROFILER is set, but there is no profiler csv at {device_profiler_csv}.")
                return EXIT_PROFILER_CSV_MISSING
            else:
                log("NOTE: no device profiler csv (TT_METAL_DEVICE_PROFILER is not set); skipping the kernel analysis.")
                parser_device_profiler_csv = None

            log("Running parser...")
            parser_rc = run_parser(
                parser_script=parser_script,
                python_exe=python_exe,
                telemetry_file=paths.telemetry_file,
                summary_file=paths.summary_file,
                output_root=output_root,
                subdir=subdir,
                slot_ms=slot_ms,
                device_id=device_id,
                trim_ms=trim_ms,
                parser_cwd=parser_cwd,
                device_profiler_csv=parser_device_profiler_csv,
                log_f=log_f,
            )

            log(f"Parser exited with return code: {parser_rc}")
            if parser_rc != 0:
                log("ERROR: Parser failed.")
                return parser_rc

            write_run_complete(paths.run_dir, settings)
            log("")
            log("Workflow completed successfully.")
            log(f"All results are under: {paths.run_dir}")
            return 0

        except KeyboardInterrupt:
            log("Interrupted by user.")
            return 130

        finally:
            stop_processes([app_proc, telemetry_proc])


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def raise_keyboard_interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main() -> int:
    # Treat SIGTERM like Ctrl+C, so the cleanup in run_workflow's finally block still stops
    # telemetry and the app.
    signal.signal(signal.SIGTERM, raise_keyboard_interrupt)

    parser = argparse.ArgumentParser(
        description="Run telemetry, run TT-Metal app, save summary output, and invoke parser."
    )

    parser.add_argument(
        "--telemetry-exe",
        required=True,
        type=validate_existing_path,
        help="Path to telemetry executable.",
    )
    parser.add_argument(
        "--telemetry-freq",
        type=positive_int,
        default=DEFAULT_TELEMETRY_FREQ_HZ,
        help="Telemetry sampling frequency in Hz; converted to the polling period the telemetry tool expects. "
        f"Default: {DEFAULT_TELEMETRY_FREQ_HZ}",
    )

    parser.add_argument(
        "--app-exe",
        required=True,
        type=validate_existing_path,
        help="Path to application executable.",
    )
    parser.add_argument(
        "--app-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed to the application. Put this option last.",
    )
    parser.add_argument(
        "--fixed-blocks-per-core",
        "--fixed-tiles-per-core",
        dest="fixed_blocks_per_core",
        type=int,
        default=None,
        help=(
            "If set, appends this value as the 5th positional arg to the app, "
            "enabling fixed-per-core mode in long_matmul. "
            "Each core always computes exactly this many output blocks (tiles, at the default "
            "1x1 block) regardless of grid size, so power scales linearly with core count."
        ),
    )

    parser.add_argument(
        "--parser-script",
        required=True,
        type=validate_existing_file,
        help="Path to parser Python script.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root output directory.",
    )
    parser.add_argument(
        "--subdir",
        required=True,
        type=subdir_name,
        help="Subdirectory name for this run (letters, digits, '_', '.', '-').",
    )

    parser.add_argument(
        "--slot-ms",
        type=int,
        default=DEFAULT_SLOT_MS,
        help=f"slot-ms passed to the parser. Default: {DEFAULT_SLOT_MS}",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=DEFAULT_DEVICE_ID,
        help=(
            f"Device whose telemetry the parser uses. Default: {DEFAULT_DEVICE_ID}, the device long_matmul runs on; "
            "without it the parser would mix every chip's samples on a multi-chip board."
        ),
    )
    parser.add_argument(
        "--trim-ms",
        type=float,
        default=DEFAULT_TRIM_MS,
        help=f"Trim applied to each side of every interval, in ms. Default: {DEFAULT_TRIM_MS}",
    )

    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run the parser. Default: current interpreter",
    )

    parser.add_argument(
        "--tt-venv-activate",
        type=validate_existing_file,
        required=True,
        help="Path to the activation script of a Python environment with tt-metal's Python dependencies.",
    )
    parser.add_argument(
        "--tt-metal-root",
        required=True,
        type=validate_existing_path,
        help="Path to tt-metal repository root containing tt_metal/.",
    )

    parser.add_argument(
        "--telemetry-workdir",
        type=Path,
        default=None,
        help="Working directory for telemetry. Default: tt-metal root",
    )
    parser.add_argument(
        "--app-workdir",
        type=Path,
        default=None,
        help="Working directory for app. Default: tt-metal root",
    )
    parser.add_argument(
        "--parser-cwd",
        type=Path,
        default=None,
        help="Optional working directory for parser.",
    )

    parser.add_argument(
        "--device-profiler-csv",
        type=Path,
        default=None,
        help=(
            "Path to tt-metal's device profiler CSV (generated/profiler/.logs/"
            "profile_log_device.csv). Deleted before the app runs (if it already exists) "
            "so it only contains this run's data, then passed to the parser afterwards. "
            "Default: $TT_METAL_PROFILER_DIR/.logs/profile_log_device.csv, or "
            "<tt-metal-root>/generated/profiler/.logs/profile_log_device.csv (where tt-metal writes it)"
        ),
    )

    args = parser.parse_args()

    # --app-args takes everything after it, so an auto.py option written after it would silently
    # become an app argument. Refuse that instead.
    misplaced = misplaced_options(parser, args.app_args)
    if misplaced:
        parser.error(f"--app-args must come last; these auto.py options follow it: {' '.join(misplaced)}")

    tt_metal_root = args.tt_metal_root.expanduser().resolve()
    telemetry_workdir = args.telemetry_workdir.expanduser().resolve() if args.telemetry_workdir else tt_metal_root
    app_workdir = args.app_workdir.expanduser().resolve() if args.app_workdir else tt_metal_root

    if args.device_profiler_csv:
        device_profiler_csv = args.device_profiler_csv.expanduser().resolve()
    else:
        # tt-metal uses a relative TT_METAL_PROFILER_DIR as is, so it is relative to the app's cwd.
        profiler_dir = os.environ.get("TT_METAL_PROFILER_DIR")
        base = app_workdir / profiler_dir if profiler_dir else tt_metal_root / "generated" / "profiler"
        device_profiler_csv = base / ".logs" / "profile_log_device.csv"

    app_args = list(args.app_args)
    if args.app_exe.name == LONG_MATMUL_APP:
        # Without M N K iterations long_matmul does a one-grid smoke run, which has no idle pause
        # to measure the base current in.
        if args.fixed_blocks_per_core is not None:
            if len(app_args) != 4:
                parser.error("--fixed-blocks-per-core needs exactly 4 --app-args (M N K iterations)")
            app_args.append(str(args.fixed_blocks_per_core))
        try:
            long_matmul_args(app_args)
            check_long_matmul_device(args.app_exe, args.device_id)
        except ValueError as e:
            parser.error(str(e))
    elif args.fixed_blocks_per_core is not None:
        parser.error("--fixed-blocks-per-core only applies to long_matmul")

    settings = run_settings(
        args.app_exe, app_args, os.environ, args.telemetry_freq, args.trim_ms, args.slot_ms, args.device_id
    )

    return run_workflow(
        settings=settings,
        telemetry_exe=args.telemetry_exe.expanduser().resolve(),
        telemetry_freq_hz=args.telemetry_freq,
        app_exe=args.app_exe.expanduser().resolve(),
        app_args=app_args,
        parser_script=args.parser_script.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        subdir=args.subdir,
        slot_ms=args.slot_ms,
        device_id=args.device_id,
        trim_ms=args.trim_ms,
        python_exe=args.python_exe,
        tt_venv_activate=args.tt_venv_activate.expanduser().resolve(),
        tt_metal_root=tt_metal_root,
        telemetry_workdir=telemetry_workdir,
        app_workdir=app_workdir,
        parser_cwd=args.parser_cwd.expanduser().resolve() if args.parser_cwd else None,
        device_profiler_csv=device_profiler_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
