#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


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


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def stop_process(proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def wait_for_file_to_appear(path: Path, timeout_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(0.1)
    return path.exists()


def build_bash_command(
    command: List[str],
    activate_script: Path,
    runtime_root: Path,
    workdir: Path,
) -> str:
    cmd_str = shell_join(command)
    return (
        f"source {shlex.quote(str(activate_script))} && "
        f"export TT_METAL_RUNTIME_ROOT={shlex.quote(str(runtime_root))} && "
        f"cd {shlex.quote(str(workdir))} && "
        f"{cmd_str}"
    )


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
        ["bash", "-lc", bash_cmd],
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

    telemetry_cmd = [
        "stdbuf",
        "-oL",
        "-eL",
        str(telemetry_exe),
        "-o",
        str(paths.telemetry_file),
        "-f",
        str(telemetry_freq_hz),
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

            time.sleep(1.0)

            if telemetry_proc.poll() is not None:
                log("ERROR: Telemetry exited immediately.")
                return 10

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
            log("Waiting 10 seconds before stopping telemetry...")
            time.sleep(10.0)

            log("Stopping telemetry...")
            stop_process(telemetry_proc, timeout_s=10.0)

            if not wait_for_file_to_appear(paths.telemetry_file, timeout_s=5.0):
                log("ERROR: Telemetry output file was not created.")
                return 11

            if not paths.summary_file.exists() or paths.summary_file.stat().st_size == 0:
                log("ERROR: Summary output file is missing or empty.")
                return 12

            if device_profiler_csv.exists():
                log(f"Device profiler csv found: {device_profiler_csv}")
                parser_device_profiler_csv: Optional[Path] = device_profiler_csv
            else:
                log(
                    "NOTE: no device profiler csv found -- was the app run with "
                    "TT_METAL_DEVICE_PROFILER=1? Skipping kernel start/end analysis."
                )
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

            log("")
            log("Workflow completed successfully.")
            log(f"All results are under: {paths.run_dir}")
            return 0

        except KeyboardInterrupt:
            log("Interrupted by user.")
            return 130

        finally:
            if app_proc is not None and app_proc.poll() is None:
                stop_process(app_proc, timeout_s=5.0)
            if telemetry_proc is not None and telemetry_proc.poll() is None:
                stop_process(telemetry_proc, timeout_s=10.0)


def main() -> int:
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
        type=int,
        default=50,
        help="Telemetry frequency in Hz. Default: 50",
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
        "--fixed-tiles-per-core",
        type=int,
        default=None,
        help=(
            "If set, appends this value as the 5th positional arg to the app, "
            "enabling fixed-per-core mode in high_power_matmul. "
            "Each core always computes exactly this many tiles regardless of grid size, "
            "so power scales linearly with core count."
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
        help="Subdirectory name for this run.",
    )

    parser.add_argument(
        "--slot-ms",
        type=int,
        default=1,
        help="slot-ms passed to the parser. Default: 1",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=None,
        help="Optional device-id passed to the parser.",
    )
    parser.add_argument(
        "--trim-ms",
        type=float,
        default=6.0,
        help="trim-ms passed to the parser. Default: 6.0",
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
            "Default: <app-workdir>/generated/profiler/.logs/profile_log_device.csv"
        ),
    )

    args = parser.parse_args()

    tt_metal_root = args.tt_metal_root.expanduser().resolve()
    telemetry_workdir = args.telemetry_workdir.expanduser().resolve() if args.telemetry_workdir else tt_metal_root
    app_workdir = args.app_workdir.expanduser().resolve() if args.app_workdir else tt_metal_root

    device_profiler_csv = (
        args.device_profiler_csv.expanduser().resolve()
        if args.device_profiler_csv
        else app_workdir / "generated" / "profiler" / ".logs" / "profile_log_device.csv"
    )

    app_args = list(args.app_args)
    if args.fixed_tiles_per_core is not None:
        app_args.append(str(args.fixed_tiles_per_core))

    return run_workflow(
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
