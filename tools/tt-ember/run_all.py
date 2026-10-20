#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from common import (
    DEFAULT_DEVICE_ID,
    DEFAULT_DPI,
    DEFAULT_SLOT_MS,
    DEFAULT_TELEMETRY_FREQ_HZ,
    DEFAULT_TRIM_MS,
    check_long_matmul_device,
    RUN_COMPLETE_FILE,
    auto_py_command,
    exit_reason,
    reset_hardware,
    run_settings,
    run_status,
    shell_join,
)


@dataclass
class RunSpec:
    subdir: str
    app_args: List[str]


PREFILL_RUNS: List[RunSpec] = [
    RunSpec(subdir="prefill_1024_2048_2048_160", app_args=["1024", "2048", "2048", "160"]),
    RunSpec(subdir="prefill_2048_4096_4096_160", app_args=["2048", "4096", "4096", "160"]),
    RunSpec(subdir="prefill_4096_8192_8192_160", app_args=["4096", "8192", "8192", "160"]),
]

FIXED_TPC_RUNS: List[RunSpec] = [
    RunSpec(subdir="fixed_tpc_100", app_args=["2048", "4096", "4096", "160", "100"]),
    RunSpec(subdir="fixed_tpc_200", app_args=["2048", "4096", "4096", "160", "200"]),
    RunSpec(subdir="fixed_tpc_400", app_args=["2048", "4096", "4096", "160", "400"]),
]


def settings_for(spec: RunSpec, args: argparse.Namespace) -> dict:
    return run_settings(
        args.app_exe, spec.app_args, os.environ, args.telemetry_freq, args.trim_ms, args.slot_ms, args.device_id
    )


def is_complete(spec: RunSpec, args: argparse.Namespace) -> bool:
    return run_status(args.output_root / spec.subdir, settings_for(spec, args)) == "done"


def run_compare(
    compare_script: Path,
    out_root: Path,
    output_dir: Path,
    filter_prefix: str,
    dpi: int,
    overcompute: bool,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(compare_script),
        "--out-root",
        str(out_root),
        "--output-dir",
        str(output_dir),
        "--filter",
        filter_prefix,
        "--dpi",
        str(dpi),
    ]
    if not overcompute:
        cmd.append("--no-overcompute")
    print(f"\n[COMPARE] Running: {shell_join(cmd)}", flush=True)
    return subprocess.run(cmd).returncode


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run all prefill and fixed-tpc telemetry sweeps sequentially via auto.py.")

    ap.add_argument(
        "--auto-script",
        type=Path,
        default=Path(__file__).parent / "auto.py",
        help="Path to auto.py. Default: auto.py next to this script.",
    )
    ap.add_argument(
        "--telemetry-exe",
        required=True,
        type=Path,
        help="Path to telemetry executable.",
    )
    ap.add_argument(
        "--telemetry-freq",
        type=int,
        default=DEFAULT_TELEMETRY_FREQ_HZ,
        help=f"Telemetry frequency in Hz. Default: {DEFAULT_TELEMETRY_FREQ_HZ}",
    )
    ap.add_argument(
        "--app-exe",
        required=True,
        type=Path,
        help="Path to application executable.",
    )
    ap.add_argument(
        "--parser-script",
        required=True,
        type=Path,
        help="Path to parser.py.",
    )
    ap.add_argument(
        "--tt-venv-activate",
        required=True,
        type=Path,
        help="Path to the Python venv activation script used by auto.py.",
    )
    ap.add_argument(
        "--tt-metal-root",
        required=True,
        type=Path,
        help="Path to tt-metal repository root.",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Root output directory.",
    )
    ap.add_argument(
        "--slot-ms",
        type=int,
        default=DEFAULT_SLOT_MS,
        help=f"Bin width in ms passed to parser. Default: {DEFAULT_SLOT_MS}",
    )
    ap.add_argument(
        "--device-id",
        type=int,
        default=DEFAULT_DEVICE_ID,
        help=f"Device ID passed to parser. Default: {DEFAULT_DEVICE_ID}",
    )
    ap.add_argument(
        "--trim-ms",
        type=float,
        default=DEFAULT_TRIM_MS,
        help=f"Trim applied to each side of every interval, in ms. Default: {DEFAULT_TRIM_MS}",
    )
    ap.add_argument(
        "--tt-smi",
        default="tt-smi",
        help="tt-smi executable name or path. Default: tt-smi",
    )
    ap.add_argument(
        "--compare-script",
        type=Path,
        default=Path(__file__).parent / "compare_runs.py",
        help="Path to compare_runs.py. Default: compare_runs.py next to this script.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run runs that are already complete, or were made with different settings.",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for comparison figures. Default: {DEFAULT_DPI}",
    )

    args = ap.parse_args(argv)
    try:
        check_long_matmul_device(args.app_exe, args.device_id)
    except ValueError as e:
        ap.error(str(e))
    print(f"Output root: {args.output_root}", flush=True)

    # --- Runs: a run is done only if auto.py recorded it as complete with the same settings ---
    pending: List[RunSpec] = []
    for spec in PREFILL_RUNS + FIXED_TPC_RUNS:
        status = run_status(args.output_root / spec.subdir, settings_for(spec, args))
        if status == "done" and not args.force:
            print(f"[SKIP] '{spec.subdir}' is already complete.", flush=True)
        elif status == "different" and not args.force:
            print(
                f"ERROR: '{spec.subdir}' holds a run made with different settings (see its {RUN_COMPLETE_FILE}). "
                "Use another --output-root, or --force to replace it.",
                file=sys.stderr,
            )
            return 1
        else:
            pending.append(spec)

    for i, spec in enumerate(pending, start=1):
        print(f"\n{'=' * 60}\n Run {i}/{len(pending)}: {spec.subdir}\n{'=' * 60}", flush=True)
        rc = reset_hardware(args.tt_smi)
        if rc != 0:
            print(f"ERROR: Hardware reset failed before run '{spec.subdir}'. Aborting.", file=sys.stderr)
            return rc
        cmd = auto_py_command(
            auto_script=args.auto_script,
            telemetry_exe=args.telemetry_exe,
            telemetry_freq=args.telemetry_freq,
            app_exe=args.app_exe,
            app_args=spec.app_args,
            parser_script=args.parser_script,
            tt_venv_activate=args.tt_venv_activate,
            tt_metal_root=args.tt_metal_root,
            output_root=args.output_root,
            subdir=spec.subdir,
            slot_ms=args.slot_ms,
            device_id=args.device_id,
            trim_ms=args.trim_ms,
        )
        print(f"[RUN] {shell_join(cmd)}", flush=True)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"ERROR: Run '{spec.subdir}' failed with return code {exit_reason(rc)}. Aborting.", file=sys.stderr)
            return rc

    # --- Comparisons: cheap, so redo them whenever their whole group is complete ---
    for name, group, output_dir in (
        ("prefill", PREFILL_RUNS, args.output_root / "comparison_prefill"),
        ("fixed_tpc", FIXED_TPC_RUNS, args.output_root / "comparison_fixed_tpc"),
    ):
        if not all(is_complete(spec, args) for spec in group):
            print(f"[SKIP] {name} comparison: not all {name} runs are complete yet.", flush=True)
            continue
        # The overcompute reference only applies when work grows with the core count (fixed mode).
        rc = run_compare(
            args.compare_script, args.output_root, output_dir, name, args.dpi, overcompute=name == "fixed_tpc"
        )
        if rc != 0:
            print(f"ERROR: Comparison '{name}' failed with return code {rc}.", file=sys.stderr)
            return rc
        print(f"[COMPARE] Written to: {output_dir}", flush=True)

    print(f"\n{'=' * 60}\n Done. Results are under: {args.output_root}\n{'=' * 60}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
