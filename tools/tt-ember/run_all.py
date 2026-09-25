#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

RESET_COOLDOWN_S = 10


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


def check_pending_runs(all_runs: List[RunSpec], output_root: Path) -> List[RunSpec]:
    pending: List[RunSpec] = []
    for spec in all_runs:
        run_dir = output_root / spec.subdir
        if run_dir.exists():
            print(f"[SKIP] '{spec.subdir}' already exists. Skipping.", flush=True)
        else:
            pending.append(spec)
    return pending


def check_pending_comparisons(
    prefill_runs: List[RunSpec],
    fixed_runs: List[RunSpec],
    output_root: Path,
) -> List[tuple]:
    pending = []

    prefill_complete = all((output_root / s.subdir).exists() for s in prefill_runs)
    fixed_complete = all((output_root / s.subdir).exists() for s in fixed_runs)

    comp_prefill_dir = output_root / "comparison_prefill"
    comp_fixed_dir = output_root / "comparison_fixed_tpc"

    if prefill_complete and not comp_prefill_dir.exists():
        pending.append(("prefill", comp_prefill_dir))
        print(f"[COMPARE] Prefill runs complete, comparison missing — will run.", flush=True)
    elif prefill_complete:
        print(f"[SKIP] Prefill comparison already exists at {comp_prefill_dir}. Skipping.", flush=True)
    else:
        print(f"[SKIP] Prefill comparison skipped — not all prefill runs are complete yet.", flush=True)

    if fixed_complete and not comp_fixed_dir.exists():
        pending.append(("fixed_tpc", comp_fixed_dir))
        print(f"[COMPARE] Fixed-tpc runs complete, comparison missing — will run.", flush=True)
    elif fixed_complete:
        print(f"[SKIP] Fixed-tpc comparison already exists at {comp_fixed_dir}. Skipping.", flush=True)
    else:
        print(f"[SKIP] Fixed-tpc comparison skipped — not all fixed-tpc runs are complete yet.", flush=True)

    return pending


def run_compare(
    compare_script: Path,
    out_root: Path,
    output_dir: Path,
    filter_prefix: str,
    dpi: int,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(compare_script),
        "--out-root", str(out_root),
        "--output-dir", str(output_dir),
        "--filter", filter_prefix,
        "--dpi", str(dpi),
    ]
    print(f"\n[COMPARE] Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def reset_hardware(tt_smi_cmd: str = "tt-smi") -> int:
    print(f"\n[RESET] Running hardware reset: {tt_smi_cmd} -r", flush=True)
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


def run_single(
    auto_script: Path,
    telemetry_exe: Path,
    telemetry_freq: int,
    app_exe: Path,
    parser_script: Path,
    tt_venv_activate: Path,
    tt_metal_root: Path,
    output_root: Path,
    slot_ms: int,
    device_id: Optional[int],
    trim_ms: float,
    spec: RunSpec,
) -> int:
    cmd: List[str] = [
        sys.executable,
        str(auto_script),
        "--telemetry-exe", str(telemetry_exe),
        "--telemetry-freq", str(telemetry_freq),
        "--app-exe", str(app_exe),
        "--parser-script", str(parser_script),
        "--tt-venv-activate", str(tt_venv_activate),
        "--tt-metal-root", str(tt_metal_root),
        "--output-root", str(output_root),
        "--subdir", spec.subdir,
        "--slot-ms", str(slot_ms),
        "--trim-ms", str(trim_ms),
    ]

    if device_id is not None:
        cmd += ["--device-id", str(device_id)]

    # --app-args must be last (argparse.REMAINDER in auto.py)
    cmd += ["--app-args"] + spec.app_args

    print(f"\n{'=' * 60}", flush=True)
    print(f" Starting: {spec.subdir}", flush=True)
    print(f"{'=' * 60}", flush=True)

    result = subprocess.run(cmd)
    return result.returncode


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run all prefill and fixed-tpc telemetry sweeps sequentially via auto.py."
    )

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
        default=50,
        help="Telemetry frequency in Hz. Default: 50",
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
        default=1,
        help="Bin width in ms passed to parser. Default: 1",
    )
    ap.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Device ID passed to parser. Default: 0",
    )
    ap.add_argument(
        "--trim-ms",
        type=float,
        default=6.0,
        help="Trim applied to each interval side in ms. Default: 6.0",
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
        "--dpi",
        type=int,
        default=150,
        help="DPI for comparison figures. Default: 150",
    )

    args = ap.parse_args(argv)

    all_runs = PREFILL_RUNS + FIXED_TPC_RUNS

    print(f"Output root: {args.output_root}", flush=True)
    print(f"\nChecking simulation runs...", flush=True)
    pending_runs = check_pending_runs(all_runs, args.output_root)

    print(f"\nChecking comparisons...", flush=True)
    pending_comparisons = check_pending_comparisons(PREFILL_RUNS, FIXED_TPC_RUNS, args.output_root)

    if not pending_runs and not pending_comparisons:
        print(
            f"\nAll runs and comparisons already exist under {args.output_root}. Nothing to do.\n"
            f"Delete the relevant subdirectories if you want to re-run them.",
            flush=True,
        )
        return 0

    # --- Simulation runs ---
    if pending_runs:
        total = len(pending_runs)
        print(f"\n{total} simulation run(s) pending.", flush=True)

        for i, spec in enumerate(pending_runs, start=1):
            print(f"\n{'=' * 60}", flush=True)
            print(f" Run {i}/{total}: {spec.subdir}", flush=True)
            print(f"{'=' * 60}", flush=True)

            rc = reset_hardware(args.tt_smi)
            if rc != 0:
                print(f"ERROR: Hardware reset failed before run '{spec.subdir}'. Aborting.", file=sys.stderr)
                return rc

            print(f"[RUN] Launching auto.py for: {spec.subdir}", flush=True)
            rc = run_single(
                auto_script=args.auto_script,
                telemetry_exe=args.telemetry_exe,
                telemetry_freq=args.telemetry_freq,
                app_exe=args.app_exe,
                parser_script=args.parser_script,
                tt_venv_activate=args.tt_venv_activate,
                tt_metal_root=args.tt_metal_root,
                output_root=args.output_root,
                slot_ms=args.slot_ms,
                device_id=args.device_id,
                trim_ms=args.trim_ms,
                spec=spec,
            )
            if rc != 0:
                print(f"ERROR: Run '{spec.subdir}' failed with return code {rc}. Aborting.", file=sys.stderr)
                return rc

            print(f"[RUN] Completed: {spec.subdir}", flush=True)

        print(f"\n{'=' * 60}", flush=True)
        print(f" All {total} pending simulation run(s) completed successfully.", flush=True)
        print(f"{'=' * 60}", flush=True)

    # --- Re-evaluate pending comparisons after new runs may have completed ---
    pending_comparisons = check_pending_comparisons(PREFILL_RUNS, FIXED_TPC_RUNS, args.output_root)

    if pending_comparisons:
        print(f"\n[COMPARE] Running comparisons...", flush=True)
        for filter_prefix, output_dir in pending_comparisons:
            rc = run_compare(
                compare_script=args.compare_script,
                out_root=args.output_root,
                output_dir=output_dir,
                filter_prefix=filter_prefix,
                dpi=args.dpi,
            )
            if rc != 0:
                print(f"ERROR: Comparison '{filter_prefix}' failed with return code {rc}.", file=sys.stderr)
                return rc
            print(f"[COMPARE] Written to: {output_dir}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f" Done. Results are under: {args.output_root}", flush=True)
    print(f"{'=' * 60}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
