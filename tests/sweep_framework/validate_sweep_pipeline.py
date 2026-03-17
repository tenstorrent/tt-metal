#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sweep Validation Pipeline Orchestrator

Chains the four manual steps of the sweep-vs-model-trace validation workflow
into a single command:

    1. Generate sweep vectors from a model trace
    2. Run the sweep test under the operations tracer
    3. Split the resulting sweep trace by operation
    4. Ensure the model trace is also split, then print file pairs
       ready for cursor-rule validation

Usage:
    python tests/sweep_framework/validate_sweep_pipeline.py \\
        --model-trace model_tracer/traced_operations/ttnn_operations_master_UF_EV_B9_GWH01_deepseek.json \\
        --module-name model_traced.all_gather_async_model_traced \\
        --suite model_traced \\
        --mesh-shape 4x8 \\
        --arch-name wormhole_b0
"""

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRACED_OPS_DIR = REPO_ROOT / "model_tracer" / "traced_operations"
VECTORS_EXPORT_DIR = REPO_ROOT / "tests" / "sweep_framework" / "vectors_export"

SEPARATOR = "=" * 70


def _run(cmd: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    """Run a subprocess, streaming output. Abort on failure."""
    merged_env = {**os.environ, **(env or {})}
    display_env = " ".join(f"{k}={v}" for k, v in (env or {}).items())
    display_cmd = " ".join(cmd)
    if display_env:
        display_cmd = f"{display_env} {display_cmd}"

    print(f"\n  $ {display_cmd}\n")

    if dry_run:
        return

    result = subprocess.run(cmd, env=merged_env)
    if result.returncode != 0:
        print(f"\nERROR: Command exited with code {result.returncode}")
        sys.exit(result.returncode)


def _module_basename(module_name: str) -> str:
    """Extract the last dotted component: 'model_traced.foo_model_traced' -> 'foo_model_traced'."""
    return module_name.rsplit(".", 1)[-1]


def step_clean_previous_outputs(
    module_name: str,
    sweep_trace_output: Path,
    sweep_trace_split_dir: Path,
    dry_run: bool,
) -> None:
    """Remove all artifacts from a previous pipeline run so we start fresh."""
    import shutil

    print(f"\n{SEPARATOR}")
    print("Step 0: Clean previous pipeline outputs")
    print(SEPARATOR)

    removed = 0

    # 1. Sweep trace JSON
    if sweep_trace_output.is_file():
        print(f"  Removing sweep trace: {sweep_trace_output}")
        if not dry_run:
            sweep_trace_output.unlink()
        removed += 1
    else:
        print(f"  Sweep trace already absent: {sweep_trace_output}")

    # 2. Sweep trace split directory
    if sweep_trace_split_dir.is_dir():
        print(f"  Removing sweep trace split dir: {sweep_trace_split_dir}")
        if not dry_run:
            shutil.rmtree(sweep_trace_split_dir)
        removed += 1
    else:
        print(f"  Sweep trace split dir already absent: {sweep_trace_split_dir}")

    # 3. Vector export files matching this module
    pattern = str(VECTORS_EXPORT_DIR / f"{module_name}__*")
    matches = sorted(glob.glob(pattern))
    if matches:
        for path in matches:
            print(f"  Removing vector file: {path}")
            if not dry_run:
                os.remove(path)
        removed += len(matches)
    else:
        print(f"  No existing vector files matching: {module_name}__*")

    print(f"  Cleaned {removed} artifact(s)")


def step_generate_vectors(
    module_name: str,
    model_trace: Path,
    dry_run: bool,
    suite_name: str | None = None,
) -> None:
    print(f"\n{SEPARATOR}")
    print("Step 1/4: Generate sweep vectors")
    print(SEPARATOR)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tests" / "sweep_framework" / "sweeps_parameter_generator.py"),
        "--module-name",
        module_name,
        "--master-trace",
        str(model_trace),
    ]
    if suite_name:
        cmd += ["--suite-name", suite_name]
    _run(cmd, dry_run=dry_run)


def step_run_sweep_under_tracer(
    module_name: str,
    suite: str,
    mesh_shape: str,
    arch_name: str,
    sweep_trace_output: Path,
    dry_run: bool,
) -> None:
    print(f"\n{SEPARATOR}")
    print("Step 2/4: Run sweep under operations tracer")
    print(SEPARATOR)

    tracer_script = str(REPO_ROOT / "model_tracer" / "generic_ops_tracer.py")
    runner_script = str(REPO_ROOT / "tests" / "sweep_framework" / "sweeps_runner.py")

    cmd = [
        sys.executable,
        tracer_script,
        runner_script,
        "-o",
        str(sweep_trace_output),
        "--",
        "--module-name",
        module_name,
        "--suite-name",
        suite,
        "--vector-source",
        "vectors_export",
        "--result-dest",
        "results_export",
        "--main-proc-verbose",
    ]

    extra_env = {
        "MESH_DEVICE_SHAPE": mesh_shape,
        "LEAD_MODELS_RUN": "1",
        "ARCH_NAME": arch_name,
    }

    _run(cmd, env=extra_env, dry_run=dry_run)


def step_split_trace(
    trace_json: Path,
    split_dir: Path,
    label: str,
    dry_run: bool,
    force: bool = False,
) -> None:
    """Split a trace JSON by operation using split_model_trace.py."""
    print(f"\n{SEPARATOR}")
    print(f"Step: Split {label}")
    print(SEPARATOR)

    if split_dir.is_dir() and any(split_dir.iterdir()):
        if force:
            print(f"  --force: removing stale split directory: {split_dir}")
            if not dry_run:
                import shutil

                shutil.rmtree(split_dir)
        elif trace_json.is_file() and split_dir.stat().st_mtime < trace_json.stat().st_mtime:
            print(f"  Split directory is older than source JSON, re-splitting: {split_dir}")
            if not dry_run:
                import shutil

                shutil.rmtree(split_dir)
        else:
            print(f"  Split directory already exists, skipping: {split_dir}")
            return

    cmd = [
        sys.executable,
        str(REPO_ROOT / "model_tracer" / "traced_operations" / "split_model_trace.py"),
        str(trace_json),
        "-o",
        str(split_dir),
    ]
    _run(cmd, dry_run=dry_run)


def step_print_validation_pairs(
    model_trace_split_dir: Path,
    sweep_trace_split_dir: Path,
    dry_run: bool,
) -> None:
    print(f"\n{SEPARATOR}")
    print("Validation file pairs")
    print(SEPARATOR)

    if dry_run:
        print("  (dry-run: split directories have not been created yet)")
        print(f"  Model trace split dir: {model_trace_split_dir}")
        print(f"  Sweep trace split dir: {sweep_trace_split_dir}")
        return

    if not sweep_trace_split_dir.is_dir():
        print(f"  WARNING: Sweep trace split directory not found: {sweep_trace_split_dir}")
        return

    sweep_ops = sorted(d.name for d in sweep_trace_split_dir.iterdir() if d.is_dir())

    model_ops = set()
    if model_trace_split_dir.is_dir():
        model_ops = {d.name for d in model_trace_split_dir.iterdir() if d.is_dir()}

    matched = []
    sweep_only = []

    for op_dir_name in sweep_ops:
        sweep_file = sweep_trace_split_dir / op_dir_name / "ttnn_operations_master.json"
        model_file = model_trace_split_dir / op_dir_name / "ttnn_operations_master.json"

        if model_file.is_file():
            matched.append((op_dir_name, model_file, sweep_file))
        else:
            sweep_only.append(op_dir_name)

    model_only = sorted(model_ops - set(sweep_ops))

    print(f"\n  Matched operations: {len(matched)}")
    print(f"  Sweep-only (no model match): {len(sweep_only)}")
    print(f"  Model-only (not tested): {len(model_only)}")

    if matched:
        print("\n  Ready for validation. Copy/paste the exact chat input below:\n")
        for op_dir_name, model_file, sweep_file in matched:
            op_display = op_dir_name.replace("_", ".")
            chat_input = (
                f"Use @validate-sweep-trac.mdc to validate {op_display}. " f"Model: {model_file} Sweep: {sweep_file}"
            )
            print(f"    {op_display}:")
            print(f"      Chat input: {chat_input}")

    if sweep_only:
        print("\n  Operations in sweep trace but NOT in model trace split:")
        for name in sweep_only:
            print(f"    {name.replace('_', '.')}")

    if model_only:
        print("\n  Operations in model trace but NOT in sweep trace (expected):")
        for name in model_only[:15]:
            print(f"    {name.replace('_', '.')}")
        if len(model_only) > 15:
            print(f"    ... and {len(model_only) - 15} more")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="Sweep Validation Pipeline",
        description=(
            "Orchestrate the full sweep-vs-model-trace validation workflow: "
            "generate vectors, run sweep under tracer, split traces, and "
            "print file pairs for cursor-rule validation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model-trace",
        required=True,
        type=str,
        help="Path to the fresh model trace JSON (e.g., ttnn_operations_master_UF_EV_B9_GWH01_deepseek.json)",
    )
    parser.add_argument(
        "--module-name",
        required=True,
        type=str,
        help="Sweep module name (e.g., model_traced.all_gather_async_model_traced)",
    )
    parser.add_argument(
        "--suite",
        default="model_traced",
        type=str,
        help="Suite name to run (default: model_traced)",
    )
    parser.add_argument(
        "--mesh-shape",
        default="4x8",
        type=str,
        help="Mesh device shape, e.g., 4x8 (default: 4x8)",
    )
    parser.add_argument(
        "--arch-name",
        default="wormhole_b0",
        type=str,
        help="Architecture name (default: wormhole_b0)",
    )
    parser.add_argument(
        "--sweep-trace-output",
        type=str,
        default=None,
        help="Override path for the sweep trace JSON output (default: auto-derived from module name)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without running them",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-split of trace directories even if they already exist",
    )

    args = parser.parse_args()

    model_trace = Path(args.model_trace).resolve()
    if not args.dry_run and not model_trace.is_file():
        print(f"ERROR: Model trace file not found: {model_trace}")
        return 1

    basename = _module_basename(args.module_name)

    if args.sweep_trace_output:
        sweep_trace_output = Path(args.sweep_trace_output).resolve()
    else:
        sweep_trace_output = TRACED_OPS_DIR / f"sweep_trace_{basename}.json"

    sweep_trace_split_dir = sweep_trace_output.parent / f"{sweep_trace_output.stem}_split"
    model_trace_split_dir = model_trace.parent / f"{model_trace.stem}_split"

    print(SEPARATOR)
    print("Sweep Validation Pipeline")
    print(SEPARATOR)
    print(f"  Model trace:        {model_trace}")
    print(f"  Module:             {args.module_name}")
    print(f"  Suite:              {args.suite}")
    print(f"  Mesh shape:         {args.mesh_shape}")
    print(f"  Arch:               {args.arch_name}")
    print(f"  Sweep trace output: {sweep_trace_output}")
    print(f"  Sweep trace split:  {sweep_trace_split_dir}")
    print(f"  Model trace split:  {model_trace_split_dir}")
    if args.dry_run:
        print(f"  Mode:               DRY RUN")

    # Step 0: Remove all artifacts from a previous run
    step_clean_previous_outputs(
        module_name=args.module_name,
        sweep_trace_output=sweep_trace_output,
        sweep_trace_split_dir=sweep_trace_split_dir,
        dry_run=args.dry_run,
    )

    # Step 1: Generate vectors
    step_generate_vectors(
        module_name=args.module_name,
        model_trace=model_trace,
        dry_run=args.dry_run,
        suite_name=args.suite,
    )

    # Step 2: Run sweep under tracer
    step_run_sweep_under_tracer(
        module_name=args.module_name,
        suite=args.suite,
        mesh_shape=args.mesh_shape,
        arch_name=args.arch_name,
        sweep_trace_output=sweep_trace_output,
        dry_run=args.dry_run,
    )

    # Step 3: Split sweep trace
    step_split_trace(
        trace_json=sweep_trace_output,
        split_dir=sweep_trace_split_dir,
        label="sweep trace (3/4)",
        dry_run=args.dry_run,
        force=args.force,
    )

    # Step 4: Ensure model trace is split
    step_split_trace(
        trace_json=model_trace,
        split_dir=model_trace_split_dir,
        label="model trace (4/4)",
        dry_run=args.dry_run,
        force=args.force,
    )

    # Print validation pairs
    step_print_validation_pairs(
        model_trace_split_dir=model_trace_split_dir,
        sweep_trace_split_dir=sweep_trace_split_dir,
        dry_run=args.dry_run,
    )

    print(f"\n{SEPARATOR}")
    print("Pipeline complete")
    print(SEPARATOR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
