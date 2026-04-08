#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Run gen_util_report over a grid of (seqlen, batch_size) for simple_text_demo (device-perf).

This is distinct from tools/tracy/profile_sweeps.py (YAML op-level sweeps).

Example (prefill):
  python tools/sweep/run_llm_util_sweep.py \\
    --experiment-dir ./exp/prefill \\
    --mode prefill \\
    --seqlens 256,1k,2k \\
    --batch-sizes 1

Example (decode):
  python tools/sweep/run_llm_util_sweep.py \\
    --experiment-dir ./exp/decode \\
    --mode decode \\
    --seqlens 1k \\
    --batch-sizes 1,8,32 \\
    --max-generated-tokens 128
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

from sweep_common import (
    apply_profiler_env,
    build_pytest_argv,
    check_environment,
    grid_points,
    parse_comma_ints,
    parse_comma_strs,
    pytest_command_string,
    resolve_seqlen_with_max,
    run_collect_only,
    run_gen_util_report,
    tt_metal_home,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM util-report sweep via gen_util_report + simple_text_demo")
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Root output directory for all points")
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--seqlens", type=str, required=True, help="Comma list of presets (1k,256,…) or json paths")
    parser.add_argument("--batch-sizes", type=str, default="1", help="Comma-separated batch sizes (default: 1)")
    parser.add_argument("--num-layers", type=int, default=1, help="pytest --num_layers (default: 1)")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override max_seq_len for all points. Each preset has a default "
        "large enough for the full prompt (required for custom prompt paths).",
    )
    parser.add_argument(
        "--max-generated-tokens",
        type=int,
        default=None,
        help="pytest --max_generated_tokens (decode default: 128; prefill default: 2)",
    )
    parser.add_argument("--instruct", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-trace", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--steady-state", action="store_true", help="Pass --steady-state to gen_util_report")
    parser.add_argument("--single-model-iteration", action="store_true", help="Highest trace session only")
    parser.add_argument("--dry-run", action="store_true", help="pytest --collect-only per grid point")
    parser.add_argument("--skip-collect", action="store_true", help="Do not run collect_model_util_reports at end")
    parser.add_argument("--pytest-args", type=str, default="", help="Extra pytest args (shell-quoted string)")
    parser.add_argument("--hf-model", type=str, default=None, help="Sets HF_MODEL for child processes")
    parser.add_argument(
        "--max-prefill-chunk-size",
        type=int,
        default=None,
        help="Sets MAX_PREFILL_CHUNK_SIZE env for child processes",
    )
    parser.add_argument(
        "--noc-timeout",
        type=int,
        default=None,
        help="Timeout in seconds for the NOC trace pass in gen_util_report. "
        "If exceeded the report is produced without NOC metrics.",
    )

    args = parser.parse_args()

    ttm = tt_metal_home()
    check_environment(ttm)
    apply_profiler_env()
    os.chdir(ttm)

    if args.hf_model:
        os.environ["HF_MODEL"] = args.hf_model
    if args.max_prefill_chunk_size is not None:
        os.environ["MAX_PREFILL_CHUNK_SIZE"] = str(args.max_prefill_chunk_size)

    seqlen_specs = parse_comma_strs(args.seqlens)
    batches = parse_comma_ints(args.batch_sizes)
    extra = shlex.split(args.pytest_args) if args.pytest_args.strip() else []

    if args.mode == "decode":
        mgt = args.max_generated_tokens if args.max_generated_tokens is not None else 128
    else:
        mgt = args.max_generated_tokens if args.max_generated_tokens is not None else 2

    experiment = args.experiment_dir.resolve()
    experiment.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, str]] = []  # (label, cmd_string, status)

    for seqlen_spec, batch in grid_points(seqlen_specs, batches):
        label_sq, msl, input_prompts = resolve_seqlen_with_max(seqlen_spec, args.max_seq_len)
        label = f"seqlen_{label_sq}_batch_{batch}_layers_{args.num_layers}"
        out = experiment / label

        argv = build_pytest_argv(
            mode=args.mode,
            batch_size=batch,
            num_layers=args.num_layers,
            input_prompts=input_prompts,
            max_seq_len=msl,
            max_generated_tokens=mgt,
            instruct=args.instruct,
            enable_trace=args.enable_trace,
            extra_args=extra,
        )

        if args.dry_run:
            cmd_str = pytest_command_string(argv + ["--collect-only"])
            try:
                run_collect_only(ttm, argv + ["--collect-only"])
                results.append((label, cmd_str, "OK"))
            except SystemExit:
                results.append((label, cmd_str, "FAILED"))
            continue

        pytest_cmd = pytest_command_string(argv)
        cmd_str = shlex.join(
            [sys.executable, "tools/hw_debug/gen_util_report.py", "-o", str(out), "-c", pytest_cmd]
            + (["--steady-state"] if args.steady_state else [])
            + (["--single-model-iteration"] if args.single_model_iteration else [])
        )
        try:
            run_gen_util_report(
                ttm,
                out,
                pytest_cmd,
                steady_state=args.steady_state,
                single_model_iteration=args.single_model_iteration,
                noc_timeout=args.noc_timeout,
            )
            results.append((label, cmd_str, "OK"))
        except subprocess.CalledProcessError:
            results.append((label, cmd_str, "FAILED"))

    print("\n" + "=" * 80)
    print("SWEEP SUMMARY")
    print("=" * 80)
    for label, cmd_str, status in results:
        marker = "PASS" if status == "OK" else "FAIL"
        print(f"  [{marker}] {label}")
        print(f"         {cmd_str}")
    passed = sum(1 for _, _, s in results if s == "OK")
    failed = sum(1 for _, _, s in results if s != "OK")
    print(f"\n  {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 80)

    if not args.dry_run and not args.skip_collect and passed > 0:
        collect_script = ttm / "tools" / "sweep" / "collect_model_util_reports.py"
        subprocess.run([sys.executable, str(collect_script), str(experiment)], cwd=ttm, check=True)


if __name__ == "__main__":
    main()
