# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Matched long-context capacity runs, each in a fresh model process.

The previous mixed-geometry run fragmented DRAM before its 128K/B16 cache
allocation. Isolate every cell to distinguish cold-process capacity from
capacity after changing geometry, and measure both existing recurrence paths.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep

CASES = ((131072, 16), (32768, 32), (262016, 8))
POLICIES = {"native": "precision_accurate_decode.json", "single-step": "precision_single_step_gdn.json"}


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini")]
    files.extend(path for path in model.rglob("*") if path.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(path.relative_to(args.source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    status = dict(state="queued", source=str(args.source), source_sha256=hashes, precision_change=False, runs=[])
    save(status_path, status)
    try:
        wait_for_sweep(
            args.after_unit,
            args.after_results,
            status,
            status_path,
            receipt_names=tuple(
                f"{i}-{variant}/reader.json" for i, variant in enumerate(("native", "kv4", "kv8", "kv16", "native"))
            ),
        )
        if json.loads((args.after_results / "queue.json").read_text()).get("state") != "completed":
            raise RuntimeError("Reader queue lacks its completed comparison")
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")

        from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import compare
        from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import render as render_comparison
        from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render, save_report
        from models.demos.qwen38_27b_qb2.tt.precision import load_precision

        policies = [load_precision(model / "config" / filename) for filename in POLICIES.values()]
        if [p["decode_recurrence"] for p in policies] != ["native", "single_step"] or (
            {k: v for k, v in policies[0].items() if k not in ("config_id", "decode_recurrence")}
            != {k: v for k, v in policies[1].items() if k not in ("config_id", "decode_recurrence")}
        ):
            raise ValueError("Capacity comparison must preserve precision and vary only recurrence")
        env = environment(args.task, args.source, args.weights)
        env.pop("TT_METAL_KERNEL_PATH", None)
        python = str(args.task / "python_env/bin/python")
        status["state"] = "cpu_validation"
        save(status_path, status)
        subprocess.run(
            [
                python,
                "-m",
                "pytest",
                str(model / "tests/unit"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-q",
                f"--junitxml={args.results}/unit.xml",
            ],
            cwd=args.source,
            env=env,
            check=True,
            timeout=600,
        )
        for length, batch in CASES:
            pair = args.results / f"s{length}-b{batch}"
            paths = []
            for variant, policy_file in POLICIES.items():
                directory = pair / variant
                plan = make_plan(1, batches=(batch,), input_lengths=(length,), max_pool_tokens=2359296)
                plan.update(
                    recurrence_variant=variant,
                    qualification_scope="Fresh-process full-model capacity/performance; no online-eval or full-Galaxy claim",
                    allocation_history="No earlier request geometry in this process",
                )
                save_report(plan, directory)
                render(plan, directory)
                variant_env = dict(env, QWEN_PRECISION_CONFIG=str(model / "config" / policy_file))
                command = [
                    python,
                    "-m",
                    "models.demos.qwen38_27b_qb2.demo.run_sweep_attempts",
                    "--task",
                    str(args.task),
                    "--source",
                    str(args.source),
                    "--results",
                    str(directory),
                    "--timeout",
                    "7200",
                ]
                status["runs"].append(
                    dict(input_tokens=length, batch=batch, variant=variant, directory=str(directory), command=command)
                )
                status.update(
                    state="hardware_sweep", active_input_tokens=length, active_batch=batch, active_variant=variant
                )
                save(status_path, status)
                subprocess.run(command, env=variant_env, cwd=args.source, check=True, timeout=7500)
                receipt = directory / "sweep.json"
                measured = json.loads(receipt.read_text())
                if (
                    measured.get("state") not in ("completed", "completed_with_oom")
                    or measured.get("cleanup_completed") is not True
                ):
                    raise RuntimeError("Capacity cell lacks a terminal sweep and clean-device receipt")
                paths.append(receipt)
            comparison = compare(*paths)
            render_comparison(comparison, pair / "comparison")
            status.setdefault("comparisons", []).append(comparison)
            save(status_path, status)
        status.update(state="completed", cleanup_completed=True, promoted_to_serving=False)
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "after-results"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
