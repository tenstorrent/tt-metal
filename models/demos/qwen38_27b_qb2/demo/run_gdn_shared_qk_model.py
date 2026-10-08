# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare opt-in shared Q/K against single-step GDN at fixed precision and batch."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep
from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import compare
from models.demos.qwen38_27b_qb2.tests.compare_gdn_sweeps import render as render_comparison
from models.demos.qwen38_27b_qb2.tests.gdn_shared_qk import compare as compare_layer
from models.demos.qwen38_27b_qb2.tests.sweep_report import make_plan, render, save_report
from models.demos.qwen38_27b_qb2.tt.precision import load_precision

CASES = ((32768, 32), (16384, 32), (131072, 16), (262016, 8))
POLICIES = {
    "single-step": "precision_single_step_gdn.json",
    "shared-qk": "precision_single_step_shared_qk.json",
}
RECURRENCES = ("single_step", "single_step_shared_qk")


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="queued",
        source=str(args.source),
        source_sha256=hashes,
        precision_change=False,
        promoted_to_serving=False,
        runs=[],
        hardware_lock="/tmp/tt-device.lock",
    )
    save(status_path, status)
    try:
        wait_for_sweep(
            args.after_unit, args.after_results, status, status_path, receipt_names=("queue.json", "layer.json")
        )
        dependency = json.loads((args.after_results / "queue.json").read_text())
        layer = json.loads((args.after_results / "layer.json").read_text())
        if layer.get("passed") is not True or len(layer.get("cases", [])) != 6:
            raise RuntimeError("Require the passing physical real-weight GDN layer comparison")
        for i, batch in enumerate((32, 16)):
            group = layer["cases"][3 * i : 3 * i + 3]
            checked = compare_layer(group)
            if checked["batch"] != batch or (checked["qualified_speedup"] or 0) < 1.02:
                raise RuntimeError("Real-layer control drift or speedup gate failed")
            if any(
                row["projected_output_sha256_per_rank"] != group[0]["projected_output_sha256_per_rank"] for row in group
            ):
                raise RuntimeError("Real-layer projected output differs")
        for name, digest in dependency["source_sha256"].items():
            if "/tt/gdn_step/" in name and not name.endswith("/workspace.py") and hashes.get(name) != digest:
                raise RuntimeError("Model kernel differs from the passing real-layer experiment")
        policies = [load_precision(model / "config" / name) for name in POLICIES.values()]
        if tuple(p["decode_recurrence"] for p in policies) != RECURRENCES or (
            {k: v for k, v in policies[0].items() if k not in ("config_id", "decode_recurrence")}
            != {k: v for k, v in policies[1].items() if k not in ("config_id", "decode_recurrence")}
        ):
            raise ValueError("Shared Q/K comparison must preserve every precision setting")
        env = environment(args.task, args.source, args.weights)
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(key, None)
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
                "-o",
                "addopts=",
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
                for name, digest in hashes.items():
                    if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                        raise RuntimeError(f"Frozen source changed: {name}")
                directory = pair / variant
                plan = make_plan(1, batches=(batch,), input_lengths=(length,), max_pool_tokens=2359296)
                plan.update(
                    recurrence_variant=variant,
                    qualification_scope="Fresh-process full-model throughput and token equality; online eval and physical eight-replica scaling pending",
                    allocation_history="No earlier request geometry in this process",
                )
                save_report(plan, directory)
                render(plan, directory)
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
                status.update(state="hardware_sweep_or_waiting_for_lock", active_case=[length, batch, variant])
                save(status_path, status)
                run_capture(
                    command,
                    cwd=args.source,
                    env=dict(env, QWEN_PRECISION_CONFIG=str(model / "config" / policy_file)),
                    root=directory,
                    timeout=7380,
                )
                paths.append(directory / "sweep.json")
            comparison = compare(
                *paths, variants=tuple(POLICIES), recurrence_policies=RECURRENCES, require_same_output=True
            )
            render_comparison(comparison, pair / "comparison")
            status.setdefault("comparisons", []).append(comparison)
            save(status_path, status)
        status.update(state="completed", cleanup_completed=True, output_equivalence_passed=True)
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "after-results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
