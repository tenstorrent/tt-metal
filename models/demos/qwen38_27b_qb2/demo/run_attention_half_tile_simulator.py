# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent CPU-only partial-tile numerical screen; never takes the device lock."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.attention_half_tile import VARIANTS, build_overlay, compare_simulator


def run(args):
    args.results.mkdir()
    path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json")]
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="cpu_validation",
        source=str(args.source),
        source_sha256=hashes,
        physical_devices_accessed=False,
        promoted_to_model=False,
        runs=[],
    )
    save(path, status)
    try:
        env = environment(args.task, args.source, args.weights)
        for name in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(name, None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        python = str(args.task / "python_env/bin/python")
        # Unit tests only import runtime libraries; they never open a device.
        command = [
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
        ]
        subprocess.run(command, cwd=args.source, env=env, check=True, timeout=600)
        reports = []
        for variant in VARIANTS:
            for name, digest in hashes.items():
                if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                    raise RuntimeError("Queued simulator source changed")
            directory = args.results / variant
            directory.mkdir()
            manifest = build_overlay(args.task / "metal", directory / "overlay", variant)
            simenv = dict(env)
            simenv.update(
                TT_METAL_SIMULATOR=str(args.simulator),
                TT_METAL_SLOW_DISPATCH_MODE="1",
                TT_METAL_DISABLE_SFPLOADMACRO="1",
                TT_METAL_INSPECTOR_RPC="0",
                TT_METAL_KERNEL_PATH=manifest["overlay"],
                TT_METAL_CACHE=str(directory / "jit-cache"),
            )
            command = [
                python,
                "-m",
                "models.demos.qwen38_27b_qb2.demo.probe_attention_half_tile",
                "--manifest",
                str(directory / "overlay/manifest.json"),
                "--output",
                str(directory / "probe.json"),
            ]
            status["runs"].append(dict(variant=variant, command=command, overlay=manifest))
            status.update(state="simulator", active_variant=variant)
            save(path, status)
            subprocess.run(command, cwd=args.source, env=simenv, check=True, timeout=2700)
            reports.append(json.loads((directory / "probe.json").read_text()))
        comparison = compare_simulator(reports)
        (args.results / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
        status.update(
            state="completed", cleanup_completed=True, candidate_accuracy_passed=comparison["candidate_accuracy_passed"]
        )
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "simulator"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
