# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded reader-barrier A/B tests after the placement diagnostic closes."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep
from models.demos.qwen38_27b_qb2.tests.attention_reader import VARIANTS, build_overlay, compare_readers


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
        wait_for_sweep(args.after_unit, args.after_results, status, status_path, receipt_names=("placement.json",))
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Queued source changed: {relative}")
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
        reports = []
        for index, variant in enumerate((*VARIANTS, "native")):
            directory = args.results / f"{index}-{variant}"
            directory.mkdir()
            manifest = build_overlay(args.task / "metal", directory / "overlay", variant)
            # Separate cache per process: identical public kernel names must not
            # reuse a binary built from another source override.
            variant_env = dict(env)
            variant_env.update(
                TT_METAL_KERNEL_PATH=manifest["overlay"],
                TT_METAL_CACHE=str(directory / "jit-cache"),
                QWEN_ATTENTION_READER="1",
                QWEN_ATTENTION_READER_MANIFEST=str(directory / "overlay/manifest.json"),
                QWEN_ATTENTION_READER_RECEIPT=str(directory / "reader.json"),
            )
            command = [
                "timeout",
                "--signal=TERM",
                "--kill-after=180",
                "1800",
                "/bin/bash",
                str(args.task / "source/scripts/run_safe_pytest.sh"),
                str(model / "tests/test_attention_reader.py"),
                f"--rootdir={args.source}",
                "-c",
                str(args.source / "pytest.ini"),
                "-vv",
                "-s",
                "--tb=short",
                "--timeout=1620",
                f"--junitxml={directory}/hardware.xml",
            ]
            status["runs"].append(dict(variant=variant, directory=str(directory), command=command, overlay=manifest))
            status.update(state="hardware_diagnostic", active_variant=variant, active_index=index)
            save(status_path, status)
            subprocess.run(command, cwd=args.source, env=variant_env, check=True, timeout=2100)
            receipt = json.loads((directory / "reader.json").read_text())
            if (
                receipt.get("passed") is not True
                or receipt.get("state") != "completed"
                or receipt.get("cleanup_completed") is not True
                or not receipt.get("compilation_evidence")
            ):
                raise RuntimeError("Reader variant lacks a passing compiled-override and clean-device receipt")
            reports.append(receipt)
        status.update(state="completed", comparisons=compare_readers(reports), promoted_to_model=False)
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
