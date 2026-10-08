# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent physical partial-query screen gated by completed simulator evidence."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tests.attention_half_tile import (
    COMMON,
    HARDWARE_CASES,
    build_overlay,
    compare_simulator,
)


def run(args):
    args.results.mkdir()
    path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    status = dict(
        state="checking_simulator",
        promoted_to_model=False,
        source=str(args.source),
        hardware_lock="/tmp/tt-device.lock",
        planned_geometries=HARDWARE_CASES,
    )
    save(path, status)
    try:
        reports = [
            json.loads((args.sim_results / name / "probe.json").read_text()) for name in ("native", "accurate_partial")
        ]
        comparison = compare_simulator(reports)
        if not comparison["candidate_accuracy_passed"]:
            raise ValueError("Simulator numerical gate did not pass")
        manifest = build_overlay(args.task / "metal", args.results / "overlay", "accurate_partial")
        if manifest["overlay_sha256"][str(COMMON)] != reports[1]["overlay"]["overlay_sha256"][str(COMMON)]:
            raise ValueError("Hardware candidate differs from the completed simulator experiment")
        status["simulator_gate"] = comparison
        files = [args.source / n for n in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
        files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
        hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        status.update(state="cpu_validation", source_sha256=hashes, overlay=manifest)
        save(path, status)
        env = environment(args.task, args.source, args.weights)
        for key in (
            "TT_METAL_SIMULATOR",
            "TT_METAL_SLOW_DISPATCH_MODE",
            "TT_METAL_DISABLE_SFPLOADMACRO",
            "TT_METAL_KERNEL_PATH",
        ):
            env.pop(key, None)
        python = str(args.task / "python_env/bin/python")
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
        for name, digest in hashes.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Frozen hardware diagnostic source changed")
        env.update(
            TT_METAL_KERNEL_PATH=manifest["overlay"],
            TT_METAL_CACHE=str(args.results / "jit-cache"),
            QWEN_HALF_TILE_HARDWARE="1",
            QWEN_HALF_TILE_MANIFEST=str(args.results / "overlay/manifest.json"),
            QWEN_HALF_TILE_RECEIPT=str(args.results / "hardware.json"),
        )
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            "10800",
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_attention_half_tile_hardware.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=2700",
            f"--junitxml={args.results}/hardware.xml",
        ]
        status.update(state="hardware_diagnostic_or_waiting_for_lock", command=command)
        save(path, status)
        subprocess.run(command, cwd=args.source, env=env, check=True, timeout=11100)
        report = json.loads((args.results / "hardware.json").read_text())
        if (
            report.get("state") != "completed"
            or not report.get("passed")
            or not report.get("cleanup_completed")
            or len(report.get("comparisons", [])) != len(HARDWARE_CASES)
            or not report.get("compilation_evidence")
        ):
            raise RuntimeError("Incomplete physical hardware diagnostic")
        suites = ET.parse(args.results / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise RuntimeError("Physical hardware JUnit did not pass")
        status.update(
            state="completed", cleanup_completed=True, all_candidates_qualified=report["all_candidates_qualified"]
        )
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:4000]))
        raise
    finally:
        save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "sim-results"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
