# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistently run full GPQA only after exact-source eight-replica qualification."""

import argparse
import hashlib
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.galaxy_serving import qualified_groups, verify_qualified_source
from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / name for name in ("conftest.py", "pytest.ini")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json", ".sh"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(state="cpu_validation", source=str(args.source), source_sha256=hashes, passed=False)
    save(status_path, status)
    try:
        policy = model / "config/precision_single_step_shared_qk.json"
        env = environment(args.task, args.source, args.weights)
        env.update(QWEN_PRECISION_CONFIG=str(policy))
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(key, None)
        # Qualification hashes include the effective override; use the same
        # explicit artifact for parent validation and every child worker.
        os.environ["QWEN_PRECISION_CONFIG"] = str(policy)
        subprocess.run(
            [
                str(args.task / "python_env/bin/python"),
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
        while True:
            result = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    args.after_unit,
                    "-p",
                    "LoadState",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                    "-p",
                    "MainPID",
                    "-p",
                    "Result",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            props = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            status.update(state="waiting_for_g0", dependency=props)
            save(status_path, status)
            missing = props.get("LoadState") == "not-found"
            terminal = props.get("ActiveState") in ("inactive", "failed") and props.get("MainPID") == "0"
            if missing or terminal:
                if not missing and props.get("Result") != "success":
                    raise RuntimeError(f"G0 service failed: {props}")
                break
            time.sleep(20)
        qualification_path = args.after_results / "full-model.json"
        qualification = json.loads(qualification_path.read_text())
        groups = qualified_groups(qualification)
        verify_qualified_source(qualification, model)
        suites = ET.parse(args.after_results / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise RuntimeError("G0 test process lacks a clean passing JUnit receipt")
        for name, digest in hashes.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Frozen serving source changed: {name}")
        command = [
            "/bin/bash",
            str(model / "demo/run_galaxy_serving.sh"),
            str(args.task),
            str(args.results / "evaluation"),
            str(qualification_path),
            str(args.source),
            "--exit-after-eval",
            "--port",
            str(args.port),
        ]
        status.update(state="serving_and_evaluation_or_waiting_for_lock", command=command, qualified_groups=groups)
        save(status_path, status)
        run_capture(command, cwd=args.source, env=env, root=args.results, timeout=21600)
        report = json.loads((args.results / "evaluation/deployment.json").read_text())
        gpqa = report.get("gpqa", {})
        if (
            report.get("state") != "evaluation_completed"
            or not report.get("owned_processes_stopped")
            or (gpqa.get("completed_samples") != 198 or gpqa.get("full_dataset") is not True)
        ):
            raise RuntimeError("Full evaluation or owned-worker shutdown did not complete")
        status.update(
            state="completed",
            passed=report["passed"],
            gpqa=gpqa,
            resident_endpoint=False,
            owned_processes_stopped=True,
            hardware_reset_required=Path("/tmp/tt-device.dirty").exists(),
        )
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
    parser.add_argument("--port", type=int, default=8078)
    run(parser.parse_args())
