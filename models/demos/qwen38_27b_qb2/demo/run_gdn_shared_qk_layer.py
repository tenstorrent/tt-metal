# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistently advance passing shared-Q/K gains to a real-weight GDN block."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save, wait_for_sweep
from models.demos.qwen38_27b_qb2.tests.gdn_shared_qk import BATCHES, compare


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / p for p in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(state="cpu_validation", source=str(args.source), source_sha256=hashes, promoted_to_model=False)
    save(status_path, status)
    try:
        env = environment(args.task, args.source, args.weights)
        for key in ("TT_METAL_SIMULATOR", "TT_METAL_KERNEL_PATH", "TT_METAL_DISABLE_SFPLOADMACRO"):
            env.pop(key, None)
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
        wait_for_sweep(
            args.after_unit, args.after_results, status, status_path, receipt_names=("queue.json", "shared.json")
        )
        dependency = json.loads((args.after_results / "queue.json").read_text())
        component = json.loads((args.after_results / "shared.json").read_text())
        if component.get("passed") is not True or component.get("long_horizon", [{}])[-1].get("steps") != 4096:
            raise RuntimeError("Component accuracy and long-horizon gate is incomplete")
        for name, digest in dependency["source_sha256"].items():
            if "/tt/gdn_step/" in name and hashes.get(name) != digest:
                raise RuntimeError("Real-weight kernel source differs from the passing component")
        comparisons = [compare(component["cases"][i : i + 3]) for i in range(0, 3 * len(BATCHES), 3)]
        if comparisons != component["comparisons"]:
            raise RuntimeError("Component summary differs from raw comparisons")
        eligible = [c for c in comparisons if c["batch"] in (16, 32) and (c["qualified_speedup"] or 0) >= 1.02]
        status["component_comparisons"] = comparisons
        if not eligible:
            status.update(
                state="completed",
                cleanup_completed=True,
                hardware_run=False,
                reason="No qualified >=2% adapter gain at B16/B32; preserve the candidate without further hardware work",
            )
            return
        for name, digest in hashes.items():
            if hashlib.sha256((args.source / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Frozen real-weight source changed")
        env.update(QWEN_GDN_SHARED_QK_LAYER="1", QWEN_GDN_LAYER_RECEIPT=str(args.results / "layer.json"))
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            "10800",
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_gdn_shared_qk_layer.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=2400",
            f"--junitxml={args.results}/hardware.xml",
        ]
        status.update(state="real_weight_diagnostic_or_waiting_for_lock", command=command, hardware_run=True)
        save(status_path, status)
        run_capture(command, cwd=args.source, env=env, root=args.results, timeout=10980)
        report = json.loads((args.results / "layer.json").read_text())
        if (
            report.get("state") != "completed"
            or not report.get("passed")
            or not report.get("cleanup_completed")
            or len(report.get("cases", [])) != 6
            or len(report.get("comparisons", [])) != 2
        ):
            raise RuntimeError("Incomplete real-weight GDN comparison")
        for index, batch in enumerate((32, 16)):
            group = report["cases"][3 * index : 3 * index + 3]
            checked = compare(group)
            checked.update(scope=report["scope"], projected_output_bit_identical=True, real_weights=True)
            if checked != report["comparisons"][index] or checked["batch"] != batch:
                raise RuntimeError("Real-weight summary differs from raw comparisons")
            if any(
                row["projected_output_sha256_per_rank"] != group[0]["projected_output_sha256_per_rank"] for row in group
            ):
                raise RuntimeError("Projected GDN block output changed")
        suites = ET.parse(args.results / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise RuntimeError("Real-weight hardware test did not pass")
        status.update(state="completed", cleanup_completed=True, comparisons=report["comparisons"])
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results", "after-results"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--after-unit", required=True)
    run(parser.parse_args())
