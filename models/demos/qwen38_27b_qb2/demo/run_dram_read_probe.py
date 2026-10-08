# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, persistent raw read calibration; never promotes a model policy."""

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from models.demos.qwen38_27b_qb2.demo.run_bounded_layer_profile import run_capture
from models.demos.qwen38_27b_qb2.demo.run_long_context_capacity import environment, save
from models.demos.qwen38_27b_qb2.tt.dram_read_probe import variants


def run(args):
    args.results.mkdir()
    status_path = args.results / "queue.json"
    model = args.source / "models/demos/qwen38_27b_qb2"
    files = [args.source / p for p in ("conftest.py", "pytest.ini", "scripts/run_safe_pytest.sh")]
    files.extend(p for p in model.rglob("*") if p.suffix in (".py", ".cpp", ".h", ".hpp", ".json"))
    hashes = {str(p.relative_to(args.source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    status = dict(
        state="cpu_validation",
        source=str(args.source),
        source_sha256=hashes,
        hardware_lock="/tmp/tt-device.lock",
        promoted_to_model=False,
        scope="raw BFP8-sized page read calibration; no attention or precision change",
    )
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
        for relative, digest in hashes.items():
            if hashlib.sha256((args.source / relative).read_bytes()).hexdigest() != digest:
                raise RuntimeError("Frozen read-probe source changed")
        env.update(QWEN_DRAM_READ_PROBE="1", QWEN_DRAM_READ_RECEIPT=str(args.results / "probe.json"))
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=180",
            "14400",
            "/bin/bash",
            str(args.source / "scripts/run_safe_pytest.sh"),
            str(model / "tests/test_dram_read_probe.py"),
            f"--rootdir={args.source}",
            "-c",
            str(args.source / "pytest.ini"),
            "-vv",
            "-s",
            "--tb=short",
            "--timeout=1800",
            f"--junitxml={args.results}/hardware.xml",
        ]
        status.update(state="hardware_diagnostic_or_waiting_for_lock", command=command)
        save(status_path, status)
        run_capture(command, cwd=args.source, env=env, root=args.results, timeout=14580)
        report = json.loads((args.results / "probe.json").read_text())
        if (
            report.get("state") != "completed"
            or not report.get("passed")
            or not report.get("cleanup_completed")
            or len(report.get("validation", [])) != len(variants())
            or len(report.get("cases", [])) != 3 * (len(variants()) + 1)
            or len(report.get("comparisons", [])) != 3
            or not all(row.get("full_bytes_equal_all_four_ranks") for row in report["validation"])
            or not all(row.get("packet_markers_passed_all_four_ranks") for row in report["cases"])
        ):
            raise RuntimeError("Incomplete raw read integrity or timing evidence")
        suites = ET.parse(args.results / "hardware.xml").getroot().findall(".//testsuite")
        if sum(int(s.get("tests", 0)) for s in suites) != 1 or any(
            int(s.get(k, 0)) for s in suites for k in ("failures", "errors", "skipped")
        ):
            raise RuntimeError("Read-probe hardware test did not pass")
        status.update(state="completed", cleanup_completed=True, comparisons=report["comparisons"])
    except BaseException as error:
        status.update(state="failed", error=dict(type=type(error).__name__, message=str(error)[:3000]))
        raise
    finally:
        save(status_path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("task", "source", "weights", "results"):
        parser.add_argument("--" + name, required=True, type=Path)
    run(parser.parse_args())
