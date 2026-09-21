# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sweep all 32 Galaxy devices using the unchanged, single-device reproducer."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def qualify(output, device, iterations, code, dry_run):
    result = dict(device=device, exit_code=code, status="error", iterations=0, faults=0)
    if dry_run and code == 0:
        result["status"] = "dry-run"
        return result
    try:
        report = json.loads((output / "report.json").read_text())
        launcher = json.loads((output / "launcher.json").read_text())
        count = report.get("iterations")
        faults = report.get("faults")
        qualified = (
            code in (0, 1)
            and launcher.get("result_qualified") is True
            and launcher.get("child_exit_code") == code
            and report.get("stage") == "complete"
            and report.get("device") == device
            and report.get("source_unchanged_end") is True
            and report.get("requested_iterations") == iterations
            and type(count) is int
            and 0 < count <= iterations
            and isinstance(faults, list)
            and code == int(bool(faults))
            and (code == 1 or count == iterations)
        )
        if qualified:
            result.update(status="fault" if code else "clean", iterations=count, faults=len(faults))
        else:
            result["error"] = "Incomplete or unqualified device result; inspect report.json and launcher.json"
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result["error"] = str(exc)
    return result


def run_sweep(checkout, output, iterations, dry_run=False):
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    output.mkdir(parents=True, exist_ok=False)
    devices = list(range(32))
    identity = dict(
        runner_name=os.environ.get("RUNNER_NAME"),
        container_hostname=platform.node(),
        platform=platform.platform(),
        github_sha=os.environ.get("GITHUB_SHA"),
        github_run_id=os.environ.get("GITHUB_RUN_ID"),
        github_run_attempt=os.environ.get("GITHUB_RUN_ATTEMPT"),
        device_namespace="UMD physical device ID",
        devices=devices,
        requested_iterations_per_device=iterations,
        dry_run=dry_run,
    )
    (output / "runner.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps(identity), flush=True)
    summary = dict(stage="running", devices=devices, iterations_per_device=iterations, dry_run=dry_run, results=[])

    def checkpoint():
        summary["clean_devices"] = [r["device"] for r in summary["results"] if r["status"] == "clean"]
        summary["fault_devices"] = [r["device"] for r in summary["results"] if r["status"] == "fault"]
        summary["error_devices"] = [r["device"] for r in summary["results"] if r["status"] == "error"]
        summary["total_iterations"] = sum(r["iterations"] for r in summary["results"])
        temporary = output / "sweep.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output / "sweep.json")

    # Separate processes preserve the original one-device workload and close each
    # mesh before proceeding. A fault or setup error must not hide later devices.
    for device in devices:
        summary["current_device"] = device
        checkpoint()
        device_output = output / f"device-{device:02d}"
        command = [
            sys.executable,
            str(Path(__file__).with_name("repro") / "run.py"),
            "--tt-metal",
            str(checkout),
            "--device",
            str(device),
            "--iterations",
            str(iterations),
            "--output",
            str(device_output),
        ]
        if dry_run:
            command.append("--dry-run")
        print(f"Galaxy sweep: starting device {device}/31", flush=True)
        try:
            child = subprocess.run(command, cwd=checkout)
            result = qualify(device_output, device, iterations, child.returncode, dry_run)
        except OSError as exc:
            result = dict(device=device, exit_code=2, status="error", iterations=0, faults=0, error=str(exc))
        summary["results"].append(result)
        checkpoint()
        print(json.dumps(result), flush=True)

    summary["stage"] = "dry-run" if dry_run else "complete"
    summary["current_device"] = None
    checkpoint()
    code = 2 if summary["error_devices"] else 1 if summary["fault_devices"] else 0
    print(
        json.dumps(dict(stage=summary["stage"], exit_code=code, total_iterations=summary["total_iterations"])),
        flush=True,
    )
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tt-metal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100000, help="Iteration budget for each of the 32 devices")
    parser.add_argument("--dry-run", action="store_true", help="Check all commands without importing Torch/TTNN")
    args = parser.parse_args()
    return run_sweep(args.tt_metal.resolve(), args.output.resolve(), args.iterations, args.dry_run)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Galaxy sweep setup error: {exc}", file=sys.stderr)
        raise SystemExit(2)
