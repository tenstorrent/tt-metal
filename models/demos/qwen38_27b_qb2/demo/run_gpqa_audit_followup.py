# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only response audit after the exact owned head-control service exits."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def predecessor_ready(properties, receipt, invocation):
    if properties.get("InvocationID") and properties["InvocationID"] != invocation:
        raise ValueError("Head-control service was replaced")
    if (
        properties.get("LoadState") not in ("loaded", "not-found")
        or properties.get("MainPID") != "0"
        or properties.get("ActiveState") not in ("inactive", "failed")
    ):
        return False
    if properties.get("LoadState") == "loaded" and properties.get("Result") != "success":
        raise ValueError("Head-control service did not exit successfully")
    if not receipt or receipt.get("state") != "completed":
        raise ValueError("Head-control queue did not complete")
    stages = receipt.get("steps", [])
    if (
        len(stages) != 2
        or {row.get("name") for row in stages} != {"native-g0-run", "native-control"}
        or any(row.get("state") != "completed" for row in stages)
        or receipt.get("native-control", {}).get("owned_processes_stopped") is not True
    ):
        raise ValueError("Head-control hardware and evaluation stages are incomplete")
    return True


def follow(args):
    args.output.mkdir()
    status_path = args.output / "status.json"
    state = dict(
        state="waiting",
        predecessor_unit=args.unit,
        predecessor_invocation=args.invocation,
        hardware_opened=False,
        survives_disconnect=True,
        resumes_after_reboot=False,
        started_at=time.time(),
    )

    def save():
        state["updated_at"] = time.time()
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(status_path)

    save()
    deadline = time.monotonic() + args.wait_timeout
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Head-control wait exceeded its bound")
            try:
                output = subprocess.check_output(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        args.unit,
                        "-p",
                        "InvocationID",
                        "-p",
                        "LoadState",
                        "-p",
                        "ActiveState",
                        "-p",
                        "MainPID",
                        "-p",
                        "Result",
                    ],
                    text=True,
                    timeout=15,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                state["observation_error"] = type(error).__name__
                save()
                time.sleep(30)
                continue
            properties = dict(line.split("=", 1) for line in output.splitlines() if "=" in line)
            receipt = json.loads(args.receipt.read_text()) if args.receipt.exists() else None
            state["last_service"] = properties
            save()
            if predecessor_ready(properties, receipt, args.invocation):
                break
            time.sleep(30)
        if hashlib.sha256(args.audit_source.read_bytes()).hexdigest() != args.audit_sha256:
            raise ValueError("Frozen response-audit source changed")
        command = [
            sys.executable,
            str(args.audit_source),
            "--receipts",
            str(args.results / "gpqa-responses.jsonl"),
            "--private-responses",
            str(args.results / "private-responses"),
            "--summary",
            str(args.results / "summary.json"),
            "--count",
            "198",
            "--max-output-tokens",
            "65536",
            "--max-model-len",
            "262144",
            "--output",
            str(args.output / "audit.json"),
        ]
        state.update(state="auditing", command=command)
        save()
        with (args.output / "audit.log").open("x") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        report = json.loads((args.output / "audit.json").read_text())
        state.update(state="completed", qualification=report["qualification"], finished_at=time.time())
    except BaseException as error:
        state.update(state="failed", error=type(error).__name__, detail=str(error)[:2000])
        raise
    finally:
        save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("audit-source", "receipt", "results", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("unit", "invocation", "audit-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--wait-timeout", type=float, default=64800)
    follow(parser.parse_args())
