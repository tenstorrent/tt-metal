# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize precision jobs and retain exact commands; stop at runtime failures."""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/datatype_sweep"
parser = argparse.ArgumentParser()
parser.add_argument("configs", nargs="+")
modes = parser.add_mutually_exclusive_group()
modes.add_argument("--smoke", action="store_true")
modes.add_argument("--context", action="store_true", help="Full context gate; numeric failures reject a policy")
modes.add_argument("--batch-check", action="store_true", help="Full batch32 capability gate")
args = parser.parse_args()
manifest = DOC / "commands.jsonl"


def record_capability_check(name, output):
    checks_path = DOC / ("capability_checks.json" if args.context else "batch_capability_checks.json")
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else []
    check = dict(config_id=name, evidence_path=str(output.relative_to(DOC)))
    if check not in checks:
        checks.append(check)
        checks_path.write_text(json.dumps(checks, indent=2) + "\n")


for name in args.configs:
    config = DOC / "configs" / f"{name}.json"
    output = (
        DOC
        / ("context_checks" if args.context else "batch_checks" if args.batch_check else "runs")
        / f'{name}{"_smoke" if args.smoke else ""}.json'
    )
    log = output.with_suffix(".log")
    output.parent.mkdir(exist_ok=True)
    if output.exists() and json.loads(output.read_text()).get("status") in (
        "pass",
        "accuracy_fail",
        "capability_gate_fail",
        "smoke_pass_not_full_model",
    ):
        saved = json.loads(output.read_text())
        if saved.get("precision_config") != json.loads(config.read_text()):
            raise SystemExit(f"Saved policy differs for {output}; preserve it and use a fresh config ID")
        hashes = saved.get("implementation_sha256", {})
        if not hashes or any(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()
        ):
            raise SystemExit(f"Saved implementation differs for {output}; preserve it and use a fresh config ID")
        prior = (
            [
                json.loads(line)
                for line in manifest.read_text().splitlines()
                if str(output) in json.loads(line)["command"]
            ]
            if manifest.exists()
            else []
        )
        expected_rc = 1 if saved["status"] == "capability_gate_fail" else 0
        if not prior or prior[-1]["returncode"] != expected_rc:
            raise SystemExit(
                f"No matching completed process result for {output}; inspect and preserve the prior attempt"
            )
        if args.context or args.batch_check:
            record_capability_check(name, output)
        print("REUSE_COMPLETED", output, flush=True)
        continue
    command = [
        sys.executable,
        str(ROOT / "tests/run_datatype_candidate.py"),
        "--config",
        str(config),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    if args.smoke:
        command.append("--smoke")
    if args.context or args.batch_check:
        runner = "check_full_context.py" if args.context else "check_full_batch.py"
        command = [sys.executable, str(ROOT / "tests" / runner), "--output", str(output)]
        environment["MUSE_PRECISION_CONFIG"] = str(config)
        environment.update(
            TT_METAL_TRACE_ALLOC_TRACKING="1",
            TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE="0",
            TT_METAL_TRACE_ALLOC_TRACEBACKS="1",
        )
    row = dict(command=command, log=str(log), started=datetime.datetime.now(datetime.timezone.utc).isoformat())
    row["environment"] = {
        key: environment.get(key)
        for key in (
            "MUSE_PRECISION_CONFIG",
            "TT_METAL_TRACE_ALLOC_TRACKING",
            "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
            "TT_METAL_TRACE_ALLOC_TRACEBACKS",
        )
    }
    print(
        "START",
        name,
        "context" if args.context else "batch32" if args.batch_check else "smoke" if args.smoke else "full",
        flush=True,
    )
    with log.open("w") as handle:
        rc = subprocess.call(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    row.update(returncode=rc, finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with manifest.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    print("FINISH", name, "returncode", rc, flush=True)
    if args.batch_check:
        append_path = output.with_suffix(".append.json")
        if rc == 1 and append_path.exists():
            append = json.loads(append_path.read_text())
            if append["status"] == "capability_gate_fail":
                output.write_text(json.dumps(append, indent=2) + "\n")
        if output.exists():
            result = json.loads(output.read_text())
            if (rc == 0 and result.get("status") == "pass") or (
                rc == 1 and result.get("status") == "capability_gate_fail"
            ):
                record_capability_check(name, output)
                if rc:
                    print("REJECT_BATCH_APPEND", name, result["cosine_min"], result["argmax_equal"], flush=True)
                    continue
    if args.context and output.exists():
        result = json.loads(output.read_text())
        if result["status"] in ("pass", "capability_gate_fail"):
            record_capability_check(name, output)
            if rc and result["status"] == "capability_gate_fail":
                print("REJECT_CONTEXT", name, result["prefill_decode_cosine"], flush=True)
                continue
    if rc:
        sys.exit(rc)
