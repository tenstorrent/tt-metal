# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-safe planner/serial launcher for an explicit projection-kernel manifest."""

import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
DOC = ROOT / "doc/optimized_multichip_decoder"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_report(path):
    data = json.loads(path.read_text())
    require(data.get("pass") is True and data.get("status") == "pass", f"Failed/PCC-failed report: {path}")
    require(bool(data.get("cases")), f"No measured cases: {path}")
    for case in data["cases"]:
        require(
            case.get("status") == "pass" and case.get("minimum_pcc", 0) >= 0.995, f"Unchanged PCC gate failed: {path}"
        )
        require(
            case.get("changed_input_trace") and case.get("first_last_trace_equal"),
            f"Missing changed-input/first-last trace evidence: {path}",
        )
    return data


def source_check(manifest):
    for name, expected in manifest["source_sha256"].items():
        require(
            hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected,
            f"Source changed: {name}; preserve this queue and prepare a fresh manifest",
        )
    for policy in manifest["policies"]:
        require(
            json.loads((REPO / policy["path"]).read_text()) == policy["value"], f"Policy file changed: {policy['path']}"
        )


def metadata(path):
    info = path.stat()
    return {"size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def table(manifest, destination):
    fields = [
        "job",
        "result",
        "layer",
        "role",
        "rows",
        "family",
        "weight_layout",
        "output_dtype",
        "median_us",
        "us_per_row",
        "minimum_pcc",
        "status",
        "program",
    ]
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for job in manifest["jobs"]:
            path = REPO / job["output"]
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            for case in data.get("cases", []):
                row = {k: case.get(k) for k in fields}
                row.update(job=job["name"], result=job["output"], layer=job["layer"])
                row["program"] = json.dumps(case.get("program"), sort_keys=True)
                writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-file", type=Path, default=DOC / "STOP")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    source_check(manifest)
    require(len({j["name"] for j in manifest["jobs"]}) == len(manifest["jobs"]), "Duplicate job names")
    for job in manifest["jobs"]:
        require((REPO / job["output"]).resolve().is_relative_to(DOC.resolve()), "Output outside stage5")
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "plan_valid",
                    "jobs": len(manifest["jobs"]),
                    "run": False,
                    "rows": manifest["rows"],
                    "limitations": manifest["limitations"],
                }
            )
        )
        return
    import fcntl

    with (DOC / ".prefill_chunk_queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for job in manifest["jobs"]:
            if args.stop_file.exists():
                print(json.dumps({"status": "stopped_between_jobs", "next_job": job["name"]}))
                return
            source_check(manifest)
            output = REPO / job["output"]
            receipt = output.with_suffix(".queue_receipt.json")
            if output.exists() or receipt.exists():
                require(
                    args.resume and output.exists() and receipt.exists(),
                    "Preserve existing output; choose fresh job names",
                )
                saved = json.loads(receipt.read_text())
                require(
                    saved.get("pass")
                    and saved.get("command") == job["command"]
                    and saved.get("output_metadata") == metadata(output),
                    "Failed/stale receipt; prepare a missing-only or adapted queue",
                )
                checked_report(output)
                continue
            started = datetime.datetime.now(datetime.timezone.utc).isoformat()
            clock = time.monotonic()
            environment = dict(os.environ)
            for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
                environment.pop(key, None)
            environment["MULTICHIP_JOB_TIMEOUT"] = str(job["timeout_seconds"])
            cmd = [
                "bash",
                str(DOC / "run_job.sh"),
                job["name"],
                "env",
                "TT_METAL_TRACE_ALLOC_TRACKING=0",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0",
                *job["command"],
            ]
            code = subprocess.run(cmd, cwd=REPO, env=environment, check=False).returncode
            saved = {
                "started_utc": started,
                "elapsed_seconds": time.monotonic() - clock,
                "command": job["command"],
                "wrapper_command": cmd,
                "exit_code": code,
                "pass": False,
                "output_metadata": metadata(output) if output.exists() else None,
            }
            try:
                require(code == 0, f"Job failed: {job['name']} (exit {code}); remaining jobs not launched")
                checked_report(output)
                source_check(manifest)
                saved["pass"] = True
            finally:
                receipt.write_text(json.dumps(saved, indent=2) + "\n")
                table(manifest, args.manifest.with_suffix(".csv"))
            print(
                json.dumps({"job": job["name"], "pass": True, "elapsed_seconds": saved["elapsed_seconds"]}), flush=True
            )
        print(json.dumps({"status": "all_planned_jobs_passed", "jobs": len(manifest["jobs"])}))


if __name__ == "__main__":
    main()
