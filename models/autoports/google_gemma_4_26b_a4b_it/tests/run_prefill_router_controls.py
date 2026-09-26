# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sequential parent-owned hardware driver for four paired prefill-router controls."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tt/optimized_decoder.py"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if digest(RUNTIME) != args.runtime_sha256:
        parser.error("Runtime differs from the requested immutable snapshot")
    journal_path = args.output_dir / "commands.json"
    if journal_path.exists():
        parser.error("Execution journal already exists; choose a fresh directory")
    paths = [RUNTIME, ROOT / "tt/fused_decoder.py", Path(__file__)]
    paths += [
        Path(__file__).with_name(name)
        for name in (
            "probe_optimized_prefill_router.py",
            "probe_optimized_prefill_pairs.py",
            "probe_optimized_minimal_advice.py",
        )
    ]
    hashes = {str(path): digest(path) for path in paths}
    jobs = []
    for layer in (0, 5):
        for candidate in ("hifi2", "producer_l1"):
            output = args.output_dir / f"layer{layer}_{candidate}.json"
            if output.exists():
                parser.error(f"Existing output: {output}")
            command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_prefill_router",
                "--router-prefill",
                candidate,
                "--pairs",
                str(args.pairs),
                "--defaults",
                "--real",
                "--layer",
                str(layer),
                "--length",
                "4096",
                "--decode",
                "--steps",
                "128",
                "--input-fixture",
                str(ROOT / f"doc/optimized_decoder/actual_text_layer{layer}_4096_128.pt"),
                "--verify-program-cache",
                "--timing",
                "--output",
                str(output),
            ]
            jobs.append(dict(layer=layer, candidate=candidate, command=command, output=str(output)))
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"))
    }
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1")
    plan = dict(
        status="planned_not_executed",
        hashes=hashes,
        jobs=jobs,
        environment_overrides={
            key: environment[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "HF_HUB_OFFLINE")
        },
        profiler_and_watcher_unset=True,
    )
    write(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(args.output_dir / "plan.json")
        return
    journal = []
    for job in jobs:
        assert all(digest(path) == expected for path, expected in hashes.items())
        log = Path(job["output"]).with_suffix(".log")
        entry = dict(
            **job,
            log=str(log),
            hashes=hashes,
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            status="running",
        )
        journal.append(entry)
        write(journal_path, journal)
        start = time.monotonic()
        print(f"Starting layer{job['layer']} {job['candidate']}", flush=True)
        with log.open("w") as stream:
            completed = subprocess.run(
                job["command"], stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=1800
            )
        output = Path(job["output"])
        report = json.loads(output.read_text()) if output.exists() else {}
        entry.update(
            returncode=completed.returncode,
            elapsed_seconds=time.monotonic() - start,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        if output.exists():
            entry["report_sha256"] = digest(output)
        if completed.returncode == 0:
            assert report["runtime_sha256"] == args.runtime_sha256
            assert report["passed"] and report["decode"]["passed"] and report["decode"]["repeated_equal"]
            pair = report["paired_prefill"]
            assert pair["program_cache_misses_forbidden"] and len(pair["samples"]) == 2 * args.pairs
            entry.update(
                status="passed",
                prefill_pcc=report["pcc"],
                minimum_decode_pcc=report["decode"]["min_pcc"],
                median_paired_delta_us=pair["median_paired_delta_us"],
                candidate_faster_pairs=pair["candidate_faster_pairs"],
            )
        elif report.get("passed") is False or report.get("decode", {}).get("passed") is False:
            entry.update(status="accuracy_failed", error=report.get("router_prefill_error"))
        else:
            entry.update(status="unexpected_error", error=report.get("router_prefill_error"))
        write(journal_path, journal)
        print(job["layer"], job["candidate"], entry["status"], entry.get("median_paired_delta_us"), flush=True)
        assert all(digest(path) == expected for path, expected in hashes.items())
        if entry["status"] == "unexpected_error":
            raise SystemExit(completed.returncode or 1)
    plan.update(status="controls_complete_selection_review_pending", journal_sha256=digest(journal_path))
    write(args.output_dir / "plan.json", plan)


if __name__ == "__main__":
    main()
