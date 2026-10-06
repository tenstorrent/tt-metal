# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Validate HiFi2 prefill QKV on actual 512-step stress and maximum context."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_minimal_advice_controls import ROOT, RUNTIME, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    assert digest(RUNTIME) == args.runtime_sha256
    journal_path = args.output_dir / "commands.json"
    if journal_path.exists():
        parser.error("Execution journal exists; choose a fresh output directory")
    probes = [
        Path(__file__).with_name(name)
        for name in ("probe_optimized_minimal_advice.py", "probe_optimized_minimal_advice_contract.py")
    ]
    hashes = {str(path): digest(path) for path in [RUNTIME, Path(__file__), *probes]}
    jobs = []
    for phase in ("stress", "long"):
        for layer in (0, 5):
            output = args.output_dir / f"layer{layer}_{phase}_hifi2.json"
            module = "probe_optimized_minimal_advice_contract" if phase == "long" else "probe_optimized_minimal_advice"
            command = [
                sys.executable,
                "-m",
                f"models.autoports.google_gemma_4_26b_a4b_it.tests.{module}",
                "--minimal-advice",
                "hifi2",
                "--defaults",
                "--layer",
                str(layer),
            ]
            if phase == "stress":
                command += [
                    "--length",
                    "1025",
                    "--real",
                    "--decode",
                    "--steps",
                    "512",
                    "--input-fixture",
                    str(ROOT / f"doc/optimized_decoder/actual_text_layer{layer}_1025_512.pt"),
                    "--verify-program-cache",
                    "--timing",
                ]
            else:
                command += [
                    "--contract",
                    "long_context",
                    "--length",
                    "262144",
                    "--threads",
                    "4",
                    "--input-fixture",
                    str(ROOT / f"doc/optimized_decoder/actual_text_long/actual_text_layer{layer}_262144_0.pt"),
                    "--reference-file",
                    str(ROOT / f"doc/optimized_decoder/actual_text_long/actual_text_layer{layer}_262144_reference.pt"),
                ]
            command += ["--output", str(output)]
            jobs.append(dict(layer=layer, phase=phase, output=str(output), command=command))
    plan = dict(status="planned_not_executed", hashes=hashes, jobs=jobs)
    write_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(args.output_dir / "plan.json")
        return
    env = {key: value for key, value in os.environ.items() if not key.startswith("TT_METAL_WATCHER")}
    env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TT_METAL_DEVICE_PROFILER="0")
    journal = []
    for job in jobs:
        assert all(digest(path) == expected for path, expected in hashes.items())
        log = Path(job["output"]).with_suffix(".log")
        print(f"Starting {job['phase']} layer{job['layer']}", flush=True)
        start = time.monotonic()
        with log.open("w") as stream:
            result = subprocess.run(job["command"], stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=3600)
        output = Path(job["output"])
        report = json.loads(output.read_text()) if output.exists() else {}
        entry = dict(**job, returncode=result.returncode, seconds=time.monotonic() - start, log=str(log))
        if output.exists():
            entry["report_sha256"] = digest(output)
        if result.returncode == 0:
            assert report["passed"] and report["runtime_sha256"] == args.runtime_sha256
            if job["phase"] == "stress":
                assert (
                    report["decode"]["passed"]
                    and report["decode"]["repeated_equal"]
                    and report["program_cache_miss_guard"]
                )
                entry["minimum_decode_pcc"] = report["decode"]["min_pcc"]
            else:
                rows = report["sampled_row_diagnostics"]
                assert len(rows) == 291 and all(row["passed"] and row["pcc"] >= 0.995 for row in rows)
                assert report["prefill_sampled_rows_passed"] and all(row["passed"] for row in report["decode"])
                entry.update(sampled_rows=len(rows), minimum_sampled_pcc=min(row["pcc"] for row in rows))
            entry["status"] = "passed"
        else:
            entry.update(status="failed", error=report.get("minimal_advice_error"))
        journal.append(entry)
        write_json(journal_path, journal)
        print(job["phase"], job["layer"], entry["status"], flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
    plan.update(status="acceptance_passed", journal_sha256=digest(journal_path))
    write_json(args.output_dir / "plan.json", plan)


if __name__ == "__main__":
    main()
