# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run six selected-policy prefill QKV advice controls sequentially."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "tt/optimized_decoder.py"
PROBE = Path(__file__).with_name("probe_optimized_minimal_advice.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if digest(RUNTIME) != args.runtime_sha256:
        parser.error("Runtime hash differs from the requested snapshot")
    journal_path = args.output_dir / "commands.json"
    if journal_path.exists():
        parser.error("Execution journal exists; choose a fresh output directory")
    jobs = []
    for layer in (0, 5):
        for candidate in ("baseline", "hifi2", "grid110"):
            output = args.output_dir / f"layer{layer}_{candidate}.json"
            command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice",
                "--minimal-advice",
                candidate,
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
                "--prefill-timing",
                "--timing",
                "--verify-program-cache",
                "--output",
                str(output),
            ]
            jobs.append(dict(layer=layer, candidate=candidate, command=command, output=str(output)))
    hashes = dict(runtime=digest(RUNTIME), probe=digest(PROBE), driver=digest(__file__))
    plan = dict(status="planned_not_executed", hashes=hashes, jobs=jobs)
    write_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(args.output_dir / "plan.json")
        return
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TT_METAL_WATCHER")}
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TT_METAL_DEVICE_PROFILER="0")
    journal = []
    for job in jobs:
        assert hashes == dict(runtime=digest(RUNTIME), probe=digest(PROBE), driver=digest(__file__))
        log = Path(job["output"]).with_suffix(".log")
        start = time.monotonic()
        print(f"Starting layer{job['layer']} {job['candidate']}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(
                job["command"], stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=1800
            )
        output = Path(job["output"])
        report = json.loads(output.read_text()) if output.exists() else {}
        entry = dict(**job, log=str(log), returncode=result.returncode, seconds=time.monotonic() - start, hashes=hashes)
        if output.exists():
            entry["report_sha256"] = digest(output)
        if result.returncode == 0:
            assert report["passed"] and report["decode"]["passed"] and report["decode"]["repeated_equal"]
            assert report["runtime_sha256"] == args.runtime_sha256
            entry.update(
                status="passed",
                prefill_pcc=report["pcc"],
                min_decode_pcc=report["decode"]["min_pcc"],
                prefill_host_median_us=statistics.median(report["warmed_prefill_host_us"]),
                decode_host_median_us=statistics.median(report["traced_decode_host_us"]),
            )
        elif report.get("passed") is False or report.get("decode", {}).get("passed") is False:
            entry.update(
                status="accuracy_failed",
                prefill_pcc=report.get("pcc"),
                min_decode_pcc=report.get("decode", {}).get("min_pcc"),
            )
        else:
            entry.update(status="unexpected_error", error=report.get("minimal_advice_error"))
        journal.append(entry)
        write_json(journal_path, journal)
        print(job["layer"], job["candidate"], entry["status"], entry.get("prefill_host_median_us"), flush=True)
        if entry["status"] == "unexpected_error":
            raise SystemExit(result.returncode or 1)
    plan.update(status="controls_complete_selection_review_pending", journal_sha256=digest(journal_path))
    write_json(args.output_dir / "plan.json", plan)


if __name__ == "__main__":
    main()
