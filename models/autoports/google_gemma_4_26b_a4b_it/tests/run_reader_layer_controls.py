# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the bounded22 whole-layer integrations after the isolated reader matrix."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers import (
    ROOT,
    RUNTIME,
    digest,
    write_json,
)


def candidates(layer):
    yield "baseline", "baseline", 1, 11
    # K11 reader1 is independently proven to exceed static L1. K1 retains a
    # legal one-reader integration, not an equal-K reader-speed comparison.
    for readers, block in ((1, 1), (2, 11), (3, 11)):
        yield f"qkv_r{readers}_k{block}", "qkv", readers, block
    for readers in (1, 2, 3):
        yield f"output_r{readers}", "output", readers, 11
    selected = 1 if layer == 0 else 2
    for role in ("shared_gate_up", "shared_down"):
        for readers in (1, 2, 3):
            if readers != selected:
                yield f"{role}_r{readers}", role, readers, 11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, choices=(0, 5), nargs="+", default=[0, 5])
    parser.add_argument("--runtime-sha256", default=digest(RUNTIME))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if digest(RUNTIME) != args.runtime_sha256:
        parser.error("Runtime does not match expected hash")
    jobs = []
    for layer in args.layers:
        for name, role, readers, block in candidates(layer):
            output = args.output_dir / f"layer{layer}_{name}.json"
            command = [
                sys.executable,
                "-m",
                "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_reader_layer",
                "--reader-role",
                role,
                "--readers",
                str(readers),
                "--reader-qkv-block",
                str(block),
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
            jobs.append(
                dict(
                    layer=layer, name=name, role=role, readers=readers, block=block, command=command, output=str(output)
                )
            )
    plan = dict(
        status="planned_not_executed",
        runtime_sha256=args.runtime_sha256,
        driver_sha256=digest(__file__),
        probe_sha256=digest(Path(__file__).with_name("probe_optimized_reader_layer.py")),
        jobs=jobs,
        scope="22 actual4096/128 one-role integrations. Host trace samples are whole-layer; isolated native times remain separate. Full shared_down_r3 retains gate reader2.",
    )
    journal_path = args.output_dir / "commands.json"
    if journal_path.exists():
        parser.error("Existing execution journal: use a new output directory rather than rewriting evidence")
    write_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(args.output_dir / "plan.json")
        return
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TT_METAL_WATCHER")}
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1", TT_METAL_DEVICE_PROFILER="0")
    journal = []
    for job in jobs:
        if (
            digest(RUNTIME) != args.runtime_sha256
            or digest(Path(__file__).with_name("probe_optimized_reader_layer.py")) != plan["probe_sha256"]
        ):
            raise RuntimeError("Published runtime/probe changed during controls")
        log_path = Path(job["output"]).with_suffix(".log")
        start = time.monotonic()
        print(f"Starting layer{job['layer']} {job['name']}", flush=True)
        with log_path.open("w") as stream:
            result = subprocess.run(
                job["command"], stdout=stream, stderr=subprocess.STDOUT, env=environment, timeout=1800
            )
        entry = {**job, "log": str(log_path), "returncode": result.returncode, "seconds": time.monotonic() - start}
        output = Path(job["output"])
        report = json.loads(output.read_text()) if output.exists() else {}
        if result.returncode == 0:
            assert report["passed"] and report["decode"]["passed"] and report["decode"]["repeated_equal"]
            assert report["runtime_sha256"] == args.runtime_sha256
            entry.update(
                status="passed",
                prefill_pcc=report["pcc"],
                min_decode_pcc=report["decode"]["min_pcc"],
                host_median_us=statistics.median(report["traced_decode_host_us"]),
                host_samples_us=report["traced_decode_host_us"],
                report_sha256=digest(output),
            )
        elif any(
            token in report.get("reader_layer_error", "")
            for token in ("Statically allocated circular buffers", "L1 buffer", "No free DRAM reader")
        ):
            entry.update(status="invalid_setup", error=report["reader_layer_error"])
        elif report.get("passed") is False or report.get("decode", {}).get("passed") is False:
            entry.update(
                status="accuracy_failed",
                prefill_pcc=report.get("pcc"),
                min_decode_pcc=report.get("decode", {}).get("min_pcc"),
            )
        else:
            entry.update(status="unexpected_error", error=report.get("reader_layer_error"))
        journal.append(entry)
        write_json(journal_path, journal)
        print(job["layer"], job["name"], entry["status"], entry.get("host_median_us"), flush=True)
        if entry["status"] == "unexpected_error":
            raise SystemExit(result.returncode or 1)
    plan["status"] = "controls_complete_selection_review_pending"
    plan["execution_journal_sha256"] = digest(journal_path)
    write_json(args.output_dir / "plan.json", plan)


if __name__ == "__main__":
    main()
