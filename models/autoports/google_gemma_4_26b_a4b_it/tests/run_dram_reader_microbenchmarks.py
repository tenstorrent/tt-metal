# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized capture/Tracy/CPU-summary driver for exact-input reader controls."""

import argparse
import json
import os
import shutil
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

MODULE = "models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_dram_readers"


def commands(output_dir, fixture_dir, layer):
    prefix = output_dir / f"layer{layer}"
    capture = prefix.with_suffix(".inputs.pt")
    report = prefix.with_suffix(".json")
    common = ["--layer", str(layer), "--capture", str(capture)]
    capture_command = [
        sys.executable,
        "-m",
        MODULE,
        *common,
        "--input-fixture",
        str(fixture_dir / f"actual_text_layer{layer}_4096_128.pt"),
        "--capture-only",
        "--report",
        str(prefix.with_suffix(".capture.json")),
    ]
    raw = output_dir / f"tracy_layer{layer}" / "raw"
    profile_command = [
        sys.executable,
        "-m",
        "tracy",
        "-r",
        "-p",
        "-v",
        "--op-support-count",
        "100000",
        "--no-op-info-cache",
        "--disable-device-data-dump-to-files",
        "--disable-device-data-push-to-tracy",
        "-o",
        str(raw),
        "-n",
        "readers",
        "-m",
        MODULE,
        *common,
        "--profile",
        "--rounds",
        "6",
        "--replays",
        "20",
        "--roles",
        "qkv",
        "output",
        "shared_gate_up",
        "shared_down",
        "--qkv-blocks",
        "1",
        "11",
        "--report",
        str(report),
    ]
    csv_path = output_dir / f"tracy_layer{layer}" / "ops.csv"
    summary_command = [sys.executable, "-m", MODULE, "--summarize-csv", str(csv_path), "--report", str(report)]
    return dict(
        layer=layer,
        capture=capture_command,
        profile=profile_command,
        summary=summary_command,
        raw=str(raw),
        csv=str(csv_path),
        report=str(report),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "doc/optimized_decoder")
    parser.add_argument("--layers", type=int, choices=(0, 5), nargs="+", default=[0, 5])
    parser.add_argument("--runtime-sha256", default=digest(RUNTIME))
    parser.add_argument(
        "--plan-only", action="store_true", help="Write commands and provenance; do not import TTNN or run a subprocess"
    )
    args = parser.parse_args()
    if digest(RUNTIME) != args.runtime_sha256:
        parser.error("Expected runtime hash does not match the selected runtime")
    plan = dict(
        status="planned_not_executed",
        runtime_sha256=args.runtime_sha256,
        driver_sha256=digest(__file__),
        probe_sha256=digest(Path(__file__).with_name("probe_optimized_dram_readers.py")),
        workload="Real recorded 4096+1 production capture; isolated one-op traces at exact captured precision",
        commands=[commands(args.output_dir, args.fixture_dir, layer) for layer in args.layers],
        environment=dict(
            OMP_NUM_THREADS="4",
            MKL_NUM_THREADS="4",
            HF_HUB_OFFLINE="1",
            capture_profiler="0",
            profile_profiler="1",
            watcher="removed",
        ),
        parent_owns_device_serialization=True,
    )
    write_json(args.output_dir / "plan.json", plan)
    if args.plan_only:
        print(args.output_dir / "plan.json")
        return
    journal_path = args.output_dir / "commands.json"
    if journal_path.exists():
        parser.error("This output directory already has an execution journal; choose a new directory for a new run")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("TT_METAL_WATCHER")}
    environment.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", HF_HUB_OFFLINE="1")
    journal = []

    def run(layer, phase, command):
        if digest(RUNTIME) != args.runtime_sha256:
            raise RuntimeError("Runtime changed while the reader driver was running")
        env = {**environment, "TT_METAL_DEVICE_PROFILER": "1" if phase == "profile" else "0"}
        log_path = args.output_dir / f"layer{layer}.{phase}.log"
        start = time.monotonic()
        entry = dict(layer=layer, phase=phase, command=command, log=str(log_path), runtime_sha256=args.runtime_sha256)
        print(f"Starting layer{layer} {phase}", flush=True)
        try:
            with log_path.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, timeout=1800)
            entry["returncode"] = result.returncode
        except subprocess.TimeoutExpired:
            entry["timeout_seconds"] = 1800
            raise
        finally:
            entry["seconds"] = time.monotonic() - start
            journal.append(entry)
            write_json(journal_path, journal)
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f"Finished layer{layer} {phase}: {entry['seconds']:.1f}s", flush=True)

    for item in plan["commands"]:
        layer, raw = item["layer"], Path(item["raw"])
        run(layer, "capture", item["capture"])
        previous = set(raw.rglob("ops_perf_results_*.csv"))
        run(layer, "profile", item["profile"])
        produced = set(raw.rglob("ops_perf_results_*.csv")) - previous
        if len(produced) != 1:
            raise RuntimeError(f"Expected exactly one newly generated native CSV for layer{layer}: {produced}")
        original = produced.pop()
        shutil.copyfile(original, item["csv"])
        write_json(
            Path(item["csv"]).with_suffix(".provenance.json"),
            dict(
                source=str(original),
                source_sha256=digest(original),
                copy=item["csv"],
                copy_sha256=digest(item["csv"]),
                runtime_sha256=args.runtime_sha256,
            ),
        )
        run(layer, "summary", item["summary"])
        report = json.loads(Path(item["report"]).read_text())
        if len(report["cases"]) != 5 or any(
            case["status"] not in ("measured_device_and_host", "all_readers_invalid") for case in report["cases"]
        ):
            raise RuntimeError("Isolated reader evidence is incomplete")
    plan["status"] = "isolated_device_evidence_complete_integration_decision_pending"
    plan["execution_journal_sha256"] = digest(journal_path)
    write_json(args.output_dir / "plan.json", plan)


if __name__ == "__main__":
    main()
