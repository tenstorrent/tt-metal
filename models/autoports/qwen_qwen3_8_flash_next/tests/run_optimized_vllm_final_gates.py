# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialize remaining stage10 gates after the already-running baseline arm."""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    model = Path(__file__).resolve().parents[1]
    stage = model / "doc/optimized_vllm"
    readiness = model / "readiness_vllm"
    manifest = stage / "final_gates_execution.json"
    assert not manifest.exists(), "Preserve existing gate execution evidence"
    module = "models.autoports.qwen_qwen3_8_flash_next.tests."
    report = dict(passed=False, steps=[], start_utc=datetime.now(timezone.utc).isoformat())

    def save():
        manifest.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        deadline = time.monotonic() + 10000
        while True:
            baseline = json.loads((readiness / "stage10_matched_before/execution.json").read_text())
            if baseline.get("error"):
                raise RuntimeError(f"Baseline failed: {baseline['error']}")
            if baseline.get("passed") and baseline.get("source_restored"):
                break
            assert time.monotonic() < deadline, "Baseline completion deadline"
            time.sleep(5)
        steps = [
            (
                "matched_after",
                "run_optimized_vllm_measurement",
                [
                    "--budget",
                    "9",
                    "--output-dir",
                    readiness / "stage10_matched_after",
                    "--serial-reference",
                    readiness / "stage10_matched_before/vllm_prefill_burst_benchmark.json",
                ],
            ),
            (
                "compare",
                "compare_optimized_vllm",
                [
                    "--before-dir",
                    readiness / "stage10_matched_before",
                    "--after-dir",
                    readiness / "stage10_matched_after",
                    "--full-model",
                    stage / "full_model_comparison_v3.json",
                    "--output",
                    stage / "perf_summary.json",
                ],
            ),
            (
                "tracked_sync",
                "run_optimized_vllm_replay_checks",
                ["--budget", "9", "--output-dir", readiness / "stage10_final_tracked_sync"],
            ),
            (
                "tracked_async",
                "run_optimized_vllm_replay_checks",
                [
                    "--budget",
                    "9",
                    "--output-dir",
                    readiness / "stage10_final_tracked_async",
                    "--async-control",
                    readiness / "stage10_final_tracked_sync/history.json",
                ],
            ),
            (
                "acceptance",
                "run_optimized_vllm_acceptance",
                ["--budget", "9", "--output-dir", readiness / "stage10_final_acceptance"],
            ),
        ]
        for name, entry, arguments in steps:
            command = [sys.executable, "-m", module + entry, *map(str, arguments)]
            log_path = stage / f"final_gate_{name}.log"
            row = dict(name=name, command=command, log=str(log_path), start_utc=datetime.now(timezone.utc).isoformat())
            report["steps"].append(row)
            save()
            print(f"START {name}", flush=True)
            with log_path.open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            row.update(exit=result.returncode, end_utc=datetime.now(timezone.utc).isoformat())
            save()
            assert result.returncode == 0, f"Gate {name} failed; inspect evidence before continuing"
            print(f"DONE {name}", flush=True)
        report["passed"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        report["end_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
