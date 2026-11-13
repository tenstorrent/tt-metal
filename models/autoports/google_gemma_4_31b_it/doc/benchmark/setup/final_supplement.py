#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Admit one final profile using observed startup and original row predictions."""
import json
import math
import subprocess
import time
from pathlib import Path

root = Path.cwd()
model = root / "models/autoports/google_gemma_4_31b_it"
base = model / "doc/benchmark"
run = base / "run"
python = str(root / "python_env/bin/python")
timing = json.loads((base / "stage_timing.json").read_text())
prior = json.loads((base / "setup/supplements.json").read_text())
assert "completed_unix" in prior and all(x["exit_code"] == 0 for x in prior["actions"])
row = next(x for x in json.loads((base / "plan.json").read_text())["rows"] if x["id"] == "b32-c1-i131072-swe")
pred = row["predicted"]
startup = json.loads((run / "perf-b32-server.json").read_text())["startup_seconds"]
row_estimate = (
    1.5 * (1 + math.ceil(row["requests"] / row["concurrency"])) * (pred["ttft_s"] + 127 * pred["step_ms"] / 1000)
)
deadline = timing["started_monotonic"] + 10080
remaining = deadline - time.monotonic()
ledger = {
    "observed_unix": time.time(),
    "original_stage_deadline": timing["deadline_monotonic"],
    "measurement_deadline": deadline,
    "closure_reserve_seconds": 720,
    "observed_prior_startup_seconds": startup,
    "startup_margin_seconds": 60,
    "predicted_whole_row_seconds": row_estimate,
    "row_margin_seconds": 60,
    "remaining_seconds": remaining,
    "actions": [],
}
ledger["admitted"] = startup + 60 + row_estimate + 60 <= remaining
record = base / "setup/final_supplement.json"


def save():
    record.write_text(json.dumps(ledger, indent=2) + "\n")


def invoke(name, argv):
    start = time.monotonic()
    with (base / "setup" / f"{name}.log").open("w") as log:
        result = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT)
    ledger["actions"].append(
        {
            "name": name,
            "command": argv,
            "started_monotonic": start,
            "finished_monotonic": time.monotonic(),
            "exit_code": result.returncode,
        }
    )
    save()
    result.check_returncode()


save()
if ledger["admitted"]:
    directory = run / "supplement-b32"
    directory.mkdir(exist_ok=False)
    invoke(
        "final_b32_server",
        [
            python,
            str(model / "tools/benchmark_server.py"),
            "--max-num-seqs",
            "32",
            "--base-url",
            "http://127.0.0.1:8000",
            "--output",
            str(directory / "perf-b32-server.json"),
        ],
    )
    invoke(
        "final_b32_check",
        [
            python,
            str(model / "tools/collect_benchmark_roofline.py"),
            "--run-dir",
            str(directory),
            "--concurrency",
            "32",
            "--action",
            "check",
        ],
    )
    invoke(
        "final_b32_grid",
        [
            python,
            str(model / "tools/benchmark_grid.py"),
            "--run-dir",
            str(directory),
            "--concurrency",
            "32",
            "--action",
            "collect",
            "--grid-only",
            "--deadline-seconds",
            "10080",
            "--closure-reserve-seconds",
            "720",
        ],
    )
    invoke("final_b32_stop", [python, str(model / "tools/benchmark_server.py"), "--stop"])
ledger["completed_unix"] = time.time()
save()
print(json.dumps(ledger), flush=True)
