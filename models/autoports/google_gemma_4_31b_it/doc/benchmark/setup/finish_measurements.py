#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the remaining serial measurements under the original stage deadline."""
import json
import os
import re
import subprocess
import time
from pathlib import Path

root = Path.cwd()
model = root / "models/autoports/google_gemma_4_31b_it"
base = model / "doc/benchmark"
run = base / "run"
python = str(root / "python_env/bin/python")
skill = Path(os.environ["TT_MODEL_BRINGUP_ROOT"])
timing = json.loads((base / "stage_timing.json").read_text())
ledger = {"started_unix": time.time(), "actions": []}


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
    (base / "setup/supplements.json").write_text(json.dumps(ledger, indent=2) + "\n")
    result.check_returncode()


while True:
    try:
        native = json.loads((base / "setup/native_fallback.json").read_text())
    except (OSError, ValueError):
        native = {}
    if "exit_code" in native:
        assert native["exit_code"] == 0 and native["within_original_native_boundary"], native
        break
    assert time.monotonic() < timing["deadline_monotonic"] - 1800
    time.sleep(1)
assert json.loads((run / "summary.json").read_text())["status"] == "completed"
invoke(
    "release_b1",
    [
        python,
        str(model / "tools/benchmark_grid.py"),
        "--run-dir",
        str(run),
        "--concurrency",
        "1",
        "--action",
        "collect",
        "--grid-only",
        "--release-only",
        "--deadline-seconds",
        "8100",
    ],
)
invoke("anomaly_run", [python, str(model / "tools/benchmark_anomalies.py")])
shared = json.loads((base / "qualitative_outputs.json").read_text())
identity = json.loads((run / "perf-b1-server.json").read_text())
invoke(
    "logging_check",
    [
        python,
        str(skill / "scripts/reply_digest.py"),
        "--check-log",
        "--log",
        identity["server_log"],
        "--id",
        shared[0]["response"]["id"],
        "--record",
        str(model / "doc/agentic_evals/logging_check.json"),
    ],
)
invoke(
    "grid_b1",
    [
        python,
        str(model / "tools/benchmark_grid.py"),
        "--run-dir",
        str(run),
        "--concurrency",
        "1",
        "--action",
        "collect",
        "--grid-only",
        "--deadline-seconds",
        "8100",
    ],
)
if time.monotonic() < timing["started_monotonic"] + 9000:
    directory = run / "supplement-b8"
    directory.mkdir(exist_ok=False)
    invoke(
        "supplement_b8_server",
        [
            python,
            str(model / "tools/benchmark_server.py"),
            "--max-num-seqs",
            "8",
            "--base-url",
            "http://127.0.0.1:8000",
            "--output",
            str(directory / "perf-b8-server.json"),
        ],
    )
    identity = json.loads((directory / "perf-b8-server.json").read_text())
    log = Path(identity["server_log"]).read_text(errors="replace")
    values = re.findall(r"TT KV cache size: ([\d,]+) tokens", log)
    assert values, "Missing actual B8 capacity"
    capacity = int(values[-1].replace(",", ""))
    plan = json.loads((base / "plan.json").read_text())
    for row in plan["rows"]:
        if row["batch"] == 8 and row["concurrency"] == 8:
            needed = row["concurrency"] * (row["input_tokens"] + row["output_tokens"])
            if needed > capacity:
                row[
                    "missing_reason"
                ] = f"Unchanged live B8 KV pool has {capacity} token capacity; {needed} tokens required for the simultaneous prompt/output working set. Full context remains262144; concurrency1 rows are retained."
            else:
                row.pop("missing_reason", None)
    (base / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (base / "setup/b8_capacity.json").write_text(
        json.dumps(
            {
                "tokens": capacity,
                "server_log": identity["server_log"],
                "observed_unix": time.time(),
                "prediction_fields_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
    invoke(
        "supplement_b8_check",
        [
            python,
            str(model / "tools/collect_benchmark_roofline.py"),
            "--run-dir",
            str(directory),
            "--concurrency",
            "8",
            "--action",
            "check",
        ],
    )
    invoke(
        "grid_b8",
        [
            python,
            str(model / "tools/benchmark_grid.py"),
            "--run-dir",
            str(directory),
            "--concurrency",
            "8",
            "--action",
            "collect",
            "--grid-only",
            "--deadline-seconds",
            "9600",
        ],
    )
invoke("stop_owned", [python, str(model / "tools/benchmark_server.py"), "--stop"])
ledger["completed_unix"] = time.time()
(base / "setup/supplements.json").write_text(json.dumps(ledger, indent=2) + "\n")
print(json.dumps(ledger), flush=True)
