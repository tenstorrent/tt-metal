#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Stands in for a Google Benchmark binary in unit tests.

$FAKE_BENCH_PLAN names a JSON file: {"metric": {...decl...}, "runs": [{case: value, ...}, ...]}. Each invocation
consumes the next entry of "runs", so a test can give the retry different values from the first run.
"""

import json
import os
import re
import sys
from pathlib import Path

args = dict(arg.split("=", 1) for arg in sys.argv[1:] if arg.startswith("--") and "=" in arg)
plan_path = Path(os.environ["FAKE_BENCH_PLAN"])
plan = json.loads(plan_path.read_text())
call_path = plan_path.with_suffix(".calls")
calls = json.loads(call_path.read_text()) if call_path.exists() else []
values = plan["runs"][min(len(calls), len(plan["runs"]) - 1)]
calls.append(sys.argv[1:])
call_path.write_text(json.dumps(calls))

repetitions = int(args.get("--benchmark_repetitions", 1))
case_filter = args.get("--benchmark_filter")
metric = plan["metric"]
rows = []
for case, value in values.items():
    if case_filter and not re.search(case_filter, case):
        continue
    if value is None:
        rows.append(
            {"name": case, "run_name": case, "run_type": "iteration", "error_occurred": True, "error_message": "boom"}
        )
        continue
    for rep in range(repetitions):
        rows.append(
            {
                "name": case,
                "run_name": case,
                "run_type": "iteration",
                "repetitions": repetitions,
                "repetition_index": rep,
                metric["name"]: value * (1 + 0.001 * rep),
                "ctx_aiclk_mhz": 1000,
            }
        )
document = {
    "context": {f"perf.metric.{metric['name']}": metric["decl"], "perf.context.iommu": "on"},
    "benchmarks": rows,
}
Path(args["--benchmark_out"]).write_text(json.dumps(document))
