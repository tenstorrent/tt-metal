# SPDX-License-Identifier: Apache-2.0
"""Record phase boundaries around the unchanged shared HTTP readiness runner."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--stages", required=True)
p.add_argument("--label", required=True)
p.add_argument("--events", required=True)
p.add_argument("--max-num-seqs", type=int, default=32)
p.add_argument("--no-ci", action="store_true")
a = p.parse_args()
root = Path(os.environ["MODEL_DIR"])
out = root / "readiness_vllm"
events = Path(a.events)
before = len(events.read_text().splitlines())
cmd = [
    sys.executable,
    "-m",
    "readiness_check.run_vllm_server",
    "--stages",
    a.stages,
    "--server-url",
    "http://localhost:8000",
    "--model-dir",
    str(root),
    "--hf-model",
    os.environ["KOLIBRI_CHECKPOINT_DIR"],
    "--max-num-seqs",
    str(a.max_num_seqs),
    "--sampling-profile",
    "full",
]
if a.no_ci:
    cmd.append("--no-benchmark-ci-serving")
report = dict(command=cmd, started=time.time(), event_offset=before)
path = out / (a.label + "-checks.json")
path.write_text(json.dumps(report, indent=2) + "\n")
with (out / (a.label + "-checks.log")).open("w") as log:
    code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
new = [json.loads(x) for x in events.read_text().splitlines()[before:]]
report.update(
    finished=time.time(),
    exit_code=code,
    events=len(new),
    host_compat_events=sum(x["event"] == "explicit_host_compat" for x in new),
    capture_events=sum(x["event"] == "capture" for x in new),
    retirement_events=sum(x["event"] in ("shutdown_release", "retirement", "eviction") for x in new),
    largest_active_decode_rows=max([sum(p >= 0 for p in x["positions"]) for x in new if x["event"] == "decode"] or [0]),
)
path.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if "benchmark" in a.stages.split(","):
    assert report["host_compat_events"] == 0, "Benchmark took host sampling compatibility path"
assert report["capture_events"] == 0 and report["retirement_events"] == 0, report
raise SystemExit(code)
