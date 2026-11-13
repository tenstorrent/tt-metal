#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extend the natural-ending sample check while the grid coordinator is paused."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path.cwd()
model = root / "models/autoports/google_gemma_4_31b_it"
base = model / "doc/benchmark"
sys.path.insert(0, str(model / "tools"))
from benchmark_server import idle, process

while process(12377) is not None:
    time.sleep(1)
assert len(json.loads((base / "anomaly_check.json").read_text())["replies"]) == 3
assert len(json.loads((base / "qualitative_outputs.json").read_text())) == 6
for _ in range(3):
    assert idle("http://127.0.0.1:8000")
    time.sleep(1)
with (base / "setup/long_anomaly_run.log").open("w") as log:
    subprocess.run(
        [str(root / "python_env/bin/python"), str(model / "tools/benchmark_anomalies.py")],
        stdout=log,
        stderr=subprocess.STDOUT,
        check=True,
    )
sys.path.insert(0, str(Path(os.environ["TT_MODEL_BRINGUP_ROOT"]) / "scripts"))
from reply_digest import responses

identity = json.loads((base / "run/perf-b1-server.json").read_text())
logged = {body["id"]: body for body in responses(Path(identity["server_log"]).read_text(errors="replace").splitlines())}
record = {"server_log": identity["server_log"], "responses": []}
for path in sorted((base / "anomaly_replies").glob("*-response.json")):
    body = json.loads(path.read_text())
    matches = logged.get(body["id"]) == body
    record["responses"].append({"response_id": body["id"], "exact_complete_body_in_log": matches, "source": str(path)})
(base / "setup/long_response_logging.json").write_text(json.dumps(record, indent=2) + "\n")
assert all(row["exact_complete_body_in_log"] for row in record["responses"]), record
boundary = json.loads((base / "setup/long_sample_boundary.json").read_text())
current = process(boundary["pid"])
assert current and current["start_ticks"] == boundary["start_ticks"]
os.kill(boundary["pid"], signal.SIGCONT)
boundary["resumed_unix"] = time.time()
(base / "setup/long_sample_boundary.json").write_text(json.dumps(boundary, indent=2) + "\n")
print("Extended anomaly check and full-body logging passed; grid coordinator resumed.", flush=True)
