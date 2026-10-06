# SPDX-License-Identifier: Apache-2.0
"""Audit explicit trace shutdown and absence of serving device holders."""

import argparse
import json
import os
import shutil
import time
from collections import Counter
from pathlib import Path

import psutil

p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--runner-pid", required=True, type=int)
a = p.parse_args()
root = Path(os.environ["MODEL_DIR"]) / "readiness_vllm"
events = [json.loads(line) for line in (root / (a.label + "-events.jsonl")).read_text().splitlines()]
captures = [x for x in events if x["event"] == "capture"]
releases = [x for x in events if x["event"] == "shutdown_release"]
leftovers = []
for proc in psutil.process_iter(["pid", "name", "cmdline"]):
    cmd = proc.info["cmdline"] or []
    if (proc.info["name"] or "").startswith("VLLM::Engine") or any(
        x in cmd for x in ("vllm.entrypoints.openai.api_server", "readiness_check.run_vllm_server")
    ):
        leftovers.append(proc.info)
captured = Counter((x["pid"], x["trace_id"]) for x in captures)
released = Counter((x["pid"], x["trace_id"]) for x in releases)
result = dict(
    time=time.time(),
    method=f"SIGINT shared runner {a.runner_pid}; runner completion checked by caller",
    leftover_processes=leftovers,
    captures=captures,
    shutdown_releases=releases,
    exact_once_release=captured == released and len(captured) == 8 and all(n == 1 for n in captured.values()),
)
shutil.copy2(root / "server.log", root / (a.label + "-server.log"))
(root / (a.label + "-cleanup.json")).write_text(json.dumps(result, indent=2) + "\n")
assert not leftovers, leftovers
assert result["exact_once_release"], result
print("CLEANUP_PASS", a.label, "eight traces released exactly once; no serving processes")
