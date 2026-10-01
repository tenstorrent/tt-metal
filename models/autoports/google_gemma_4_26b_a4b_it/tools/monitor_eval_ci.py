# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read-only CI status and payload-free serving-counter monitor."""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=int)
    parser.add_argument("--repo", default="tenstorrent/tt-agentic-bringup-qb2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server", help="Verified HTTP server alias when it differs from the Actions runner name")
    args = parser.parse_args()
    names = (
        "num_requests_running|num_requests_waiting|prompt_tokens_total|generation_tokens_total|"
        "request_success_total|time_to_first_token_seconds_count|time_to_first_token_seconds_sum|"
        "e2e_request_latency_seconds_sum"
    )
    metric_pattern = re.compile(r"^vllm:(" + names + r")\{([^}]*)\} ([\d.eE+\-]+)$")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = {"utc": datetime.now(timezone.utc).isoformat()}
        done = False
        try:
            run = json.loads(
                subprocess.check_output(["gh", "api", f"repos/{args.repo}/actions/runs/{args.run}"], timeout=15)
            )
            jobs = json.loads(
                subprocess.check_output(["gh", "api", f"repos/{args.repo}/actions/runs/{args.run}/jobs"], timeout=15)
            )["jobs"]
            row.update(status=run["status"], conclusion=run["conclusion"])
            row["jobs"] = [
                {
                    "id": j["id"],
                    "status": j["status"],
                    "conclusion": j["conclusion"],
                    "runner": j["runner_name"],
                    "active_steps": [s["name"] for s in j["steps"] if s["status"] == "in_progress"],
                }
                for j in jobs
                if j["runner_name"] and "/ run-tests /" in j["name"]
            ]
            done = run["status"] == "completed"
            running = [j for j in row["jobs"] if j["status"] == "in_progress"]
            if len(running) == 1:
                row["server"] = args.server or f"http://{running[0]['runner']}:8000"
                endpoint = row["server"].rstrip("/") + "/metrics"
                try:
                    with urlopen(endpoint, timeout=3) as response:
                        metrics = {}
                        for line in response.read().decode().splitlines():
                            match = metric_pattern.match(line)
                            if match:
                                reason = re.search(r'finished_reason="([^"]+)"', match[2])
                                key = match[1] + (":" + reason[1] if reason else "")
                                metrics[key] = metrics.get(key, 0) + float(match[3])
                        row["metrics"] = metrics
                except OSError as error:
                    row["metrics_error"] = type(error).__name__
        except (subprocess.SubprocessError, ValueError, KeyError) as error:
            row["status_error"] = type(error).__name__
        encoded = json.dumps(row)
        with args.output.open("a") as stream:
            stream.write(encoded + "\n")
        print(encoded, flush=True)
        if done:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
