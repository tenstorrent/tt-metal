# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Read-only bounded-interval CI and vLLM metrics monitor; stops when jobs finish."""

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
    parser.add_argument("--job", action="append", required=True)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    if not 10 <= args.interval <= 60:
        parser.error("Use a 10–60 second monitoring interval")
    names = {
        "num_requests_running",
        "num_requests_waiting",
        "prompt_tokens_total",
        "generation_tokens_total",
        "request_success_total",
        "e2e_request_latency_seconds_sum",
        "time_to_first_token_seconds_sum",
        "time_to_first_token_seconds_count",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        row = {"utc": datetime.now(timezone.utc).isoformat(), "jobs": [], "server": args.server}
        for job in args.job:
            result = subprocess.run(
                ["gh", "api", f"repos/tenstorrent/tt-agentic-bringup-qb2/actions/jobs/{job}"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode:
                row["jobs"].append({"id": job, "status": "query_error"})
                continue
            data = json.loads(result.stdout)
            row["jobs"].append(
                {
                    "id": job,
                    "status": data["status"],
                    "conclusion": data["conclusion"],
                    "runner_name": data.get("runner_name"),
                    "active_steps": [s["name"] for s in data["steps"] if s["status"] == "in_progress"],
                }
            )
        try:
            with urlopen(args.server.rstrip("/") + "/metrics", timeout=5) as response:
                metrics = response.read().decode()
            values = {}
            for line in metrics.splitlines():
                match = re.fullmatch(r"vllm:([a-z0-9_]+)\{([^}]*)\} ([0-9.e+\-]+)", line)
                if match and match[1] in names:
                    reason = re.search(r'finished_reason="([^"]+)"', match[2])
                    key = match[1] + (":" + reason[1] if reason else "")
                    values[key] = float(match[3])
            row["metrics"] = values
        except Exception as error:
            row["metrics_error"] = type(error).__name__
        with args.output.open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if all(job["status"] == "completed" for job in row["jobs"]):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
