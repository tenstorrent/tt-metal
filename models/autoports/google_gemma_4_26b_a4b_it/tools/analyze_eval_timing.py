# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Reconstruct mini-swe timing without treating sampled Running=0 as idle."""

import argparse
import json
import re
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path


def epoch(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def analyze(path, server_samples):
    data = json.loads(path.read_text())
    result = json.loads((path.parent.parent / "result.json").read_text())
    phase = result["agent_execution"]
    wall = epoch(phase["finished_at"]) - epoch(phase["started_at"])
    requests, tools, commands = [], [], []
    previous_end = None
    for message in data["messages"]:
        extra = message.get("extra", {})
        response = extra.get("response")
        if response:
            end = extra["timestamp"]
            start = response["created"]
            usage = response["usage"]
            requests.append(
                {
                    "start_epoch_floor_s": start,
                    "end_epoch_s": end,
                    "request_s_upper": end - start,
                    "prompt_tokens": usage["prompt_tokens"],
                    "output_tokens": usage["completion_tokens"],
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "finish_reason": response["choices"][0]["finish_reason"],
                }
            )
            previous_end = end
            commands.extend(
                a["command"] if isinstance(a["command"], str) else json.dumps(a["command"], sort_keys=True)
                for a in extra.get("actions", [])
            )
        elif message["role"] == "tool" and previous_end:
            tools.append({"duration_s": extra["timestamp"] - previous_end, "returncode": extra.get("returncode")})
            previous_end = extra["timestamp"]
    counts = Counter(commands)
    upper = sum(r["request_s_upper"] for r in requests)
    samples = [s for s in server_samples if epoch(phase["started_at"]) <= s[0] <= epoch(phase["finished_at"])]
    return {
        "task": result["task_name"],
        "wall_s": wall,
        "reward": result.get("verifier_result"),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
        "responses": len(requests),
        "prompt_tokens": sum(r["prompt_tokens"] for r in requests),
        "output_tokens": sum(r["output_tokens"] for r in requests),
        "request_s_lower": upper - len(requests),
        "request_s_upper": upper,
        "tool_s": sum(t["duration_s"] for t in tools),
        "tool_returncodes": dict(Counter(t["returncode"] for t in tools)),
        "repeated_commands": sum(n - 1 for n in counts.values()),
        "top_repeats": counts.most_common(5),
        "prompt_tokens_median": statistics.median(r["prompt_tokens"] for r in requests) if requests else None,
        "prompt_tokens_max": max((r["prompt_tokens"] for r in requests), default=None),
        "output_tokens_median": statistics.median(r["output_tokens"] for r in requests) if requests else None,
        "last_response_to_timeout_s": epoch(phase["finished_at"]) - requests[-1]["end_epoch_s"] if requests else None,
        "response_timestamp_after_agent_end_s": [
            r["end_epoch_s"] - epoch(phase["finished_at"])
            for r in requests
            if r["end_epoch_s"] > epoch(phase["finished_at"])
        ],
        "server_generated_tokens_10s_approx": round(sum(s[2] * 10 for s in samples)),
        "server_prompt_tokens_10s_approx": round(sum(s[1] * 10 for s in samples)),
        "server_decode_active_s_10s_approx": sum(10 for s in samples if s[2] > 20),
        "requests": requests,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    server_samples = []
    pattern = re.compile(
        r"(?:INFO|DEBUG) (\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+)"
    )
    for path in args.artifact_root.glob("tt_triage/vllm_*.log"):
        year = re.search(r"vllm_(\d{4})-", path.name)[1]
        for line in path.read_text().splitlines():
            match = pattern.search(line)
            if match:
                stamp = epoch(year + "-" + match[1].replace(" ", "T") + "+00:00")
                server_samples.append((stamp, float(match[2]), float(match[3])))
    rows = [
        analyze(p, server_samples)
        for p in sorted(args.artifact_root.glob("**/swe_bench*/*/agent/mini-swe-agent.trajectory.json"))
    ]
    if args.summary_only:
        for row in rows:
            row.pop("requests")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
