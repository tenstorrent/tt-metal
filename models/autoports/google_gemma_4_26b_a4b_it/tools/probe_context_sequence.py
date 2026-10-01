# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Replay consecutive recorded requests with native stopping and unchanged limits."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from replay_eval_requests import TOOLS, post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    path = next(args.artifact_root.glob("**/swe_bench*/django*/agent/mini-swe-agent.trajectory.json"))
    messages = json.loads(path.read_text())["messages"]
    indices = [i for i, m in enumerate(messages) if i >= 322 and m.get("extra", {}).get("response")][: args.count]
    rows = []
    for index in indices:
        context = [
            {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}}
            for m in messages[:index]
        ]
        started = time.monotonic()
        first, last, usage, finish = None, None, None, None
        chunks = []
        with post(
            args.base_url,
            "/v1/chat/completions",
            {
                "model": "google/gemma-4-26B-A4B-it",
                "messages": context,
                "tools": TOOLS,
                "max_tokens": 32768,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "seed": 9472,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        ) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                now = time.monotonic()
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("tool_calls"):
                        first = now if first is None else first
                        last = now
                        chunks.append(delta)
                    finish = choice.get("finish_reason") or finish
                if now - started > 180:
                    break
        row = {
            "index": index,
            "usage": usage,
            "elapsed_s": time.monotonic() - started,
            "first_visible_delta_s": None if first is None else first - started,
            "decode_visible_s": None if first is None else last - first,
            "finish_reason": finish,
            "deltas_sha256": hashlib.sha256(json.dumps(chunks).encode()).hexdigest(),
            "tool_deltas": [d["tool_calls"] for d in chunks if d.get("tool_calls")],
            "protocol": "C1 native stop, same sampling/output limit, seeded diagnostic, tools not executed; 180s request wall cap",
        }
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "tool_deltas"}), flush=True)


if __name__ == "__main__":
    main()
