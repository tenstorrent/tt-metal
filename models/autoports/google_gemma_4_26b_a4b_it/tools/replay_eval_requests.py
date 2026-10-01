# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Bounded C1 serving probe using real SWE trajectory contexts (no task execution)."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
                "required": ["command"],
            },
        },
    }
]


def post(base, endpoint, payload):
    return urlopen(
        Request(base + endpoint, json.dumps(payload).encode(), {"Content-Type": "application/json"}), timeout=900
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()
    model = "google/gemma-4-26B-A4B-it"
    rows = []
    for task, target in [
        ("django", 2000),
        ("django", 12000),
        ("django", 21000),
        ("astropy", 32000),
        ("scikit-learn", 49000),
    ]:
        path = next(args.artifact_root.glob(f"**/swe_bench*/{task}*/agent/mini-swe-agent.trajectory.json"))
        data = json.loads(path.read_text())
        candidates = [
            (i, m["extra"]["response"]["usage"]["prompt_tokens"])
            for i, m in enumerate(data["messages"])
            if m.get("extra", {}).get("response")
        ]
        index, recorded_tokens = min(candidates, key=lambda item: abs(item[1] - target))
        messages = [
            {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}}
            for m in data["messages"][:index]
        ]
        encoded = json.loads(
            post(
                args.base_url,
                "/tokenize",
                {
                    "model": model,
                    "messages": messages,
                    "tools": TOOLS,
                    "add_generation_prompt": True,
                },
            ).read()
        )
        tokens = encoded["tokens"]
        for repeat in range(args.repeat):
            payload = {
                "model": model,
                "prompt": tokens,
                "max_tokens": 128,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "seed": 9472,
                "ignore_eos": True,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            start = time.monotonic()
            first, last, usage = None, None, None
            chunks = []
            with post(args.base_url, "/v1/completions", payload) as response:
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    now = time.monotonic()
                    for choice in event.get("choices", []):
                        if choice.get("text"):
                            first = now if first is None else first
                            last = now
                            chunks.append(choice["text"])
                    usage = event.get("usage") or usage
            end = time.monotonic()
            row = {
                "task": task,
                "message_index": index,
                "recorded_prompt_tokens": recorded_tokens,
                "rendered_prompt_tokens": len(tokens),
                "repeat": repeat,
                "elapsed_s": end - start,
                "ttft_s": first - start,
                "decode_s": last - first,
                "usage": usage,
                "output_sha256": hashlib.sha256("".join(chunks).encode()).hexdigest(),
                "output_preview": "".join(chunks)[:240],
                "protocol": "C1 real context; diagnostic output fixed128 ignore_eos; not an accuracy eval",
            }
            rows.append(row)
            args.output.write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
