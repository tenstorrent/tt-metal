# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay one saved task context through native chat/tool parsing; execute no tools."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from probe_loop_recovery import wall_deadline
from replay_eval_requests import TOOLS, post


def request_messages(messages):
    """Keep template-visible reasoning while excluding local trajectory metadata."""
    fields = {"role", "content", "tool_calls", "tool_call_id", "reasoning", "reasoning_content"}
    return [{k: v for k, v in message.items() if k in fields} for message in messages]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--messages", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--seconds", type=float, default=180)
    args = parser.parse_args()
    messages = json.loads(args.trajectory.read_text())["messages"][: args.messages]
    payload = {
        "model": "google/gemma-4-26B-A4B-it",
        "messages": request_messages(messages),
        "tools": TOOLS,
        "chat_template_kwargs": {"enable_thinking": True},
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "seed": 9472,
        "max_tokens": 32768,
        "repetition_detection": {"min_pattern_size": 16, "max_pattern_size": 1024, "min_count": 8},
    }
    deadline, response = {"expired": False}, None
    start = time.monotonic()
    with wall_deadline(args.seconds, deadline):
        with post(args.base_url, "/v1/chat/completions", payload) as stream:
            response = json.load(stream)
    report = {
        "scope": "Native parsed next-action diagnostic only; no tool execution or solve/reward claim",
        "trajectory": str(args.trajectory),
        "message_count": len(messages),
        "payload_sha256": hashlib.sha256(json.dumps(payload).encode()).hexdigest(),
        "sampling": {k: payload[k] for k in ("temperature", "top_p", "top_k", "seed", "max_tokens")},
        "elapsed_s": time.monotonic() - start,
        "deadline": deadline,
        "response": response,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "response"}), flush=True)


if __name__ == "__main__":
    main()
