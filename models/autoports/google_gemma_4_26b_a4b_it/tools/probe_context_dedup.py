# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Quantify exact repeated command/output compaction on recorded contexts.

Only /tokenize is called: this does not generate tokens or execute tools. The
first full output remains in history. Candidate quality must be tested separately.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from replay_eval_requests import TOOLS, post


def compact(messages, minimum_chars=256):
    result, commands, seen, changes = copy.deepcopy(messages), {}, {}, []
    for index, message in enumerate(result):
        for call in message.get("tool_calls") or []:
            commands[call["id"]] = call["function"]
        content = message.get("content")
        call_id = message.get("tool_call_id")
        if message["role"] != "tool" or call_id not in commands or not isinstance(content, str):
            continue
        if len(content) < minimum_chars:
            continue
        fingerprint = hashlib.sha256(json.dumps([commands[call_id], content], sort_keys=True).encode()).hexdigest()
        if fingerprint not in seen:
            seen[fingerprint] = call_id
            continue
        message["content"] = (
            f"The identical command returned exactly the same output as tool call {seen[fingerprint]}. "
            "The full observed output is retained there; this repeated check observed no change."
        )
        changes.append({"message_index": index, "original_chars": len(content), "retained_call_id": seen[fingerprint]})
    return result, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.artifact_root.glob("**/agent/mini-swe-agent.trajectory.json")):
        saved = json.loads(path.read_text())["messages"]
        messages = [
            {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}} for m in saved
        ]
        # Reconstruct the input preceding the last saved assistant response.
        last = max(i for i, m in enumerate(messages) if m["role"] == "assistant")
        messages = messages[:last]
        candidate, changes = compact(messages)
        counts = []
        for context in (messages, candidate):
            counts.append(
                json.loads(
                    post(
                        args.base_url,
                        "/tokenize",
                        {
                            "model": "google/gemma-4-26B-A4B-it",
                            "messages": context,
                            "tools": TOOLS,
                            "add_generation_prompt": True,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    ).read()
                )["count"]
            )
        rows.append(
            {
                "task": path.parent.parent.name,
                "baseline_tokens": counts[0],
                "candidate_tokens": counts[1],
                "removed_tokens": counts[0] - counts[1],
                "changes": changes,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
