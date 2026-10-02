# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""CPU-only exact-token opportunity audit; does not alter or replay agent trials.

Supply a JSON array of saved trajectories on stdin. Only compact statistics
are written; histories and commands remain in their original evidence files.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

from probe_context_dedup import compact
from probe_native_eval_action import request_messages
from replay_eval_requests import TOOLS


def canonical(messages):
    result = copy.deepcopy(messages)
    for message in result:
        if isinstance(message.get("content"), str):
            message["content"] = [{"type": "text", "text": message["content"]}]
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"].get("arguments"), str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return result


def common_prefix_blocks(previous, current, block=32):
    """Token-match opportunity only; does not assert cache residency or support."""
    common = 0
    for before, after in zip(previous, current):
        if before != after:
            break
        common += 1
    return min(common, max(0, len(current) - 1)) // block * block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    model = "google/gemma-4-26B-A4B-it"
    revision = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True)

    def encode(messages):
        return tokenizer.apply_chat_template(
            canonical(messages),
            tools=TOOLS,
            add_generation_prompt=True,
            enable_thinking=True,
            tokenize=True,
            return_dict=False,
        )

    report = {
        "scope": "offline token opportunity, not latency or reward improvement",
        "revision": revision,
        "trials": [],
    }
    for item in json.load(sys.stdin):
        saved = item["messages"]
        messages = request_messages(saved)
        rows = []
        previous = []
        for index, message in enumerate(saved):
            usage = message.get("extra", {}).get("response", {}).get("usage")
            if message["role"] != "assistant" or not usage:
                continue
            original = messages[:index]
            candidate, changes = compact(original)
            tokens = encode(original)
            before = len(tokens)
            after = len(encode(candidate)) if changes else before
            rows.append(
                {
                    "message_index": index,
                    "recorded_tokens": usage["prompt_tokens"],
                    "baseline_tokens": before,
                    "candidate_tokens": after,
                    "removed_tokens": before - after,
                    "compacted_outputs": len(changes),
                    "previous_input_matching_block_tokens": common_prefix_blocks(previous, tokens),
                }
            )
            previous = tokens
        total = sum(row["baseline_tokens"] for row in rows)
        removed = sum(row["removed_tokens"] for row in rows)
        summary = {
            "trial": item["trial"],
            "responses": len(rows),
            "baseline_input_tokens": total,
            "removed_input_tokens": removed,
            "removed_fraction": removed / total if total else 0,
            "canonical_counts_match_saved": all(row["recorded_tokens"] == row["baseline_tokens"] for row in rows),
            "previous_input_matching_block_tokens": sum(row["previous_input_matching_block_tokens"] for row in rows),
            "rows": rows,
        }
        report["trials"].append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
