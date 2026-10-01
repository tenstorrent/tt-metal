# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Audit native repetition stops inside saved response fields, without inference.

Read a JSON list of saved mini-swe trajectories on stdin. Parsed response fields
are not the original raw model token stream, so this checks internal patterns,
not exact native stopping positions or an exhaustive false-positive guarantee.
"""

import argparse
import json
import sys
from collections import Counter

from transformers import AutoTokenizer

from vllm.sampling_params import RepetitionDetectionParams
from vllm.v1.core.sched.utils import check_sequence_repetition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pattern-size", type=int, default=128)
    args = parser.parse_args()
    trajectories = json.load(sys.stdin)
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-4-26B-A4B-it", revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52", local_files_only=True
    )
    params = RepetitionDetectionParams(min_pattern_size=16, max_pattern_size=args.max_pattern_size, min_count=8)
    counts, flags = Counter(), []
    for item in trajectories:
        for index, message in enumerate(item["trajectory"]["messages"]):
            response = message.get("extra", {}).get("response")
            if not response:
                continue
            counts["responses"] += 1
            for choice in response["choices"]:
                parsed = choice["message"]
                fields = {key: parsed[key] for key in ("content", "reasoning", "reasoning_content") if parsed.get(key)}
                fields.update(
                    {
                        f"tool_{i}": call["function"]["arguments"]
                        for i, call in enumerate(parsed.get("tool_calls") or [])
                    }
                )
                for field, text in fields.items():
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    counts["fields"] += 1
                    counts["field_tokens"] += len(tokens)
                    for end in range(128, len(tokens) + 1):
                        if check_sequence_repetition(tokens[max(0, end - args.max_pattern_size * 8) : end], params):
                            flags.append(
                                {
                                    "task": item["task"],
                                    "message_index": index,
                                    "field": field,
                                    "first_stop_token": end,
                                    "field_tokens": len(tokens),
                                }
                            )
                            break
    print(
        json.dumps(
            {
                "counts": counts,
                "flagged_fields": flags,
                "params": {"min_pattern_size": 16, "max_pattern_size": args.max_pattern_size, "min_count": 8},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
