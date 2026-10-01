# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Compare bounded next-action recovery on a recorded repetitive SWE context."""

import argparse
import json
import time
from pathlib import Path

from replay_eval_requests import TOOLS, post


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="django")
    parser.add_argument("--message-index", type=int, default=322)
    parser.add_argument("--cases", nargs="+", default=["control", "repetition_feedback", "thinking_enabled"])
    args = parser.parse_args()
    path = next(args.artifact_root.glob(f"**/swe_bench*/{args.task}*/agent/mini-swe-agent.trajectory.json"))
    data = json.loads(path.read_text())
    # Default: the 12028-token Django context from the timing replay.
    messages = [
        {k: v for k, v in m.items() if k in {"role", "content", "tool_calls", "tool_call_id"}}
        for m in data["messages"][: args.message_index]
    ]
    intervention = {
        "role": "user",
        "content": "Loop check: the same command has been repeated with the same unsuccessful result. Repeating it again without new evidence or a relevant change will not add evidence. Choose a different inspection or test, update your hypothesis using the observations already present, and continue solving the original issue. Do not submit until you have made and checked the required fix.",
    }
    generation_recovery = {
        "role": "user",
        "content": "Your previous response repeated the same text extensively without completing a tool action. It was interrupted before any tool command was executed. Reassess using the observations above. Choose one concrete inspection, edit, or test that produces new evidence. Avoid repeating the same explanation.",
    }
    rows = []
    for name, context, thinking in [
        ("control", messages, False),
        ("repetition_feedback", messages + [intervention], False),
        ("thinking_enabled", messages, True),
        ("generation_recovery", messages + [generation_recovery], False),
    ]:
        if name not in args.cases:
            continue
        tokens = json.loads(
            post(
                args.base_url,
                "/tokenize",
                {
                    "model": "google/gemma-4-26B-A4B-it",
                    "messages": context,
                    "tools": TOOLS,
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": thinking},
                },
            ).read()
        )["tokens"]
        start = time.monotonic()
        text, usage, finish = "", None, None
        capped = False
        with post(
            args.base_url,
            "/v1/completions",
            {
                "model": "google/gemma-4-26B-A4B-it",
                "prompt": tokens,
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
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    text += choice.get("text", "")
                    finish = choice.get("finish_reason") or finish
                if time.monotonic() - start > 120:
                    capped = True
                    break
        row = {
            "case": name,
            "prompt_tokens": len(tokens),
            "elapsed_s": time.monotonic() - start,
            "output": text,
            "usage": usage,
            "finish_reason": finish,
            "diagnostic_wall_cap_reached": capped,
            "prompt_format": "pinned HF chat template with mini-swe bash schema",
            "quality_scope": "next-action diagnostic only; no tools executed; no solve/reward claim",
        }
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "output"}), flush=True)


if __name__ == "__main__":
    main()
