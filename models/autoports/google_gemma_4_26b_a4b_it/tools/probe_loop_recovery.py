# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Compare bounded next-action recovery on a recorded repetitive SWE context."""

import argparse
import copy
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
    parser.add_argument("--trajectory", type=Path, help="Explicit local trial trajectory instead of an artifact glob")
    parser.add_argument("--message-index", type=int, default=322)
    parser.add_argument("--cases", nargs="+", default=["control", "repetition_feedback", "thinking_enabled"])
    parser.add_argument("--repetition-detection", type=json.loads)
    parser.add_argument("--wall-cap-sec", type=float, default=120)
    parser.add_argument("--greedy", action="store_true", help="Greedy HF-control comparison, not scored eval policy")
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="Diagnostic output allowance only")
    args = parser.parse_args()
    path = args.trajectory or next(
        args.artifact_root.glob(f"**/swe_bench*/{args.task}*/agent/mini-swe-agent.trajectory.json")
    )
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
    format_error = {
        "role": "user",
        "content": "No tool calls found in the response. Every response MUST include at least one tool call.",
    }
    execution_policy = copy.deepcopy(messages)
    policy = (
        "\nExecution guidance: the task's testbed Python environment and dependencies are already installed "
        "and activated. Use the existing project and its tests. Keep bash commands focused on executable "
        "inspection, editing, or testing; do not put extended reasoning or repeated commentary in shell "
        "comments. Use observations already in the history instead of repeatedly reading unchanged files. "
        "After a failed attempt, change the hypothesis or command based on the actual error."
    )
    if execution_policy and execution_policy[0]["role"] == "system":
        execution_policy[0]["content"] += policy
    else:
        execution_policy.insert(0, {"role": "system", "content": policy.strip()})
    rows = []
    for name, context, thinking in [
        ("control", messages, False),
        ("repetition_feedback", messages + [intervention], False),
        ("thinking_enabled", messages, True),
        ("thinking_control", messages, True),
        ("execution_policy", execution_policy, True),
        ("generation_recovery", messages + [generation_recovery], False),
        ("format_error_recovery", messages + [format_error], False),
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
        first, last = None, None
        capped = False
        with post(
            args.base_url,
            "/v1/completions",
            {
                "model": "google/gemma-4-26B-A4B-it",
                "prompt": tokens,
                "max_tokens": args.max_new_tokens,
                "temperature": 0.0 if args.greedy else 1.0,
                "top_p": 1.0 if args.greedy else 0.95,
                "top_k": 1 if args.greedy else 20,
                "seed": 9472,
                "stream": True,
                "stream_options": {"include_usage": True},
                **({"repetition_detection": args.repetition_detection} if args.repetition_detection else {}),
            },
        ) as response:
            for line in response:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                event = json.loads(line[6:])
                usage = event.get("usage") or usage
                for choice in event.get("choices", []):
                    chunk = choice.get("text", "")
                    if chunk:
                        last = time.monotonic()
                        first = last if first is None else first
                        text += chunk
                    finish = choice.get("finish_reason") or finish
                if time.monotonic() - start > args.wall_cap_sec:
                    capped = True
                    break
        row = {
            "case": name,
            "sampling": {
                "temperature": 0.0 if args.greedy else 1.0,
                "top_p": 1.0 if args.greedy else 0.95,
                "top_k": 1 if args.greedy else 20,
                "max_tokens": args.max_new_tokens,
                "seed": 9472,
            },
            "prompt_tokens": len(tokens),
            "elapsed_s": time.monotonic() - start,
            "ttft_s": None if first is None else first - start,
            "decode_s": None if first is None else last - first,
            "output": text,
            "usage": usage,
            "finish_reason": finish,
            "diagnostic_wall_cap_reached": capped,
            "prompt_format": "pinned HF chat template with mini-swe bash schema",
            "quality_scope": "next-action diagnostic only; no tools executed; no solve/reward claim",
            "repetition_detection": args.repetition_detection,
        }
        rows.append(row)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "output"}), flush=True)


if __name__ == "__main__":
    main()
