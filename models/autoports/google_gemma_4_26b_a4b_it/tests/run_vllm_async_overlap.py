#!/usr/bin/env python3
"""Focused traced-async state test against an already-running vLLM server."""

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import openai


def degeneracy_reasons(text: str) -> list[str]:
    """Return mechanical repetition failures, including doubled tokens/phrases."""
    words = re.findall(r"[\w']+", text.casefold())
    reasons = []
    doubled = [words[index] for index in range(1, len(words)) if words[index] == words[index - 1]]
    if doubled:
        reasons.append(f"consecutive doubled token(s): {sorted(set(doubled))}")
    for width in range(2, min(13, len(words) // 2 + 1)):
        for start in range(len(words) - 2 * width + 1):
            if words[start : start + width] == words[start + width : start + 2 * width]:
                reasons.append(f"adjacent repeated {width}-word phrase: {' '.join(words[start:start + width])!r}")
                return reasons
    if len(words) >= 12:
        word, count = Counter(words).most_common(1)[0]
        if count >= 5 and count / len(words) > 0.20:
            reasons.append(f"dominant repeated token: {word!r} appears {count}/{len(words)} times")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = openai.OpenAI(base_url=f"{args.server_url.rstrip('/')}/v1", api_key="dummy")

    cases = [
        {
            "name": "cross_page_boundary",
            "prompt": (
                "Explain, in a connected short essay of at least 140 words, how a brass telescope helps an "
                "astronomer observe the Moon. Discuss its lenses, focusing, mounting, and careful observation."
            ),
            "max_tokens": 96,
            "ignore_eos": True,
        },
        {
            "name": "early_completion",
            "prompt": "Reply with exactly these two words and nothing else: Brass telescope",
            "max_tokens": 32,
            "ignore_eos": False,
        },
    ]

    def request(case, delay=0.0):
        if delay:
            time.sleep(delay)
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": case["prompt"]}],
            max_tokens=case["max_tokens"],
            temperature=0.0,
            extra_body={"ignore_eos": case["ignore_eos"]},
        )
        return {
            "text": response.choices[0].message.content or "",
            "completion_tokens": response.usage.completion_tokens,
            "finish_reason": response.choices[0].finish_reason,
        }

    isolated = [request(case) for case in cases]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, cases[0], 0.0), pool.submit(request, cases[1], 0.15)]
        overlapped = [future.result() for future in futures]

    matches = [expected["text"] == actual["text"] for expected, actual in zip(isolated, overlapped)]
    degeneracy = [degeneracy_reasons(result["text"]) for result in isolated + overlapped]
    crossed_page = isolated[0]["completion_tokens"] > 64 and overlapped[0]["completion_tokens"] > 64
    finished_early = (
        isolated[1]["completion_tokens"] < cases[1]["max_tokens"]
        and overlapped[1]["completion_tokens"] < cases[1]["max_tokens"]
        and isolated[1]["finish_reason"] == "stop"
        and overlapped[1]["finish_reason"] == "stop"
    )
    passed = all(matches) and not any(degeneracy) and crossed_page and finished_early
    report = {
        "verdict": "pass" if passed else "fail",
        "server_mode": "async scheduling + trace + sample_on_device=all",
        "prompt_mode": "OpenAI chat completions (Gemma instruct chat template)",
        "invariants": {
            "overlapped_outputs_match_isolated_controls": all(matches),
            "decode_crosses_64_token_page_boundary": crossed_page,
            "short_request_reaches_eos_before_long_request": finished_early,
            "no_repeated_or_doubled_token_degeneracy": not any(degeneracy),
        },
        "state_evidence": (
            "Direct persistent-input, changed/unchanged page-table, and deferred-read counters are recorded "
            "separately by test_reduced_mixed_prompt_and_inactive_slot_probe."
        ),
        "cases": [
            {
                **case,
                "matches_isolated_control": matches[index],
                "isolated_sha256": hashlib.sha256(isolated[index]["text"].encode()).hexdigest(),
                "overlapped_sha256": hashlib.sha256(overlapped[index]["text"].encode()).hexdigest(),
                "isolated_completion_tokens": isolated[index]["completion_tokens"],
                "overlapped_completion_tokens": overlapped[index]["completion_tokens"],
                "isolated_finish_reason": isolated[index]["finish_reason"],
                "overlapped_finish_reason": overlapped[index]["finish_reason"],
                "isolated_degeneracy_reasons": degeneracy[index],
                "overlapped_degeneracy_reasons": degeneracy[index + len(cases)],
                "completion": overlapped[index]["text"],
            }
            for index, case in enumerate(cases)
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit("async overlap state/quality gate failed; inspect output artifact")


if __name__ == "__main__":
    main()
