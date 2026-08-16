#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does vLLM's OWN presence-penalty implementation pass the failing conformance row?

The arithmetic probes settle that the device computes vLLM's presence rule, in bfloat16.
That leaves the behavioural question the conformance row actually asserts:

    unique_ratio(presence_penalty=1.2) >= unique_ratio(baseline) * 0.90

on "Write a very repetitive story.".  If that assertion is a property of the *penalty*, then
vLLM's own fp32 implementation of it must satisfy it on this model.  If it is a property of
*this model's* answer to that prompt, then vLLM's implementation fails it too, and the row is
a behavioural heuristic rather than a defect in the TT path.

`logprobs: true` routes the request to vLLM's host sampler on this 4-die mesh, where the
penalty is applied by `vllm/model_executor/layers/utils.py::apply_penalties` in float32 --
the reference implementation, with no Tenstorrent sampling or penalty code involved.  The
plain request runs the same model through the on-device path.  Same prompt, same payload,
same statistics function as the test, several seeds.

The comparison is only meaningful when both arms actually produced a full answer, so
`finish_reason` and length are recorded: a truncated or early-stopped response has an
inflated type-token ratio and passes the assertion for the wrong reason.

Usage::

    python presence_penalty_reference_behaviour.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections import Counter

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
MESSAGES = [{"role": "user", "content": "Write a very repetitive story."}]
PENALTY = 1.2


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def repetition_stats(text: str) -> dict:
    """Verbatim from tt-inference-server llm_module/test_vllm_chat_completions.py."""
    tokens = text.lower().split()
    counts = Counter(tokens)
    total = len(tokens)
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values()) if total else 0.0
    return {
        "len": total,
        "unique": len(set(tokens)),
        "unique_ratio": len(set(tokens)) / total if total else 0,
        "most_common": counts.most_common(3),
        "entropy": round(entropy, 4),
    }


def run(seed, temperature, host_sampler: bool, penalty: float, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": MESSAGES,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        body["seed"] = seed
    if penalty:
        body["presence_penalty"] = penalty
    if host_sampler:
        body["logprobs"] = True
        body["top_logprobs"] = 1
    ch = post(body)["choices"][0]
    text = ch["message"]["content"] or ""
    st = repetition_stats(text)
    st["finish_reason"] = ch.get("finish_reason")
    st["text"] = text
    return st


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="*", default=[1234, 1, 42, 7, 2024])
    args = ap.parse_args()

    arms = []
    configs = [("device_sampler", False), ("host_sampler_vllm_reference", True)]
    trials = [("greedy_temp0", None, 0.0)] + [(f"seed{s}_temp0.1", s, 0.1) for s in args.seeds]

    for arm_name, host in configs:
        for tname, seed, temp in trials:
            base = run(seed, temp, host, 0.0, args.max_tokens)
            pen = run(seed, temp, host, PENALTY, args.max_tokens)
            ratio = (pen["unique_ratio"] / base["unique_ratio"]) if base["unique_ratio"] else None
            arms.append(
                {
                    "arm": arm_name,
                    "trial": tname,
                    "seed": seed,
                    "temperature": temp,
                    "base": {k: v for k, v in base.items() if k != "text"},
                    "penalty": {k: v for k, v in pen.items() if k != "text"},
                    "assertion_ratio": ratio,
                    "assertion_passes": (ratio is not None and ratio >= 0.90),
                    "base_text_head": base["text"][:300],
                    "penalty_text_head": pen["text"][:300],
                }
            )
            print(
                json.dumps(
                    {
                        "arm": arm_name,
                        "trial": tname,
                        "base_len": base["len"],
                        "pen_len": pen["len"],
                        "base_ur": round(base["unique_ratio"], 4),
                        "pen_ur": round(pen["unique_ratio"], 4),
                        "ratio": None if ratio is None else round(ratio, 4),
                        "passes": arms[-1]["assertion_passes"],
                        "pen_finish": pen["finish_reason"],
                    }
                ),
                flush=True,
            )

    summary = {}
    for arm_name, _ in configs:
        rows = [a for a in arms if a["arm"] == arm_name]
        summary[arm_name] = {
            "trials": len(rows),
            "passes": sum(a["assertion_passes"] for a in rows),
            "fails": sum(not a["assertion_passes"] for a in rows),
            "ratios": [None if a["assertion_ratio"] is None else round(a["assertion_ratio"], 4) for a in rows],
        }

    out = {
        "_what": __doc__.strip().splitlines()[0],
        "prompt": MESSAGES[0]["content"],
        "presence_penalty": PENALTY,
        "max_tokens": args.max_tokens,
        "assertion": "unique_ratio(penalty) >= unique_ratio(base) * 0.90",
        "host_sampler_note": "logprobs:true routes to vLLM's host sampler on this 4-die mesh; the penalty "
        "is then vllm/model_executor/layers/utils.py::apply_penalties in float32, no TT sampling code involved",
        "summary": summary,
        "trials": arms,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
