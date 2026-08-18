# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""What does a *penalised* request actually cost, on the primary workload?

The stage's headline benchmark (128/128/1, ``max_num_seqs=1``) never sets a
penalty, so it measures the fast path and says nothing about the penalised one.
The penalty stage adds per-step **host** work — it stages full-vocabulary
operands before every decode step — and host work on this path is serial with the
trace replay, so it lands directly on TPOT. This probe measures that, in situ,
through the same HTTP serving path, at the same shape.

Three legs against one live server, so the comparison is internally controlled
(same harness, same prompt, same token count, same server process):

* ``none`` — no penalty. ``_penalty_mode == 0``: the ops are not in the captured
  trace. This leg also validates the harness, because it must land on the
  `vllm bench serve` headline.
* ``repetition_only`` — ``_penalty_mode == 1``: one full-vocabulary operand.
  This is the realistic case: this checkpoint's ``generation_config.json``
  injects ``repetition_penalty=1.05`` into every request that does not override
  it, so any server run *without* ``--generation-config vllm`` is on this path.
* ``all_three`` — ``_penalty_mode == 3``: two operands.

The prompt is passed as **explicit token ids**, not text, so every leg decodes
exactly 128 tokens after exactly 128 prompt tokens regardless of what the
penalties do to the output — TPOT is then a like-for-like comparison rather than
a comparison of two different-length generations. ``ignore_eos`` keeps the length
fixed for the same reason.

TTFT is the time to the first streamed token; TPOT is the mean inter-token
latency over the remaining tokens, which is the same definition
``vllm bench serve`` uses for a single request.

Needs a live server at ``max_num_seqs=1``; does not open a device itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: 128 arbitrary but valid, non-special token ids. Explicit ids rather than text
#: so the prompt length is exactly 128 on every leg.
PROMPT_TOKENS = list(range(1000, 1128))

LEGS = [
    ("none", {}),
    # The checkpoint's own generation_config default.
    ("repetition_only", {"repetition_penalty": 1.05}),
    ("all_three", {"repetition_penalty": 1.05, "frequency_penalty": 0.5, "presence_penalty": 0.5}),
]


def stream_once(url: str, model: str, max_tokens: int, extra: dict) -> dict:
    body = {
        "model": model,
        "prompt": PROMPT_TOKENS,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        **extra,
    }
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            choice = chunk.get("choices", [{}])[0]
            if choice.get("text"):
                stamps.append(time.perf_counter())
    if len(stamps) < 2:
        raise RuntimeError(f"only {len(stamps)} streamed chunks for {extra!r}")
    ttft = (stamps[0] - start) * 1e3
    itl = [(b - a) * 1e3 for a, b in zip(stamps, stamps[1:])]
    return {
        "tokens_streamed": len(stamps),
        "ttft_ms": ttft,
        "tpot_ms": statistics.fmean(itl),
        "itl_p50_ms": statistics.median(itl),
        "itl_p99_ms": sorted(itl)[max(0, int(0.99 * len(itl)) - 1)],
        "tsu": 1000.0 / statistics.fmean(itl),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", default=str(HERE / "penalty_serving_cost_probe.json"))
    args = parser.parse_args()

    # One warm request so no leg pays the first-touch cost of the server.
    stream_once(args.url, args.model, 16, {})

    legs: dict = {}
    for name, extra in LEGS:
        runs = [stream_once(args.url, args.model, args.max_tokens, extra) for _ in range(args.repeats)]
        best = min(runs, key=lambda r: r["tpot_ms"])
        legs[name] = {
            "penalties": extra,
            "repeats": args.repeats,
            "tokens_streamed": best["tokens_streamed"],
            # Median across repeats for the headline figures, so one slow run
            # (server-side GC, host scheduling) does not decide the answer.
            "ttft_ms": round(statistics.median(r["ttft_ms"] for r in runs), 3),
            "tpot_ms": round(statistics.median(r["tpot_ms"] for r in runs), 3),
            "itl_p50_ms": round(statistics.median(r["itl_p50_ms"] for r in runs), 3),
            "itl_p99_ms": round(statistics.median(r["itl_p99_ms"] for r in runs), 3),
            "tsu": round(1000.0 / statistics.median(r["tpot_ms"] for r in runs), 3),
            "all_runs_tpot_ms": [round(r["tpot_ms"], 3) for r in runs],
        }
        print(
            f"[{name:16s}] TTFT {legs[name]['ttft_ms']:8.3f}  TPOT {legs[name]['tpot_ms']:7.3f}  "
            f"t/s/u {legs[name]['tsu']:7.3f}  (runs {legs[name]['all_runs_tpot_ms']})"
        )

    base = legs["none"]["tpot_ms"]
    results = {
        "url": args.url,
        "model": args.model,
        "workload": {"prompt_len": 128, "output_len": args.max_tokens, "num_requests": 1, "concurrency": 1},
        "legs": legs,
        "penalty_tpot_overhead_ms": {
            name: round(legs[name]["tpot_ms"] - base, 3) for name in ("repetition_only", "all_three")
        },
        "penalty_tpot_overhead_pct": {
            name: round(100.0 * (legs[name]["tpot_ms"] - base) / base, 1) for name in ("repetition_only", "all_three")
        },
        "unpenalised_tokens_streamed": legs["none"]["tokens_streamed"],
    }
    results["all_legs_same_token_count"] = len({legs[n]["tokens_streamed"] for n in legs}) == 1
    Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v for k, v in results.items() if k != "legs"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
