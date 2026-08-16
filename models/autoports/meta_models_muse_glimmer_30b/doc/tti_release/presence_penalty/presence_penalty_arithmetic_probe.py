#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which arithmetic does the on-device presence penalty actually implement?

`presence_penalty_argmax_probe.py` showed the device and vLLM's host sampler make the
same greedy choice for 159 steps and then split on a 0.05-logit gap.  Two families of
explanation survive that, and this probe separates them by *prediction*, over the whole
159-step prefix where the two paths agree plus the step where they do not:

  WRONG-FORMULA family.  The device penalises a different token set or by a different
  rule.  Emulated here as four candidate rules, each scored against the device's own
  token sequence:
      presence  : logit - P * [token in generated output]      (vLLM's rule)
      count     : logit - P * count(token in generated output) (frequency's rule)
      prompt+out: logit - P * [token in prompt OR output]      (repetition's token set)
      none      : logit                                        (penalty never lands)

  PRECISION family.  The device applies vLLM's rule, but in bfloat16.  Emulated exactly:
  bf16(1.2) is 1.203125, the logits sit on a bf16 grid, the subtraction result is rounded
  back onto that grid, and exact ties are broken by lowest global token id
  (`TTSampling._adjust_values_for_tiebreak`, k==1 path).

Both families are evaluated against the SAME evidence: the raw (pre-penalty) top-k
logprobs vLLM returns from its host sampler, teacher-forced on the host token sequence,
compared against the DEVICE token sequence.

The probe also measures the bf16 grid the logits live on, since the size of the rounding
error -- and therefore whether the precision family can explain a 0.05 gap at all -- is
set by that grid and by nothing else.  Logprob = logit - logsumexp, so differences of
logprobs within a step are differences of logits: the spacing of those differences is the
logit ULP.

Usage::

    python presence_penalty_arithmetic_probe.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import math
import urllib.request
from collections import Counter

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
PROMPT = [{"role": "user", "content": "Write a very repetitive story."}]
TOKENIZER_DIR = (
    "/home/ttuser/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/"
    "snapshots/f84ecc3a0ea984a4c04542a84269e3d065350a6e"
)
# bfloat16 keeps 8 mantissa bits, so float(1.2) rounds to 1 + 26/128.
BF16_PENALTY_1_2 = 1.203125


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def round_to_grid(x: float, grid: float) -> float:
    """Round x onto a uniform grid, round-half-to-even -- what bf16 rounding reduces to
    while the exponent is fixed (which it is here: the penalty moves a ~16-32 logit by 1.2)."""
    q = x / grid
    r = math.floor(q)
    frac = q - r
    if frac > 0.5 or (frac == 0.5 and r % 2 == 1):
        r += 1
    return r * grid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--penalty", type=float, default=1.2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--top-logprobs", type=int, default=20)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    base = {
        "model": MODEL,
        "messages": PROMPT,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "presence_penalty": args.penalty,
        "return_token_ids": True,
    }
    host_r = post({**base, "logprobs": True, "top_logprobs": args.top_logprobs})
    dev_r = post(dict(base))
    host, device = host_r["choices"][0], dev_r["choices"][0]

    host_ids = list(host["token_ids"])
    device_ids = list(device["token_ids"])
    prompt_ids = set(host_r.get("prompt_token_ids") or [])
    steps = host["logprobs"]["content"]

    def cand_id(entry):
        s = bytes(entry["bytes"]).decode("utf-8", errors="replace")
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    # ---------- 1. the bf16 grid the logits live on ----------
    # Every within-step logprob difference must be an integer multiple of the logit ULP.
    # Only candidates in the SAME bf16 binade share a grid, so measure by rank: the decision is
    # always between the top few, and those sit at the top of the logit range together.
    by_rank = {r: Counter() for r in range(1, 6)}
    rank_n = Counter()
    for step in steps:
        vals = sorted((e["logprob"] for e in step["top_logprobs"]), reverse=True)
        for r in range(2, min(6, len(vals) + 1)):
            d = vals[0] - vals[r - 1]
            rank_n[r] += 1
            for g in (0.125, 0.0625, 0.03125):
                q = d / g
                if abs(q - round(q)) < 1e-3:
                    by_rank[r][g] += 1
    grid = {
        "_what": "spacing of within-step logprob differences == spacing of the logits themselves "
        "(logprob = logit - logsumexp, so the shift cancels)",
        "top1_minus_topN_is_a_multiple_of": {
            f"rank{r}": {
                "steps": rank_n[r],
                "of_0.125": by_rank[r][0.125],
                "of_0.0625": by_rank[r][0.0625],
                "of_0.03125": by_rank[r][0.03125],
            }
            for r in range(2, 6)
        },
        "implied_logit_ulp_at_the_top_of_the_distribution": 0.125,
        "implied_logit_magnitude": "|logit| in [16, 32) -- the only bf16 binade whose ULP is 0.125",
        "note": "lower-ranked candidates sit in lower binades with finer bf16 spacing, so their "
        "differences are multiples of a finer grid; they never decide a greedy step.",
    }

    # ---------- 2. rule emulation, teacher-forced on the host sequence ----------
    n = min(len(host_ids), len(device_ids))
    first_div = next((i for i in range(n) if host_ids[i] != device_ids[i]), -1)
    horizon = len(steps) if first_div < 0 else first_div + 1  # includes the divergent step

    P = args.penalty
    rules = ["presence_fp32", "count_fp32", "prompt_plus_output_fp32", "no_penalty", "presence_bf16_device"]
    agree_device = {r: 0 for r in rules}
    agree_host = {r: 0 for r in rules}
    first_mismatch_device = {r: None for r in rules}
    div_step_pick = {}

    seen_counts: Counter = Counter()
    for t in range(horizon):
        cands = []
        for e in steps[t]["top_logprobs"]:
            cid = cand_id(e)
            if cid is None:
                continue
            cands.append((cid, e["logprob"], e["token"]))
        if not cands:
            continue
        top_lp = max(c[1] for c in cands)

        picks = {}
        for rule in rules:
            best = None
            for cid, lp, tokstr in cands:
                if rule == "presence_fp32":
                    s = lp - (P if seen_counts[cid] > 0 else 0.0)
                elif rule == "count_fp32":
                    s = lp - P * seen_counts[cid]
                elif rule == "prompt_plus_output_fp32":
                    s = lp - (P if (seen_counts[cid] > 0 or cid in prompt_ids) else 0.0)
                elif rule == "no_penalty":
                    s = lp
                else:  # presence_bf16_device: bf16 subtraction on the logit grid
                    # Snap first: the reported logprob is logit - logsumexp in fp32, so it carries
                    # ~1e-7 of logsumexp noise on top of an exact multiple of the 0.125 logit grid.
                    # Snapping restores the true logit difference, so ties come out exactly tied
                    # and the device's documented tiebreak -- not float noise -- resolves them.
                    d = round_to_grid(lp - top_lp, 0.125)
                    s = round_to_grid(d - BF16_PENALTY_1_2, 0.125) if seen_counts[cid] > 0 else d
                # exact ties broken by lowest global token id (TTSampling greedy tiebreak)
                key = (-s, cid)
                if best is None or key < best[0]:
                    best = (key, cid, tokstr, s)
            picks[rule] = best

        for rule in rules:
            _, cid, _, _ = picks[rule]
            if cid == device_ids[t]:
                agree_device[rule] += 1
            elif first_mismatch_device[rule] is None:
                first_mismatch_device[rule] = {
                    "step": t,
                    "rule_picked_id": cid,
                    "rule_picked": tok.decode([cid]),
                    "device_picked_id": device_ids[t],
                    "device_picked": tok.decode([device_ids[t]]),
                }
            if cid == host_ids[t]:
                agree_host[rule] += 1

        if t == first_div:
            div_step_pick = {r: {"id": picks[r][1], "token": picks[r][2], "score": picks[r][3]} for r in rules}
            # full bf16 arithmetic trace of the two contenders at the divergent step
            lut = {cid: lp for cid, lp, _ in cands}
            h, dv = host_ids[t], device_ids[t]
            dh, dd = round_to_grid(lut[h] - top_lp, 0.125), round_to_grid(lut[dv] - top_lp, 0.125)
            div_step_pick["_bf16_trace"] = {
                "host_token": tok.decode([h]),
                "device_token": tok.decode([dv]),
                "host_token_id": h,
                "device_token_id": dv,
                "host_token_generated_count_so_far": seen_counts[h],
                "device_token_generated_count_so_far": seen_counts[dv],
                "logit_offsets_from_top (exact, on 0.125 grid)": {"host": dh, "device": dd},
                "fp32_reference": {
                    "host_score": dh - P * (seen_counts[h] > 0),
                    "device_score": dd - P * (seen_counts[dv] > 0),
                    "winner": "host_token",
                },
                "bf16_device": {
                    "penalty_as_bf16": BF16_PENALTY_1_2,
                    "host_score": round_to_grid(dh - BF16_PENALTY_1_2, 0.125) if seen_counts[h] else dh,
                    "device_score": round_to_grid(dd - BF16_PENALTY_1_2, 0.125) if seen_counts[dv] else dd,
                },
            }
            b = div_step_pick["_bf16_trace"]["bf16_device"]
            b["exact_tie_after_bf16_rounding"] = b["host_score"] == b["device_score"]
            b["tiebreak_rule"] = "lowest global token id wins (TTSampling._adjust_values_for_tiebreak, k==1)"
            b["tiebreak_winner"] = "device_token" if dv < h else "host_token"

        seen_counts[host_ids[t]] += 1

    out = {
        "_what": __doc__.strip().splitlines()[0],
        "prompt": PROMPT[0]["content"],
        "presence_penalty": P,
        "max_tokens": args.max_tokens,
        "steps_scored": horizon,
        "first_divergence_step": first_div,
        "logit_grid": grid,
        "rule_vs_device_tokens": {
            r: {
                "steps_matching_device": agree_device[r],
                "of": horizon,
                "steps_matching_host": agree_host[r],
                "first_mismatch_vs_device": first_mismatch_device[r],
            }
            for r in rules
        },
        "at_the_divergent_step": div_step_pick,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
