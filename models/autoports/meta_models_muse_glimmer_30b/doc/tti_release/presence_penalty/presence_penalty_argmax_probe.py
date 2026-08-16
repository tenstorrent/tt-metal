#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Is the on-device presence penalty *wrong*, or merely bf16-rounded?

`presence_penalty_greedy_equivalence.json` showed that with `presence_penalty=1.2`
and `temperature=0` the on-device sampler and vLLM's host sampler agree for 725
characters and then diverge.  Two explanations fit that:

  (A) the device applies a *different* penalty (wrong magnitude, wrong token set,
      count instead of presence) -- then the two paths pick tokens separated by a
      large score gap, and the divergence is arithmetic;
  (B) the device applies the *same* penalty in bfloat16 -- then the two paths only
      ever disagree at a near-tie, i.e. a gap on the order of one bf16 ULP.

This probe measures the gap, and independently re-derives vLLM's own decision from
its raw (pre-penalty) logprobs so we know exactly what the reference computes.

Method
------
* Run A: greedy + presence_penalty, `logprobs=true` -> forces vLLM's HOST sampler
  on this 4-die mesh, and returns `token_ids` plus the top-k RAW logprobs (vLLM
  reports `raw_logprobs`, computed before penalties, by design).
* Run B: identical request without `logprobs` -> the DEVICE sampler.
* For every step of run A, rebuild the penalized score of each candidate as
      score(v) = raw_logprob(v) - presence_penalty * [v in tokens generated so far]
  and check that argmax(score) is the token vLLM actually emitted.  That validates
  the reference model of the penalty against vLLM itself.
* At the first step where the device's token differs from the host's, report the
  penalized-score gap between the two tokens, in units of the bf16 ULP at that
  logit magnitude.

Usage::

    python presence_penalty_argmax_probe.py --out <path.json>
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-models/Muse-Glimmer-30B"
PROMPT = [{"role": "user", "content": "Write a very repetitive story."}]
TOKENIZER_DIR = (
    "/home/ttuser/.cache/huggingface/hub/models--meta-models--Muse-Glimmer-30B/"
    "snapshots/f84ecc3a0ea984a4c04542a84269e3d065350a6e"
)


def post(body: dict) -> dict:
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def bf16_ulp(x: float) -> float:
    """Spacing of the bfloat16 grid at |x| (bf16 = fp32 truncated to 8 mantissa bits)."""
    if x == 0:
        return 2.0**-133
    bits = struct.unpack("<I", struct.pack("<f", abs(x)))[0]
    exp = (bits >> 23) & 0xFF
    return 2.0 ** (exp - 127 - 7)


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
    host = post({**base, "logprobs": True, "top_logprobs": args.top_logprobs})["choices"][0]
    device = post(dict(base))["choices"][0]

    host_ids = list(host["token_ids"])
    device_ids = list(device["token_ids"])
    steps = host["logprobs"]["content"]
    assert len(steps) == len(host_ids), (len(steps), len(host_ids))

    # --- reference model check: rebuild vLLM's own decision from raw logprobs ---
    # Candidate ids come from re-encoding the reported token bytes.  Keep only
    # candidates that round-trip to a single token id, so the mapping is exact.
    def cand_id(entry):
        s = bytes(entry["bytes"]).decode("utf-8", errors="replace")
        ids = tok.encode(s, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    seen: set[int] = set()
    recon_ok = 0
    recon_bad = []
    per_step = []
    for t, step in enumerate(steps):
        cands = []
        for e in step["top_logprobs"]:
            cid = cand_id(e)
            score = e["logprob"] - (args.penalty if cid is not None and cid in seen else 0.0)
            cands.append({"id": cid, "tokstr": e["token"], "raw": e["logprob"], "score": score})
        cands.sort(key=lambda c: -c["score"])
        chosen = host_ids[t]
        top = cands[0]
        ok = top["id"] == chosen
        recon_ok += int(ok)
        if not ok:
            recon_bad.append({"step": t, "chosen_id": chosen, "recon_id": top["id"], "recon_tok": top["tokstr"]})
        per_step.append({"cands": cands, "gap_top1_top2": cands[0]["score"] - cands[1]["score"]})
        seen.add(chosen)

    # --- the divergence ---
    n = min(len(host_ids), len(device_ids))
    first = next((i for i in range(n) if host_ids[i] != device_ids[i]), -1)

    div = None
    if first >= 0:
        cands = per_step[first]["cands"]
        by_id = {c["id"]: c for c in cands}
        h, d = host_ids[first], device_ids[first]
        hc, dc = by_id.get(h), by_id.get(d)
        gap = None if (hc is None or dc is None) else hc["score"] - dc["score"]
        ref_mag = max(abs(hc["raw"]) if hc else 0.0, abs(dc["raw"]) if dc else 0.0)
        div = {
            "step": first,
            "host_token_id": h,
            "host_token": tok.decode([h]),
            "device_token_id": d,
            "device_token": tok.decode([d]),
            "host_penalized_score": None if hc is None else hc["score"],
            "device_penalized_score": None if dc is None else dc["score"],
            "host_raw_logprob": None if hc is None else hc["raw"],
            "device_raw_logprob": None if dc is None else dc["raw"],
            "host_token_previously_generated": h in set(host_ids[:first]),
            "device_token_previously_generated": d in set(host_ids[:first]),
            "penalized_score_gap": gap,
            "device_token_in_host_top_k": dc is not None,
            "top_candidates": cands[:5],
            # scale references for judging "near tie"
            "bf16_ulp_at_that_logit": bf16_ulp(ref_mag),
            "gap_in_bf16_ulps": None if gap is None else gap / bf16_ulp(ref_mag),
            "penalty_magnitude": args.penalty,
        }

    # distribution of top1-top2 gaps, for context on how tight this decode is
    gaps = sorted(s["gap_top1_top2"] for s in per_step)
    out = {
        "_what": __doc__.strip().splitlines()[0],
        "prompt": PROMPT[0]["content"],
        "presence_penalty": args.penalty,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "reference_model_check": {
            "_what": "argmax(raw_logprob - penalty*[token already generated]) vs the token vLLM's host sampler emitted",
            "steps": len(steps),
            "steps_reproduced": recon_ok,
            "mismatches": recon_bad[:10],
            "num_mismatches": len(recon_bad),
        },
        "divergence": div,
        "host_len": len(host_ids),
        "device_len": len(device_ids),
        "gap_top1_top2_quantiles": {
            "min": gaps[0],
            "p10": gaps[len(gaps) // 10],
            "median": gaps[len(gaps) // 2],
        }
        if gaps
        else None,
        "host_text": host["message"]["content"],
        "device_text": device["message"]["content"],
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("host_text", "device_text")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
