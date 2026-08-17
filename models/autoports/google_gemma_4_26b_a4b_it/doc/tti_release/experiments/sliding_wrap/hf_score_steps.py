# SPDX-License-Identifier: Apache-2.0
"""Score every TT decode step against the HF BF16 reference on CPU.

For each token the TT device produced, this asks the reference model the only
question that matters for a greedy trajectory: given the identical prefix, would
HF have chosen the same token, and if not, by how much did it lose?

Emits per-step arrays plus the summary the bringup pipeline never collected:
flip rate versus decode index, the HF top1/top2 margin distribution at flips,
and whether flips cluster at sliding-window boundaries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

HF_MODEL = "google/gemma-4-26B-A4B-it"


def _percentiles(values: list[float], points=(1, 5, 25, 50, 75, 95, 99)) -> dict[str, float]:
    if not values:
        return {}
    tensor = torch.tensor(values, dtype=torch.float64).sort().values
    out = {}
    for p in points:
        idx = min(len(values) - 1, max(0, int(round((p / 100.0) * (len(values) - 1)))))
        out[f"p{p}"] = float(tensor[idx])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-run", type=Path, required=True, help="JSON written by tt_longgen.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=1024, help="Prefill chunk size for the reference forward.")
    ap.add_argument("--limit", type=int, default=0, help="Score only the first N generated tokens (0 = all).")
    args = ap.parse_args()

    run = json.loads(args.tt_run.read_text(encoding="utf-8"))
    prompt_ids: list[int] = run["prompt_token_ids"]
    gen_ids: list[int] = run["generated_token_ids"]
    if args.limit:
        gen_ids = gen_ids[: args.limit]
    seq = prompt_ids + gen_ids
    prompt_len = len(prompt_ids)
    print(f"prompt {prompt_len} + generated {len(gen_ids)} = {len(seq)} tokens", flush=True)

    print("loading HF reference (bf16, cpu)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL, dtype=torch.bfloat16, trust_remote_code=True).eval()

    # Positions prompt_len-1 .. len(seq)-2 predict gen_ids[0..]. Run the sequence
    # in chunks with a cache so peak logits memory stays bounded.
    input_ids = torch.tensor([seq], dtype=torch.long)
    records = []
    past = None
    start = 0
    with torch.no_grad():
        while start < len(seq) - 1:
            end = min(start + args.chunk, len(seq))
            chunk = input_ids[:, start:end]
            out = model(input_ids=chunk, past_key_values=past, use_cache=True)
            past = out.past_key_values
            logits = out.logits[0].float()  # [chunk, vocab]
            for local, absolute in enumerate(range(start, end)):
                gen_idx = absolute - (prompt_len - 1)
                if gen_idx < 0 or gen_idx >= len(gen_ids):
                    continue
                row = logits[local]
                top2 = torch.topk(row, k=2)
                hf_top1 = int(top2.indices[0])
                tt_token = gen_ids[gen_idx]
                tt_logit = float(row[tt_token])
                # Rank is only meaningful when it is small; cap the search cost.
                rank = int((row > row[tt_token]).sum()) + 1
                records.append(
                    {
                        "i": gen_idx,
                        "tt": tt_token,
                        "hf": hf_top1,
                        "match": hf_top1 == tt_token,
                        "top1_top2_margin": float(top2.values[0] - top2.values[1]),
                        "top1_minus_tt": float(top2.values[0]) - tt_logit,
                        "tt_rank": rank,
                    }
                )
            del out, logits
            start = end
            print(f"  scored through generated index {records[-1]['i'] if records else -1}", flush=True)

    flips = [r for r in records if not r["match"]]
    matched = [r for r in records if r["match"]]
    bucket = 256
    buckets = {}
    for r in records:
        key = (r["i"] // bucket) * bucket
        entry = buckets.setdefault(f"{key}-{key + bucket - 1}", {"n": 0, "flips": 0})
        entry["n"] += 1
        entry["flips"] += 0 if r["match"] else 1
    for entry in buckets.values():
        entry["flip_rate"] = entry["flips"] / entry["n"]

    near_wrap = [r for r in flips if min(r["i"] % 1024, 1024 - (r["i"] % 1024)) <= 4]
    summary = {
        "tt_run": str(args.tt_run),
        "scored_steps": len(records),
        "flips": len(flips),
        "flip_rate": len(flips) / max(len(records), 1),
        "first_flip_index": flips[0]["i"] if flips else None,
        "flip_indices": [r["i"] for r in flips][:200],
        "tt_rank_histogram": {str(k): sum(1 for r in flips if r["tt_rank"] == k) for k in (2, 3, 4, 5)},
        "tt_rank_gt5": sum(1 for r in flips if r["tt_rank"] > 5),
        "flip_rate_by_bucket": buckets,
        "margin_at_flips": _percentiles([r["top1_top2_margin"] for r in flips]),
        "margin_at_matches": _percentiles([r["top1_top2_margin"] for r in matched]),
        "top1_minus_tt_at_flips": _percentiles([r["top1_minus_tt"] for r in flips]),
        "flips_within_4_of_1024_boundary": len(near_wrap),
        "expected_flips_near_boundary_if_uniform": len(flips) * (9.0 / 1024.0),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "steps": records}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
