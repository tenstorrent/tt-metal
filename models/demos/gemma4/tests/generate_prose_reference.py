# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Writes the fp32 HF reference of ``tests/test_prose_accuracy.py``: the first ``length`` tokens of A Tale of Two Cities
(from "It was the best of times", the checkpoint's tokenizer) and the HF model's fp32 top-5 next-token ids per position,
plus the bars (recorded TT agreement minus a margin). Either loads a saved fp32 logits dump ({"hf": [1, S, V] or
[S, V], "input_ids"}) or runs the HF model in fp32 on the host (~48 GB RAM for the 12B).

    python generate_prose_reference.py --model google/gemma-4-12B-it --length 2048 \\
        --logits _logs/g6_logit_dumps/gemma-4-12B-it_natural_hf_fp32.pt --top1-all 0.xx --top5-all 0.xx --top1-last64 0.xx
"""

import argparse
import bz2
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
TALE = HERE.parents[2] / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"


def prose_ids(tokenizer, length):
    text = bz2.open(TALE).read().decode("utf-8", "replace")
    text = text[text.find("It was the best of times") :]
    ids = tokenizer.encode(text[: length * 8], return_tensors="pt")[:, :length]
    assert ids.shape[1] == length, f"only {ids.shape[1]} prose tokens for {length}"
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--logits", help="saved fp32 logits dump; without it the HF model runs in fp32 on the host")
    ap.add_argument("--top1-all", type=float, required=True, help="recorded TT top-1 agreement over all rows")
    ap.add_argument("--top5-all", type=float, required=True, help="recorded fp32 top-1 in TT top-5 over all rows")
    ap.add_argument(
        "--top1-last64", type=float, required=True, help="recorded TT top-1 agreement over the last 64 rows"
    )
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--out", help="default tests/reference_outputs/prose_fp32_<tag>_<length>.refpt")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = prose_ids(tokenizer, args.length)
    if args.logits:
        d = torch.load(args.logits, map_location="cpu")
        logits = d["hf"].reshape(-1, d["hf"].shape[-1])[: args.length].float()
        saved = d["input_ids"].reshape(-1)[: args.length]
        assert torch.equal(saved, ids[0]), "the saved dump's prompt differs from the tokenizer's prose"
    else:
        from transformers import AutoModelForCausalLM

        hf = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, trust_remote_code=True)
        hf.eval()
        with torch.no_grad():
            logits = hf(ids).logits[0].float()
    top5 = logits.topk(5, dim=-1).indices.to(torch.int32)
    tag = args.model.split("/")[-1]
    out = Path(args.out) if args.out else HERE / "reference_outputs" / f"prose_fp32_{tag}_{args.length}.refpt"
    torch.save(
        {
            "model": tag,
            "length": args.length,
            "source": "tale-of-two-cities from 'It was the best of times'",
            "input_ids": ids[0].to(torch.int32),
            "top5": top5,
            "reference": "HF fp32 on the host",
            "recorded": {"top1_all": args.top1_all, "top5_all": args.top5_all, "top1_last64": args.top1_last64},
            "bars": {
                "top1_all": round(args.top1_all - args.margin, 4),
                "top5_all": round(args.top5_all - args.margin, 4),
                "top1_last64": round(args.top1_last64 - 2 * args.margin, 4),
            },
        },
        out,
    )
    print(f"wrote {out}: {args.length} tokens, top-5 ids, bars {torch.load(out)['bars']}")


if __name__ == "__main__":
    main()
