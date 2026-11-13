# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B of the unconfirmed conventions (oracle.Conv): one layer-streaming pass of the whole
model over a natural-text prompt, N variant streams sharing each layer's weights. The variant with
the clearly lowest next-token cross-entropy is the right convention (a wrong convention in any of
~77 layers wrecks the language model). Usage: python -m ...variant_ab OUT.json [--layers N]"""
import json
import sys
import time
from dataclasses import replace

import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi, mtp_forward

TEXT = (
    "The Great Barrier Reef is the world's largest coral reef system, composed of over 2,900 individual reefs "
    "and 900 islands stretching for over 2,300 kilometres off the coast of Queensland, Australia. "
    "It can be seen from outer space and is the world's biggest single structure made by living organisms. "
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"
    "In mathematics, a prime number is a natural number greater than 1 that is not a product of two smaller natural numbers."
)

VARIANTS = {
    "base": {},
    "rope_neox": dict(rope_interleave=False),
    "gu_interleave": dict(gate_up="interleave"),
    "gu_swap": dict(gate_up="swap"),
    "hc_eps_hc": dict(hc_norm_eps="hc"),
    "no_sink": dict(sink_in_denominator=False),
    "gate_in_raw": dict(attn_gate_in="raw"),
    "no_clamp": dict(swiglu_clamp=False),
    "no_attn_gate": dict(attn_gate_act="none"),
    "no_hc_pre_eps": dict(hc_pre_eps=False),
}


def main():
    out = sys.argv[1]
    nl = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 78
    torch.set_num_threads(32)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    ids = torch.tensor([[120000] + tok.encode(TEXT).ids])
    o = Hy4Oracle()
    names = list(VARIANTS)
    convs = [replace(Conv(), **VARIANTS[n]) for n in names]
    x4s = [o.embed_ids(ids) for _ in names]
    sts = [{} for _ in names]
    t = time.time()
    for L in range(nl):
        x4s = decoder_layers_multi(o.ck, L, x4s, convs, sts)
        print(
            f"layer {L} done {time.time()-t:.0f}s norms {[round(x.norm(dim=-1).mean().item(),2) for x in x4s[:3]]}",
            flush=True,
        )
    res = {"n_tokens": ids.shape[1], "layers": nl, "variants": {}}
    hs = {}
    for n, c, x4 in zip(names, convs, x4s):
        h, hn = o.head(x4, c)
        lg = o.logits(hn)
        ce = F.cross_entropy(lg[0, :-1], ids[0, 1:]).item()
        acc = (lg[0, :-1].argmax(-1) == ids[0, 1:]).float().mean().item()
        res["variants"][n] = {"ce": ce, "top1": acc, "last_top5": lg[0, -1].topk(5)[1].tolist()}
        hs[n] = (h, hn, lg)
        print(n, res["variants"][n], flush=True)
    if nl == 78:
        h, hn, lg = hs["base"]
        for hsrc, cat in (("hc_head", "e_h"), ("normed", "e_h"), ("hc_head", "h_e"), ("normed", "h_e")):
            c = replace(Conv(), mtp_h_source=hsrc, mtp_cat=cat)
            hh = (h if hsrc == "hc_head" else hn)[:, :-1]
            m = mtp_forward(o.ck, hh, ids[:, 1:], c, embed=o.embed)
            ml = o.logits(m)  # predicts ids[t+2] at position t
            tgt = ids[0, 2:]
            ce = F.cross_entropy(ml[0, :-1], tgt).item()
            agree = (ml[0, :-1].argmax(-1) == lg[0, 1:-1].argmax(-1)).float().mean().item()
            top1 = (ml[0, :-1].argmax(-1) == tgt).float().mean().item()
            res.setdefault("mtp", {})[f"{hsrc}/{cat}"] = {
                "ce": ce,
                "top1_vs_truth": top1,
                "agree_with_main_next": agree,
            }
            print("mtp", hsrc, cat, res["mtp"][f"{hsrc}/{cat}"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
