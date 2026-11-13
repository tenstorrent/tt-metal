# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Confirms the DSA indexer conventions on real weights without the full model: run layers 0..5 on a
~2600-token natural text (> index_topk=2048, so selection is active) and, at full-indexer layers 1 and 5,
measure how much of the dense (post-sink) attention probability mass the indexer's top-2048 captures for
queries past position 2048. A correct indexer captures far more than recency/random selection.
    python -m ...indexer_check OUT.json"""
import json
import sys
from dataclasses import replace

import torch

from models.demos.blackhole.hy4_preview.reference import oracle as O

VARS = {
    "default(rope_first,halfsplit)": {},
    "rope_last": dict(idx_rope_first=False),
    "rope_first_interleave": dict(idx_rope_interleave=True),
    "rope_last_interleave": dict(idx_rope_first=False, idx_rope_interleave=True),
}


def main():
    torch.set_num_threads(32)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    txt = open("/mnt/tt-data/Hy4-Preview/README.md").read() + open("/mnt/tt-data/Hy4-Preview/finetune/README.md").read()
    ids = torch.tensor([([120000] + tok.encode(txt).ids)[:2700]])
    S = ids.shape[1]
    o = O.Hy4Oracle()
    cfg, conv = o.cfg, O.Conv()
    qs = list(range(2100, S, 40))
    x4 = o.embed_ids(ids)
    res = {"S": S, "layers": {}}
    st = {}
    for L in range(6):
        sd = o.ck.layer_state(L)
        if cfg.is_full_indexer(L):
            pre, post = O.hc_pre_post(x4, O.hc_params(sd, "attn"), cfg, conv)
            h = O.rms_norm(O.hc_collapse(x4, pre), sd["input_layernorm.weight"], cfg.norm_eps)
            qa = O.rms_norm(
                torch.nn.functional.linear(h, sd["self_attn.q_a_proj.weight"]),
                sd["self_attn.q_a_layernorm.weight"],
                cfg.norm_eps,
            )
            cs = O.rope_cos_sin(S, cfg.rope, cfg.rope_theta)
            idxs = {n: O.indexer_topk(h, qa, sd, cfg, replace(conv, **kw), *cs) for n, kw in VARS.items()}
            masses = {n: [] for n in list(VARS) + ["recency", "random"]}
            g = torch.Generator().manual_seed(0)

            def probe(b, s0, s1, p):
                for q in qs:
                    if s0 <= q < s1:
                        pr = p[:, q - s0]  # [H,S]
                        for n, ix in idxs.items():
                            sel = ix[0, q]
                            masses[n].append(pr[:, sel[sel >= 0]].sum(-1).mean().item())
                        masses["recency"].append(pr[:, q - 2047 : q + 1].sum(-1).mean().item())
                        rnd = torch.randperm(q + 1, generator=g)[:2048]
                        masses["random"].append(pr[:, rnd].sum(-1).mean().item())

            O.attention(h, sd, cfg, conv, None, probe=probe)
            res["layers"][L] = {n: sum(v) / len(v) for n, v in masses.items()}
            print(L, res["layers"][L], flush=True)
            st["idx"] = idxs["default(rope_first,halfsplit)"]
        x4 = O.decoder_layers_multi(o.ck, L, [x4], [conv], [st])[0]
    json.dump(res, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
