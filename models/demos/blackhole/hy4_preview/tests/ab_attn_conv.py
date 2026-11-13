# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end A/B of the ATTENTION conventions the ref-oracle did not cover (task layer-attn), same method as reference/variant_ab.py (cross-entropy of a natural-text prompt through all 78 layers):
sink domain (post-scale logit | x scale | / scale) and attention scale (256^-0.5 | 192^-0.5). Monkeypatches ``oracle.attention`` (the oracle files are not edited).
    PYTHONPATH=<oracle worktree> python ab_attn_conv.py OUT.json"""
import json
import sys
import time
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.reference import oracle as O
from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle, decoder_layers_multi
from models.demos.blackhole.hy4_preview.reference.variant_ab import TEXT


@dataclass(frozen=True)
class Conv2(Conv):
    sink_mode: str = "logit"  # logit | x_scale | div_scale
    scale_dim: int = 256


def attention2(h, sd, cfg, conv, topk_idx, probe=None):
    B, S, D = h.shape
    H, nope, rope, vd = cfg.n_heads, cfg.nope, cfg.rope, cfg.v_dim
    qk = nope + rope
    qa = O.rms_norm(F.linear(h, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
    q = F.linear(qa, sd["self_attn.q_b_proj.weight"]).view(B, S, H, qk)
    kv = F.linear(h, sd["self_attn.kv_a_proj_with_mqa.weight"])
    lat, k_pe = kv[..., : cfg.kv_lora], kv[..., cfg.kv_lora :]
    lat = O.rms_norm(lat, sd["self_attn.kv_a_layernorm.weight"], cfg.norm_eps)
    kvb = F.linear(lat, sd["self_attn.kv_b_proj.weight"]).view(B, S, H, nope + vd)
    k_nope, v = kvb[..., :nope], kvb[..., nope:]
    cos, sin = O.rope_cos_sin(S, rope, cfg.rope_theta)
    q_pe = O.apply_rope(q[..., nope:], cos, sin, conv.rope_interleave)
    k_pe = O.apply_rope(k_pe, cos, sin, conv.rope_interleave)
    qf = torch.cat([q[..., :nope], q_pe], -1)
    kf = torch.cat([k_nope, k_pe.unsqueeze(2).expand(B, S, H, rope)], -1)
    scale = conv.scale_dim**-0.5
    sink = sd["self_attn.learnable_sink_param"].float()
    if conv.sink_mode == "x_scale":
        sink = sink * scale
    elif conv.sink_mode == "div_scale":
        sink = sink / scale
    o = torch.empty(B, S, H, vd)
    causal = torch.ones(S, S, dtype=torch.bool).tril()
    for b in range(B):
        sc = torch.einsum("shd,thd->hst", qf[b], kf[b]) * scale
        allowed = causal
        if topk_idx is not None:
            sel = torch.zeros(S, S + 1, dtype=torch.bool)
            ix = topk_idx[b]
            sel.scatter_(1, torch.where(ix >= 0, ix, torch.full_like(ix, S)), True)
            allowed = allowed & sel[:, :S]
        sc = sc.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        full = torch.cat([sc, sink.view(H, 1, 1).expand(H, S, 1)], -1)
        p = full.softmax(-1)[..., :S]
        o[b] = torch.einsum("hst,thd->shd", p, v[b])
    return o.reshape(B, S, H * vd), h


O.attention = attention2


def main():
    out = sys.argv[1]
    torch.set_num_threads(24)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    ids = torch.tensor([[120000] + tok.encode(TEXT).ids])
    o = Hy4Oracle()
    variants = {
        "base": {},
        "sink_x_scale": dict(sink_mode="x_scale"),
        "sink_div_scale": dict(sink_mode="div_scale"),
        "scale_192": dict(scale_dim=192),
    }
    names = list(variants)
    convs = [replace(Conv2(), **variants[n]) for n in names]
    x4s = [o.embed_ids(ids) for _ in names]
    sts = [{} for _ in names]
    t = time.time()
    for L in range(78):
        x4s = decoder_layers_multi(o.ck, L, x4s, convs, sts)
        print(
            f"layer {L} done {time.time()-t:.0f}s norms {[round(x.norm(dim=-1).mean().item(),2) for x in x4s]}",
            flush=True,
        )
    res = {}
    for n, c, x4 in zip(names, convs, x4s):
        h, hn = o.head(x4, c)
        lg = o.logits(hn)
        res[n] = {
            "ce": F.cross_entropy(lg[0, :-1], ids[0, 1:]).item(),
            "top1": (lg[0, :-1].argmax(-1) == ids[0, 1:]).float().mean().item(),
        }
        print(n, res[n], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
