# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Oracle MTP golden for the device block test: per prompt (ids [N] incl. BOS, post-norm main hidden H [N,D]) the step-0 MTP block over rows t = 0..N-2
(token t+1, hidden H[t], position t) with intermediates. Usage: python -m ...spec.mtp_golden OUT.pt DUMP_DIR [prompt indices] [--long]"""
import sys

import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.reference.dump_ref import SHORT, long_prompt
from models.demos.blackhole.hy4_preview.reference.oracle import (
    Conv,
    Hy4Oracle,
    attn_block,
    indexer_topk,
    moe_multi,
    rms_norm,
    rope_cos_sin,
)


def mtp_taps(o, ids, H, conv=None):
    conv = conv or Conv()
    ck, cfg = o.ck, o.ck.cfg
    sd = ck.layer_state(-1)
    tok = ids[None, 1:]
    h = H[None, :-1]
    e = rms_norm(o.embed[tok].float(), sd["enorm.weight"], cfg.norm_eps)
    hh = rms_norm(h, sd["hnorm.weight"], cfg.norm_eps)
    x0 = F.linear(torch.cat([e, hh], -1), sd["eh_proj.weight"])
    S = x0.shape[1]
    hn = rms_norm(x0, sd["input_layernorm.weight"], cfg.norm_eps)
    qa = rms_norm(F.linear(hn, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
    cos_i, sin_i = rope_cos_sin(S, cfg.rope, cfg.rope_theta)
    idx = indexer_topk(hn, qa, sd, cfg, conv, cos_i, sin_i) if S > cfg.idx_topk else None
    a = attn_block(hn, hn, sd, cfg, conv, idx)
    x1 = x0 + a
    hm = rms_norm(x1, sd["post_attention_layernorm.weight"], cfg.norm_eps)
    y, routes = moe_multi(ck, -1, [hm], [sd], cfg, [conv])
    x2 = x1 + y[0]
    out = rms_norm(x2, sd["final_layernorm.weight"], cfg.norm_eps)
    d1 = o.logits(out)[0].argmax(-1)
    main = o.logits(H).argmax(-1)  # main[r] predicts token r+1
    return dict(
        ids=ids,
        H=H,
        x0=x0[0],
        a=a[0],
        x1=x1[0],
        y=y[0][0],
        x2=x2[0],
        out=out[0],
        d1=d1,
        main=main,
        routes=routes[0][0],
        idx=None if idx is None else idx[0],
    )


def main():
    out = sys.argv[1]
    dump = sys.argv[2]
    sel = [int(a) for a in sys.argv[3:] if a.isdigit()]
    from tokenizers import Tokenizer

    torch.set_num_threads(24)
    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    texts = SHORT + [long_prompt()]
    o = Hy4Oracle()
    res = {}
    for p in sel or range(len(texts)):
        ids = torch.tensor([120000] + tok.encode(texts[p]).ids)
        H = torch.load(f"{dump}/final_hidden_p{p}.pt")["final_normed"][0]
        assert H.shape[0] == ids.shape[0]
        res[p] = mtp_taps(o, ids, H)
        print("prompt", p, ids.shape[0], flush=True)
        torch.save(res, out)


if __name__ == "__main__":
    main()
