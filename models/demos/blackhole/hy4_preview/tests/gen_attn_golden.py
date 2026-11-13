# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Golden dumps of the attention sublayer of layers 0, 1, 2 from the fp32 CPU oracle (task layer-attn). CPU only.

Run with the ORACLE worktree first on sys.path (the oracle package lives on branch ssinghal/hy4-preview-ref):

    PYTHONPATH=/mnt/tt-data/ssinghal/wt/hy4_ref HY4_THREADS=16 \
      python gen_attn_golden.py OUT_DIR [--tokens 3072] [--layers 3]

One long natural-text prompt (the oracle dump_ref long prompt, truncated to --tokens, BOS first). Per layer L the oracle's iHC stream is
advanced through the whole layer (attention + MLP), and for the attention sublayer the script saves (fp32, [S, ...], batch squeezed):
    L{L}.pt = {h (post input_layernorm, the attention input), qa (post q_a_layernorm), lat (post kv_a_layernorm, 512), k_pe (post rope, 64),
               q (post rope, [S,64,256]), o (attention output before the gate, [S,16384]), gate (post sigmoid, [S,16384]), out (o_proj output [S,6144]),
               idx (indexer top-k ids of the layer's index source, [S,2048] int64, -1 = unused), idx_source (layer index), conv (dict), tokens}
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from models.demos.blackhole.hy4_preview.reference import oracle as O
from models.demos.blackhole.hy4_preview.reference.dump_ref import long_prompt
from models.demos.blackhole.hy4_preview.reference.oracle import Conv, Hy4Oracle

VARIANTS = {
    1: "Chapter {i}. The engineering team measured a latency of {a} ms across {b} nodes, then rewrote the scheduler loop and agreed to revisit the cache policy after the next release. ",
    2: "def handler_{i}(x):\n    # scale the input by {a} and shift by {b}\n    return x * {a} + {b}\n\n",
    3: "In the year {i}, the village of Ashford harvested {a} bushels of wheat and {b} of barley, which the elders recorded in the ledger before the first frost arrived. ",
}


def variant_text(v):
    if v == 0:
        return long_prompt(120)
    return "".join(VARIANTS[v].format(i=i, a=(i * 7) % 23 + 1, b=(i * 5) % 17 + 1) for i in range(1, 200))


def attn_taps(ck, layer, x4, st, conv):
    cfg = ck.cfg
    sd = ck.layer_state(layer)
    hp = O.hc_params(sd, "attn")
    pre, post = O.hc_pre_post(x4, hp, cfg, conv)
    raw = O.hc_collapse(x4, pre)
    h = O.rms_norm(raw, sd["input_layernorm.weight"], cfg.norm_eps)
    B, S, _ = h.shape
    qa = O.rms_norm(F.linear(h, sd["self_attn.q_a_proj.weight"]), sd["self_attn.q_a_layernorm.weight"], cfg.norm_eps)
    if cfg.is_full_indexer(layer):
        cos_i, sin_i = O.rope_cos_sin(S, cfg.rope, cfg.rope_theta)
        st["idx"] = O.indexer_topk(h, qa, sd, cfg, conv, cos_i, sin_i)
        st["idx_source"] = layer
    idx = st["idx"] if S > cfg.idx_topk else None
    H, nope, rope = cfg.n_heads, cfg.nope, cfg.rope
    q = F.linear(qa, sd["self_attn.q_b_proj.weight"]).view(B, S, H, nope + rope)
    kv = F.linear(h, sd["self_attn.kv_a_proj_with_mqa.weight"])
    lat = O.rms_norm(kv[..., : cfg.kv_lora], sd["self_attn.kv_a_layernorm.weight"], cfg.norm_eps)
    cos, sin = O.rope_cos_sin(S, rope, cfg.rope_theta)
    q_pe = O.apply_rope(q[..., nope:], cos, sin, conv.rope_interleave)
    k_pe = O.apply_rope(kv[..., cfg.kv_lora :], cos, sin, conv.rope_interleave)
    qf = torch.cat([q[..., :nope], q_pe], -1)
    o, _ = O.attention(h, sd, cfg, conv, idx)
    gate = torch.sigmoid(F.linear(h, sd["self_attn.linear_gate.weight"]))
    out = F.linear(o * gate, sd["self_attn.o_proj.weight"])
    chk = O.attn_block(h, raw, sd, cfg, conv, idx)
    assert torch.allclose(chk, out, atol=1e-4, rtol=1e-4), (chk - out).abs().max()
    return {
        "h": h[0],
        "qa": qa[0],
        "lat": lat[0],
        "k_pe": k_pe[0],
        "q": qf[0],
        "o": o[0],
        "gate": gate[0],
        "out": out[0],
        "idx": st["idx"][0] if S > cfg.idx_topk else None,
        "idx_source": st["idx_source"],
        "sink": sd["self_attn.learnable_sink_param"].float(),
    }


def main():
    out_dir = sys.argv[1]
    n_tok = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 3072
    n_layers = int(sys.argv[sys.argv.index("--layers") + 1]) if "--layers" in sys.argv else 3
    torch.set_num_threads(int(os.environ.get("HY4_THREADS", "16")))
    os.makedirs(out_dir, exist_ok=True)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file("/mnt/tt-data/Hy4-Preview/tokenizer.json")
    variant = int(sys.argv[sys.argv.index("--variant") + 1]) if "--variant" in sys.argv else 0
    ids = [120000] + tok.encode(variant_text(variant)).ids
    assert len(ids) >= n_tok, len(ids)
    ids = torch.tensor([ids[:n_tok]])
    conv = Conv(**json.loads(os.environ.get("HY4_CONV", "{}")))
    o = Hy4Oracle()
    x4 = o.embed_ids(ids)
    st = {}
    json.dump({"tokens": ids[0].tolist(), "conv": conv.__dict__}, open(os.path.join(out_dir, "meta.json"), "w"))
    t0 = time.time()
    for L in range(n_layers):
        taps = attn_taps(o.ck, L, x4, st, conv)
        torch.save({**taps, "conv": conv.__dict__}, os.path.join(out_dir, f"L{L}.pt"))
        print(f"layer {L} attention taps saved {time.time()-t0:.0f}s", flush=True)
        x4 = O.decoder_layers_multi(o.ck, L, [x4], [conv], [st])[0]
        print(f"layer {L} advanced {time.time()-t0:.0f}s norm {x4.norm(dim=-1).mean().item():.3f}", flush=True)


if __name__ == "__main__":
    main()
