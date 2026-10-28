# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""L0 CPU study: compressed-KV rows stored as format A (reference fp4 e2m1 + e4m3 scale per 16) vs format B
(bfloat4_b, one exponent per 16) vs bf16 (today's device), against the reference's NATIVE rows (golden s10240,
numerics native). Attention impact on the real attention queries of the dumped layers (chunk 1, positions 5120+).
No ttnn import (it opens the cluster). Run:
  PYTHONPATH=~/repos/tt-metal python -I /scratch/dnijemcevic/ckv_derisk/study.py"""

import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/repos/tt-metal"))
import codec  # noqa: E402

from models.demos.deepseek_v41_flash_d_p.reference import kernels as K  # noqa: E402
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config, rope, rope_table, yarn_freqs  # noqa: E402
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint  # noqa: E402

torch.set_num_threads(32)
HF = "/scratch/dnijemcevic/bringup/deepseek_v41_flash_d_p/hf_full"
G = "/data/dnijemcevic/bringup/deepseek_v41_flash_d_p/golden_full/galaxy_8x4/s10240_c5120"
C1 = 5120  # chunk 1 start
NQ = 256

cfg = Config.from_dir(HF)
ck = Checkpoint(HF)
r_ = cfg.rope
freqs = yarn_freqs(cfg.rope_dim, r_["compress_theta"], r_["original"], r_["factor"], r_["beta_fast"], r_["beta_slow"])


def kvc(layer, name):
    with safe_open(f"{G}/kv_cache/layer_{layer}.safetensors", "pt") as h:
        return h.get_tensor(f"{name}_cache_layer_{layer}").float()


def chunk1(layer, name):
    with safe_open(f"{G}/chunk_01/layer_{layer:02d}.safetensors", "pt") as h:
        return h.get_tensor(name)


def rel(a, b):
    return float((a - b).norm() / b.norm())


FORMATS = {
    "bf16": lambda x: x.bfloat16().float(),
    "A": lambda x: codec.decode_a(codec.encode_a(x)),
    "B": lambda x: codec.decode_b(codec.encode_b(x)),
    "B_trunc": lambda x: codec.decode_b(codec.encode_b(x, truncate=True)),
    "B_from_bf16": lambda x: codec.decode_b(codec.encode_b(x.bfloat16().float())),
}

out = {"rows": {}, "attn": {}}
rows = {}
for src in (2, 8, 14, 20):
    x = kvc(src, "compress_kv")
    assert torch.equal(K.fp4_quant_dequant(x, 16, e4m3_scale=True), codec.decode_a(codec.encode_a(x)))
    q = {f: fn(x) for f, fn in FORMATS.items()}
    rows[src] = {"native": x, **q}
    nope, rp = slice(0, 448), slice(448, 512)
    d = {f: rel(v, x) for f, v in q.items()}
    d["A_vs_B"] = rel(q["B"], q["A"])
    d["A_nope"], d["A_rope"] = rel(q["A"][:, nope], x[:, nope]), rel(q["A"][:, rp], x[:, rp])
    d["B_nope"], d["B_rope"] = rel(q["B"][:, nope], x[:, nope]), rel(q["B"][:, rp], x[:, rp])
    # per-row worst (1 - cosine)
    for f in ("A", "B"):
        cos = torch.nn.functional.cosine_similarity(q[f], x, dim=-1)
        d[f + "_row_cos_min"] = float(cos.min())
    # mean value bias (norm ratio) and B's saturation rate (group max element clamped at 7)
    d["A_norm_ratio"] = float(q["A"].norm() / x.norm())
    d["B_norm_ratio"] = float(q["B"].norm() / x.norm())
    out["rows"][src] = {"n": x.shape[0], **d}
    print("rows", src, json.dumps({k: round(v, 5) for k, v in d.items()}), flush=True)

g = torch.Generator().manual_seed(1)
for layer in (2, 3, 14, 20, 21, 24):
    src, isrc = cfg.kv_source_of(layer), cfg.index_source_of(layer)
    r = cfg.ratio(layer)
    pos = torch.sort(torch.randperm(C1 - 128, generator=g)[:NQ] + C1 + 127).values  # window inside chunk 1
    qr = chunk1(layer, "q_resid")[pos - C1]
    kv = chunk1(layer, "kv")  # window keys, native
    wq = ck.weight(f"layers.{layer}.attn.wq_b.weight", torch.float32)
    sink = ck.weight(f"layers.{layer}.attn.attn_sink", torch.float32)
    rot = rope_table(freqs, pos)
    qh = rope((qr @ wq.t()).view(NQ, cfg.n_heads, cfg.head_dim), rot)
    topk = kvc(isrc, "topk_idx")[pos].long()[:, : cfg.index_topk]
    valid_c = topk >= 0
    wpos = pos.unsqueeze(-1) - torch.arange(127, -1, -1)  # [NQ, 128]
    win = kv[wpos - C1]
    valid = torch.cat([torch.ones(NQ, 128, dtype=torch.bool), valid_c], dim=1)
    scale = cfg.head_dim**-0.5
    res, ref_o, ref_s = {}, None, None
    for f in ["native", *FORMATS]:
        ck_rows = rows[src][f][topk.clamp_min(0)]  # [NQ, 512, 512]
        keys = torch.cat([win, ck_rows], dim=1)
        o = K.attend(qh, keys, valid, sink, scale)
        s = torch.einsum("bhd,btd->bht", qh, ck_rows) * scale  # compressed-key logits
        s = s.masked_fill(~valid_c[:, None, :], 0.0)
        # attention mass on the compressed keys (as attend: shared max over window + compressed + sink)
        if f == "native":
            ref_o, ref_s = o, s
            full = torch.einsum("bhd,btd->bht", qh, keys) * scale
            full = full.masked_fill(~valid[:, None, :], float("-inf"))
            pr = torch.softmax(torch.cat([full, sink[None, :, None].expand(NQ, -1, 1)], dim=-1), dim=-1)
            res["mass_compressed"] = float(pr[..., 128:-1].sum(-1).mean())
            ref_top = s.masked_fill(~valid_c[:, None, :], float("-inf")).topk(32, dim=-1).indices
            continue
        top = s.masked_fill(~valid_c[:, None, :], float("-inf")).topk(32, dim=-1).indices
        ov = (ref_top.unsqueeze(-1) == top.unsqueeze(-2)).any(-1).float().mean()
        res[f] = {
            "o_rel": rel(o, ref_o),
            "logit_rel": rel(s, ref_s),
            "logit_maxabs": float((s - ref_s).abs().max()),
            "top32_overlap": float(ov),
        }
    out["attn"][layer] = {"src": src, "idx_src": isrc, "ratio": r, **res}
    print("attn", layer, json.dumps(out["attn"][layer]), flush=True)

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "study_results.json"), "w"), indent=1)
