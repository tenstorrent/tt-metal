# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU study: the window KV, the index keys and the index query at the reference's width, in the device's formats,
against the reference's NATIVE values (2x2 scaled golden s4096, chunk 1 = positions 2048..4095).

- window KV (reference: e4m3 with a power-of-two scale per 32): rows rel L2 and the attention output over the window
  (+ native compressed rows) for bf16 (today), the reference's own rounding, bfloat8_b (one exponent per 16) rounded to
  nearest even or half away (the packer with bfp8_pack_precise).
- index keys / query (reference: e2m1 with a power-of-two scale per 32): the index scores and the top-512 overlap with
  the native selection for bf16 (today), the reference's fp4, bfloat4_b (RNE / truncated as the packer alone) and
  bfloat8_b; keys and query rounded together and one at a time.
No ttnn import. Run: python -m models.demos.deepseek_v41_flash_d_p.tools.lowbit_study"""

import json
import os

import torch
from safetensors import safe_open

from models.demos.deepseek_v41_flash_d_p.reference import kernels as K
from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import (
    Config,
    candidate_mask,
    lowest_index_topk,
    rope,
    rope_table,
    yarn_freqs,
)
from models.demos.deepseek_v41_flash_d_p.reference.weights import Checkpoint

torch.set_num_threads(int(os.environ.get("STUDY_THREADS", "16")))
ROOT = "/localdev/dnijemcevic/bringup/deepseek_v41_flash_d_p"
HF = f"{ROOT}/hf_scaled/quietbox_2x2"
G = f"{ROOT}/golden_scaled/quietbox_2x2/s4096_c2048"
C1, S = 2048, 2048  # chunk 1
NQ = 256
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bringup", "lowbit_derisk", "study_results.json")

cfg = Config.from_dir(HF)
ck = Checkpoint(HF)
r_ = cfg.rope
FREQS = {
    0: yarn_freqs(cfg.rope_dim, r_["theta"], 0, r_["factor"], r_["beta_fast"], r_["beta_slow"]),
    1: yarn_freqs(cfg.rope_dim, r_["compress_theta"], r_["original"], r_["factor"], r_["beta_fast"], r_["beta_slow"]),
}


def bfp(x: torch.Tensor, mbits: int, mode: str = "rne", group: int = 16) -> torch.Tensor:
    """x (fp32, last dim a multiple of group) on tt-metal's bfp grid: one shared exponent (the group's max) per group
    along the last dim, mbits magnitude bits (bfloat8_b 7, bfloat4_b 3) incl. the leading one; mode rne (the host
    packer), away (half away from zero: the packer with bfp8_pack_precise), trunc."""
    shape = x.shape
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    exp, sign, man = (u >> 23) & 0xFF, u >> 31, (u & 0x7FFFFF) | (1 << 23)
    eg = exp.view(-1, group)
    shared = eg.amax(dim=-1, keepdim=True)
    m = man.view(-1, group) >> (shared - eg).clamp(max=62)
    sh = 24 - mbits
    q, rv, half = m >> sh, m & ((1 << sh) - 1), 1 << (sh - 1)
    if mode == "rne":
        q = q + ((rv > half) | ((rv == half) & ((q & 1) == 1))).to(torch.int64)
    elif mode == "away":
        q = q + (rv >= half).to(torch.int64)
    q = q.clamp(max=(1 << mbits) - 1)
    q = torch.where(eg == 0, torch.zeros_like(q), q)
    mag = q.double() * torch.pow(2.0, (shared - 127 - (mbits - 1)).double())
    return torch.where(sign.view(-1, group) == 1, -mag, mag).float().view(shape)


def b16(x):
    return x.bfloat16().float()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def chunk1(layer, name):
    with safe_open(f"{G}/chunk_01/layer_{layer:02d}.safetensors", "pt") as h:
        return h.get_tensor(name).float()


def kvc(layer, name):
    with safe_open(f"{G}/kv_cache/layer_{layer}.safetensors", "pt") as h:
        return h.get_tensor(f"{name}_cache_layer_{layer}").float()


LAYERS = [int(f[6:8]) for f in sorted(os.listdir(f"{G}/chunk_01")) if f.startswith("layer_")]
out = {"window": {}, "index": {}}
g = torch.Generator().manual_seed(1)
pos = torch.sort(torch.randperm(S - 128, generator=g)[:NQ] + C1 + 127).values  # whole window inside chunk 1

# ------------------------------------------------------------------------------------------------ window KV
WIN = {
    "bf16": b16,
    "ref_e4m3_pow2_32": lambda x: K.act_quant_dequant(x, 32),
    "bfp8_rne": lambda x: bfp(b16(x), 7, "rne"),
    "bfp8_away": lambda x: bfp(b16(x), 7, "away"),
    "bfp8_trunc": lambda x: bfp(b16(x), 7, "trunc"),
}
for layer in LAYERS:
    r = cfg.ratio(layer)
    kv = chunk1(layer, "kv")  # [2048, 512] native window keys of chunk 1
    res = {"ratio": r, "rows": {f: rel(fn(kv), kv) for f, fn in WIN.items()}}
    wq = ck.weight(f"layers.{layer}.attn.wq_b.weight", torch.float32)
    sink = ck.weight(f"layers.{layer}.attn.attn_sink", torch.float32)
    qh = rope(
        (chunk1(layer, "q_resid")[pos - C1] @ wq.t()).view(NQ, cfg.n_heads, cfg.head_dim),
        rope_table(FREQS[1 if r else 0], pos),
    )
    wrow = pos.unsqueeze(-1) - torch.arange(127, -1, -1) - C1  # [NQ, 128]
    extra, evalid = None, None
    if r:
        src = cfg.kv_source_of(layer)
        ckv = kvc(src, "compress_kv")
        topk = kvc(cfg.index_source_of(layer), "topk_idx")[pos].long()[:, : cfg.index_topk]
        extra, evalid = ckv[topk.clamp_min(0)], topk >= 0
    o_ref = None
    res["attn"] = {}
    for f, fn in [("native", lambda x: x), *WIN.items()]:
        keys = fn(kv)[wrow]
        valid = torch.ones(NQ, 128, dtype=torch.bool)
        if r:
            keys, valid = torch.cat([keys, extra], 1), torch.cat([valid, evalid], 1)
        o = K.attend(qh, keys, valid, sink, cfg.head_dim**-0.5)
        if o_ref is None:
            o_ref = o
            continue
        res["attn"][f] = rel(o, o_ref)
    out["window"][layer] = res
    print("window", layer, json.dumps(res), flush=True)

# ------------------------------------------------------------------------------------------------ index K / Q
FMT = {
    "bf16": b16,
    "ref_fp4_pow2_32": lambda x: K.fp4_quant_dequant(x, 32),
    "bfp4_rne": lambda x: bfp(b16(x), 3, "rne"),
    "bfp4_trunc": lambda x: bfp(b16(x), 3, "trunc"),
    "bfp8_rne": lambda x: bfp(b16(x), 7, "rne"),
}
COMBOS = [(f, f) for f in FMT] + [
    ("bf16", "ref_fp4_pow2_32"),
    ("ref_fp4_pow2_32", "bf16"),
    ("bf16", "bfp4_rne"),
    ("bfp4_rne", "bf16"),
]  # (query, keys)
nh, hd = cfg.index_heads, cfg.index_dim
for layer in [i for i in LAYERS if cfg.ratio(i) and cfg.is_index_source(i)]:
    r = cfg.ratio(layer)
    keys = kvc(cfg.kv_source_of(layer), "index_k")
    lens = (pos + 1) // r
    nk = int(lens.max())
    keys = keys[:nk]
    wq = ck.weight(f"layers.{layer}.attn.indexer.wq_b.weight", torch.float32)
    wp = ck.weight(f"layers.{layer}.attn.indexer.weights_proj.weight", torch.float32)
    q = rope((chunk1(layer, "q_resid")[pos - C1] @ wq.t()).view(NQ, nh, hd), rope_table(FREQS[1], pos))
    wts = (chunk1(layer, "attn_norm")[pos - C1] @ wp.t()) * (hd**-0.5 * nh**-0.5)
    mask = torch.arange(nk) >= lens.unsqueeze(-1)
    if cfg.uses_candidates(layer):
        blocks = kvc(cfg.cand_source, "candidates")[pos].long()
        mask |= ~candidate_mask(blocks, nk, cfg.cand_block)

    def score(qq, kk):
        s = (torch.einsum("bhd,td->bht", qq, kk).relu_() * wts.unsqueeze(-1)).sum(1)
        return s.masked_fill(mask, float("-inf"))

    s_ref = score(q, keys)
    top_ref = lowest_index_topk(s_ref, cfg.index_topk)
    fin = torch.isfinite(s_ref)
    res = {"ratio": r, "n_keys": nk, "q_rows": {}, "k_rows": {}, "sel": {}}
    for f, fn in FMT.items():
        res["q_rows"][f], res["k_rows"][f] = rel(fn(q), q), rel(fn(keys), keys)
    for fq, fk in COMBOS:
        s = score(FMT[fq](q), FMT[fk](keys))
        top = lowest_index_topk(s, cfg.index_topk)
        vis = (top_ref >= 0).sum(-1).clamp_min(1)
        miss = 1 - ((top_ref.unsqueeze(-1) == top.unsqueeze(-2)).any(-1) & (top_ref >= 0)).sum(-1) / vis
        res["sel"][f"{fq}|{fk}"] = {
            "score_rel": rel(s[fin], s_ref[fin]),
            "missed": float(miss.mean()),
            "missed_worst": float(miss.max()),
        }
    out["index"][layer] = res
    print("index", layer, json.dumps(res), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT)
