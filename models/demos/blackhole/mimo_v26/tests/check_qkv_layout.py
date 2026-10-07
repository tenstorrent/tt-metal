# SPDX-License-Identifier: Apache-2.0
"""CPU check of the fused qkv row layout of the real checkpoint (layers 1 = SWA, 5 = GA).
Hypothesis A: rank-blocked [q_r|k_r|v_r] x4 (track-0 loader deinterleave). Hypothesis B: rows already contiguous Q|K|V (dequantised with the same per-rank scales).
A wrong layout cannot be detected by comparing device vs torch (both use the same weights), so we measure how structured the resulting attention is on real
text: entropy and max probability of the softmax rows, sink excluded. A correct layout gives clearly peaked heads (previous-token / first-token / local heads); a wrong
one mixes unrelated rows into q and k and gives near-uniform, high-entropy attention."""
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attn_ref import DQK, NQ, apply_rope, rope_cos_sin  # noqa: E402
from ref_loader_copy import Ckpt, deinterleave_qkv, dequant_fp8_qkv_ranked, qkv_dims  # noqa: E402

T = 512
ck = Ckpt()
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.environ.get("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL"))
text = ""
for f in sorted(glob.glob("/mnt/tt-data/ssinghal/mimo/wt/attn/docs/**/*.rst", recursive=True)):
    text += open(f, errors="ignore").read() + "\n\n"
    if len(text) > 20000:
        break
ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"][:T])
assert len(ids) == T, len(ids)


def stats(x, q_w, k_w, nkv, theta, window):
    q = (x @ q_w.T).view(T, NQ, DQK).transpose(0, 1)
    k = (x @ k_w.T).view(T, nkv, DQK).transpose(0, 1)
    c, s = rope_cos_sin(torch.arange(T), theta)
    q, k = apply_rope(q, c, s), apply_rope(k, c, s)
    qi, kj = torch.arange(T)[:, None], torch.arange(T)[None, :]
    mask = kj <= qi
    if window:
        mask &= kj > qi - window
    ents, mx, loc, first = [], [], [], []
    for h in range(NQ):
        sc = (q[h] @ k[h // (NQ // nkv)].T) * DQK**-0.5
        w = torch.softmax(sc.masked_fill(~mask, float("-inf")), 1)[64:]
        ents.append(float(-(w * (w + 1e-30).log()).sum(1).mean()))
        mx.append(float(w.max(1).values.mean()))
        full = torch.softmax(sc.masked_fill(~mask, float("-inf")), 1)
        rel = torch.stack([full[torch.arange(64, T), torch.arange(64, T) - o] for o in range(4)], 0).sum(
            0
        )  # mass on offsets 0..3
        loc.append(float(rel.mean()))
        first.append(float(full[64:, 0].mean()))
    nvalid = mask[64:].sum(1).float()
    uniform = float(nvalid.log().mean())
    return dict(
        max_head_mass_on_offsets_0to3=round(max(loc), 3),
        mean_head_mass_on_offsets_0to3=round(sum(loc) / NQ, 3),
        max_head_mass_on_token0=round(max(first), 3),
        mean_entropy=round(sum(ents) / NQ, 3),
        uniform_entropy=round(uniform, 3),
        mean_maxprob=round(sum(mx) / NQ, 3),
        frac_heads_entropy_below_half_uniform=round(sum(e < 0.5 * uniform for e in ents) / NQ, 3),
    )


for L, kind, theta, window in ((1, "swa", 1e4, 128), (5, "ga", 1e7, 0)):
    p = f"model.layers.{L}."
    nq, nk, nv = qkv_dims(L)
    raw = dequant_fp8_qkv_ranked(
        ck.get(p + "self_attn.qkv_proj.weight"), ck.get(p + "self_attn.qkv_proj.weight_scale_inv"), nq, nk, nv
    )
    A = deinterleave_qkv(raw, nq, nk, nv)
    ln = ck.get(p + "input_layernorm.weight").float()
    h = ck.get("model.embed_tokens.weight")[ids].float()
    x = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * ln
    nkv = nk // DQK
    for name, W in (("A_rank_blocked_deinterleaved", A), ("B_contiguous_raw", raw)):
        r = stats(x, W[:nq], W[nq : nq + nk], nkv, theta, window)
        qblocks = W[:nq].view(NQ, DQK, -1).abs().mean((1, 2))
        r["q_head_absmean_cv"] = round(float(qblocks.std() / qblocks.mean()), 3)
        print("LAYOUT", kind, L, name, r, flush=True)
