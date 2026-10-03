# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: build the index-key state and one decode query of an index-source layer with the CHECKPOINT's own Compressor / Indexer
code and real weights, and save everything the device test compares against.

    python -m models.demos.blackhole.deepseek_v41_flash.tests.indexer_ref_state --layer 2 --entries 2048 --out DIR

State layout (``DIR/layer{L}_N{entries}.pt``): ``x_q`` [5120] normed decode hidden, ``qr`` [1280] q-lora latent, ``index_k`` [N,128] (RoPE'd +
fp4-simulated keys exactly as the reference writes its k_cache), ``scores`` [N] fp32 (reference formula with fp4 q), ``scores_noq`` (same without the
q fp4 simulation), ``topk`` [512] sorted entry ids from ``Indexer.forward`` itself (reference module call), ``pos`` decode position.
Tokens are random embedding rows pushed through attn_norm: realistic scale, random content.
"""

import argparse
import os
import time

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.loader import rope_freqs
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


def dq(sh, key):
    w, s = sh.get(key + ".weight"), sh.get(key + ".scale")
    return ref_kernels.dequant_fp8_weight(w, s, w.size(0) // s.size(0)).to(torch.bfloat16)


def load_params(module, sh, prefix):
    for name, p in module.named_parameters():
        t = sh.get(prefix + name)
        if p.dtype == torch.float8_e4m3fn:
            p.data = t
        else:
            p.data = t.to(p.dtype)


def build(layer, N, seed, out_dir):
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.bfloat16)
    mod = R.load_model_module()
    ratio = R.model_args().compress_ratios[layer]
    S = N * ratio  # prefilled tokens; decode position = S, compressed length (S + 1) // ratio = N
    args = R.model_args(max_batch_size=1, max_seq_len=S + 64)
    sh = _Shards()
    p = f"layers.{layer}."
    t0 = time.time()
    emb = sh.get("embed.weight")
    toks = torch.randint(0, emb.shape[0], (S + 1,))
    h = emb[toks].to(torch.bfloat16)
    norm = mod.RMSNorm(args.dim, args.norm_eps)
    norm.weight.data = sh.get(p + "attn_norm.weight").to(norm.weight.dtype)
    x = norm(h)  # [S+1, 5120]
    comp = mod.Compressor(args, layer)
    load_params(comp, sh, p + "attn.compressor.")
    with torch.no_grad():
        lat = comp(x[None, :S], 0)  # [1, S/ratio, 512] RoPE-free latents (pre-quantisation)
    print(f"compressor done {time.time() - t0:.0f}s lat {tuple(lat.shape)}", flush=True)
    idxr = mod.Indexer(args, layer)
    load_params(idxr, sh, p + "attn.indexer.")
    freqs = rope_freqs(layer, S + 64)
    idxr.freqs_cis = freqs
    rd = args.rope_head_dim
    with torch.no_grad():
        k = idxr.k_norm(idxr.wk(lat))  # [1, N, 128]
        mod.apply_rotary_emb(k[..., -rd:], freqs[: S - S % ratio : ratio])
        mod.fp4_act_quant(k, mod.fp4_block_size, True)
        idxr.k_cache[:1, :N] = k
        mod.shared_attn.index_k = idxr.k_cache
        # decode query
        x_q = x[S : S + 1][None]  # [1,1,5120]
        wq_a = dq(sh, p + "attn.wq_a")
        qn = mod.RMSNorm(args.q_lora_rank, args.norm_eps)
        qn.weight.data = sh.get(p + "attn.q_norm.weight").to(qn.weight.dtype)
        qr = qn(x_q @ wq_a.T)  # [1,1,1280]
        # reference module: indices of the top-k entries (offset 0 here)
        ids = idxr(x_q, qr, None, S, 0)  # [1,1,topk] int32, sorted by position

        # explicit score formula (same as Indexer.forward) for score-level comparison
        def scores_for(quant_q):
            q = idxr.wq_b(qr).unflatten(-1, (args.index_n_heads, args.index_head_dim)).clone()
            mod.apply_rotary_emb(q[..., -rd:], freqs[S : S + 1])
            if quant_q:
                mod.fp4_act_quant(q, mod.fp4_block_size, True)
            wts = idxr.weights_proj(x_q) * (idxr.softmax_scale * args.index_n_heads**-0.5)
            sc = torch.einsum("bshd,btd->bsht", q, k[:, :N])
            return (sc.relu_() * wts.unsqueeze(-1)).sum(dim=2)[0, 0].float(), q[0, 0].float(), wts[0, 0].float()

        sc, q_ref, w_ref = scores_for(True)
        sc_noq, _, _ = scores_for(False)
    topk_formula = sc.topk(min(512, N)).indices.sort().values
    assert (
        torch.equal(topk_formula.int(), ids[0, 0].sort().values.int())
        or (topk_formula.int() == ids[0, 0].int()).float().mean() > 0.99
    ), "formula != module"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"layer{layer}_N{N}.pt")
    torch.save(
        {
            "x_q": x_q[0, 0].float(),
            "qr": qr[0, 0].float(),
            "index_k": k[0].float(),
            "scores": sc,
            "scores_noq": sc_noq,
            "topk": ids[0, 0].long(),
            "q_ref": q_ref,
            "w_ref": w_ref,
            "pos": S,
            "N": N,
            "ratio": ratio,
            "layer": layer,
        },
        path,
    )
    inter = len(set(ids[0, 0].tolist()) & set(sc_noq.topk(512).indices.tolist())) / 512
    print(
        f"saved {path} in {time.time() - t0:.0f}s; ref top-512 vs top-512 without q fp4 simulation: set agreement {inter:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=2)
    ap.add_argument("--entries", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/mnt/tt-data/ssinghal/dsv4-kv-state")
    a = ap.parse_args()
    build(a.layer, a.entries, a.seed, a.out)
