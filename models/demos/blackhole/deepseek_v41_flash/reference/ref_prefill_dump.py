# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU oracle for the DEVICE PREFILL: like ``ref_chain.py`` (same tokens for the same seed/S/B/steps, same per-layer
``layer_L.pt`` keys, so ``tests/test_decode_steps.py`` can also run on these directories) plus everything the prefill
tests compare against.

Per layer L, ``layer_{L}.pt`` additionally holds ``prefill`` = {
    "h_in":  [B,S,4,D] bf16  layer input streams (AFTER the Engram write of layers 1/14),  "pm_in":  [B,S,4] fp32 (the pre mix)
    "h_out": [B,S,4,D] bf16  layer output streams,                                          "pm_out": [B,S,4] fp32 (ffn pre)
    "attn_in":  [B,S,D] bf16 input of the attention sub-block (after hc_pre + attn_norm),   "attn_out": [B,S,D] bf16 (before hc_post)
    "ffn_in":   [B,S,D] bf16 input of the MoE (after hc_pre + ffn_norm),                    "ffn_out":  [B,S,D] bf16 (before hc_post)
    "routing_idx": [B*S, 6] int expert ids of the prefill tokens
}
and ``state`` is the decode state the layer holds after the prefill (window ring [B,128,512] with position p at slot
p % 128, compressed latents [B,S//ratio,512] RoPE'd, kv_state/score_state [B,ratio,512] of the ratio-2 compressor).
``final.pt`` (with --head) = {"prefill_logits": [B,vocab] logits of the LAST prompt token, "prefill_argmax": [B], plus the
decode-step logits/argmax exactly like ref_chain}. ``tokens.pt`` = {"prefill_tokens", "decode_tokens"}.

Run (CPU only; S=128, 16 users ~ 1 min/layer):
    python -m models.demos.blackhole.deepseek_v41_flash.reference.ref_prefill_dump --S 128 --engram --head --out /mnt/tt-data/ssinghal/dsv4-prefill-s128
"""

import argparse
import gc
import os
import time

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.ref_chain import parse_layers
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _Shards


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-39")
    ap.add_argument("--S", type=int, default=9)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--engram", action="store_true")
    ap.add_argument("--head", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3, help="teacher-forced decode steps after the prefill")
    ap.add_argument(
        "--max_seq_len",
        type=int,
        default=0,
        help="reference cache length (default: S + steps rounded up to 32, at least 256)",
    )
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    ref_kernels.FAKE_QUANT = False
    max_seq = a.max_seq_len or max(256, -(-(a.S + a.steps) // 32) * 32)
    g = torch.Generator().manual_seed(a.seed)
    pre_tok = torch.randint(1000, 100000, (a.B, a.S), generator=g)
    dec_tok = torch.randint(1000, 100000, (a.B, a.steps), generator=g)
    torch.save({"prefill_tokens": pre_tok, "decode_tokens": dec_tok}, os.path.join(a.out, "tokens.pt"))

    h_pre, pm_pre = R.embed_tokens(pre_tok)
    emb = [R.embed_tokens(dec_tok[:, i : i + 1]) for i in range(a.steps)]
    h_decs, pm_decs = [e[0] for e in emb], [e[1] for e in emb]
    engram = hashes_pre = None
    hashes_decs = []
    if a.engram:
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostEngram

        engram = HostEngram(max_batch_size=a.B, max_seq_len=max_seq)
        hashes_pre = engram.hashes(pre_tok, 0)
        hashes_decs = [engram.hashes(dec_tok[:, i : i + 1], a.S + i) for i in range(a.steps)]

    sh = _Shards()
    layers = parse_layers(a.layers)
    sa = R.load_model_module().shared_attn
    fields = ("compress_kv", "index_k", "topk_idxs", "candidates")
    phase = {"pre": {k: None for k in fields}}
    phase.update({("dec", i): {k: None for k in fields} for i in range(a.steps)})

    def enter(name):
        for k in fields:
            v = phase[name][k]
            if v is None and name != "pre":
                v = phase["pre"][k]
            setattr(sa, k, v)

    def leave(name):
        phase[name] = {k: getattr(sa, k) for k in fields}

    bf = lambda t: t.detach().to(torch.bfloat16).clone()
    t_all = time.time()
    for L in layers:
        t0 = time.time()
        blk = R.build_layer(L, max_batch_size=a.B, max_seq_len=max_seq)
        if engram is not None and L in engram.mods:
            h_pre = engram.apply(L, h_pre, hashes_pre)
            h_decs = [engram.apply(L, h, hs) for h, hs in zip(h_decs, hashes_decs)]
        cap = {}
        hooks = [
            blk.ffn.register_forward_hook(lambda m, i, o: cap.update(ffn_in=bf(i[0]), ffn_out=bf(o))),
            blk.attn.register_forward_hook(lambda m, i, o: cap.update(attn_in=bf(i[0]), attn_out=bf(o))),
            blk.ffn.gate.register_forward_hook(
                lambda m, i, o: cap.update(routing_idx=o[1].detach().reshape(-1, o[1].shape[-1]).clone())
            ),
        ]
        h_in, pm_in = bf(h_pre), pm_pre.clone().float()
        enter("pre")
        h_pre, pm_pre = blk(h_pre, 0, pm_pre, None)
        leave("pre")
        for h in hooks:
            h.remove()
        x = cap["ffn_in"].reshape(-1, 5120).float()
        w = sh.get(f"layers.{L}.ffn.gate.weight").float()
        b = sh.get(f"layers.{L}.ffn.gate.bias").float()
        rank = torch.nn.functional.softplus(x @ w.T).sqrt() + b
        K = float(rank.topk(7, -1).values[:, 5].mean())
        ratio = blk.attn.compress_ratio
        state = {"window": blk.attn.window_kv_cache.clone().float()}
        if ratio and blk.attn.is_kv_source:
            state["comp"] = blk.attn.compress_kv_cache[:, : a.S // ratio].clone().float()
            if ratio > 1:
                state["kv_state"] = blk.attn.compressor.kv_state.clone()
                state["score_state"] = blk.attn.compressor.score_state.clone()
            if getattr(blk.attn, "indexer", None) is not None and blk.attn.indexer.owns_k:
                state["index_k"] = (
                    blk.attn.indexer.k_cache[:, : a.S // ratio].clone().float()
                )  # index keys (RoPE'd, fp4-simulated) of the key owners
        prefill = {
            "h_in": h_in,
            "pm_in": pm_in,
            "h_out": bf(h_pre),
            "pm_out": pm_pre.clone().float(),
            "attn_in": cap["attn_in"],
            "attn_out": cap["attn_out"],
            "ffn_in": cap["ffn_in"],
            "ffn_out": cap["ffn_out"],
            "routing_idx": cap["routing_idx"],
        }
        dec_in, pre_in = h_decs[0].clone(), pm_decs[0].clone()
        steps_in = [(h.clone(), p.clone()) for h, p in zip(h_decs, pm_decs)]
        gate_out = {}
        ghook = blk.ffn.gate.register_forward_hook(
            lambda m, i, o: gate_out.update(w=o[0].detach().clone(), idx=o[1].detach().clone())
        )
        steps_out = []
        for i in range(a.steps):
            enter(("dec", i))
            h_decs[i], pm_decs[i] = blk(h_decs[i], a.S + i, pm_decs[i], None)
            leave(("dec", i))
            steps_out.append((h_decs[i].clone(), pm_decs[i].clone()))
            if i == 0:
                ghook.remove()
        out = {
            "S": a.S,
            "prefill": prefill,
            "state": state,
            "gate_cutoff": K,
            "ratio": ratio,
            "is_kv_source": bool(ratio and blk.attn.is_kv_source),
        }
        if a.steps:
            out.update(
                {
                    "dec_in": dec_in,
                    "pre_in": pre_in,
                    "dec_out": steps_out[0][0],
                    "pre_out": steps_out[0][1],
                    "dec_in_steps": steps_in,
                    "dec_out_steps": steps_out,
                    "routing_idx": gate_out["idx"].reshape(a.B, -1),
                    "routing_wt": gate_out["w"].reshape(a.B, -1),
                }
            )
        torch.save(out, os.path.join(a.out, f"layer_{L}.pt"))
        print(
            f"layer {L:2d} ratio {ratio} K={K:.3f}  {time.time() - t0:.0f}s  (total {time.time() - t_all:.0f}s)",
            flush=True,
        )
        del blk, cap, x, rank, out, prefill
        gc.collect()

    if a.head and layers[-1] == 39:
        from models.demos.blackhole.deepseek_v41_flash.tt.host_model import HostHead

        head = HostHead()
        fin = {}
        pl = head(h_pre[:, -1:], pm_pre[:, -1:])  # logits of the last prompt token
        fin.update({"prefill_logits": pl, "prefill_argmax": pl.argmax(-1)})
        if a.steps:
            all_logits = [head(h, p) for h, p in zip(h_decs, pm_decs)]
            fin.update(
                {
                    "logits": all_logits[0],
                    "argmax": all_logits[0].argmax(-1),
                    "logits_steps": torch.stack(all_logits),
                    "argmax_steps": torch.stack([l.argmax(-1) for l in all_logits]),
                }
            )
        torch.save(fin, os.path.join(a.out, "final.pt"))
        print("prefill argmax (first generated token):", pl.argmax(-1).tolist())


if __name__ == "__main__":
    main()
