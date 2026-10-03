"""CPU attribution of the routed-expert error on real layers (no device): which part comes from weight format, which from
bf16 intermediates, and how much a free output-channel reordering of w0/w1 (+ matching w2 rows) recovers.

python tests/analyze_routed_error.py   (env DSV41_CHAIN, DSV41_ANALYZE_LAYERS="1,8", DSV41_ANALYZE_TOKENS=6)
"""
import os

import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_kernels
from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.reference.bfp_emulation import bfp_roundtrip
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import _fp4_expert, _Shards

torch.set_num_threads(24)
CHAIN = os.environ.get("DSV41_CHAIN", "/mnt/tt-data/ssinghal/dsv4-chain-e")
LAYERS = [int(x) for x in os.environ.get("DSV41_ANALYZE_LAYERS", "1,8").split(",")]
NTOK = int(os.environ.get("DSV41_ANALYZE_TOKENS", "6"))
ref_kernels.FAKE_QUANT = False
sh = _Shards()
bf = lambda t: t.to(torch.bfloat16).float()
rel = lambda a, b: float((a - b).norm() / b.norm())


def expert_out(x, w0, w1, w2, act_bf16=False):
    g, u = x @ w0, x @ w1
    if act_bf16:
        g, u = bf(g), bf(u)
        h = bf(bf(torch.nn.functional.silu(g)) * u)
        return bf(h @ w2)
    return (torch.nn.functional.silu(g) * u) @ w2


for L in LAYERS:
    d = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
    blk = R.build_layer(L, max_batch_size=16, max_seq_len=256)
    st = d["state"]
    blk.attn.window_kv_cache.copy_(st["window"])
    cap = {}
    blk.ffn.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach()))
    blk(d["dec_in"].to(torch.bfloat16), 9, d["pre_in"], None)
    x = cap["x"].reshape(16, 5120).float()[:NTOK]
    wg, bg = sh.get(f"layers.{L}.ffn.gate.weight").float(), sh.get(f"layers.{L}.ffn.gate.bias").float()
    s = torch.nn.functional.softplus(x @ wg.T).sqrt()
    idx = (s + bg).topk(6, -1).indices
    wt = s.gather(1, idx)
    wt = wt / wt.sum(-1, keepdim=True) * 1.5
    experts = sorted(set(idx.flatten().tolist()))
    print(f"layer {L}: {NTOK} tokens, {len(experts)} distinct experts", flush=True)

    variants = {
        k: torch.zeros(NTOK, 5120)
        for k in (
            "exact",
            "bfp4 weights",
            "bfp4 w0/w1 only",
            "bfp4 w2 only",
            "bfp4 + channel reorder",
            "bfp8 weights",
            "exact + bf16 intermediates",
            "bfp4 + bf16 intermediates",
        )
    }
    for e in experts:
        p = f"layers.{L}.ffn.experts.{e}."
        w0, w1, w2 = (
            _fp4_expert(sh, p + n, torch.float32) for n in ("w1", "w3", "w2")
        )  # w0=gate(w1) w1=up(w3) [in,out]
        # channel statistic from the checkpoint scales (per output channel n, one scale per 32 input rows)
        s0 = sh.get(p + "w1.scale").float().log2().mean(1)
        s1 = sh.get(p + "w3.scale").float().log2().mean(1)
        perm = torch.argsort(s0 + s1)  # group channels of similar scale into the same 16-block
        q = lambda w: bfp_roundtrip(w, 3)
        cases = {
            "exact": (w0, w1, w2, False),
            "bfp4 weights": (q(w0), q(w1), q(w2), False),
            "bfp4 w0/w1 only": (q(w0), q(w1), w2, False),
            "bfp4 w2 only": (w0, w1, q(w2), False),
            "bfp4 + channel reorder": (q(w0[:, perm]), q(w1[:, perm]), q(w2[perm, :]), False),
            "bfp8 weights": (bfp_roundtrip(w0, 7), bfp_roundtrip(w1, 7), bfp_roundtrip(w2, 7), False),
            "exact + bf16 intermediates": (w0, w1, w2, True),
            "bfp4 + bf16 intermediates": (q(w0), q(w1), q(w2), True),
        }
        sel = idx == e
        toks = sel.any(1).nonzero().flatten()
        for name, (a, b, c, act) in cases.items():
            out = expert_out(x[toks], a, b, c, act)
            variants[name][toks] += wt[toks][sel[toks]].unsqueeze(1) * out
    for name, v in variants.items():
        print(f"RESULT layer {L}: {name:30s} rel err vs exact {rel(v, variants['exact']):.4f}", flush=True)
    del blk
