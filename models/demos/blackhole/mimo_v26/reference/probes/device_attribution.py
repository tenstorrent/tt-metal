"""Attribute the device-vs-oracle gap per layer (teacher-forced on DEVICE captures).
(a) MoE: oracle MoE on the DEVICE ffn_in with the DEVICE routing (idx, w) forced -> compare to device ffn_out (= expert-compute numerics only); the same with a
    bf16 oracle (forced) vs the fp32 forced oracle is the floor. Also the free-routing error split into tokens whose device expert set equals the fp32 oracle's
    and flipped tokens (oracle router on the device ffn_in).
(b) Attention: attention on the DEVICE attn_in with weights (fp32 | bf16 | bfp8-quantised qkv/o, bf16 activations) vs device attn_out; floor = bf16 vs fp32.
Output: jsonl per layer. Usage: ... device_attribution <device_dir> <out.jsonl> --layers 16,41,6-8,17,47
"""
import argparse
import json
import os
import time

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch
import torch.nn.functional as F

from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R
from models.demos.blackhole.mimo_v26.reference.bfp import quant_weight_bfp8
from models.demos.blackhole.mimo_v26.reference.ref_prefill_dump import parse_layers

torch.set_num_threads(32)


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def moe_forced(x, sd, idx, w):
    T = x.shape[0]
    out = torch.zeros(T, 4096, dtype=torch.float32)
    flat = idx.flatten()
    order = flat.argsort(stable=True)
    counts = torch.bincount(flat, minlength=256).tolist()
    tok = order // idx.shape[1]
    ws = w.flatten()[order]
    p = 0
    for e, c in enumerate(counts):
        if c:
            t = tok[p : p + c]
            xe = x[t]
            y = (
                F.silu(xe @ sd[f"mlp.experts.{e}.gate_proj.weight"].T) * (xe @ sd[f"mlp.experts.{e}.up_proj.weight"].T)
            ) @ sd[f"mlp.experts.{e}.down_proj.weight"].T
            out.index_add_(0, t, y.float() * ws[p : p + c, None])
        p += c
    return out.to(x.dtype)


def tok_rel(a, b):  # per-token relative error [T]
    return (a.float() - b.float()).norm(dim=-1) / b.float().norm(dim=-1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev")
    ap.add_argument("out")
    ap.add_argument("--layers", default="16,41,6-8,17,47")
    a = ap.parse_args()
    ck = Ld.Ckpt()
    done = {json.loads(l)["L"] for l in open(a.out)} if os.path.exists(a.out) else set()
    layers = []
    for part in a.layers.split(","):
        layers += parse_layers(part)
    for L in layers:
        if L in done:
            continue
        t0 = time.time()
        d = torch.load(f"{a.dev}/layer_{L}.pt")["prefill"]
        sd32 = Ld.load_layer(L, torch.float32, ck)
        cast = lambda s, dt: {
            k: (v.to(dt) if v.is_floating_point() and "e_score" not in k else v) for k, v in s.items()
        }
        sd16 = cast(sd32, torch.bfloat16)
        sdq = dict(sd16)
        for k in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
            sdq[k] = quant_weight_bfp8(sd32[k]).to(torch.bfloat16)
        sdq32 = dict(sd32)
        for k in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
            sdq32[k] = quant_weight_bfp8(sd32[k])
        row = {"L": L, "type": "swa" if R.is_swa(L) else "global"}
        # ---- (b) attention on the device attn_in
        ain = d["attn_in"]
        a32 = R.attention(ain.float(), sd32, L)[0]
        a16 = R.attention(ain, sd16, L)[0]
        aq16 = R.attention(ain, sdq, L)[0]
        aq32 = R.attention(ain.float(), sdq32, L)[0]
        dev = d["attn_out"]
        fl = rel(a16, a32)
        row["attn"] = {
            "floor_bf16_vs_fp32": fl,
            "dev_vs_fp32": rel(dev, a32),
            "dev_vs_bf16": rel(dev, a16),
            "dev_vs_bfp8w_bf16act": rel(dev, aq16),
            "dev_vs_bfp8w_fp32act": rel(dev, aq32),
            "bfp8w_bf16act_vs_fp32": rel(aq16, a32),
            "bfp8w_fp32act_vs_fp32": rel(aq32, a32),
            "ratio_dev_fp32_over_floor": rel(dev, a32) / fl,
            "ratio_dev_bfp8w_over_floor": rel(dev, aq16) / fl,
        }
        # ---- (a) MoE on the device ffn_in
        if L > 0:
            fin = d["ffn_in"].reshape(-1, 4096)
            di, dw = d["routing_idx"], d["routing_w"].float()
            dev_f = d["ffn_out"].reshape(-1, 4096)
            f32 = moe_forced(fin.float(), sd32, di, dw)
            f16 = moe_forced(fin, sd16, di, dw)
            row["moe_forced_device_routing"] = {
                "floor_bf16_vs_fp32": rel(f16, f32),
                "dev_vs_fp32": rel(dev_f, f32),
                "dev_vs_bf16": rel(dev_f, f16),
                "ratio_dev_over_floor": rel(dev_f, f32) / rel(f16, f32),
            }
            idx_o, w_o, gap = R.router(fin.float(), sd32)
            eq = (di.sort(-1).values == idx_o.sort(-1).values).all(-1)
            free32, _, _, _ = R.moe(fin.float(), sd32)
            tr = tok_rel(dev_f, free32)
            row["moe_free_routing_split"] = {
                "frac_equal_sets": eq.float().mean().item(),
                "rel_equal_tokens": rel(dev_f[eq], free32[eq]) if eq.any() else None,
                "rel_flipped_tokens": rel(dev_f[~eq], free32[~eq]) if (~eq).any() else None,
                "tokrel_median_equal": tr[eq].median().item(),
                "tokrel_median_flipped": tr[~eq].median().item() if (~eq).any() else None,
            }
        row["secs"] = round(time.time() - t0)
        open(a.out, "a").write(json.dumps(row) + "\n")
        at = row["attn"]
        print(
            L,
            row["type"],
            "attn rel dev/fp32 %.4f dev/bfp8w %.4f floor %.4f ratio %.1f->%.1f"
            % (
                at["dev_vs_fp32"],
                at["dev_vs_bfp8w_bf16act"],
                at["floor_bf16_vs_fp32"],
                at["ratio_dev_fp32_over_floor"],
                at["ratio_dev_bfp8w_over_floor"],
            ),
            "| moe forced dev/fp32 %.4f floor %.4f"
            % (row["moe_forced_device_routing"]["dev_vs_fp32"], row["moe_forced_device_routing"]["floor_bf16_vs_fp32"])
            if L > 0
            else "",
            flush=True,
        )
        del sd32, sd16, sdq, sdq32


if __name__ == "__main__":
    main()
