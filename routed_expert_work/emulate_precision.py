# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Host-side (no device) torch emulation of the routed-expert numeric pipeline, to attribute the
error of moe_fused_swiglu vs. the fp32 reference to individual quantization points.

Quantization points emulated (each is a real round-trip through ttnn's host block-float packer):
  x     -> bfp8 (tilize path, both ops)
  W     -> bfp4 or bfp8 (both ops)
  gate/up partials per K-shard -> bfp8 (fused op only; 8 shards = KGROUPS)   <-- hypothesis
  gate/up sum -> bf16 (slice CBs), silu -> bf16
  h     -> bfp8 (both ops)
  out   -> bfp8 (both ops)

Usage: python_env/bin/python routed_expert_work/emulate_precision.py [--wdtype bfp4|bfp8] [--m 256]
"""
import argparse

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert


def q(t: torch.Tensor, dtype) -> torch.Tensor:
    """Round-trip a 2D fp32 tensor through a ttnn host tensor of block-float `dtype` (TILE layout)."""
    if dtype is None:
        return t
    if dtype == "bf16":
        return t.to(torch.bfloat16).float()
    tt = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt).float()


def metrics(ref, out):
    ref = ref.float()
    out = out.float()
    vx = ref - ref.mean()
    vy = out - out.mean()
    pcc = (vx * vy).sum() / (vx.norm() * vy.norm())
    return float(pcc), float((out - ref).norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdtype", default="bfp4")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--emb", type=int, default=7168)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--kgroups", type=int, default=8)
    ap.add_argument("--wscale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    wd = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b, "bf16": "bf16", "fp32": None}[args.wdtype]

    torch.manual_seed(args.seed)
    emb, hidden, m = args.emb, args.hidden, args.m
    weights = {
        "gate_proj": torch.randn(hidden, emb) * args.wscale,
        "up_proj": torch.randn(hidden, emb) * args.wscale,
        "down_proj": torch.randn(emb, hidden) * args.wscale,
    }
    x = torch.randn(m, emb)
    with torch.no_grad():
        ref = TorchExpert(emb, hidden, weights, activation=ACTIVATION_SILU)(x)

    wg = q(weights["gate_proj"].T, wd)  # [emb, hidden]
    wu = q(weights["up_proj"].T, wd)
    wdn = q(weights["down_proj"].T, wd)  # [hidden, emb]
    xq = q(x, ttnn.bfloat8_b)

    def pipeline(partial_dtype, h_dtype, out_dtype, sum_dtype="bf16", xq_=xq):
        """partial_dtype: None -> exact fp32 partials summed (old-op-like, bf16 accumulator approximated
        by one bf16 round at the end); ttnn.bfloat8_b -> each K-shard partial rounded to bfp8 first."""
        kr = emb // args.kgroups
        g = torch.zeros(m, hidden)
        u = torch.zeros(m, hidden)
        for s in range(args.kgroups):
            ks = slice(s * kr, (s + 1) * kr)
            gp = xq_[:, ks] @ wg[ks, :]
            up = xq_[:, ks] @ wu[ks, :]
            g += q(gp, partial_dtype)
            u += q(up, partial_dtype)
        g = q(g, sum_dtype)
        u = q(u, sum_dtype)
        sg = q(torch.nn.functional.silu(g), "bf16")
        h = q(sg * u, h_dtype)
        out = q(h @ wdn, out_dtype)
        return out

    rows = [
        ("fp32 everything (x bfp8, W quantized only)", pipeline(None, None, None, None)),
        ("old-op-like: partials exact, sum bf16, h bfp8, out bfp8", pipeline(None, ttnn.bfloat8_b, ttnn.bfloat8_b)),
        (
            "fused-op-like: partials bfp8 (8 shards), h bfp8, out bfp8",
            pipeline(ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b),
        ),
        ("fused-op-like but partials bf16", pipeline("bf16", ttnn.bfloat8_b, ttnn.bfloat8_b)),
        ("partials bfp8, h bf16, out bfp8", pipeline(ttnn.bfloat8_b, "bf16", ttnn.bfloat8_b)),
        ("partials exact, h bf16, out bfp8", pipeline(None, "bf16", ttnn.bfloat8_b)),
        ("partials exact, h bfp8, out bf16", pipeline(None, ttnn.bfloat8_b, "bf16")),
    ]
    print(f"M={m} emb={emb} hidden={hidden} W={args.wdtype} kgroups={args.kgroups}")
    print(f"{'variant':<62} {'pcc':>10} {'rel_rms':>9}")
    for name, out in rows:
        pcc, rr = metrics(ref, out)
        print(f"{name:<62} {pcc:10.6f} {rr:9.5f}")


if __name__ == "__main__":
    main()
