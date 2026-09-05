# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Second, device-faithful torch emulation of the two routed-expert ops (host only).

Models, per matmul: LoFi operand truncation (SrcOrder::Reverse -> in0=x/h in SrcB keeps hidden+6
mantissa bits, in1=W in SrcA keeps hidden+4), per-tile-K-step DEST accumulation with bf16 rounding
(fp32_dest_acc_en=false), packer L1-accumulation in bf16 between K-blocks, and the bfp8 pack
(round-to-nearest with bfp8_pack_precise, truncation without).

Fused op (moe_fused_swiglu):
  gate/up: K split in 8 shards of 28 tiles; each shard: 28 DEST steps -> pack bfp8 (precise) ->
           8 partials summed in DEST (bf16 steps) -> bf16 slice -> silu bf16 -> h = bfp8(silu*up)
  down (mrow, m_eff==8):  64 DEST steps over K=64 tiles -> pack bfp8 output
  down (short, m_eff<8):  11 K-blocks of 6 DEST steps, bf16 L1-acc between blocks -> bfp8 output
Old op (unified_routed_expert_ffn): gate/up K-blocks with bf16 L1 acc (block width in0_block_w),
  h bfp8 (truncating pack), down likewise.

Usage: python routed_expert_work/emulate_precision2.py --wdtype bfp4 --m 256
"""
import argparse

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import ACTIVATION_SILU, TorchExpert

TILE = 32


def q_block(t: torch.Tensor, dtype) -> torch.Tensor:
    tt = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(tt).float()


def bf16(t):
    return t.to(torch.bfloat16).float()


def trunc_mantissa(t: torch.Tensor, explicit_bits: int) -> torch.Tensor:
    """Truncate (toward zero) a float tensor's mantissa to `explicit_bits` explicit bits (plus hidden 1)."""
    m, e = torch.frexp(t)  # t = m * 2**e, 0.5 <= |m| < 1
    scale = 2.0 ** (explicit_bits + 1)  # hidden bit + explicit bits below the leading 1 at 2^-1
    m_q = torch.trunc(m * scale) / scale
    return torch.ldexp(m_q, e)


def bfp8_truncate(t: torch.Tensor) -> torch.Tensor:
    """bfp8 with truncation instead of round-to-nearest: shared exponent per 16 consecutive elements
    of the last dim (one face row), 7-bit magnitude mantissa, truncated."""
    shp = t.shape
    g = t.reshape(-1, 16)
    mx = g.abs().amax(dim=1, keepdim=True)
    e = torch.floor(torch.log2(mx.clamp_min(1e-38))) + 1  # 2^e > max
    lsb = 2.0 ** (e - 7)
    q = torch.trunc(g / lsb) * lsb
    return q.reshape(shp)


def matmul_dest_steps(a: torch.Tensor, b: torch.Tensor, lofi=True, step_round=True, l1acc_block=None):
    """a [M,K] @ b [K,N] emulating the FPU: tile-K steps of 32 accumulated in bf16 DEST; if
    l1acc_block is given, DEST is packed to bf16 L1 every l1acc_block K-tiles and accumulated in bf16."""
    if lofi:
        a = trunc_mantissa(a, 6)  # SrcB: hidden + 6 MSB
        b = trunc_mantissa(b, 4)  # SrcA: hidden + 4 MSB
    K = a.shape[1]
    kt = K // TILE
    acc_l1 = None
    dest = None
    for k in range(kt):
        ks = slice(k * TILE, (k + 1) * TILE)
        p = a[:, ks] @ b[ks, :]
        dest = p if dest is None else dest + p
        if step_round:
            dest = bf16(dest)
        if l1acc_block is not None and (k + 1) % l1acc_block == 0:
            acc_l1 = dest if acc_l1 is None else bf16(acc_l1 + dest)
            dest = None
    if l1acc_block is not None:
        if dest is not None:
            acc_l1 = dest if acc_l1 is None else bf16(acc_l1 + dest)
        return acc_l1
    return dest


def metrics(ref, out):
    vx = ref - ref.mean()
    vy = out - out.mean()
    return float((vx * vy).sum() / (vx.norm() * vy.norm())), float((out - ref).norm() / ref.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wdtype", default="bfp4")
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--emb", type=int, default=7168)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--wscale", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    wd = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[a.wdtype]
    torch.manual_seed(a.seed)
    emb, hidden, m = a.emb, a.hidden, a.m
    weights = {
        "gate_proj": torch.randn(hidden, emb) * a.wscale,
        "up_proj": torch.randn(hidden, emb) * a.wscale,
        "down_proj": torch.randn(emb, hidden) * a.wscale,
    }
    x = torch.randn(m, emb)
    with torch.no_grad():
        ref = TorchExpert(emb, hidden, weights, activation=ACTIVATION_SILU)(x)
    wg = q_block(weights["gate_proj"].T, wd)
    wu = q_block(weights["up_proj"].T, wd)
    wdn = q_block(weights["down_proj"].T, wd)
    xq = q_block(x, ttnn.bfloat8_b)
    silu = torch.nn.functional.silu

    def fused(partial_fmt="bfp8", down="mrow", lofi=True, step_round=True, h_fmt="bfp8"):
        kr = emb // 8
        g = None
        u = None
        for s in range(8):
            ks = slice(s * kr, (s + 1) * kr)
            gp = matmul_dest_steps(xq[:, ks], wg[ks, :], lofi, step_round)
            up = matmul_dest_steps(xq[:, ks], wu[ks, :], lofi, step_round)
            if partial_fmt == "bfp8":
                gp, up = q_block(gp, ttnn.bfloat8_b), q_block(up, ttnn.bfloat8_b)
            elif partial_fmt == "bf16":
                gp, up = bf16(gp), bf16(up)
            g = gp if g is None else bf16(g + gp)  # fold in DEST, bf16 per add
            u = up if u is None else bf16(u + up)
        sg = bf16(silu(g))
        h = sg * u
        h = q_block(h, ttnn.bfloat8_b) if h_fmt == "bfp8" else bf16(h)
        if down == "mrow":
            out = matmul_dest_steps(h, wdn, lofi, step_round)
        else:
            out = matmul_dest_steps(h, wdn, lofi, step_round, l1acc_block=6)
        return q_block(out, ttnn.bfloat8_b)

    def old(block_w=7, lofi=True, step_round=True, precise_pack=False):
        pk = (lambda t: q_block(t, ttnn.bfloat8_b)) if precise_pack else bfp8_truncate
        g = matmul_dest_steps(xq, wg, lofi, step_round, l1acc_block=block_w)
        u = matmul_dest_steps(xq, wu, lofi, step_round, l1acc_block=block_w)
        h = pk(bf16(silu(g)) * u)
        out = matmul_dest_steps(h, wdn, lofi, step_round, l1acc_block=block_w)
        return pk(out)

    rows = [
        ("fused: partials bfp8, down mrow (device M>=256 path)", fused()),
        ("fused: partials bfp8, down short/L1acc (device M<=128 path)", fused(down="short")),
        ("fused: partials bf16, down mrow", fused(partial_fmt="bf16")),
        ("fused: partials bf16, down short", fused(partial_fmt="bf16", down="short")),
        ("fused: partials bfp8, mrow, h bf16", fused(h_fmt="bf16")),
        ("fused: partials bfp8, mrow, no LoFi trunc", fused(lofi=False)),
        ("fused: partials bfp8, mrow, fp32 DEST (no step round)", fused(step_round=False)),
        ("fused: partials bf16, mrow, fp32 DEST", fused(partial_fmt="bf16", step_round=False)),
        ("old-like: L1acc block 7, truncating bfp8 pack", old()),
        ("old-like: block 7, precise pack", old(precise_pack=True)),
        ("old-like: block 4", old(block_w=4)),
        ("old-like: no LoFi trunc", old(lofi=False)),
    ]
    print(f"M={m} emb={emb} hidden={hidden} W={a.wdtype}")
    print(f"{'variant':<64} {'pcc':>10} {'rel_rms':>9}")
    for name, out in rows:
        pcc, rr = metrics(ref, out)
        print(f"{name:<64} {pcc:10.6f} {rr:9.5f}")


if __name__ == "__main__":
    main()
