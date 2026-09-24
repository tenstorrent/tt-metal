# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DIAGNOSTIC (reduced sequence, one layer, one TP column's heads — never a graded result): accuracy of
the fused ``ttnn.transformer.chunk_gated_delta_rule`` op on the real checkpoint's inputs.

The q/k/v/g/beta of one GDN layer are computed on the host in fp32 from the fp32 reference's own layer
input; the op and a CPU fp32 core then both consume the *same* bf16-rounded q/k/v, so any excess
error of the op over the CPU core is the op's own. Variants: compute-kernel configs.

    python models/demos/qwen_3_8_27b/scripts/diag_gdn_core.py LAYER [N]
"""

import os
import sys

import torch
import torch.nn.functional as F

import ttnn
from models.demos.qwen_3_8_27b.config import QWEN38
from models.demos.qwen_3_8_27b.reference import qwen3_8_ref as ref
from models.demos.qwen_3_8_27b.reference.checkpoint import CheckpointReader
from models.demos.qwen_3_8_27b.tt.gdn import GDN_OP_CHUNK, _const_tiles
from models.demos.qwen_3_8_27b.tt.mesh import close_mesh, open_mesh


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    L = int(sys.argv[1])
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    ref_h = torch.load(f"/tmp/qwen38_diag/ref_hidden_{N}.pt")
    sd = CheckpointReader().layer(L)
    layer = ref.DecoderLayer(QWEN38, L)
    layer.load_state_dict(sd)
    layer = layer.float()
    g_ = layer.linear_attn
    with torch.no_grad():
        h = layer.input_layernorm(ref_h[L - 1].float())
        qkv, _ = g_.causal_conv(g_.in_proj_qkv(h).transpose(1, 2))
        qkv = qkv.transpose(1, 2)
        kd = QWEN38.linear_key_dim
        nk, nv, dk = 4, 12, 128  # one TP column's heads (QWEN38_DIAG_COL, default 0)
        col = int(os.environ.get("QWEN38_DIAG_COL", "0"))
        qs, vs = col * nk * dk, col * nv * dk
        q = qkv[..., qs : qs + nk * dk].reshape(1, N, nk, dk)
        k = qkv[..., kd + qs : kd + qs + nk * dk].reshape(1, N, nk, dk)
        v = qkv[..., 2 * kd + vs : 2 * kd + vs + nv * dk].reshape(1, N, nv, dk)
        hs = slice(col * nv, (col + 1) * nv)
        beta = g_.in_proj_b(h)[..., hs].sigmoid()
        g = (-g_.A_log.float().exp() * F.softplus(g_.in_proj_a(h).float() + g_.dt_bias))[..., hs]
        print(
            f"column {col}: g min {g.min().item():.1f}, per-chunk(32) cumsum min {g.reshape(1, -1, 32, nv).sum(2).min().item():.1f}"
        )
        rep = lambda t: t.repeat_interleave(3, dim=2)  # noqa: E731
        o32, s32 = ref.chunk_gated_delta_rule(rep(q), rep(k), v, g, beta)  # fp32 inputs: the truth
        qb, kb, vb = (t.bfloat16().float() for t in (q, k, v))
        ob, sb = ref.chunk_gated_delta_rule(rep(qb), rep(kb), vb, g, beta)  # CPU core on bf16 inputs
    print(f"layer {L} N={N} (REDUCED) rel-err vs fp32 core:   o      state")
    print(f"  CPU fp32 core, bf16 q/k/v       {rel(ob, o32):.5f}  {rel(sb, s32):.5f}")
    mesh = open_mesh((8, 4))
    try:
        rp = ttnn.ReplicateTensorToMesh(mesh)
        up = lambda t, dt: ttnn.from_torch(
            t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rp
        )  # noqa: E731
        tq, tk, tv = (
            up(q.reshape(1, N, -1), ttnn.bfloat16),
            up(k.reshape(1, N, -1), ttnn.bfloat16),
            up(v.reshape(1, N, -1), ttnn.bfloat16),
        )
        tg, tb = up(g, ttnn.float32), up(beta, ttnn.float32)
        eye, tril, ones, masks = _const_tiles(mesh)
        variants = {
            "default kernel cfg": None,
            "HiFi4 fp32acc": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
                math_approx_mode=False,
            ),
            "HiFi4 fp32acc l1acc": ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True, math_approx_mode=False
            ),
        }
        # the composed fp32 core (tt/gdn_core.py), fp32 and bf16-rounded q/k/v
        from models.demos.qwen_3_8_27b.tt.gdn_core import ChunkDeltaRule

        core = ChunkDeltaRule(mesh)
        cap = {}
        cap_all = {}
        if os.environ.get("QWEN38_GDN_DEBUG") == "1":

            def _dbg(n, t):
                cap[n] = ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
                cap_all[n] = cap[n]
                print(f"    [{n}] max|x| {cap[n].abs().max().item():.4g}")

            core.debug = _dbg
        C = core.C
        hm = lambda t: t[0].permute(1, 0, 2).reshape(t.shape[2], N // C, C, -1)  # noqa: E731
        for name, (qq, kk, vv) in {"composed fp32 q/k/v": (q, k, v), "composed bf16 q/k/v": (qb, kb, vb)}.items():
            o, s = core(
                up(hm(rep(qq)), ttnn.float32),
                up(hm(rep(kk)), ttnn.float32),
                up(hm(vv), ttnn.float32),
                up(hm(g[..., None]), ttnn.float32),
                up(hm(beta[..., None]), ttnn.float32),
            )
            od = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(nv, N, dk).permute(1, 0, 2)[None]
            sd_ = ttnn.to_torch(ttnn.get_device_tensors(s)[0]).float().reshape(1, nv, dk, dk)
            print(f"  device [{name:22s}] {rel(od, o32):.5f}  {rel(sd_, s32):.5f}  nan={torch.isnan(od).any().item()}")
            if cap:  # rerun the sequential loop on the host from the device's own intermediates
                S = torch.zeros(nv, 1, dk, dk)
                outs = []
                for c in range(cap["vp"].shape[1]):
                    vn = cap["vp"][:, c : c + 1] - cap["kc"][:, c : c + 1] @ S
                    outs.append(cap["qg"][:, c : c + 1] @ S + cap["att"][:, c : c + 1] @ vn)
                    S = S * cap["el"][:, c : c + 1] + cap["kd"][:, c : c + 1].transpose(-1, -2) @ vn
                oh = torch.cat(outs, 1).reshape(nv, N, dk).permute(1, 0, 2)[None]
                print(
                    f"  host loop on device intermediates: {rel(oh, o32):.5f}  {rel(S.reshape(1, nv, dk, dk), s32):.5f}"
                )
                # host fp32 intermediates of the same algorithm
                Cc = core.C
                qh, kh, vh = hm(rep(qq)), hm(rep(kk)), hm(vv)
                gh, bh = hm(g[..., None]), hm(beta[..., None])
                qh = F.normalize(qh, dim=-1, eps=1e-6) * dk**-0.5
                kh = F.normalize(kh, dim=-1, eps=1e-6)
                gc = torch.cumsum(gh, 2)
                tril = torch.tril(torch.ones(Cc, Cc))
                dec = torch.exp(((gc - gc.transpose(-1, -2)) * tril + (1 - tril) * -1e4).clamp(min=-80)) * tril
                M = -(kh * bh) @ kh.transpose(-1, -2) * dec * (tril - torch.eye(Cc))
                P = torch.linalg.inv(torch.eye(Cc) - M)
                href = {
                    "gc": gc,
                    "decay": dec,
                    "P": P,
                    "vp": P @ (vh * bh),
                    "kc": P @ (kh * bh * torch.exp(gc.clamp(min=-80))),
                    "att": (qh @ kh.transpose(-1, -2)) * dec,
                    "qg": qh * torch.exp(gc.clamp(min=-80)),
                    "kd": kh * torch.exp((gc[:, :, -1:] - gc).clamp(min=-80)),
                    "el": torch.exp(gc[:, :, -1:].clamp(min=-80)),
                }
                print(
                    "   "
                    + "  ".join(
                        f"{n}:{rel(cap_all[n].reshape(href[n].shape), href[n]):.4f}" for n in href if n in cap_all
                    )
                )
                cap.clear()
                cap_all.clear()
        # 4D (non-flat) inputs with q/k L2-normalised on the host in fp32 (the op then skips in-kernel norm)
        qn = F.normalize(q, dim=-1, eps=1e-6)
        kn = F.normalize(k, dim=-1, eps=1e-6)
        q4, k4, v4 = up(qn, ttnn.bfloat16), up(kn, ttnn.bfloat16), up(v, ttnn.bfloat16)
        for name, kw in {
            "4D host-l2 C32": dict(chunk_size=32),
            "4D host-l2 C64": dict(chunk_size=64),
            "flat C32 no-mcast": dict(chunk_size=32, flat=True, use_mcast=False),
        }.items():
            flat = kw.pop("flat", False)
            args = (tq, tk, tv) if flat else (q4, k4, v4)
            try:
                o, s = ttnn.transformer.chunk_gated_delta_rule(
                    *args,
                    tg,
                    tb,
                    scale=dk**-0.5,
                    output_final_state=True,
                    **kw,
                )
                od = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(1, N, nv, dk)
                sd_ = ttnn.to_torch(ttnn.get_device_tensors(s)[0]).float().reshape(1, nv, dk, dk)
                print(f"  device op [{name:22s}] {rel(od, o32):.5f}  {rel(sd_, s32):.5f}")
            except Exception as e:  # noqa: BLE001
                print(f"  device op [{name:22s}] FAIL {str(e)[:160]}")
        for name, ckc in variants.items():
            try:
                o, s = ttnn.transformer.chunk_gated_delta_rule(
                    tq,
                    tk,
                    tv,
                    tg,
                    tb,
                    scale=dk**-0.5,
                    output_final_state=True,
                    chunk_size=GDN_OP_CHUNK,
                    eye=eye,
                    tril=tril,
                    ones=ones,
                    masks=masks,
                    compute_kernel_config=ckc,
                )
                od = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float().reshape(1, N, nv, dk)
                sd_ = ttnn.to_torch(ttnn.get_device_tensors(s)[0]).float().reshape(1, nv, dk, dk)
                print(f"  device op [{name:22s}] {rel(od, o32):.5f}  {rel(sd_, s32):.5f}")
            except Exception as e:  # noqa: BLE001
                print(f"  device op [{name:22s}] FAIL {str(e)[:120]}")
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
