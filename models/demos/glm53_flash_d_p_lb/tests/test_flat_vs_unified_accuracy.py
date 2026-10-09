# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Accuracy: flat_routed_expert vs unified_routed_expert_moe, each in GLM-5.3-Flash's current config, on identical
bfp4 weight bits and identical bf16 row-major input, against an fp32 host reference on those same bits.

One chip, GLM layer LAYER's real experts 0..35 (chip 0's). The weights are driven to a bfp4 fixed point first: ttnn's
host quantizer returns them unchanged in the W and the W^T tiling, so whatever tiling either op uses holds the same
bits (asserted). Ops:
  flat     the all-gather MoE path's call (tt/experts_ag.py): C++ op, capacity 8192, indexed mode (token_index into the
           tokens), clamped_silu, LoFi, x tilized to bfp8 on the relays, fp32 DEST gate/up and down, bfp8 h,
           row-major bf16 y
  unified  the unified path's call (tt/experts.py): ttnn.bringup.unified_routed_expert_moe, HiFi4, fp32 DEST, packer
           L1 acc, high_precision (bf16 x / h / y), ClampedSiluGlu
Reference: y = (silu(min(x Wg, 10)) * clamp(x Wu, +-10)) Wd in fp32 on the bf16 x and the exact bfp4 weights.

Inputs (x, bf16): the golden's real MoE input with its real routing (and x4), then synthetic x over skewed routing
(Dirichlet 0.8 over ~5120 routed rows, scattered token ids): gauss 1 / 0.05 / 4, student-t df 2.5, outlier channels
(8 channels x 100), log-normal row scales, 90 % sparse, DC-shifted. Per input and op: PCC, rel L2, scale coefficient
<y, ref> / <ref, ref>, per-row rel error p50 / p99 / max, and flat vs unified directly; plus the error a bfp8-rounded
x alone gives the reference (the flat op's input rounding), and flat with packer stochastic rounding (pack_stochastic_rounding, not
the model's default: separates the packer's ties-away bias from bfp8 precision). GLM_ACC_DISTS (comma list) selects inputs.
"""

import os

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.component import _step
from models.demos.common.bringup.testing.harness import component_golden, spec

S = spec()
LAYER = int(os.environ.get("GLM_ACC_LAYER", "4"))
E, NG, H, I, T, CAP = 36, 288, 4096, 2048, 5120, 8192
DISTS = (
    "real",
    "real_x4",
    "gauss1",
    "gauss0.05",
    "gauss4",
    "student_t",
    "outlier_ch",
    "row_scale",
    "sparse",
    "dc_shift",
)


def _q4(w):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _q8(w):
    return ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _fixed_bf4(w):
    """bfp4 values that the host quantizer returns unchanged in both the W and the W^T tiling."""
    w = w.bfloat16().float()
    for _ in range(8):
        w = _q4(_q4(w).T.contiguous()).T.contiguous()
        if torch.equal(_q4(w), w) and torch.equal(_q4(w.T.contiguous()).T, w):
            return w
    raise AssertionError("no common bfp4 fixed point")


def _act(g, u):
    return torch.nn.functional.silu(g.clamp(max=10.0)) * u.clamp(-10.0, 10.0)


def _metrics(y, ref):
    y, ref = y.double(), ref.double()
    a, b = y.flatten() - y.mean(), ref.flatten() - ref.mean()
    pcc = float((a @ b) / (a.norm() * b.norm()))
    rel = float((y - ref).norm() / ref.norm())
    coef = float((y * ref).sum() / (ref * ref).sum())
    rr = ((y - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-30)).float()
    q = torch.quantile(rr, torch.tensor([0.5, 0.99]))
    return dict(pcc=pcc, rel=rel, coef=coef, row_p50=float(q[0]), row_p99=float(q[1]), row_max=float(rr.max()))


def _synthetic_x(kind, g):
    if kind.startswith("gauss"):
        return torch.randn(T, H, generator=g) * float(kind[5:])
    if kind == "student_t":
        return torch.distributions.StudentT(2.5).sample((T, H)) * 0.5
    if kind == "outlier_ch":
        x = torch.randn(T, H, generator=g)
        x[:, torch.randperm(H, generator=g)[:8]] *= 100.0
        return x
    if kind == "row_scale":
        return torch.randn(T, H, generator=g) * torch.exp(torch.randn(T, 1, generator=g) * 1.5)
    if kind == "sparse":
        x = torch.randn(T, H, generator=g)
        return x * (torch.rand(T, H, generator=g) < 0.1)
    if kind == "dc_shift":
        return torch.randn(T, H, generator=g) + 0.5
    raise ValueError(kind)


def _skewed_routing(g):
    p = torch.distributions.Dirichlet(torch.ones(E) * 0.8).sample()
    c = torch.floor(p * 5120).long().clamp(max=CAP)
    return [torch.randperm(T, generator=g)[: int(c[e])] for e in range(E)]


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_vs_unified_accuracy(device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    from models.demos.glm53_flash_d_p.bringup import hooks
    from models.demos.glm53_flash_d_p.reference.weights import PackedExpert

    loader, _ = hooks._loader_cfg(S)
    W = []
    for e in range(E):
        gw, uw, dw = PackedExpert(loader, LAYER, e).weights(torch.float32)  # HF [out, in]
        W.append(tuple(_fixed_bf4(w.T.contiguous()) for w in (gw, uw, dw)))  # [H, I], [H, I], [I, H]
    print(f"[acc] layer {LAYER}: {E} experts at a common bfp4 fixed point (W and W^T tilings)", flush=True)

    flat = FlatRoutedExpert(
        device, [W], m=CAP, H=H, I=I, gids=[list(range(E))], n_global=NG, wdtype="bf4", act="clamped_silu", pin=1
    )
    to_dev = lambda w: ttnn.from_torch(  # noqa: E731
        w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    uw = {k: [to_dev(W[e][i]) for e in range(E)] for i, k in enumerate(("gate", "up", "down"))}
    for i, k in enumerate(("gate", "up", "down")):  # the unified op's bits are the reference's bits
        assert torch.equal(ttnn.to_torch(uw[k][0]).float(), W[0][i]), k
    ucfg = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    rm = lambda t, d: ttnn.from_torch(  # noqa: E731
        t, dtype=d, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gidx = rm(torch.arange(E, dtype=torch.int32), ttnn.uint32)

    real_x, real_toks = None, None
    g_, c_ = component_golden(S)
    ref_ = S.hooks().reference(S, layers=[LAYER], dtype=torch.float32)
    st = _step(ref_, LAYER, "experts")
    gl = g_.layer(c_, LAYER)
    xr, rr = (gl[i].float() for i in st.inputs)
    real_x = xr.reshape(-1, H)
    rr = rr.reshape(real_x.shape[0], -1)
    real_toks = [torch.nonzero(rr[:, e] != 0).reshape(-1) for e in range(E)]
    print(
        f"[acc] golden: {real_x.shape[0]} tokens, x std {float(real_x.std()):.4f}, "
        f"{sum(len(t) for t in real_toks)} rows routed to experts 0..35",
        flush=True,
    )

    gen = torch.Generator().manual_seed(0)
    rows_out = []
    for kind in os.environ.get("GLM_ACC_DISTS", ",".join(DISTS)).split(","):
        if kind.startswith("real"):
            x = real_x * (4.0 if kind == "real_x4" else 1.0)
            toks = real_toks
        else:
            x = _synthetic_x(kind, gen)
            toks = _skewed_routing(gen)
        x = x.bfloat16()
        counts = [len(t) for t in toks]
        pads = [-(-c // 32) * 32 for c in counts]
        offs = [sum(pads[:e]) for e in range(E)]
        rows = sum(pads)
        tidx = torch.zeros(1, rows, dtype=torch.int32)
        xb = torch.zeros(rows, H, dtype=torch.bfloat16)
        cnt = torch.zeros(1, NG, dtype=torch.int32)
        reg = torch.zeros(1, NG, dtype=torch.int32)
        for e in range(E):
            tidx[0, offs[e] : offs[e] + counts[e]] = toks[e].int()
            xb[offs[e] : offs[e] + counts[e]] = x[toks[e]]
            cnt[0, e], reg[0, e] = counts[e], offs[e]

        # reference (fp32 on the exact bits), and what bfp8-rounding x alone does to it
        xq8 = _q8(xb.float())
        ref, ref8 = torch.zeros(rows, H), torch.zeros(rows, H)
        for e in range(E):
            sl = slice(offs[e], offs[e] + counts[e])
            if counts[e]:
                Wg, Wu, Wd = W[e]
                for src, dst in ((xb.float()[sl], ref), (xq8[sl], ref8)):
                    dst[sl] = _act(src @ Wg, src @ Wu) @ Wd
        mask = torch.zeros(rows, dtype=torch.bool)
        for e in range(E):
            mask[offs[e] : offs[e] + counts[e]] = True

        x_dev, t_dev, c_dev, r_dev = (
            rm(x, ttnn.bfloat16),
            rm(tidx, ttnn.uint32),
            rm(cnt, ttnn.uint32),
            rm(reg, ttnn.uint32),
        )
        y_flat = flat(x_dev, c_dev, r_dev, token_index=t_dev, y_row_major=True, down_fp32=True)
        y_flat = ttnn.to_torch(y_flat).float().reshape(rows, H)
        y_srnd = flat(
            x_dev, c_dev, r_dev, token_index=t_dev, y_row_major=True, down_fp32=True, pack_stochastic_rounding=True
        )
        y_srnd = ttnn.to_torch(y_srnd).float().reshape(rows, H)
        # the unified op's buffer holds at least one expert's capacity of rows (GLM's dispatch buffer: ~43k rows)
        xb_dev = rm(torch.cat([xb, torch.zeros(max(0, CAP - rows), H, dtype=torch.bfloat16)]), ttnn.bfloat16)
        y_uni = ttnn.bringup.unified_routed_expert_moe(
            xb_dev,
            r_dev,
            c_dev,
            gidx,
            uw["gate"],
            uw["up"],
            uw["down"],
            max_dispatched_tokens_per_expert=CAP,
            compute_kernel_config=ucfg,
            activation=ttnn.bringup.RoutedExpertActivation.ClampedSiluGlu,
            high_precision=True,
        )
        y_uni = ttnn.to_torch(y_uni).float().reshape(-1, H)[:rows]
        for t_ in (x_dev, t_dev, c_dev, r_dev, xb_dev):
            ttnn.deallocate(t_)

        R = ref[mask]
        res = {
            "flat": _metrics(y_flat[mask], R),
            "flat srnd": _metrics(y_srnd[mask], R),
            "unified": _metrics(y_uni[mask], R),
            "bfp8 x only": _metrics(ref8[mask], R),
            "flat vs unified": _metrics(y_flat[mask], y_uni[mask]),
        }
        print(
            f"[acc] == {kind}: {int(mask.sum())} rows, x std {float(x.float().std()):.4f}, "
            f"ref rms {float(R.pow(2).mean().sqrt()):.4f}, gate/up clamp hits "
            f"{float(((xb.float()[mask][:256] @ W[0][0]).abs() > 10).float().mean()):.4%} (expert 0 weights, 256 rows)",
            flush=True,
        )
        for name, m_ in res.items():
            print(
                f"[acc]   {name:16s} pcc {m_['pcc']:.7f}  rel {m_['rel']:.5f}  coef {m_['coef']:.5f}  "
                f"row rel p50 {m_['row_p50']:.5f} p99 {m_['row_p99']:.5f} max {m_['row_max']:.5f}",
                flush=True,
            )
            rows_out.append((kind, name, m_))
    print("[acc] | input | op | PCC | rel L2 | scale coef | row rel p50 | p99 | max |")
    for kind, name, m_ in rows_out:
        print(
            f"[acc] | {kind} | {name} | {m_['pcc']:.6f} | {m_['rel']:.5f} | {m_['coef']:.5f} | "
            f"{m_['row_p50']:.5f} | {m_['row_p99']:.5f} | {m_['row_max']:.5f} |"
        )
