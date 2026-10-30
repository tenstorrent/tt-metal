# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 mHC mixes (attn_hc / ffn_hc; reference/ds41_ref.py:hc_mixes, inference Block.hc_pre's coefficient
part) on the SP x TP mesh, all fp32:

    flat   = the 4 streams of a token [4H]
    mixes  = (flat @ hc_fn^T) * rsqrt(mean(flat^2) + norm_eps)                    [S, 24]
    pre    = sigmoid(mixes[:4] a0 + b[:4]) + hc_eps
    post   = 2 sigmoid(mixes[4:8] a1 + b[4:8])
    comb   = row softmax(mixes[8:] a2 + b[8:]) + hc_eps, column norm, 19 x (row, column) norms   (eps in each sum)
    out    = [pre 4 | post 4 | comb 16 (row j = input stream, column k = output)]  [S, 24]

Per chip (r, c) the streams are [1, 1, S/SP, 4 x H/TP] (tt/layout.py). The RMS has no weight, so its rsqrt commutes
past the linear: each chip computes its partial mixes (one HiFi4 fp32-acc matmul with its 4 x H/TP rows of hc_fn^T,
permuted on the host to the chip's stream-column order, padded to 32 output columns, rounded at load to the 9
mantissa bits HiFi4 multiplies of in1, DS41-42) and the exact partial sum of
squares (ttnn.bringup.mhc_pre_xing_pack, SFPU fp32) in column 24 of the same [S/SP, 32] row; one exchange over axis 1
(all_gather + ttnn.add, exact, K-30; tt/common.py:tp_sum) completes both. Then, on every chip of the row (the
output is replicated over TP), setting HC_SINKHORN:
  sfpu_fp32 (default) = ttnn.bringup.mhc_pre_xing(deepseek=True): one program from the exchanged row, mixes x
    rsqrt(sum / 4H + norm_eps), the sigmoids and the Sinkhorn in DeepSeek's order (reference/kernels.py:sinkhorn_split)
    lane-wise fp32 on the SFPU (FIX.4);
  fpu = the composed rsqrt chain, then ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn (DeepSeek-V4's kernel,
    K-20; logits capped at 80 instead of the row max), whose FPU steps read fp32 at TF32 (DS41-40).
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import build_consts

from . import ccl, host_alloc
from . import mm_configs as mm
from .ccl import dealloc
from .common import DRAM, TP_AXIS, compute_config, fpu_weight, tp_sum
from .layout import HC, streams_cols_to_chip_major
from .settings import settings

W = 32  # one tile row: 24 coefficient columns, the sum of squares in column SS_COL, the rest zero
NC = HC * HC
NG = 2 * HC + NC  # 24
SS_COL = NG


class TtHcMixes:
    """One mHC coefficient block (attn_hc or ffn_hc of one layer). Call with the per-chip streams
    [1, 1, S/SP, 4 x H/TP] fp32 TILE; returns [1, 1, S/SP, 24] fp32 TILE (pre | post | comb), replicated over
    axis 1. Does not consume its input."""

    out_sites = (None, None)  # FORK_OUT_BUFFERS=on (P.16, tt/host_alloc.reserve_forks): fused()'s hc / y sites

    def __init__(self, mesh, fn, scale, base, hidden: int, norm_eps: float, hc_eps: float, iters: int):
        fn = fn.float()
        assert fn.shape == (NG, HC * hidden), fn.shape
        tp = mesh.shape[1]
        assert hidden % (tp * 32) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        self.mesh, self.hidden = mesh, hidden
        self.inv_n = 1.0 / float(HC * hidden)
        self.norm_eps, self.hc_eps, self.iters = float(norm_eps), float(hc_eps), int(iters)
        self.ckc = compute_config(mesh)
        # MHC_FUSED=on: the residual's mhc_post(stats_weight=fn_t) multiplies x' @ fn^T at the same fidelity (fp32 DEST)
        self.post_cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        )
        # fn^T in the chip-major (c, j, k) row order, padded to 32 output columns; rows split over mesh columns
        # (axis 1), replicated over mesh rows (axis 0): chip (r, c) holds its [4 x H/TP, 32] block.
        fn_t = torch.zeros(HC * hidden, W, dtype=torch.float32)
        fn_t[:, :NG] = streams_cols_to_chip_major(fn, hidden, tp).T
        self.fn_t = ttnn.from_torch(
            fpu_weight(fn_t).reshape(1, 1, HC * hidden, W),  # in1 = srcA: 9 mantissa bits at HiFi4, RNE (DS41-42)
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
        )
        self.sinkhorn = settings.get("HC_SINKHORN")
        self.gather = ccl.glue_fold()  # GLUE_FOLD=on (P.19): the fused step's exchange as one all_gather
        # sfpu_fp32: the fork's scalars (runtime args of one program shared by every layer; fp32 values, exact)
        self.scale = [float(v) for v in scale.float().reshape(-1)]
        self.base = [float(v) for v in base.float().reshape(-1)]
        self.consts = None
        if self.sinkhorn == "sfpu_fp32":
            return
        # fpu: the Sinkhorn kernel's constants (SEL_pre / post / comb with the scales folded in, the bases, RB / CB).
        consts = build_consts(SimpleNamespace(n=HC), scale.float().reshape(-1), base.float().reshape(-1))
        self.consts = ttnn.from_torch(
            consts.reshape(8, W, W).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def partial_row(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """This chip's partial row [1, 1, R, 32]: x @ fn^T (its hidden slice) in columns 0..23, its sum of x^2 in
        column 24 (what ttnn.bringup.mhc_post(stats_weight=fn_t) emits in the residual's pass, MHC_FUSED=on)."""
        mix = mm.matmul(
            x, self.fn_t, site="hc.fn", dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=HC)  # column 24 = this chip's sum of x^2 (exact)
        dealloc(mix)
        return packed

    def fused(self, x: ttnn.Tensor, pre: ttnn.Tensor, row: ttnn.Tensor | None = None):
        """MHC_FUSED=on: (hc [1, 1, R, 24], y [1, 1, R, H/TP]) = this step's coefficients and the collapse
        y = sum_n pre[:, n] x_n with the GIVEN pre (single-pass mHC, DS41-3: the pending / previous sublayer's), from
        one exchange and ONE ttnn.bringup.mhc_pre_xing(collapse_pre=pre) program. ``row``: this chip's partial row
        when the previous residual already emitted it (mhc_post stats); None = compute it here (matmul + pack).
        Consumes ``row``, not x / pre."""
        assert self.sinkhorn == "sfpu_fp32", "MHC_FUSED=on needs HC_SINKHORN=sfpu_fp32"
        if row is None:
            row = self.partial_row(x)
        tp = self.mesh.shape[TP_AXIS]
        partials = {}
        if self.gather and tp > 1:
            # GLUE_FOLD=on (P.19): ONE all_gather of the chips' fp32 rows over TP -> [1, TP, R, 32]; mhc_pre_xing adds
            # them in mesh-column order on the SFPU (exact fp32: _gather_add's sum; no bf16 RS, no casts)
            red = ccl.all_gather(row, 1, TP_AXIS, site="mhc.row")  # (HOST_ALLOC: load-time output)
            partials = {"partials": tp}
        else:
            red = tp_sum(self.mesh, row)
        dealloc(row)
        hs, ys = self.out_sites  # FORK_OUT_BUFFERS=on (P.16): load-time outputs of an in-layer hc / y
        rows = x.shape[-2]
        hco = host_alloc.fork_out(self.mesh, hs, (1, 1, rows, HC * (HC + 2)), ttnn.float32)
        yo = host_alloc.fork_out(self.mesh, ys, (1, 1, rows, x.shape[-1] // HC), x.dtype)
        hc, y = ttnn.bringup.mhc_pre_xing(
            red,
            x,
            scale=self.scale,
            base=self.base,
            norm_width=float(HC * self.hidden),
            n=HC,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.iters,
            deepseek=True,
            collapse_pre=pre,
            hc_output=hco,
            y_output=yo,
            **partials,
        )
        dealloc(red)
        # (the kept objects: the op returns new wrappers of their buffers, DS41-149)
        return (hc if hco is None else hco), (y if yo is None else yo)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        rows = x.shape[-2]
        mix = mm.matmul(
            x, self.fn_t, site="hc.fn", dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM
        )
        packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=HC)  # column 24 = this chip's sum of x^2 (exact)
        dealloc(mix)
        red = tp_sum(self.mesh, packed)
        dealloc(packed)
        if self.sinkhorn == "sfpu_fp32":
            hc, _ = ttnn.bringup.mhc_pre_xing(
                red,
                None,
                scale=self.scale,
                base=self.base,
                norm_width=float(HC * self.hidden),
                n=HC,
                norm_eps=self.norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.iters,
                deepseek=True,
            )
            dealloc(red)
            return hc
        ss = ttnn.slice(red, [0, 0, 0, SS_COL], [1, 1, rows, SS_COL + 1])
        ms = ttnn.multiply(ss, self.inv_n, dtype=ttnn.float32)
        dealloc(ss)
        t = ttnn.add(ms, self.norm_eps, dtype=ttnn.float32)
        dealloc(ms)
        rs = ttnn.rsqrt(t)
        dealloc(t)
        mixes = ttnn.multiply(red, rs, dtype=ttnn.float32, memory_config=DRAM)  # column 24 is ignored below
        dealloc(red)
        dealloc(rs)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.consts, HC, self.iters, self.hc_eps
        )
        dealloc(mixes)
        out = ttnn.concat([pre, post, comb], dim=-1, memory_config=DRAM)
        for t_ in (pre, post, comb):
            dealloc(t_)
        return ttnn.reshape(out, [1, 1, rows, NG])


def build_hc(mesh, ck, cfg, layer: int, step: str) -> TtHcMixes:
    """step: 'attn_hc' or 'ffn_hc' (checkpoint layers.<layer>.hc_attn_* / hc_ffn_*, fp32)."""
    kind = {"attn_hc": "attn", "ffn_hc": "ffn"}[step]
    p = f"layers.{layer}.hc_{kind}_"
    fn, scale, base = (ck.weight(p + k, torch.float32) for k in ("fn", "scale", "base"))
    return TtHcMixes(mesh, fn, scale, base, cfg.dim, cfg.eps, cfg.hc_eps, cfg.sinkhorn_iters)
