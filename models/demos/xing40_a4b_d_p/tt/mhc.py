# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Xing4.0 mHC coefficients (attn_hc / ffn_hc, HF Xing4_0HyperConnection, reference/xing_ref.py:hc_weights) on the
4x2 mesh, all fp32:

    mix    = (flat [S, 4H] @ fn[24, 4H]^T) * rsqrt(mean(flat^2) + 1e-6)
    pre    = sigmoid(mix[:, :4] * s0 + b[:4])
    post   = 2 * sigmoid(mix[:, 4:8] * s1 + b[4:8])
    logits = clamp(mix[:, 8:] * s2 + b[8:], -30, 30)           (row-major 4x4, comb[i, j] at column 4 i + j)
    comb   = exp(logits - rowmax); 20 x (rows / (sum + 1e-6), columns / (sum + 1e-6))
    out    = [pre | post | comb] [S, 24]

Per chip (r, c) the streams are [1, 1, S/4, 4 x 1792] (tt/layout.py). The RMS has no weight, so its rsqrt commutes
with the linear (hy4_preview_d_p/tt/ihc.py:TtHcGates): each chip computes its partial mixes (its 7168 columns of
fn, permuted on the host to the chip's stream-column order) and its partial sum of squares, packed into one
[S/4, 32] fp32 tile row (mixes in columns 0-23, sum of squares in column 24); one ttnn.all_reduce over axis 1
completes both. The rest is redundant on both chips of a row and tiny ([S/4, 32] and [S/4, 16]).

Sinkhorn (composed): on the [S/4, 16] comb (entry (i, j) at column 4 i + j) every row / column sum is a matmul with
a 0/1 block matrix (deepseek_v3_d_p/tt/mhc/tt_mhc.py:_selection_row_col): row sums broadcast = M @ RB, column sums
= M @ CB. The row max uses three within-row cyclic column rotations (M @ ROT_k) and ttnn.maximum; the matmul reads
fp32 as tf32, so the max is rounded, but it is only a shift inside exp and cancels in the first row normalisation
(it only sizes hc_eps against the row sum, which stays >= ~1). All matmuls HiFi4 + fp32 accumulation (owner rule).

XING_HC_IMPL selects what runs after the all_reduce (and tt/collapse.py's collapse):
- fused (default, P.2b): ttnn.bringup.mhc_pre_xing (bring-up fork mhc_pre_ttnn): RMS scale, sigmoid gates, clamp,
  exp(L - rowmax) and the 20 x (row, column) Sinkhorn in one program, lane-wise fp32 on the SFPU; the collapse is the
  same entry with coefficients_given=True.
- composed: the op chain below (~100 small programs per call; the P.2 baseline, 39 ms per chunk per hc step).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.xing40_a4b_d_p.tt.settings import settings

from .layout import HC, streams_cols_to_chip_major

W = 32  # one tile row: 24 coefficient columns, the sum of squares in column SS_COL, the rest zero
NC = HC * HC  # comb entries
NG = 2 * HC + NC  # 24 coefficient columns
SS_COL = NG
HC_IMPLS = ("fused", "composed")


def hc_impl() -> str:
    mode = settings.get("HC_IMPL")
    assert mode in HC_IMPLS, f"XING_HC_IMPL={mode!r}, want one of {HC_IMPLS}"
    return mode


def hifi4(mesh):
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=getattr(ttnn.MathFidelity, settings.get("MATMUL_FIDELITY")),
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


def _replicated(mesh, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        t.float().reshape(1, 1, *t.shape[-2:]) if t.dim() >= 2 else t.float().reshape(1, 1, 1, -1),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def comb_matrices(n: int = HC) -> dict[str, torch.Tensor]:
    """[n*n, n*n] 0/1 matrices on the flattened row-major comb (entry (i, j) at column n i + j): RB (row sums
    broadcast), CB (column sums broadcast) and ROT1..ROT{n-1} ((M @ ROTk)[n i + j] = M[n i + (j + k) % n])."""
    nn_ = n * n
    rb, cb = torch.zeros(nn_, nn_), torch.zeros(nn_, nn_)
    for q in range(nn_):
        for p in range(nn_):
            rb[q, p] = float(q // n == p // n)
            cb[q, p] = float(q % n == p % n)
    out = {"RB": rb, "CB": cb}
    for k in range(1, n):
        rot = torch.zeros(nn_, nn_)
        for p in range(nn_):
            i, j = divmod(p, n)
            rot[n * i + (j + k) % n, p] = 1.0
        out[f"ROT{k}"] = rot
    return out


class TtHcWeights:
    """One mHC coefficient block (attn_hc or ffn_hc of one layer). Call with the per-chip streams
    [1, 1, S/4, 4 x H/2] fp32 TILE; returns [1, 1, S/4, 24] fp32 TILE (pre 4 | post 4 | comb 16), replicated over
    axis 1."""

    def __init__(
        self,
        mesh,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        hidden: int,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        iters: int = 20,
        clamp: tuple[float, float] = (-30.0, 30.0),
        impl: str | None = None,
    ):
        assert fn.shape == (NG, HC * hidden), fn.shape
        self.mesh, self.hidden = mesh, hidden
        self.impl = impl or hc_impl()
        self.inv_n = 1.0 / float(HC * hidden)
        self.norm_eps, self.hc_eps, self.iters = float(norm_eps), float(hc_eps), int(iters)
        self.clamp = (float(clamp[0]), float(clamp[1]))
        self.ckc = hifi4(mesh)
        # fn^T in the chip-major (c, j, k) row order, padded to 32 output columns; rows split over mesh columns
        # (axis 1), replicated over mesh rows (axis 0): chip (r, c) holds its [4 x H/2, 32] block.
        fn_t = torch.zeros(HC * hidden, W, dtype=torch.float32)
        fn_t[:, :NG] = streams_cols_to_chip_major(fn.float(), hidden).T
        self.fn_t = ttnn.from_torch(
            fn_t.reshape(1, 1, HC * hidden, W),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=(None, 2)),
        )
        scale = scale.float().reshape(-1)
        # Load-time Python scalars of the fused entry (runtime args of its program; no per-call host tensor work).
        self.scale3 = [float(v) for v in scale[:3]]
        self.base24 = [float(v) for v in base.float().reshape(-1)[:NG]]
        one_hot, a, b, mag = torch.zeros(W), torch.zeros(W), torch.zeros(W), torch.zeros(W)
        one_hot[SS_COL] = 1.0
        a[:HC], a[HC : 2 * HC], a[2 * HC : NG] = scale[0], scale[1], scale[2]
        b[:NG] = base.float()
        mag[:HC], mag[HC : 2 * HC] = 1.0, 2.0
        self.ss_onehot = _replicated(mesh, one_hot)
        self.scale_row = _replicated(mesh, a)
        self.base_row = _replicated(mesh, b)
        self.mag_row = _replicated(mesh, mag)
        self.mats = {k: _replicated(mesh, v) for k, v in comb_matrices(HC).items()}
        self.eps_bias = _replicated(mesh, torch.full((NC,), self.hc_eps))

    def _mm(self, a, b):
        return ttnn.matmul(
            a, b, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def _normalize(self, m, k):
        """m / (m @ K + hc_eps): one linear (the eps rides as its bias) and one divide."""
        d = ttnn.linear(
            m,
            self.mats[k],
            bias=self.eps_bias,
            dtype=ttnn.float32,
            compute_kernel_config=self.ckc,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.divide(m, d)
        ttnn.deallocate(d)
        ttnn.deallocate(m)
        return out

    def sinkhorn(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, S, 16] fp32 comb logits (after scale + base) -> comb [1, 1, S, 16] fp32. Consumes ``logits``."""
        lc = ttnn.clamp(logits, self.clamp[0], self.clamp[1])
        ttnn.deallocate(logits)
        mx = None
        for k in range(1, HC):
            r = self._mm(lc, self.mats[f"ROT{k}"])
            nxt = ttnn.maximum(lc if mx is None else mx, r)
            ttnn.deallocate(r)
            if mx is not None:
                ttnn.deallocate(mx)
            mx = nxt
        sh = ttnn.subtract(lc, mx)
        ttnn.deallocate(lc)
        ttnn.deallocate(mx)
        m = ttnn.exp(sh)
        ttnn.deallocate(sh)
        for _ in range(self.iters):
            m = self._normalize(m, "RB")
            m = self._normalize(m, "CB")
        return m

    def _fused(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Partial mixes (one HiFi4 fp32 matmul), sum x^2 into column 24 (mhc_pre_xing_pack, one exact pass over the
        streams), the [S/4, 32] all_reduce over axis 1, then the coefficients (mhc_pre_xing)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        mix = self._mm(x, self.fn_t)
        packed = ttnn.bringup.mhc_pre_xing_pack(mix, x, n=HC)
        ttnn.deallocate(mix)
        red = ttnn.all_reduce(packed, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(packed)
        hc, _ = ttnn.bringup.mhc_pre_xing(
            red,
            None,
            scale=self.scale3,
            base=self.base24,
            norm_width=float(HC * self.hidden),
            n=HC,
            norm_eps=self.norm_eps,
            hc_eps=self.hc_eps,
            sinkhorn_iters=self.iters,
            clamp_min=self.clamp[0],
            clamp_max=self.clamp[1],
        )
        ttnn.deallocate(red)
        return hc

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self.impl == "fused":
            return self._fused(x)
        s4 = x.shape[-2]
        dram = ttnn.DRAM_MEMORY_CONFIG
        # Partial mixes [S/4, 32] (columns 24-31 zero) and partial sum of squares [S/4, 1] over this chip's columns.
        mix = self._mm(x, self.fn_t)
        sq = ttnn.multiply(x, x, dtype=ttnn.float32, memory_config=dram)
        ss = ttnn.sum(sq, dim=-1, keepdim=True, compute_kernel_config=self.ckc, memory_config=dram)
        ttnn.deallocate(sq)
        ss_b = ttnn.multiply(ss, self.ss_onehot, dtype=ttnn.float32)
        ttnn.deallocate(ss)
        packed = ttnn.add(mix, ss_b, dtype=ttnn.float32)
        ttnn.deallocate(mix)
        ttnn.deallocate(ss_b)
        red = ttnn.all_reduce(packed, cluster_axis=1, memory_config=dram)
        ttnn.deallocate(packed)
        # rsqrt(sum / 4H + eps) down the row, then x scale + base.
        ssum = ttnn.slice(red, [0, 0, 0, SS_COL], [1, 1, s4, SS_COL + 1])
        t = ttnn.add(ttnn.multiply(ssum, self.inv_n), self.norm_eps)
        ttnn.deallocate(ssum)
        inv = ttnn.rsqrt(t)
        ttnn.deallocate(t)
        y = ttnn.multiply(red, inv, dtype=ttnn.float32)
        ttnn.deallocate(red)
        ttnn.deallocate(inv)
        y2 = ttnn.multiply(y, self.scale_row)
        ttnn.deallocate(y)
        y = ttnn.add(y2, self.base_row)
        ttnn.deallocate(y2)
        # pre / post: sigmoid x (1, 2); comb: clamp, exp(. - row max), Sinkhorn.
        sg = ttnn.sigmoid(y)
        g = ttnn.multiply(sg, self.mag_row)
        ttnn.deallocate(sg)
        pp = ttnn.slice(g, [0, 0, 0, 0], [1, 1, s4, 2 * HC])
        ttnn.deallocate(g)
        logits = ttnn.slice(y, [0, 0, 0, 2 * HC], [1, 1, s4, NG])
        ttnn.deallocate(y)
        comb = self.sinkhorn(logits)
        out = ttnn.concat([pp, comb], dim=-1, memory_config=dram)
        ttnn.deallocate(pp)
        ttnn.deallocate(comb)
        return out


def build_hc(mesh, loader, cfg, layer: int, step: str) -> TtHcWeights:
    """step: 'attn_hc' or 'ffn_hc' (model.layers.<layer>.<step>.hc_fn / hc_base / hc_scale)."""
    p = f"model.layers.{layer}.{step}.hc_"
    fn, base, scale = (loader.get(p + k).float() for k in ("fn", "base", "scale"))
    return TtHcWeights(
        mesh,
        fn,
        base,
        scale,
        cfg.hidden_size,
        norm_eps=cfg.rms_norm_eps,
        hc_eps=cfg.hc_eps,
        iters=cfg.hc_sinkhorn_iters,
        clamp=(cfg.mhc_h_res_clamp_min, cfg.mhc_h_res_clamp_max),
    )
