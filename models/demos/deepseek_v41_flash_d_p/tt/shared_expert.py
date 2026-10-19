# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 shared expert (reference/ds41_ref.py "shared_expert": _swiglu with lin=mm and no routing weight;
inference Expert.forward) on the SP x TP mesh, tensor parallel by the intermediate (plan.md "Shared expert"). Adapted
from models/demos/xing40_a4b_d_p/tt/mlp.py:TtDenseMLP with the model's fp8 act quant and clamps.

    shared_out = w2 q(silu(min(w1 q(x), 10)) * clamp(w3 q(x), +-10))      q = fp8 act quant, blocks of 32 (tt/quant)

Per chip (r, c), I/TP = 2304 / TP intermediate columns (1152 on 2x2, 576 on 4x4 / 8x4 / 2x4: multiples of 32, so the
fp8 blocks of h never straddle chips):

    x       ffn_norm [1, 1, S/SP, H] fp32 TILE (rows over axis 0, replicated over axis 1: tt/norm.py gather=True)
    q(x)    tt/quant.fp8_qdq, bf16 out (bit-exact; may be passed in when the experts already made it)
    g | u   one ttnn.linear by [W1^T_c | W3^T_c] [H, 2 I/TP] (column-parallel; bf16 weights, exact: e4m3 x 2^k),
            HiFi4 + fp32 accumulation, fp32 out (K-25)
    h       silu(min(g, 10)) * clamp(u, +-10), fp32 on the SFPU
    q(h)    fp8_qdq over the chip's own I/TP columns (the reference's blocks of 32 along I)
    partial ttnn.linear by W2^T_c [I/TP, H] (row-parallel), fp32 out: [1, 1, S/SP, H] fp32 (DS41-25: keep fp32)
    sum     over the TP axis, keep the chip's hidden columns (setting EXPERTS_SUM: gather_add = all_gather + SFPU add,
            exact fp32, K-30; reduce_scatter = one CCL). In the assembled model the partial is added to the routed
            experts' partial before that one exchange (``partial`` + TtExperts.reduce).

Output shared_out [1, 1, S/SP, H/TP] fp32 TILE (rows over axis 0, hidden columns over axis 1). No host work in the
forward; weights are laid out once at load.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, TP_AXIS, compute_config, sum_cols
from .quant import GROUP, act_q
from .settings import settings

TILE = 32


def _mapper(mesh, dims):
    return ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)


class TtSharedExpert:
    def __init__(self, mesh, w1: torch.Tensor, w3: torch.Tensor, w2: torch.Tensor, swiglu_limit: float):
        """w1, w3 [I, H], w2 [H, I] (nn.Linear layout), fp32 values on the fp8 grid (bf16-exact)."""
        tp = mesh.shape[TP_AXIS]
        inter, hidden = w1.shape
        assert w3.shape == (inter, hidden) and w2.shape == (hidden, inter), (w1.shape, w3.shape, w2.shape)
        assert inter % (TILE * tp) == 0 and (inter // tp) % GROUP == 0, f"intermediate {inter} over TP {tp}"
        assert hidden % (TILE * tp) == 0, f"hidden {hidden} over TP {tp}"
        for w in (w1, w3, w2):
            assert torch.equal(w.bfloat16().float(), w.float()), "fp8 weights must be bf16-exact"
        self.mesh, self.tp, self.H, self.I = mesh, tp, int(hidden), int(inter)
        self.ip = self.I // tp  # the chip's intermediate share
        self.lim = float(swiglu_limit)
        self.sum_mode = settings.get("EXPERTS_SUM")
        self.cfg = compute_config(mesh)
        # [W1^T | W3^T] chip-major along the last dim: chip column c gets [W1^T[:, c ip:(c+1) ip] | W3^T[same]]
        w1t = w1.float().t().reshape(hidden, tp, self.ip)
        w3t = w3.float().t().reshape(hidden, tp, self.ip)
        w13 = torch.stack([w1t, w3t], dim=2).reshape(1, 1, hidden, 2 * self.I)
        self.w13 = ttnn.from_torch(
            w13.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=_mapper(mesh, (None, 3)),  # columns over axis 1, replicated over axis 0
        )
        self.w2 = ttnn.from_torch(
            w2.float().t().reshape(1, 1, inter, hidden).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=_mapper(mesh, (None, 2)),  # rows (the chip's intermediate) over axis 1
        )

    def partial(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """This chip's share of the output: x [1, 1, s, H] fp32 TILE (or its fp8 q-dq ``xq`` bf16, then x is unused)
        -> [1, 1, s, H] fp32 TILE partial (sum over the TP axis = shared_out)."""
        cfg, lim, ip = self.cfg, self.lim, self.ip
        own = xq is None
        if own:
            xq = act_q(x, dtype=ttnn.bfloat16)  # NUMERICS=native: a cast (x itself when ffn_norm wrote bf16)
        gu = ttnn.linear(xq, self.w13, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        if own and xq is not x:
            ttnn.deallocate(xq)
        s = gu.shape[-2]
        g = ttnn.slice(gu, [0, 0, 0, 0], [1, 1, s, ip], memory_config=DRAM)
        u = ttnn.slice(gu, [0, 0, 0, ip], [1, 1, s, 2 * ip], memory_config=DRAM)
        ttnn.deallocate(gu)
        if lim > 0:
            g2 = ttnn.minimum(g, lim, memory_config=DRAM)
            u2 = ttnn.clamp(u, -lim, lim, memory_config=DRAM)
            ttnn.deallocate(g)
            ttnn.deallocate(u)
            g, u = g2, u2
        sg = ttnn.silu(g, memory_config=DRAM)
        ttnn.deallocate(g)
        h = ttnn.multiply(sg, u, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(sg)
        ttnn.deallocate(u)
        hq = act_q(h, dtype=ttnn.bfloat16)
        ttnn.deallocate(h)
        y = ttnn.linear(hq, self.w2, dtype=ttnn.float32, compute_kernel_config=cfg, memory_config=DRAM)
        ttnn.deallocate(hq)
        return y

    def reduce(self, part: ttnn.Tensor) -> ttnn.Tensor:
        """Sum partials [1, 1, s, H] fp32 over axis 1, keep the chip's hidden columns [1, 1, s, H/TP]."""
        if self.tp == 1:
            return ttnn.clone(part, memory_config=DRAM)
        if self.sum_mode == "reduce_scatter":
            return ttnn.reduce_scatter(part, dim=3, cluster_axis=TP_AXIS, memory_config=DRAM)
        return sum_cols(self.mesh, part)

    def __call__(self, x: ttnn.Tensor, xq: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """shared_out [1, 1, s, H/TP] fp32 (rows over axis 0, hidden columns over axis 1)."""
        part = self.partial(x, xq)
        out = self.reduce(part)
        ttnn.deallocate(part)
        return out


def build_shared_expert(mesh, ck, cfg, layer: int) -> TtSharedExpert:
    """Layer ``layer``'s shared expert (fp8 weights dequantized at load, exact; reference Config.swiglu_limit)."""
    p = f"layers.{layer}.ffn.shared_experts."
    w = {n: ck.weight(p + n + ".weight", torch.float32) for n in ("w1", "w3", "w2")}
    return TtSharedExpert(mesh, w["w1"], w["w3"], w["w2"], cfg.swiglu_limit)
