# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 q_a stem (reference/ds41_ref.py "q_a": q_resid = q_norm(fp8_linear(attn_norm, wq_a)) [S, 1280]) on the
SP x TP mesh. Adapted from models/demos/xing40_a4b_d_p/tt/q_a.py:TtQa and deepseek_v3_d_p/tt/mla/v4_attention_base.py:
TtV4AttentionBase._q_lora, with the model's fp8 act quant and an exact exchange.

Per chip (r, c) the input is attn_norm's column split [1, 1, S/SP, H/TP] fp32 TILE (tt/norm.py output):

    tt/quant.fp8_qdq            the reference fp8_linear's act_quant_dequant(x, 32), bit-exact: groups of 32 inside the
                                chip's H/TP columns (H/TP % 32 == 0), bf16 out (exact)
    TtAttnInProj                q(x) @ the chip's K rows of [wq_a | wkv]^T [H/TP, 1280 + 512] (bf16: e4m3 x 2^k is
                                bf16-exact), HiFi4 + fp32 accumulation, fp32 partial out. The FPU rounds each in-tile
                                partial dot to ~TF32 (DS41-16), so by default (settings ATTN_IN_MM = exact) the matmul
                                runs as tt/exact_mm.py:ExactMatmul (one product per tile-dot, fp32 SFPU adds): only the
                                summation order differs from the reference GEMM. ATTN_IN_MM = fpu: one ttnn.linear
    tt/common.py:tp_sum         all_gather over axis 1 + SFPU ttnn.add in mesh-column order (K-30: exact fp32 sum,
                                bit-identical on every chip of the row; the FPU in reduce_scatter / all_reduce reads fp32
                                at about TF32) -> [1, 1, S/SP, 1792] fp32, replicated over TP
    slice q [S/SP, 1280] -> tt/norm.py:TtDistributedRmsNorm(split=False) q_norm (fp32, replicated weight, eps 1e-20;
                                K-12: not ttnn.rms_norm on fp32)

Output q_resid [1, 1, S/SP, 1280] fp32 TILE, rows split over axis 0, replicated over axis 1 (wq_b and the indexer
need it whole). wkv rides in the same matmul and exchange (plan.md: one exchange per layer for the attention
projections; the kv step slices columns [1280, 1792) of ``TtAttnInProj``'s output). No host work in __call__.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, TP_AXIS, compute_config, tp_sum
from .exact_mm import ExactMatmul
from .norm import TtDistributedRmsNorm
from .quant import fp8_qdq
from .settings import settings


class TtAttnInProj:
    """The fp8 projections that read q(attn_norm), fused: [wq_a | wkv] (more column blocks may be appended).
    Call with q(attn_norm) [1, 1, S/SP, H/TP] bf16 TILE; returns [1, 1, S/SP, sum(widths)] fp32 TILE, the full sum
    over the hidden, replicated over TP. ``cols[name]`` = (start, end) of each block."""

    def __init__(self, mesh, weights: dict[str, torch.Tensor]):
        tp = mesh.shape[TP_AXIS]
        hidden = None
        cols, start, blocks = {}, 0, []
        for name, w in weights.items():
            n, k = w.shape  # [out, H] (nn.Linear layout)
            assert hidden in (None, k), (name, w.shape, hidden)
            assert n % 32 == 0, (name, w.shape)
            hidden = k
            cols[name] = (start, start + n)
            start += n
            blocks.append(w)
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        self.mesh, self.cols, self.width = mesh, cols, start
        self.ckc = compute_config(mesh)
        w_t = torch.cat(blocks, dim=0).t().contiguous()  # [H, width]
        assert torch.equal(w_t.bfloat16().float(), w_t.float()), "fp8 weights must be bf16-exact"
        self.exact = settings.get("ATTN_IN_MM") == "exact"
        if self.exact:  # ExactMatmul needs products of <= 11 significant bits: e4m3 x 2^k (3 mantissa bits) both sides
            assert not (w_t.float().contiguous().view(torch.int32) & 0xFFFFF).any(), "weights are not on the e4m3 grid"
            self.mm = ExactMatmul(mesh, hidden // tp, self.ckc, groups=4)
        # K (dim 2) split over the TP axis: chip column c holds rows [c H/TP, (c+1) H/TP), its attn_norm columns;
        # replicated over the SP axis.
        dims = (None, 2) if TP_AXIS == 1 else (2, None)
        self.weight = ttnn.from_torch(
            w_t.reshape(1, 1, hidden, self.width),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def __call__(self, xq: ttnn.Tensor) -> ttnn.Tensor:
        if self.exact:
            part = self.mm(xq, self.weight)
        else:
            part = ttnn.linear(xq, self.weight, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM)
        out = tp_sum(self.mesh, part)
        ttnn.deallocate(part)
        return out


class TtQa:
    """q_a of one layer. Call with attn_norm [1, 1, S/SP, H/TP] fp32 TILE; returns q_resid [1, 1, S/SP, q_lora_rank]
    fp32 TILE, replicated over TP. Does not consume its input."""

    def __init__(self, mesh, proj: TtAttnInProj, q_norm: torch.Tensor, eps: float):
        self.mesh, self.proj = mesh, proj
        self.q0, self.q1 = proj.cols["wq_a"]
        assert q_norm.numel() == self.q1 - self.q0, (q_norm.shape, proj.cols)
        self.norm = TtDistributedRmsNorm(mesh, q_norm.float(), eps, split=False)

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        xq = fp8_qdq(x, dtype=ttnn.bfloat16)
        p = self.proj(xq)
        ttnn.deallocate(xq)
        rows = p.shape[-2]
        q = ttnn.slice(p, [0, 0, 0, self.q0], [1, 1, rows, self.q1], memory_config=DRAM)
        ttnn.deallocate(p)
        y = self.norm(q)
        ttnn.deallocate(q)
        return y


def build_q_a(mesh, ck, cfg, layer: int) -> TtQa:
    from .kv import attn_in_weights  # the shared [wq_a | wkv] projection (wkv's rope rows in tt/kv.py's order)

    proj = TtAttnInProj(mesh, attn_in_weights(ck, cfg, layer))
    return TtQa(mesh, proj, ck.weight(f"layers.{layer}.attn.q_norm.weight", torch.float32), cfg.eps)
