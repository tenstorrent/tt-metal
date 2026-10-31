# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 q_a stem (reference/ds41_ref.py "q_a": q_resid = q_norm(fp8_linear(attn_norm, wq_a)) [S, 1280]) on the
SP x TP mesh. Adapted from models/demos/xing40_a4b_d_p/tt/q_a.py:TtQa and deepseek_v3_d_p/tt/mla/v4_attention_base.py:
TtV4AttentionBase._q_lora, with an exact exchange.

Per chip (r, c) the input is attn_norm's column split [1, 1, S/SP, H/TP] fp32 TILE (tt/norm.py output):

    gemm_input                  the device's matmul input: GEMM_INPUTS (FIX.7, owner 2026-10-28) bfloat8_b, the
                                device's 8-bit format where fp8_linear's act_quant_dequant(x, 32) rounds to fp8
                                (bf16 = the path before FIX.7)
    TtAttnInProj                x @ the chip's K rows of [wq_a | wkv]^T [H/TP, 1280 + 512] (DENSE_WEIGHTS), one
                                ttnn.linear at ATTN_FIDELITY + fp32 accumulation, partial out (CCL_IMPL's dtype)
    tt/common.py:tp_sum         all_gather over axis 1 + SFPU ttnn.add in mesh-column order (K-30: exact fp32 sum,
                                bit-identical on every chip of the row; the FPU in reduce_scatter / all_reduce reads fp32
                                at about TF32) -> [1, 1, S/SP, 1792] fp32, replicated over TP
    q [S/SP, 1280] = p's wq_a columns -> tt/norm.py:TtDistributedRmsNorm(split=False) q_norm (fp32, replicated weight,
                                eps 1e-20; NORM_IMPL=on: ttnn.bringup.rms_norm(input_window=) on p in place, P.18)

Output q_resid [1, 1, S/SP, 1280] fp32 TILE, rows split over axis 0, replicated over axis 1 (wq_b and the indexer
need it whole). wkv rides in the same matmul and exchange (plan.md: one exchange per layer for the attention
projections; the kv step slices columns [1280, 1792) of ``TtAttnInProj``'s output). No host work in __call__.

ATTN_SP32=on (P.10, ``full_rows``; tt/model.py): attn_norm is the chip's own rows with whole hidden rows
[1, 1, S/(SP TP), H] bf16 (one all_to_all over TP before a local attn_norm), [wq_a | wkv] replicated: ONE matmul over
the whole K (fp32 DEST) writing the activation dtype, no exchange; q_resid / kv are then the chip's rows
[1, 1, S/(SP TP), 1280 / 512], what the attention projections (ATTN_SP32_PROJ) and the indexer read.
"""

from __future__ import annotations

import torch

import ttnn

from . import ccl, host_alloc
from . import mm_configs as mm
from .ccl import dealloc
from .common import (
    DRAM,
    TP_AXIS,
    act_dtype,
    attn_compute_config,
    bf16_activations,
    dense_dtype,
    gemm_dtype,
    gemm_input,
    tp_sum,
)
from .cores import from_torch
from .norm import TtDistributedRmsNorm
from .settings import settings


class TtAttnInProj:
    """The fp8 projections that read q(attn_norm), fused: [wq_a | wkv] (more column blocks may be appended).
    Call with attn_norm [1, 1, S/SP, H/TP] bf16 TILE; returns [1, 1, S/SP, sum(widths)] fp32 TILE, the full sum
    over the hidden, replicated over TP. ``cols[name]`` = (start, end) of each block."""

    def __init__(self, mesh, weights: dict[str, torch.Tensor], full_rows: bool = False):
        tp = mesh.shape[TP_AXIS]
        # ATTN_SP32=on (P.10, ``full_rows``): the input is the chip's own rows with whole hidden rows
        # [1, 1, S/(SP TP), H]: the weight replicated, one matmul over the whole K, no TP sum
        self.full_rows = full_rows
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
        self.attn_ckc = attn_compute_config(mesh)  # ATTN_FIDELITY (P.7)
        w_t = torch.cat(blocks, dim=0).t().contiguous()  # [H, width]
        assert torch.equal(w_t.bfloat16().float(), w_t.float()), "fp8 weights must be bf16-exact"
        # K (dim 2) split over the TP axis: chip column c holds rows [c H/TP, (c+1) H/TP), its attn_norm columns;
        # replicated over the SP axis.
        dims = (None, 2) if TP_AXIS == 1 else (2, None)
        if full_rows:
            dims = (None, None)
        # DENSE_WEIGHTS (P.13): bf16 (exact) or bfloat8_b (the checkpoint's 8-bit width)
        wdt = dense_dtype(w_t, "attn." + " | ".join(cols))
        self.weight = from_torch(
            w_t.reshape(1, 1, hidden, self.width),
            dtype=wdt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )

    def __call__(self, xq: ttnn.Tensor) -> ttnn.Tensor:
        if self.full_rows:
            # ATTN_SP32=on (P.10): the whole sum over the hidden in fp32 DEST, written in the activation dtype (bf16
            # under BF16_ACTIVATIONS=on: the GPU code's fp8 GEMM writes bf16), the chip's rows only
            return mm.linear(
                xq,
                self.weight,
                site="q_a",
                dtype=act_dtype(),
                compute_kernel_config=self.attn_ckc,
                memory_config=DRAM,
                out=host_alloc.mm_out(self.mesh, "q_a.p", xq, self.weight, act_dtype()),  # (HOST_ALLOC, P.15)
            )
        # CCL_IMPL=fabric_bf16 (P.6): the partial in bf16 for the bf16 reduce-scatter (tt/ccl.py:partial_dtype)
        pdt = ccl.partial_dtype(self.mesh) if settings.get("HC_TP_SUM") == "gather_add" else ttnn.float32
        part = mm.linear(
            xq, self.weight, site="q_a", dtype=pdt, compute_kernel_config=self.attn_ckc, memory_config=DRAM
        )
        # BF16_ACTIVATIONS=on (P.20): the exchanged [wq_a | wkv] output stays bf16 (the GPU code's fp8 GEMM writes
        # bf16); q_norm / kv_norm read it in place
        out = tp_sum(self.mesh, part, fp32=not bf16_activations())
        dealloc(part)
        return out


class TtQa:
    """q_a of one layer. Call with attn_norm [1, 1, S/SP, H/TP] fp32 TILE; returns q_resid [1, 1, S/SP, q_lora_rank]
    fp32 TILE, replicated over TP. Does not consume its input."""

    def __init__(self, mesh, proj: TtAttnInProj, q_norm: torch.Tensor, eps: float):
        self.mesh, self.proj = mesh, proj
        self.q0, self.q1 = proj.cols["wq_a"]
        assert q_norm.numel() == self.q1 - self.q0, (q_norm.shape, proj.cols)
        # BF16_ACTIVATIONS=on (P.20): q_resid bf16 (the GPU code's q_norm .to(dtype)), read by wq_b and the indexer
        self.norm = TtDistributedRmsNorm(mesh, q_norm.float(), eps, split=False, dtype=act_dtype())

    def with_proj(self, x: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """(q_resid, the fused projection's output p [1, 1, S/SP, proj.width] fp32): the assembled model's q_a step
        keeps p for the kv step (tt/kv.py:TtWindowKv.from_proj), one matmul and one exchange for both. The caller
        frees p."""
        # GEMM_INPUTS (FIX.7): bfloat8_b, the reference's fp8 act quant (HOST_ALLOC=on, P.15: into a load-time output)
        xq = gemm_input(x, host_alloc.out(self.mesh, "q_a.xq", x.shape, gemm_dtype()) if self.proj.full_rows else None)
        p = self.proj(xq)
        if xq is not x:
            dealloc(xq)
        # q_norm on p's wq_a columns (NORM_IMPL=on, P.18: read in place by the fork's input_window; off: a slice)
        y = self.norm(p, window=(self.q0, self.q1))
        return y, p

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        y, p = self.with_proj(x)
        dealloc(p)
        return y


def build_q_a(mesh, ck, cfg, layer: int) -> TtQa:
    from .kv import attn_in_weights  # the shared [wq_a | wkv] projection (wkv's rope rows in tt/kv.py's order)

    proj = TtAttnInProj(mesh, attn_in_weights(ck, cfg, layer))
    return TtQa(mesh, proj, ck.weight(f"layers.{layer}.attn.q_norm.weight", torch.float32), cfg.eps)
