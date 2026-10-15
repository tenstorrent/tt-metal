# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 window KV (reference/ds41_ref.py:_window_kv; inference Attention, the window-KV path):

    kv [S, 512] = act_quant_dequant(rope(kv_norm(fp8_linear(attn_norm, wkv)), positions start ..), 32)

on the SP x TP mesh. Per chip (r, c) the input is attn_norm's column split [1, 1, S/SP, H/TP] fp32 TILE:

    tt/quant.fp8_qdq + tt/q_a.TtAttnInProj   q(attn_norm) @ the chip's K rows of wkv^T, HiFi4 + fp32 accumulation
                                  (ExactMatmul by default: the FPU's ~TF32 in-tile partials would flip ~0.8 % of
                                  the output's fp8 values, DS41-16), exact all_gather + SFPU add over TP (the q_a exchange: in the assembled model the
                                  kv step slices its columns of the projection q_a already made, ``from_proj``)
    slice kv [S/SP, 512] -> tt/norm.TtDistributedRmsNorm(split=False) kv_norm (fp32, eps 1e-20, replicated weight)
    tt/rope.rope_deinterleaved    the last 64 dims at absolute positions (theta 10000 for ratio-0 layers, 160000 +
                                  YaRN for ratio 1 / 2; cos / sin sliced from the load-time table, tt/rope.RopeTable),
                                  exact fp32 on the SFPU; the wkv rows (and kv_norm weight) of the rope dims are
                                  permuted at load to [even | odd] so the pairs are tile-aligned blocks
    tt/quant.fp8_qdq              e4m3 RNE per 32 with ue8m0 scales, bf16 out (exact)

Output: window KV [1, 1, S/SP, 512] bf16 TILE, rows split over axis 0 (mesh row r = positions [start + r S/SP, ...)),
replicated over axis 1. The served-cache write (all_gather over axis 0, the owner chip's slice_write of
[actual_start, actual_end), pad-row zeroing before the ack) belongs to the device state, which the attention step
brings (BREADCRUMBS C.swa.kv). No host work in __call__ / from_proj.

Component deviation (components.yaml named rotary_embedding_llama): that op multiplies on the FPU, which reads fp32
at about TF32 (K-30), and the fp8 rounding after it turns such last-bit errors into whole e4m3 steps (DS41-1); the
SFPU composition is exact and keeps the reference's arithmetic.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM
from .norm import TtDistributedRmsNorm
from .q_a import TtAttnInProj
from .quant import fp8_qdq
from .rope import RopeTable, deinterleave_perm, rope_deinterleaved


def attn_in_weights(ck, cfg, layer: int) -> dict[str, torch.Tensor]:
    """The fused attention-input projection's weights {wq_a, wkv} (fp32 [out, H]); wkv's rope rows in the
    de-interleaved order TtWindowKv expects (q_a slices only wq_a's columns)."""
    p = f"layers.{layer}.attn."
    perm = deinterleave_perm(cfg.head_dim, cfg.rope_dim)
    return {
        "wq_a": ck.weight(p + "wq_a.weight", torch.float32),
        "wkv": ck.weight(p + "wkv.weight", torch.float32)[perm].contiguous(),
    }


class TtWindowKv:
    """kv step of one layer. ``__call__(x, start)``: attn_norm [1, 1, S/SP, H/TP] fp32 -> window KV [1, 1, S/SP, 512]
    bf16, replicated over TP. ``from_proj(p, start)``: the same from the fused projection's output (columns
    ``proj.cols['wkv']``). Does not consume its input."""

    def __init__(self, mesh, proj: TtAttnInProj, kv_norm: torch.Tensor, eps: float, rope: RopeTable, rope_dim: int):
        self.mesh, self.proj, self.rope = mesh, proj, rope
        self.k0, self.k1 = proj.cols["wkv"]
        width = self.k1 - self.k0
        assert kv_norm.numel() == width and rope.half * 2 == rope_dim, (kv_norm.shape, width, rope.half, rope_dim)
        perm = deinterleave_perm(width, rope_dim)
        self.norm = TtDistributedRmsNorm(mesh, kv_norm.float()[perm], eps, split=False)

    def from_proj(self, p: ttnn.Tensor, start: int) -> ttnn.Tensor:
        rows = p.shape[-2]
        kv = ttnn.slice(p, [0, 0, 0, self.k0], [1, 1, rows, self.k1], memory_config=DRAM)
        kn = self.norm(kv)
        ttnn.deallocate(kv)
        cos, sin = self.rope.rows(start, rows * self.mesh.shape[0])
        kr = rope_deinterleaved(kn, cos, sin)
        for t in (kn, cos, sin):
            ttnn.deallocate(t)
        out = fp8_qdq(kr, dtype=ttnn.bfloat16)
        ttnn.deallocate(kr)
        return out

    def __call__(self, x: ttnn.Tensor, start: int) -> ttnn.Tensor:
        xq = fp8_qdq(x, dtype=ttnn.bfloat16)
        p = self.proj(xq)
        ttnn.deallocate(xq)
        out = self.from_proj(p, start)
        ttnn.deallocate(p)
        return out


def rope_freqs(cfg, layer: int) -> torch.Tensor:
    """The layer's [R/2] fp32 rope frequencies, as the reference picks them (DS41Reference.freqs / _rot)."""
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import yarn_freqs

    r = cfg.rope
    if cfg.ratio(layer):
        return yarn_freqs(cfg.rope_dim, r["compress_theta"], r["original"], r["factor"], r["beta_fast"], r["beta_slow"])
    return yarn_freqs(cfg.rope_dim, r["theta"], 0, r["factor"], r["beta_fast"], r["beta_slow"])


def build_kv(mesh, ck, cfg, layer: int, max_seq: int, rope: RopeTable | None = None) -> TtWindowKv:
    """The kv step with its own projection (component tests); the assembled model passes the shared projection and
    the shared rope table instead."""
    w = attn_in_weights(ck, cfg, layer)
    proj = TtAttnInProj(mesh, {"wkv": w["wkv"]})
    rope = rope or RopeTable(mesh, rope_freqs(cfg, layer), max_seq)
    kv_norm = ck.weight(f"layers.{layer}.attn.kv_norm.weight", torch.float32)
    return TtWindowKv(mesh, proj, kv_norm, cfg.eps, rope, cfg.rope_dim)
