# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 window KV (reference/ds41_ref.py:_window_kv; inference Attention, the window-KV path):

    kv [S, 512] = act_quant_dequant(rope(kv_norm(fp8_linear(attn_norm, wkv)), positions start ..), 32)

on the SP x TP mesh, without the released code's fp8 quant-dequant steps (the device keeps bf16 there and computes
in it: agent rule 17; FIX.8 deleted the emulation). Per chip (r, c) the input is attn_norm's column split [1, 1, S/SP, H/TP] fp32 TILE:

    bf16 cast + tt/q_a.TtAttnInProj   attn_norm @ the chip's K rows of wkv^T, ATTN_FIDELITY + fp32 accumulation, the
                                  sum over TP (the q_a exchange: in the assembled model the kv step slices its
                                  columns of the projection q_a already made, ``from_proj``)
    kv [S/SP, 512] = p's wkv columns -> tt/norm.TtDistributedRmsNorm(split=False) kv_norm (eps 1e-20, replicated weight;
                                  NORM_IMPL=on: ttnn.bringup.rms_norm(input_window=) on p in place, P.18)
    tt/rope.rope_deinterleaved    the last 64 dims at absolute positions (theta 10000 for ratio-0 layers, 160000 +
                                  YaRN for ratio 1 / 2; cos / sin sliced from the load-time table, tt/rope.RopeTable),
                                  exact fp32 on the SFPU; the wkv rows (and kv_norm weight) of the rope dims are
                                  permuted at load to [even | odd] so the pairs are tile-aligned blocks
    bf16 cast                     (ATTN_GLUE=on: the rope's last ops write bf16, re-interleaved by a PairPerm matmul)
    ROPE_IMPL=on (P.17, default)  kv_norm writes bf16, then ONE stock rotary_embedding_indexed (tt/rope.rope_stock) on
                                  the 64-dim tail in place; wkv / kv_norm in checkpoint order (no permutation)

Output: window KV [1, 1, S/SP, 512] bf16 TILE, rows split over axis 0 (mesh row r = positions [start + r S/SP, ...)),
replicated over axis 1. The served-cache write (all_gather over axis 0, the owner chip's slice_write of
[actual_start, actual_end), pad-row zeroing before the ack) belongs to the device state, which the attention step
brings (BREADCRUMBS C.swa.kv). No host work in __call__ / from_proj.

Component deviation (components.yaml named rotary_embedding_llama): the SFPU composition (tt/rope.py) predates F128;
ROPE_IMPL=on (P.17, the precision-debt report's pick J3) replaces it with the stock rotary_embedding_indexed.
"""

from __future__ import annotations

import torch

import ttnn

from .ccl import dealloc
from .common import cast, gemm_input
from .norm import TtDistributedRmsNorm
from .q_a import TtAttnInProj
from .rope import RopeTable, deinterleave_perm, rope_deinterleaved, rope_deinterleaved_bf16, rope_stock, stock


def attn_in_weights(ck, cfg, layer: int) -> dict[str, torch.Tensor]:
    """The fused attention-input projection's weights {wq_a, wkv} (fp32 [out, H]); wkv's rope rows in the
    de-interleaved order TtWindowKv expects (q_a slices only wq_a's columns; ROPE_IMPL=on: checkpoint order)."""
    p = f"layers.{layer}.attn."
    perm = torch.arange(cfg.head_dim) if stock() else deinterleave_perm(cfg.head_dim, cfg.rope_dim)
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
        # ROPE_IMPL=on (P.17): checkpoint order, kv_norm writing the bf16 the stock rope reads
        self.stock = rope.stock
        perm = torch.arange(width) if self.stock else deinterleave_perm(width, rope_dim)
        dtype = ttnn.bfloat16 if self.stock else None
        self.norm = TtDistributedRmsNorm(mesh, kv_norm.float()[perm], eps, split=False, dtype=dtype)

    def from_proj(self, p: ttnn.Tensor, start: int) -> ttnn.Tensor:
        rows = p.shape[-2]
        # kv_norm on p's wkv columns (NORM_IMPL=on, P.18: read in place by the fork's input_window; off: a slice)
        kn = self.norm(p, window=(self.k0, self.k1))
        if self.proj.full_rows:  # ATTN_SP32=on (P.10): the chip's own S/(SP TP) rows, their cos / sin rows
            cos, sin = self.rope.rows(start, rows * self.mesh.shape[0] * self.mesh.shape[1], tp_split=True)
        else:
            cos, sin = self.rope.rows(start, rows * self.mesh.shape[0])
        if self.stock:  # ROPE_IMPL=on (P.17): one stock rotary op on the bf16 norm output
            out = rope_stock(kn, cos, sin, self.rope.trans)
            dealloc(kn)
            self.rope.free(cos, sin)
            return out
        if self.rope.pairs is not None:
            # ATTN_GLUE=on (P.15): rope + cast to bf16 + re-interleave as a PairPerm matmul
            out = rope_deinterleaved_bf16(kn, cos, sin, self.rope.pairs)
            dealloc(kn)
            self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
            return out
        kr = rope_deinterleaved(kn, cos, sin)
        dealloc(kn)
        self.rope.free(cos, sin)
        out = cast(kr, ttnn.bfloat16)
        if out is not kr:
            dealloc(kr)
        return out

    def __call__(self, x: ttnn.Tensor, start: int) -> ttnn.Tensor:
        xq = gemm_input(x)  # GEMM_INPUTS (FIX.7)
        p = self.proj(xq)
        if xq is not x:
            dealloc(xq)
        out = self.from_proj(p, start)
        dealloc(p)
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
