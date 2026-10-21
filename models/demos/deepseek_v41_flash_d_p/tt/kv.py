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
    slice kv [S/SP, 512] -> tt/norm.TtDistributedRmsNorm(split=False) kv_norm (fp32, eps 1e-20, replicated weight)
    tt/rope.rope_deinterleaved    the last 64 dims at absolute positions (theta 10000 for ratio-0 layers, 160000 +
                                  YaRN for ratio 1 / 2; cos / sin sliced from the load-time table, tt/rope.RopeTable),
                                  exact fp32 on the SFPU; the wkv rows (and kv_norm weight) of the rope dims are
                                  permuted at load to [even | odd] so the pairs are tile-aligned blocks
    bf16 cast                     (ATTN_GLUE=on: the rope's last ops write bf16, re-interleaved by a PairPerm matmul)

Output: window KV [1, 1, S/SP, 512] bf16 TILE, rows split over axis 0 (mesh row r = positions [start + r S/SP, ...)),
replicated over axis 1. The served-cache write (all_gather over axis 0, the owner chip's slice_write of
[actual_start, actual_end), pad-row zeroing before the ack) belongs to the device state, which the attention step
brings (BREADCRUMBS C.swa.kv). No host work in __call__ / from_proj.

Component deviation (components.yaml named rotary_embedding_llama): the SFPU composition (tt/rope.py) predates F128;
replacing it with the stock op is the precision-debt report's pick J3.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, cast
from .norm import TtDistributedRmsNorm
from .q_a import TtAttnInProj
from .rope import RopeTable, deinterleave_perm, rope_deinterleaved, rope_deinterleaved_bf16


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
        if self.rope.pairs is not None:
            # ATTN_GLUE=on (P.15): rope + cast to bf16 + re-interleave as a PairPerm matmul
            out = rope_deinterleaved_bf16(kn, cos, sin, self.rope.pairs)
            ttnn.deallocate(kn)
            self.rope.free(cos, sin)  # INDEXER_GLUE=on: the chunk's memoized rows stay
            return out
        kr = rope_deinterleaved(kn, cos, sin)
        ttnn.deallocate(kn)
        self.rope.free(cos, sin)
        out = cast(kr, ttnn.bfloat16)
        if out is not kr:
            ttnn.deallocate(kr)
        return out

    def __call__(self, x: ttnn.Tensor, start: int) -> ttnn.Tensor:
        xq = cast(x, ttnn.bfloat16)
        p = self.proj(xq)
        if xq is not x:
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
