# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Partial RoPE for the full-attention layers.

Two decisions worth stating, because both diverge from what the sibling packages do.

**Composed, not ``rotary_embedding_llama``.** That op rotates its whole last dim in the
pairwise-interleaved ("Meta") convention, so using it means permuting every q/k weight row at load
(``convert_hf_qkv_to_meta_format_partial``) and keeping that permutation consistent with the
per-head QK-norm gains. Qwen3.5's ``rotate_half`` is the half-split convention and its rotary
width, 64 of 256, is two whole tiles — and each half is exactly one tile. So the rotation composes
from tile-aligned slices, a concat and two multiply-adds, with no weight permutation anywhere.
Fewer ops than the borrowed path would have needed, and one entire class of bug removed.

**Plain per-chunk cos/sin, not the indexed whole-cache table.** Within a chunk the sequence is
sharded *contiguously* across the SP rows (row r owns chunk tokens ``[r*s_local, (r+1)*s_local)``),
so row r's positions are a contiguous range and its cos/sin are a plain slice of the global table.
``rotary_embedding_indexed`` exists for the block-cyclic *cache* layout, which the query side never
sees. The host upload is 640 KiB per chunk.

For a text-only prompt the interleaved mrope reduces to a plain partial RoPE table — the reference
runs the real 3-row mrope and ``test_text_only_mrope_reduces_to_plain_rope`` pins the equivalence,
so this builds the plain table.
"""

from __future__ import annotations

import torch

import ttnn

from ..config import MeshConfig
from ..reference.config import Qwen35TextConfig


def build_cos_sin_torch(cfg: Qwen35TextConfig, start_pos: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 cos/sin ``[1, 1, length, rotary_dim]`` for global positions ``[start_pos, start_pos+length)``."""
    dim = cfg.rotary_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    pos = torch.arange(start_pos, start_pos + length, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)[None, None]
    return emb.cos(), emb.sin()


class RotarySetup:
    """Builds this chunk's SP-sharded cos/sin. One instance per model; call per chunk."""

    def __init__(self, mesh_device, cfg: Qwen35TextConfig, mesh_config: MeshConfig) -> None:
        self.mesh_device = mesh_device
        self.cfg = cfg
        self.mesh_config = mesh_config

    def chunk_mats(self, start_pos: int, chunk_size: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos/sin for chunk ``[start_pos, start_pos+chunk_size)``, sharded on the SP rows.

        Row r receives the rows for its own token block, which is what makes a per-row rotation
        correct without any position bookkeeping inside the attention module.
        """
        sp = self.mesh_config.sp
        assert chunk_size % sp == 0, f"chunk {chunk_size} must split across sp={sp}"
        cos, sin = build_cos_sin_torch(self.cfg, start_pos, chunk_size)
        mapper = self.mesh_config.sequence_parallel(self.mesh_device, seq_dim=2)
        to_dev = lambda t: ttnn.from_torch(  # noqa: E731
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        return to_dev(cos), to_dev(sin)


def rotate_half(x: ttnn.Tensor) -> ttnn.Tensor:
    """``[-x2, x1]`` over the last dim — the half-split convention, not pairwise interleave."""
    half = x.shape[-1] // 2
    shape = list(x.shape)
    x1 = ttnn.slice(x, [0] * len(shape), shape[:-1] + [half])
    x2 = ttnn.slice(x, [0] * (len(shape) - 1) + [half], shape)
    neg_x2 = ttnn.neg(x2)
    x2.deallocate(True)
    out = ttnn.concat([neg_x2, x1], dim=-1)
    neg_x2.deallocate(True)
    x1.deallocate(True)
    return out


def apply_partial_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate the first ``cos.shape[-1]`` dims of each head; pass the rest through unchanged.

    ``x``: ``[1, n_heads, S, head_dim]``. ``cos``/``sin``: ``[1, 1, S, rotary_dim]``, broadcast
    across heads. When ``rotary_dim == head_dim`` the pass-through half is empty and this is a
    full rotation, so the same function covers both.
    """
    rotary_dim = cos.shape[-1]
    head_dim = x.shape[-1]
    assert rotary_dim <= head_dim
    shape = list(x.shape)

    x_rot = ttnn.slice(x, [0, 0, 0, 0], shape[:-1] + [rotary_dim])
    rotated = ttnn.add(
        ttnn.multiply(x_rot, cos),
        ttnn.multiply(rotate_half(x_rot), sin),
    )
    x_rot.deallocate(True)
    if rotary_dim == head_dim:
        return rotated
    x_pass = ttnn.slice(x, [0, 0, 0, rotary_dim], shape)
    out = ttnn.concat([rotated, x_pass], dim=-1)
    rotated.deallocate(True)
    x_pass.deallocate(True)
    return out
