# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Vision rotary tables and the raster to merge-block permutation that lets qwen36's PatchMerger be reused."""

from __future__ import annotations

import torch

from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

# PaddleOCR's vision rotary embedding is built with theta 10000 over head_dim//2
# (PaddleOCRVisionRotaryEmbedding, modeling_paddleocr_vl.py:103).
VISION_ROPE_THETA = 10000.0


def raster_position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    """Raster (row, col) per patch; a local copy of transformers' get_vision_position_ids(grid_thw, 1)."""
    out = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def block_permutation(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Raster index of each token in projector merge-block order (the projector's reshape+transpose gather)."""
    m = spatial_merge_size
    out = []
    offset = 0
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        assert h % m == 0 and w % m == 0, f"grid {h}x{w} not divisible by merge size {m}"
        idx = torch.arange(t * h * w).reshape(t, h // m, m, w // m, m).transpose(2, 3).reshape(-1)
        out.append(idx + offset)
        offset += t * h * w
    return torch.cat(out, dim=0)


def vision_rope_tables(position_ids: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """HF-layout cos/sin matching PaddleOCRVisionRotaryEmbedding; the device needs meta_rope_tables instead."""
    dim = head_dim // 2
    inv_freq = 1.0 / (VISION_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = (position_ids.float().unsqueeze(-1) * inv_freq).flatten(1)  # [N, 2 * dim/2] = [N, head_dim/2]
    emb = freqs.repeat(1, 2)  # [N, head_dim]
    return emb.cos(), emb.sin()


def meta_rope_tables(position_ids: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved cos/sin for rotary_embedding_llama; must pair with the q/k permute in weight_mapping."""
    cos_hf, sin_hf = vision_rope_tables(position_ids, head_dim)
    return convert_rope_style_hf_to_meta(cos_hf, sin_hf)


def pad_to_bucket(x: torch.Tensor, bucket: int, *, value: float = 0.0, cos_pad: bool = False) -> torch.Tensor:
    """Pad to bucket rows: rotary tables with the identity rotation (cos=1, sin=0), everything else with zeros."""
    n = x.shape[0]
    if n == bucket:
        return x
    assert n < bucket, f"sequence {n} exceeds bucket {bucket}"
    pad = torch.full((bucket - n, *x.shape[1:]), 1.0 if cos_pad else value, dtype=x.dtype)
    return torch.cat([x, pad], dim=0)


def preprocess(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    bucket: int | None = None,
    permute: bool = True,
) -> dict:
    """Permutation, padded rotary tables, and unpadded/padded lengths for one image."""
    n = int(grid_thw.prod(dim=-1).sum())
    pos = raster_position_ids(grid_thw)

    perm = None
    if permute:
        perm = block_permutation(grid_thw, spatial_merge_size)
        pos = pos[perm]

    cos, sin = meta_rope_tables(pos, head_dim)

    seq_len = n if bucket is None else bucket
    if bucket is not None:
        cos = pad_to_bucket(cos, bucket, cos_pad=True)
        sin = pad_to_bucket(sin, bucket)

    return {
        "perm": perm,
        "cos": cos,
        "sin": sin,
        "unpadded_len": n,
        "seq_len": seq_len,
        "position_ids": pos,
    }
