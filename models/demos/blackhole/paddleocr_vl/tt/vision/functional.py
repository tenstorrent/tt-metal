# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side rotary tables and block-order permutation for the vision tower.

PaddleOCR merges 2x2 patch blocks in the projector, after the encoder, unlike
Qwen's in-encoder merge, so its raster token order needs permuting into
merge-block order before the tower; that lets ``qwen36``'s ``PatchMerger`` be
reused unchanged. See ``tests/test_vision_permutation.py`` for the proof.
"""

from __future__ import annotations

import torch

from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta

# PaddleOCR's vision rotary embedding is built with theta 10000 over head_dim//2
# (PaddleOCRVisionRotaryEmbedding, modeling_paddleocr_vl.py:103).
VISION_ROPE_THETA = 10000.0


def raster_position_ids(grid_thw: torch.Tensor) -> torch.Tensor:
    """``(row, col)`` per patch token, in the raster order the processor emits.

    Equivalent to transformers' ``get_vision_position_ids(grid_thw, 1)``; kept
    local so the port does not silently inherit a change to a shared helper.
    """
    out = []
    for t, h, w in grid_thw.tolist():
        t, h, w = int(t), int(h), int(w)
        hpos = torch.arange(h).unsqueeze(1).expand(-1, w).flatten()
        wpos = torch.arange(w).unsqueeze(0).expand(h, -1).flatten()
        out.append(torch.stack([hpos, wpos], dim=-1).repeat(t, 1))
    return torch.cat(out, dim=0)


def block_permutation(grid_thw: torch.Tensor, spatial_merge_size: int) -> torch.Tensor:
    """Indices that reorder raster tokens into projector merge-block order.

    ``out[i]`` is the raster index of the i-th token once tokens are grouped so
    that each consecutive run of ``merge**2`` belongs to one 2x2 block. This is
    the same gather the projector performs with reshape+transpose.
    """
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
    """cos/sin of shape ``[N, head_dim]`` in **HuggingFace** layout.

    Mirrors ``PaddleOCRVisionRotaryEmbedding`` followed by the encoder's
    ``rotary_embeddings.repeat(1, 2)`` (``modeling_paddleocr_vl.py:860-862``):
    half the head dim carries the row frequency bank and half the column bank,
    then the whole thing is duplicated for the rotate-half convention.

    This is the reference-matching form, and it is *not* what the device wants;
    see ``meta_rope_tables``.
    """
    dim = head_dim // 2
    inv_freq = 1.0 / (VISION_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
    freqs = (position_ids.float().unsqueeze(-1) * inv_freq).flatten(1)  # [N, 2 * dim/2] = [N, head_dim/2]
    emb = freqs.repeat(1, 2)  # [N, head_dim]
    return emb.cos(), emb.sin()


def meta_rope_tables(position_ids: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin in the interleaved layout ``rotary_embedding_llama`` expects.

    Two conventions have to agree for rotation to come out right, and they are
    easy to get half-right. HuggingFace splits each head in two and rotates one
    half against the other, so its tables read ``[c0..c_{d/2-1}, c0..c_{d/2-1}]``.
    tt-metal rotates adjacent *pairs*, so its tables read ``[c0, c0, c1, c1, ...]``
    and it expects q/k rows ordered to match -- which is what the q/k permute in
    ``weight_mapping._to_meta_rope_format`` arranges.

    Permuting the weights without converting the tables leaves the rotation
    applying the right angles to the wrong components. It degrades rather than
    breaks (the first encoder layer scored PCC 0.84 that way, not garbage), which
    is exactly why it is worth naming here.
    """
    cos_hf, sin_hf = vision_rope_tables(position_ids, head_dim)
    return convert_rope_style_hf_to_meta(cos_hf, sin_hf)


def pad_to_bucket(x: torch.Tensor, bucket: int, *, value: float = 0.0, cos_pad: bool = False) -> torch.Tensor:
    """Pad a ``[N, ...]`` tensor up to ``bucket`` rows.

    Rotary tables pad with cos=1 / sin=0 (an identity rotation) so the padded
    rows cannot rotate real content if they are ever read; everything else pads
    with zeros.
    """
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
    """Build everything the tower needs for one image.

    Returns the permutation (or ``None``), the padded rotary tables, and the
    unpadded/padded lengths so the caller can slice the tower output back down.
    """
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
