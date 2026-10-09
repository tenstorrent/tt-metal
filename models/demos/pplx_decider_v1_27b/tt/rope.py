# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Partial RoPE tables (first 64 of 256 head dims, theta 1e7) for one fused full-width rotary op.

HF rotates the first ``rotary_dim`` dims of each head in the neox ``rotate_half`` layout: dim j
pairs with dim j + rotary_dim/2. For text-only inputs HF's interleaved mRoPE sees three identical
position streams, so cos/sin equal plain 1D RoPE.

The TT path runs ``ttnn.experimental.rotary_embedding_llama`` over the whole 256-dim head instead
of slice -> rotate 64 dims -> concat. That op rotates adjacent pairs (2i, 2i+1). The weight adapter
therefore permutes the q/k head dims with ``rope_head_permutation`` (pair (j, j+32) -> (2j, 2j+1),
the rest unchanged) and these tables carry cos=1, sin=0 on the 192 pass-through dims. q and k get
the same permutation, so q.k and the attention output are unchanged; v is not permuted.

Tables are computed once at setup and live on device; ``forward`` only slices them.

Image requests (3D mRoPE). With an image in the prompt HF gives each token three position ids
(T, H, W; ``Qwen3_5Model.get_rope_index``), and frequency j of the 32 rotary frequencies takes its
position from stream H when ``j % 3 == 1 and j < 33``, W when ``j % 3 == 2 and j < 30``, else T
(``apply_interleaved_mrope``, sections [11, 11, 10]). ``get_rope_index`` below is a host port of
the HF function (per-request input prep, like tokenization) and ``PplxRequestRotary`` builds the
cos/sin tables of one request from those ids, in the same layout as ``PplxRotary``, and serves
them through the same ``rotary(start, length)`` call. ``start`` stays a sequence (cache / chunk)
index: after an image the rope position of a token is smaller than its index, so the table is
indexed by token index and holds the rope angle of that token. Text-only requests keep using the
setup-time ``PplxRotary`` table (identical numbers: the three streams are equal there).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


def rope_head_permutation(head_dim: int, rotary_dim: int) -> list[int]:
    """``new[p] = old[perm[p]]``: neox pairs (j, j + rotary_dim/2) become adjacent (2j, 2j+1)."""
    half = rotary_dim // 2
    perm = [0] * rotary_dim
    for j in range(half):
        perm[2 * j], perm[2 * j + 1] = j, j + half
    return perm + list(range(rotary_dim, head_dim))


@dataclass
class RotaryConfig:
    rotary_dim: int
    theta: float
    max_seq_len: int
    mesh_device: object
    head_dim: int


class PplxRotary(LightweightModule):
    def __init__(self, rotary_dim: int, theta: float, max_seq_len: int, mesh_device, *, head_dim: int):
        super().__init__()
        self.config = RotaryConfig(
            rotary_dim=rotary_dim, theta=theta, max_seq_len=max_seq_len, mesh_device=mesh_device, head_dim=head_dim
        )
        self._build_tables()

    @classmethod
    def from_config(cls, config: RotaryConfig) -> "PplxRotary":
        instance = object.__new__(cls)
        LightweightModule.__init__(instance)
        instance.config = config
        instance._build_tables()
        return instance

    def _build_tables(self) -> None:
        c = self.config
        freqs = (
            torch.arange(c.max_seq_len, dtype=torch.float32)[:, None] * text_inv_freq(c.rotary_dim, c.theta)[None, :]
        )
        self.cos_table, self.sin_table = upload_tables(freqs, c.head_dim, c.mesh_device)

    def forward(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos, sin for absolute positions ``start .. start+length-1``; each [1, 1, length, head_dim] TILE."""
        return slice_tables(self.cos_table, self.sin_table, start, length)


def text_inv_freq(rotary_dim: int, theta: float) -> torch.Tensor:
    """``Qwen3_5TextRotaryEmbedding.inv_freq``: fp32, ``rotary_dim / 2`` values."""
    return 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))


def host_tables(freqs: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 angles [S, rotary_dim/2] -> cos, sin [S, head_dim] fp32 in the pair layout.

    Interleaved pairs share one frequency; pass-through dims rotate by 0 (cos 1, sin 0).
    """
    angles = freqs.repeat_interleave(2, dim=-1)
    pad = head_dim - angles.shape[-1]
    cos = torch.nn.functional.pad(angles.cos(), (0, pad), value=1.0)
    sin = torch.nn.functional.pad(angles.sin(), (0, pad), value=0.0)
    return cos, sin


def upload_tables(freqs: torch.Tensor, head_dim: int, mesh_device) -> list[ttnn.Tensor]:
    """TILE tables [1, 1, S, head_dim] BF16: a chunk is one tile-row slice, no per-call tilize."""
    seq = freqs.shape[0]
    return [
        ttnn.from_torch(
            t.reshape(1, 1, seq, head_dim).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for t in host_tables(freqs, head_dim)
    ]


def slice_tables(cos_table: ttnn.Tensor, sin_table: ttnn.Tensor, start: int, length: int):
    rows = cos_table.shape[2]
    if start < 0 or start + length > rows:
        raise ValueError(f"Positions {start}..{start + length} exceed the RoPE table ({rows})")
    if start % 32:
        raise ValueError(f"RoPE chunk start {start} must be tile aligned")
    return tuple(t[:, :, start : start + length, :] for t in (cos_table, sin_table))


# ----------------------------------------------------------------------------------------------
# 3D mRoPE for image requests
# ----------------------------------------------------------------------------------------------


def get_rope_index(input_ids, mm_token_type_ids, image_grid_thw=None, *, spatial_merge_size: int = 2) -> torch.Tensor:
    """Host port of HF ``Qwen3_5Model.get_rope_index`` for one unpadded prompt (images only).

    ``input_ids`` / ``mm_token_type_ids``: [S] or [1, S] (type 0 text, 1 image; vision start / end
    tokens are text). ``image_grid_thw``: [num_images, 3] patch grids, in prompt order.
    Returns int64 position ids [3, S] (T, H, W). A text run of length L gets ``current + arange(L)``
    on all three streams; an image with merged grid (t, h, w) gets T = current, H = current + row,
    W = current + col (t == 1 for still images), then ``current += max(h, w)``.
    """
    ids = torch.as_tensor(input_ids).reshape(-1)
    types = torch.as_tensor(mm_token_type_ids).reshape(-1).tolist()
    if len(types) != ids.shape[0]:
        raise ValueError("mm_token_type_ids and input_ids differ in length")
    grids = iter([] if image_grid_thw is None else torch.as_tensor(image_grid_thw).reshape(-1, 3).tolist())
    parts, current, start = [], 0, 0
    for kind, group in itertools.groupby(types):
        length = len(list(group))
        if kind == 0:
            parts.append(torch.arange(length).view(1, -1).expand(3, -1) + current)
            current += length
        elif kind == 1:
            t, h, w = next(grids)
            gt, gh, gw = t, h // spatial_merge_size, w // spatial_merge_size
            if gt * gh * gw != length:
                raise ValueError(f"Image run of {length} tokens at {start} does not match grid {(t, h, w)}")
            temporal = torch.arange(gt).repeat_interleave(gh * gw) + current
            height = (torch.arange(gh) + current).repeat_interleave(gw).repeat(gt)
            width = (torch.arange(gw) + current).repeat(gh * gt)
            parts.append(torch.stack([temporal, height, width], dim=0))
            current += max(h, w) // spatial_merge_size
        else:
            raise ValueError(f"Token type {kind} (video) is not supported")
        start += length
    if next(grids, None) is not None:
        raise ValueError("More image grids than image runs in the prompt")
    return torch.cat(parts, dim=1).to(torch.int64)


def mrope_freqs(position_ids: torch.Tensor, inv_freq: torch.Tensor, mrope_section=(11, 11, 10)) -> torch.Tensor:
    """fp32 angles [S, rotary_dim/2] from 3D ids [3, S]: ``Qwen3_5TextRotaryEmbedding.forward`` up to
    ``apply_interleaved_mrope`` (each angle is one fp32 product, as HF's K=1 matmul)."""
    freqs = position_ids.float()[:, :, None] * inv_freq[None, None, :]  # [3, S, 32]
    out = freqs[0].clone()
    for stream, offset in ((1, 1), (2, 2)):
        idx = slice(offset, mrope_section[stream] * 3, 3)
        out[..., idx] = freqs[stream, ..., idx]
    return out


def pad_position_ids(position_ids: torch.Tensor, length: int) -> torch.Tensor:
    """Extend [3, S] ids to ``length`` rows with text-continuation positions (bucket padding rows).

    Padding rows come after every real token, so their values cannot reach a real row (causal);
    continuing the text positions keeps them ordinary.
    """
    real = position_ids.shape[1]
    if length < real:
        raise ValueError(f"Cannot pad {real} positions to {length}")
    tail = torch.arange(length - real).view(1, -1).expand(3, -1) + int(position_ids.max()) + 1
    return torch.cat([position_ids, tail], dim=1)


class PplxRequestRotary(LightweightModule):
    """cos/sin tables of one request (3D mRoPE), indexed by token index; same call as ``PplxRotary``."""

    def __init__(self, cos_table: ttnn.Tensor, sin_table: ttnn.Tensor, position_ids: torch.Tensor):
        super().__init__()
        self.cos_table, self.sin_table = cos_table, sin_table
        self.position_ids = position_ids  # host [3, bucket] int64 (kept for tests)

    @classmethod
    def from_position_ids(
        cls,
        position_ids: torch.Tensor,
        bucket: int,
        *,
        rotary_dim: int,
        theta: float,
        head_dim: int,
        mesh_device,
        mrope_section=(11, 11, 10),
    ) -> "PplxRequestRotary":
        """Host build + upload (input prep): ids [3, S] -> two [1, 1, bucket, head_dim] BF16 TILE tables."""
        ids = pad_position_ids(position_ids, bucket)
        freqs = mrope_freqs(ids, text_inv_freq(rotary_dim, theta), mrope_section)
        cos, sin = upload_tables(freqs, head_dim, mesh_device)
        return cls(cos, sin, ids)

    def forward(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return slice_tables(self.cos_table, self.sin_table, start, length)

    def deallocate(self) -> None:
        ttnn.deallocate(self.cos_table)
        ttnn.deallocate(self.sin_table)
