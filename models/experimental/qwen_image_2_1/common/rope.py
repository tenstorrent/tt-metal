# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-side rotary tables.

DiT (QwenImage21Rope): 3 axes (frame 16, height 56, width 56 dims), theta 10000, complex pairs are
ADJACENT interleaved (x.reshape(..., 64, 2)), i.e. out[2i] = x[2i]*cos_i - x[2i+1]*sin_i,
out[2i+1] = x[2i]*sin_i + x[2i+1]*cos_i. That matches ttnn.experimental.rotary_embedding_llama with the
adjacent-pair transformation matrix, with cos/sin each expanded to 128 by repeating every frequency twice.

Text encoder (Qwen3-VL text, mrope interleaved): for text-only input the three position ids coincide, so
it is plain 1-D RoPE (theta 5e6) in llama rotate_half form; we permute the q/k projection rows per head
(see weights.interleave_pairs_permutation) so the same adjacent-pair kernel applies.
"""
from __future__ import annotations

import torch

from .config import DIT, TE


def _freqs(dim: int, theta: float) -> torch.Tensor:
    return 1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float64) / dim)


def dit_positions(text_len: int, height: int, width: int):
    """(frame, h, w) integer positions for [text tokens..., image tokens (row-major h, w)]."""
    n_img = height * width
    frame = torch.cat([torch.arange(text_len), torch.full((n_img,), text_len)])
    hpos = torch.cat(
        [torch.arange(text_len), torch.arange(-(height - height // 2), height // 2).repeat_interleave(width)]
    )
    wpos = torch.cat([torch.arange(text_len), torch.arange(-(width - width // 2), width // 2).repeat(height)])
    return frame, hpos, wpos


def dit_angles(text_len: int, height: int, width: int) -> torch.Tensor:
    """[T + H*W, 64] rotation angles (fp64) in the model's frequency order (frame 8 | h 28 | w 28)."""
    frame, hpos, wpos = dit_positions(text_len, height, width)
    angs = []
    for pos, dim in zip((frame, hpos, wpos), DIT.axes_dims_rope):
        angs.append(pos.to(torch.float64)[:, None] * _freqs(dim, DIT.rope_theta)[None])
    return torch.cat(angs, dim=-1)


def angles_to_cos_sin(angles: torch.Tensor, dtype=torch.bfloat16):
    """[S, 64] angles -> cos, sin [S, 128] with each frequency duplicated for the adjacent pair."""
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def dit_cos_sin(text_len: int, height: int, width: int, dtype=torch.bfloat16):
    """cos/sin for the joint sequence; slice [:text_len] for text and [text_len:] for image."""
    return angles_to_cos_sin(dit_angles(text_len, height, width), dtype)


def dit_freqs_cis_reference(text_len: int, height: int, width: int) -> torch.Tensor:
    """Complex [S, 64] like diffusers' QwenImage21Rope.forward (for tests)."""
    return torch.polar(torch.ones_like(dit_angles(text_len, height, width)), dit_angles(text_len, height, width))


def te_cos_sin(seq_len: int, dtype=torch.bfloat16, head_dim: int = TE.head_dim, theta: float = TE.rope_theta):
    """Text-encoder tables for positions 0..seq_len-1 in the PERMUTED (adjacent-pair) layout:
    pair i uses inv_freq_i, i in [0, 64)."""
    pos = torch.arange(seq_len, dtype=torch.float64)
    ang = pos[:, None] * _freqs(head_dim, theta)[None]  # [S, 64]
    return angles_to_cos_sin(ang, dtype)


def rot_transformation_mat(tile: int = 32) -> torch.Tensor:
    """[1, 1, 32, 32] matrix T with (x @ T)[2k] = -x[2k+1], (x @ T)[2k+1] = x[2k] (adjacent pairs)."""
    m = torch.zeros(1, 1, tile, tile)
    m[..., torch.arange(0, tile, 2), torch.arange(1, tile, 2)] = 1.0
    m[..., torch.arange(1, tile, 2), torch.arange(0, tile, 2)] = -1.0
    return m


def apply_rope_adjacent(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Torch reference of the adjacent-pair rotation on x[..., S, D] with cos/sin [S, D]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rot * sin


# ----------------------------------------------------------------------------- general joint layouts
def joint_positions(segments):
    """(frame, h, w) positions for a joint sequence given as a list of segments, each either
    ("text", n_tokens) or ("image", height, width) in latent tokens, in sequence order (condition images
    first, target image last, text in between). Mirrors diffusers' QwenImage21Rope.forward: text tokens
    advance one shared position on all axes; every image block freezes the frame axis at the position
    reached so far, lays its tokens on a centred h/w grid, and advances the position by max(height, width)."""
    frame, hpos, wpos = [], [], []
    position = 0
    for seg in segments:
        if seg[0] == "text":
            n = seg[1]
            frame.extend(range(position, position + n))
            hpos.extend(range(position, position + n))
            wpos.extend(range(position, position + n))
            position += n
        else:
            _, height, width = seg
            frame.extend([position] * (height * width))
            hpos.extend([h for h in range(-(height - height // 2), height // 2) for _ in range(width)])
            wpos.extend([w for _ in range(height) for w in range(-(width - width // 2), width // 2)])
            position += max(height, width)
    return torch.tensor(frame), torch.tensor(hpos), torch.tensor(wpos)


def joint_angles(segments) -> torch.Tensor:
    frame, hpos, wpos = joint_positions(segments)
    angs = []
    for pos, dim in zip((frame, hpos, wpos), DIT.axes_dims_rope):
        angs.append(pos.to(torch.float64)[:, None] * _freqs(dim, DIT.rope_theta)[None])
    return torch.cat(angs, dim=-1)


def joint_cos_sin(segments, dtype=torch.bfloat16):
    """cos/sin [S_total, 128] for the whole joint sequence (adjacent-pair layout)."""
    return angles_to_cos_sin(joint_angles(segments), dtype)


def segments_from_image_pad_mask(image_pad_mask, img_shapes):
    """Turn the pipeline's VLM-slot mask (True at <|image_pad|> slots, 1 slot = 2x2 latent tokens; the target
    image's slots appended at the end) plus img_shapes [(1,h,w), ...] into the segment list used above."""
    m = [bool(x) for x in image_pad_mask]
    segments = []
    i = 0
    img_idx = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            segments.append(("text", j - i))
            i = j
        else:
            _, h, w = img_shapes[img_idx]
            n_slots = h * w // 4
            assert all(m[i : i + n_slots]), "image slots must be contiguous"
            segments.append(("image", h, w))
            i += n_slots
            img_idx += 1
    assert img_idx == len(img_shapes), (img_idx, len(img_shapes))
    return segments
