# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


def stack_cos_sin(cos, sin):
    cos = torch.stack([cos, cos], dim=-1).flatten(-2)
    sin = torch.stack([sin, sin], dim=-1).flatten(-2)
    return cos, sin


def get_rot_transformation_mat():
    # ROPE op uses a single tile
    dhead = 32
    rot_emb_matrix = torch.zeros(1, 1, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


def rope_halfsplit_to_interleaved_perm(head_dim: int) -> torch.Tensor:
    """Index permutation mapping the reference half-split RoPE layout to the interleaved
    (adjacent-pair) layout that ``ttnn.experimental.rotary_embedding_llama`` + the 32x32
    transformation matrix consume: ``out[2i] = in[i]``, ``out[2i+1] = in[i + head_dim/2]``.

    Applied identically to the Q/K projection output channels, the QK-norm affine weights,
    and the cos/sin tables. Because it is a SHARED permutation of head_dim on Q and K (and V/O
    are untouched), attention Q·Kᵀ is invariant -> the layout swap is numerically neutral.
    """
    h = head_dim // 2
    return torch.stack([torch.arange(h), torch.arange(h) + h], dim=1).flatten()


def rope_halfsplit_to_interleaved(cos: torch.Tensor, sin: torch.Tensor, head_dim: int):
    """Permute half-split cos/sin tables (cat(freqs,freqs)) into the interleaved
    [f0,f0,f1,f1,...] layout expected by rotary_embedding_llama. Host-side helper for callers."""
    perm = rope_halfsplit_to_interleaved_perm(head_dim)
    return cos[..., perm], sin[..., perm]
