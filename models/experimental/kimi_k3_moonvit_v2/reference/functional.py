# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side input preparation for the TT encoder, written as plain functions.

Each function reproduces one piece of `modeling_kimi_k3.py` in the form the device consumes, and
`tests/test_functional.py` checks it against the reference module:

  * `patches_to_rows`: the Conv2d patch embed as a matmul over flattened patches.
  * `pos_embeds`: the learned 64x64 table interpolated to the image grid.
  * `rope_cos_sin` + `rotate_pairs`: the complex-multiply 2D RoPE as `x * cos + rotate(x) * sin`,
    the form `ttnn.experimental.rotary_embedding_llama` computes with a pair-swapping `trans_mat`.
  * `pad_rows`: padding to the fixed device sequence length.

`prepare_host_inputs` combines them for one image.
"""

from __future__ import annotations

import torch

from ..tt.model_config import MoonViTConfig, check_grid
from .modeling_kimi_k3 import Rope2DPosEmbRepeated, get_rope_shape


def patches_to_rows(pixel_values: torch.Tensor) -> torch.Tensor:
    """(L, C, p, p) -> (L, C*p*p), matching `proj.weight.reshape(hidden, C*p*p)`."""
    return pixel_values.reshape(pixel_values.shape[0], -1)


def pos_embeds(weight: torch.Tensor, h: int, w: int, interpolation_mode: str) -> torch.Tensor:
    """(h*w, hidden): the position table at grid (h, w), as the reference adds it for one still image."""
    if (h, w) == tuple(weight.shape[:-1]):
        return weight.flatten(end_dim=1)
    return get_rope_shape(weight, interpolation_mode=interpolation_mode, shape=(h, w))


def rope_cos_sin(config: MoonViTConfig, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h*w, head_dim) fp32 cos and sin, each value repeated for the two lanes of its pair."""
    rope = Rope2DPosEmbRepeated(config.head_dim, config.max_side_patches, config.max_side_patches, config.rope_theta)
    freqs_cis = rope.get_freqs_cis(torch.tensor([[1, h, w]]), device=torch.device("cpu"))  # (h*w, head_dim/2)
    return freqs_cis.real.repeat_interleave(2, dim=-1), freqs_cis.imag.repeat_interleave(2, dim=-1)


def rotate_pairs(x: torch.Tensor) -> torch.Tensor:
    """(x0, x1) -> (-x1, x0) on adjacent pairs; on device this is `x @ trans_mat`."""
    pairs = x.unflatten(-1, (-1, 2))
    return torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)


def rot_transformation_mat() -> torch.Tensor:
    """[1, 1, 32, 32] tile with `x @ m == rotate_pairs(x)` per 32 lanes, the `trans_mat` of
    `rotary_embedding_llama`."""
    m = torch.zeros(1, 1, 32, 32)
    m[..., torch.arange(0, 32, 2), torch.arange(1, 32, 2)] = 1
    m[..., torch.arange(1, 32, 2), torch.arange(0, 32, 2)] = -1
    return m


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., L, head_dim) with cos/sin broadcast over the leading dims."""
    return x * cos + rotate_pairs(x) * sin


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero-pad dim 0 to `rows`."""
    if x.shape[0] > rows:
        raise ValueError(f"{x.shape[0]} rows exceed the padded length {rows}")
    return torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, rows - x.shape[0]))


def prepare_host_inputs(
    config: MoonViTConfig, pos_emb_weight: torch.Tensor, pixel_values: torch.Tensor, grid_thws: torch.Tensor
) -> dict:
    """Everything the device needs for one image, padded to `config.max_patches` rows.

    Returns `patches` (S, patch_dim), `pos_emb` (S, hidden), `cos` and `sin` (S, head_dim) and
    `logical_n`, the number of real rows.
    """
    h, w = check_grid(grid_thws, config)
    cos, sin = rope_cos_sin(config, h, w)
    rows = config.max_patches
    return {
        "patches": pad_rows(patches_to_rows(pixel_values), rows),
        "pos_emb": pad_rows(pos_embeds(pos_emb_weight, h, w, config.pos_emb_interpolation_mode), rows),
        "cos": pad_rows(cos, rows),
        "sin": pad_rows(sin, rows),
        "logical_n": h * w,
    }
