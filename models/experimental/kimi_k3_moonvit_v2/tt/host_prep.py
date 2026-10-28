# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-side input preparation for the TT encoder, written as plain functions.

Each function reproduces one piece of `modeling_kimi_k3.py` in the form the device consumes, and
`tests/test_host_prep.py` checks it against the reference module. Only torch is needed: nothing here
imports the reference or `transformers`.

  * `patches_to_rows`: the Conv2d patch embed as a matmul over flattened patches; `patch_rows` pads
    them with zero columns to `config.padded_patch_dim`.
  * `pos_embeds`: the learned 64x64 table interpolated to the image grid.
  * `rope_cos_sin` + `rotate_pairs`: the complex-multiply 2D RoPE as `x * cos + rotate(x) * sin`,
    the form `ttnn.experimental.rotary_embedding_llama` computes with a pair-swapping `trans_mat`.
  * `to_window_order` / `from_window_order`: the row order the device runs in. Each 2x2 merge window's
    four patches are consecutive, windows in raster order, so the patch merger folds four consecutive
    rows on each device and needs no collective. Attention is non-causal and every other op works on
    each row alone, so the order changes nothing else as long as patches, position embeddings and RoPE
    tables share it.
  * `pad_rows`: padding to the fixed device sequence length.
  * `rope_tables`: `rope_cos_sin` in window order and padded, as the device tables hold it.

`prepare_host_inputs` combines them for one image.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model_config import MoonViTConfig, check_grid


def patches_to_rows(pixel_values: torch.Tensor) -> torch.Tensor:
    """(L, C, p, p) -> (L, C*p*p), matching `proj.weight.reshape(hidden, C*p*p)`."""
    return pixel_values.reshape(pixel_values.shape[0], -1)


def patch_rows(config: MoonViTConfig, pixel_values: torch.Tensor) -> torch.Tensor:
    """(L, C, p, p) -> (L, padded_patch_dim): `patches_to_rows` with zero columns up to a whole tile."""
    rows = patches_to_rows(pixel_values)
    return F.pad(rows, (0, config.padded_patch_dim - rows.shape[-1]))


def pos_embeds(weight: torch.Tensor, h: int, w: int, interpolation_mode: str) -> torch.Tensor:
    """(h*w, hidden): the position table at grid (h, w), as the reference adds it for one still image
    (`Learnable2DInterpPosEmbDivided_fixed`, `get_rope_shape`)."""
    if (h, w) == tuple(weight.shape[:-1]):
        return weight.flatten(end_dim=1)
    table = weight.permute(2, 0, 1).unsqueeze(0)  # (1, hidden, H, W)
    return F.interpolate(table, size=(h, w), mode=interpolation_mode).squeeze(0).permute(1, 2, 0).flatten(end_dim=1)


def rope_cos_sin(config: MoonViTConfig, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(h*w, head_dim) fp32 cos and sin, each value repeated for the two lanes of its pair.

    `Rope2DPosEmbRepeated.get_freqs_cis` for one image: pair i of the head_dim/2 pairs rotates by
    x * freq[i // 2] for even i and y * freq[i // 2] for odd i, with x the column and y the row.
    """
    dim = config.head_dim
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))  # (head_dim/4,)
    y, x = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    angles = torch.stack((torch.outer(x.flatten(), freqs), torch.outer(y.flatten(), freqs)), dim=-1)
    angles = angles.flatten(1)  # (h*w, head_dim/2)
    return angles.cos().repeat_interleave(2, dim=-1), angles.sin().repeat_interleave(2, dim=-1)


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


def window_order(h: int, w: int, merge_kernel_size: tuple[int, int]) -> torch.Tensor:
    """Raster patch indices in window order: window (i, j) holds patches (kh*i + a, kw*j + b) for
    a < kh, b < kw, in the order `tpool_patch_merger` concatenates them."""
    kh, kw = merge_kernel_size
    return torch.arange(h * w).reshape(h // kh, kh, w // kw, kw).permute(0, 2, 1, 3).flatten()


def to_window_order(x: torch.Tensor, h: int, w: int, merge_kernel_size: tuple[int, int]) -> torch.Tensor:
    """Reorder dim 0 (h*w patches in raster order) into window order."""
    return x[window_order(h, w, merge_kernel_size)]


def from_window_order(x: torch.Tensor, h: int, w: int, merge_kernel_size: tuple[int, int]) -> torch.Tensor:
    """The inverse of `to_window_order`."""
    return x[window_order(h, w, merge_kernel_size).argsort()]


def pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """Zero-pad dim 0 to `rows`."""
    if x.shape[0] > rows:
        raise ValueError(f"{x.shape[0]} rows exceed the padded length {rows}")
    return torch.nn.functional.pad(x, (0, 0) * (x.ndim - 1) + (0, rows - x.shape[0]))


def rope_tables(config: MoonViTConfig, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(max_patches, head_dim) cos and sin for the device: `rope_cos_sin` in window order, zero-padded."""
    return tuple(
        pad_rows(to_window_order(t, h, w, config.merge_kernel_size), config.max_patches)
        for t in rope_cos_sin(config, h, w)
    )


def prepare_host_inputs(
    config: MoonViTConfig, pos_emb_weight: torch.Tensor, pixel_values: torch.Tensor, grid_thws: torch.Tensor
) -> dict:
    """Everything the device needs for one image, in window order and padded to `config.max_patches` rows.

    Returns `patches` (S, padded_patch_dim), `pos_emb` (S, hidden), `cos` and `sin` (S, head_dim) and
    `logical_n`, the number of real rows.
    """
    h, w = check_grid(grid_thws, config)
    cos, sin = rope_tables(config, h, w)
    pos_emb = pos_embeds(pos_emb_weight, h, w, config.pos_emb_interpolation_mode)
    rows = config.max_patches

    def prepare(x: torch.Tensor) -> torch.Tensor:
        return pad_rows(to_window_order(x, h, w, config.merge_kernel_size), rows)

    return {
        "patches": prepare(patch_rows(config, pixel_values)),
        "pos_emb": prepare(pos_emb),
        "cos": cos,
        "sin": sin,
        "logical_n": h * w,
    }
