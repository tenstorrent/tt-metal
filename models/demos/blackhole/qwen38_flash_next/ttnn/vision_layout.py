# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side layout of the vision tower's weights and rotary tables for the device (pure torch, no TTNN import).

Head dim 72 is not a tile multiple.  q, k and v are zero-padded to 96 per head inside the fused qkv projection's
output columns, the attention output projection takes the padded head concatenation as its input rows, and q and k
are permuted inside each head to the interleaved rotary layout (the rotate-half pair (i, i + 36) becomes the adjacent
pair (2i, 2i + 1)) so the device rotary op's 32x32 pair-swap transformation reproduces rotate-half exactly; v and the
output projection keep the checkpoint order (q . k is invariant under one permutation of both).  The MLP width 4304
is zero-padded to 4320.  Row counts are padded to tile multiples; cos / sin carry the identity rotation (cos 1, sin 0)
in the padded head columns and the padded rows.  ``tests/test_vision_reference_no_device.py`` proves each of these
equivalences against ``vision_reference.py``.
"""

from __future__ import annotations

from typing import Mapping

import torch

from models.demos.blackhole.qwen38_flash_next.vision_reference import VisionTowerConfig

TILE = 32
PADDED_HEAD_DIM = 96
PADDED_INTERMEDIATE = 4320


def padded_rows(rows: int) -> int:
    return -(-rows // TILE) * TILE


def interleave_permutation(head_dim: int) -> torch.Tensor:
    """``perm`` with ``x_interleaved = x[..., perm]``: ``perm[2i] = i``, ``perm[2i + 1] = i + head_dim / 2``."""

    half = head_dim // 2
    return torch.stack([torch.arange(half), torch.arange(half) + half], dim=1).reshape(-1)


def rotary_transformation_tile() -> torch.Tensor:
    """``[1, 1, 32, 32]``: ``x @ T`` maps the pair ``(a, b)`` at ``(2i, 2i + 1)`` to ``(-b, a)`` (rotate-half on pairs)."""

    matrix = torch.zeros(1, 1, TILE, TILE, dtype=torch.float32)
    matrix[..., torch.arange(0, TILE, 2), torch.arange(1, TILE, 2)] = 1.0
    matrix[..., torch.arange(1, TILE, 2), torch.arange(0, TILE, 2)] = -1.0
    return matrix


def interleave_cos_sin(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half ``[N, head_dim]`` cos / sin (``[c_0..c_35, c_0..c_35]``) -> interleaved (``[c_0, c_0, c_1, c_1, ..]``)."""

    half = cos.shape[-1] // 2
    return cos[..., :half].repeat_interleave(2, dim=-1), sin[..., :half].repeat_interleave(2, dim=-1)


def device_cos_sin(
    cos: torch.Tensor, sin: torch.Tensor, *, rows: int, padded_head_dim: int = PADDED_HEAD_DIM
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[1, 1, rows, padded_head_dim]`` FP32 interleaved cos / sin with the identity rotation in every pad."""

    cos_meta, sin_meta = interleave_cos_sin(cos.to(torch.float32), sin.to(torch.float32))
    n, head_dim = cos_meta.shape
    if rows < n or rows % TILE:
        raise ValueError(f"rows {rows} must be a tile multiple of at least {n}")
    cos_out = torch.ones(rows, padded_head_dim, dtype=torch.float32)
    sin_out = torch.zeros(rows, padded_head_dim, dtype=torch.float32)
    cos_out[:n, :head_dim] = cos_meta
    sin_out[:n, :head_dim] = sin_meta
    return cos_out.reshape(1, 1, rows, padded_head_dim), sin_out.reshape(1, 1, rows, padded_head_dim)


def rotate_interleaved(x: torch.Tensor, cos_meta: torch.Tensor, sin_meta: torch.Tensor) -> torch.Tensor:
    """What the device rotary computes on interleaved ``[.., N, D]`` input (D a tile multiple): ``x cos + (x T) sin``."""

    width = x.shape[-1]
    if width % TILE:
        raise ValueError(f"interleaved rotary width {width} must be a tile multiple")
    tile = rotary_transformation_tile()[0, 0]
    transformation = torch.block_diag(*([tile] * (width // TILE)))
    return x * cos_meta + (x @ transformation) * sin_meta


def _pad_heads(weight_rows: torch.Tensor, heads: int, head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """``[heads * head_dim, ...]`` -> ``[heads * padded_head_dim, ...]`` with zero rows after each head."""

    shaped = weight_rows.reshape(heads, head_dim, *weight_rows.shape[1:])
    padded = torch.zeros(heads, padded_head_dim, *weight_rows.shape[1:], dtype=weight_rows.dtype)
    padded[:, :head_dim] = shaped
    return padded.reshape(heads * padded_head_dim, *weight_rows.shape[1:])


def fused_qkv_weight(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    heads: int,
    head_dim: int,
    padded_head_dim: int = PADDED_HEAD_DIM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``qkv.weight [3 hidden, hidden]`` / ``bias`` -> transposed ``[hidden, 3 heads padded]`` and ``[3 heads padded]``:
    q and k rows permuted to the interleaved rotary order inside each head, every head zero-padded to
    ``padded_head_dim``, v unpermuted."""

    hidden = heads * head_dim
    if tuple(weight.shape) != (3 * hidden, hidden) or tuple(bias.shape) != (3 * hidden,):
        raise ValueError(
            f"qkv weight/bias shapes {tuple(weight.shape)} / {tuple(bias.shape)} do not match {heads} x {head_dim}"
        )
    perm = interleave_permutation(head_dim)
    weight = weight.to(torch.float32)
    bias = bias.to(torch.float32)
    pieces_w, pieces_b = [], []
    for part in range(3):
        rows_w = weight[part * hidden : (part + 1) * hidden].reshape(heads, head_dim, hidden)
        rows_b = bias[part * hidden : (part + 1) * hidden].reshape(heads, head_dim)
        if part < 2:
            rows_w, rows_b = rows_w[:, perm], rows_b[:, perm]
        pieces_w.append(_pad_heads(rows_w.reshape(hidden, hidden), heads, head_dim, padded_head_dim))
        pieces_b.append(_pad_heads(rows_b.reshape(hidden, 1), heads, head_dim, padded_head_dim).reshape(-1))
    return torch.cat(pieces_w, dim=0).transpose(0, 1).contiguous(), torch.cat(pieces_b, dim=0).contiguous()


def attention_output_weight(
    weight: torch.Tensor, *, heads: int, head_dim: int, padded_head_dim: int = PADDED_HEAD_DIM
) -> torch.Tensor:
    """``proj.weight [hidden, hidden]`` -> transposed ``[heads padded, hidden]`` with zero rows for the head pads."""

    hidden = heads * head_dim
    if tuple(weight.shape) != (hidden, hidden):
        raise ValueError(f"proj weight shape {tuple(weight.shape)} does not match {heads} x {head_dim}")
    return _pad_heads(
        weight.to(torch.float32).transpose(0, 1).contiguous(), heads, head_dim, padded_head_dim
    ).contiguous()


def padded_linear(
    weight: torch.Tensor, bias: torch.Tensor, *, out_features: int | None = None, in_features: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` / ``[out]`` -> transposed ``[in_features, out_features]`` and ``[out_features]``, zero-padded."""

    out_size, in_size = weight.shape
    out_features = out_size if out_features is None else out_features
    in_features = in_size if in_features is None else in_features
    if out_features < out_size or in_features < in_size or tuple(bias.shape) != (out_size,):
        raise ValueError(f"cannot pad {tuple(weight.shape)} / {tuple(bias.shape)} to {in_features} x {out_features}")
    transposed = torch.zeros(in_features, out_features, dtype=torch.float32)
    transposed[:in_size, :out_size] = weight.to(torch.float32).transpose(0, 1)
    padded_bias = torch.zeros(out_features, dtype=torch.float32)
    padded_bias[:out_size] = bias.to(torch.float32)
    return transposed.contiguous(), padded_bias


def block_layout(
    state_dict: Mapping[str, torch.Tensor], index: int, config: VisionTowerConfig
) -> dict[str, torch.Tensor]:
    """The FP32 host tensors of block ``index`` in device layout (transposed, padded, permuted as documented)."""

    prefix = f"blocks.{index}."
    qkv_w, qkv_b = fused_qkv_weight(
        state_dict[prefix + "attn.qkv.weight"],
        state_dict[prefix + "attn.qkv.bias"],
        heads=config.num_heads,
        head_dim=config.head_dim,
    )
    fc1_w, fc1_b = padded_linear(
        state_dict[prefix + "mlp.linear_fc1.weight"],
        state_dict[prefix + "mlp.linear_fc1.bias"],
        out_features=PADDED_INTERMEDIATE,
    )
    fc2_w, fc2_b = padded_linear(
        state_dict[prefix + "mlp.linear_fc2.weight"],
        state_dict[prefix + "mlp.linear_fc2.bias"],
        in_features=PADDED_INTERMEDIATE,
    )
    return {
        "norm1_weight": state_dict[prefix + "norm1.weight"].to(torch.float32),
        "norm1_bias": state_dict[prefix + "norm1.bias"].to(torch.float32),
        "qkv_weight": qkv_w,
        "qkv_bias": qkv_b,
        "proj_weight": attention_output_weight(
            state_dict[prefix + "attn.proj.weight"], heads=config.num_heads, head_dim=config.head_dim
        ),
        "proj_bias": state_dict[prefix + "attn.proj.bias"].to(torch.float32),
        "norm2_weight": state_dict[prefix + "norm2.weight"].to(torch.float32),
        "norm2_bias": state_dict[prefix + "norm2.bias"].to(torch.float32),
        "fc1_weight": fc1_w,
        "fc1_bias": fc1_b,
        "fc2_weight": fc2_w,
        "fc2_bias": fc2_b,
    }


def tower_layout(state_dict: Mapping[str, torch.Tensor], config: VisionTowerConfig) -> dict[str, object]:
    """Every device-resident tensor of the tower as FP32 host tensors in device layout (the position table stays a
    host table: its per-grid resampling is host work)."""

    patch_w, patch_b = padded_linear(
        state_dict["patch_embed.proj.weight"].reshape(config.hidden_size, config.patch_dim),
        state_dict["patch_embed.proj.bias"],
    )
    merger_fc1_w, merger_fc1_b = padded_linear(
        state_dict["merger.linear_fc1.weight"], state_dict["merger.linear_fc1.bias"]
    )
    merger_fc2_w, merger_fc2_b = padded_linear(
        state_dict["merger.linear_fc2.weight"], state_dict["merger.linear_fc2.bias"]
    )
    return {
        "patch_weight": patch_w,
        "patch_bias": patch_b,
        "blocks": [block_layout(state_dict, index, config) for index in range(config.depth)],
        "merger_norm_weight": state_dict["merger.norm.weight"].to(torch.float32),
        "merger_norm_bias": state_dict["merger.norm.bias"].to(torch.float32),
        "merger_fc1_weight": merger_fc1_w,
        "merger_fc1_bias": merger_fc1_b,
        "merger_fc2_weight": merger_fc2_w,
        "merger_fc2_bias": merger_fc2_b,
        "rotary_transformation": rotary_transformation_tile(),
    }
