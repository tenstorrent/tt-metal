# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Pure-torch reference implementations used by the PCC unit tests in ``tests/``.

These intentionally mirror the corresponding HF ``transformers`` modules
(``Lfm2ShortConv``, ``Lfm2MLP``/``Lfm2VlMultiModalProjector``) closely enough to be used
as golden references even when ``transformers`` does not yet ship LFM2-VL support in the
active environment (LFM2-VL requires ``transformers>=4.53``/``v5.1``, see ``README.md``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def round_ffn_dim_to_multiple(
    intermediate_size: int, multiple_of: int = 256, ffn_dim_multiplier: float | None = 1.0
) -> int:
    """LLaMA-style SwiGLU hidden-dim auto-adjustment ("block_auto_adjust_ff_dim" in LFM2's config).

    ``intermediate_size=12288`` -> ``8192`` for LFM2.5-VL-1.6B's text backbone.
    """
    ff_dim = int(2 * intermediate_size / 3)
    if ffn_dim_multiplier is not None:
        ff_dim = int(ffn_dim_multiplier * ff_dim)
    return multiple_of * ((ff_dim + multiple_of - 1) // multiple_of)


def _cast(t: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Match checkpoint weights (often bfloat16) to the reference input dtype (float32)."""
    return t.to(dtype) if t is not None else None


def swiglu_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor) -> torch.Tensor:
    """LFM2 (and tt_transformers) SwiGLU MLP: ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

    Weights are given in ``nn.Linear`` layout, i.e. ``w1: [ff, dim]``, ``w3: [ff, dim]``, ``w2: [dim, ff]``.
    """
    w1, w2, w3 = _cast(w1, x.dtype), _cast(w2, x.dtype), _cast(w3, x.dtype)
    gate = F.silu(F.linear(x, w1))
    up = F.linear(x, w3)
    return F.linear(gate * up, w2)


def short_conv(
    x: torch.Tensor,
    in_proj_w: torch.Tensor,
    out_proj_w: torch.Tensor,
    conv_w: torch.Tensor,
    in_proj_b: torch.Tensor | None = None,
    out_proj_b: torch.Tensor | None = None,
    conv_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference for LFM2's ``ShortConv`` operator.

    Args:
        x: [B, S, H]
        in_proj_w: [3H, H] (nn.Linear layout)
        out_proj_w: [H, H]
        conv_w: [H, 1, K] (depthwise nn.Conv1d layout, groups=H)
    Returns:
        y: [B, S, H]
    """
    hidden_size = x.shape[-1]
    kernel_size = conv_w.shape[-1]

    in_proj_w, out_proj_w, conv_w = _cast(in_proj_w, x.dtype), _cast(out_proj_w, x.dtype), _cast(conv_w, x.dtype)
    in_proj_b, out_proj_b, conv_b = _cast(in_proj_b, x.dtype), _cast(out_proj_b, x.dtype), _cast(conv_b, x.dtype)

    BCx = F.linear(x, in_proj_w, in_proj_b).transpose(-1, -2)  # [B, 3H, S]
    B_gate, C_gate, x_gate = BCx.split(hidden_size, dim=-2)  # each [B, H, S]
    Bx = B_gate * x_gate

    conv_out = F.conv1d(Bx, conv_w, bias=conv_b, padding=kernel_size - 1, groups=hidden_size)
    conv_out = conv_out[..., : Bx.shape[-1]]  # causal: drop the extra right-padding

    y = C_gate * conv_out  # [B, H, S]
    y = y.transpose(-1, -2)  # [B, S, H]
    return F.linear(y, out_proj_w, out_proj_b)


def pixel_unshuffle(x: torch.Tensor, factor: int = 2) -> torch.Tensor:
    """Space-to-depth pixel-unshuffle used by the LFM2-VL projector.

    Args:
        x: [B, H, W, C] (channels-last, as produced by reshaping the vision-tower sequence output
           back into its 2D patch grid).
    Returns:
        [B, H // factor, W // factor, C * factor * factor]
    """
    batch, height, width, channels = x.shape
    assert height % factor == 0 and width % factor == 0, (height, width, factor)
    x = x.reshape(batch, height // factor, factor, width // factor, factor, channels)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.reshape(batch, height // factor, width // factor, channels * factor * factor)


def lfm2_vl_projector(
    vision_features: torch.Tensor,
    linear_1_w: torch.Tensor,
    linear_2_w: torch.Tensor,
    linear_1_b: torch.Tensor | None = None,
    linear_2_b: torch.Tensor | None = None,
    downsample_factor: int = 2,
) -> torch.Tensor:
    """Reference for ``Lfm2VlMultiModalProjector`` (``projector_use_layernorm=False``).

    ``vision_features``: [B, num_patches, vision_dim] (SigLIP2 encoder output, square patch grid).
    Returns: [B, num_patches // factor**2, text_dim]
    """
    batch, num_patches, vision_dim = vision_features.shape
    side = int(round(num_patches**0.5))
    assert side * side == num_patches, num_patches

    dtype = vision_features.dtype
    linear_1_w, linear_2_w = _cast(linear_1_w, dtype), _cast(linear_2_w, dtype)
    linear_1_b, linear_2_b = _cast(linear_1_b, dtype), _cast(linear_2_b, dtype)

    x = vision_features.reshape(batch, side, side, vision_dim)
    x = pixel_unshuffle(x, factor=downsample_factor)
    x = x.reshape(batch, -1, vision_dim * downsample_factor * downsample_factor)

    x = F.linear(x, linear_1_w, linear_1_b)
    x = F.gelu(x)
    x = F.linear(x, linear_2_w, linear_2_b)
    return x
