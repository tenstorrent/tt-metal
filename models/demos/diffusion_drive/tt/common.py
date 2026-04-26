# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared TTNN pre-processing utilities for DiffusionDrive.

Two entry points:
  fold_bn(conv, bn) -> (w_folded, b_folded)
      Fuses BatchNorm into the preceding Conv2d weight/bias at fp32 precision.

  custom_preprocessor(torch_model, name) -> parameters
      Used by ttnn.model_preprocessing.preprocess_model_parameters.
      Handles Conv2d (with BN-fold for Conv+BN pairs) and Linear.
      LayerNorm is intentionally NOT handled here — let the default handler
      call preprocess_layernorm_parameter() which reshapes weight/bias to
      (1, d_model) TILE_LAYOUT satisfying ttnn.layer_norm's shape constraint.

Notes from CLAUDE.md:
  - preprocess_model_parameters keyword is `initialize_model=`, not `init_model=`
  - LayerNorm: default handler is correct; do NOT add a branch here
  - Conv has bias=False in ResNet-34; fold_bn creates a zero-bias tensor
  - Compute fold in fp32; cast to bfloat16 at the end to preserve precision
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# ttnn is imported lazily inside custom_preprocessor so that the fold_bn
# helper (and its unit tests) can be used without an active ttnn import.


# ---------------------------------------------------------------------------
# BN-fold helper (gap 28–30 from plan)
# ---------------------------------------------------------------------------


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse BatchNorm parameters into Conv2d weight/bias.

    Always computed in fp32; cast to bfloat16 at the end.

    Handles:
      - Conv2d with bias=False (creates a zero-bias tensor)
      - Any kernel size (1×1 downsampling shortcuts, 3×3 blocks, 7×7 stem)

    Returns:
        w_folded: (out_ch, in_ch/groups, kH, kW) bfloat16
        b_folded: (out_ch,)                        bfloat16
    """
    # Work in fp32 to avoid precision loss in sqrt / division
    w = conv.weight.float()  # (out_ch, in_ch/groups, kH, kW)
    b_conv = conv.bias.float() if conv.bias is not None else torch.zeros(conv.out_channels, dtype=torch.float32)

    g = bn.weight.float()
    mu = bn.running_mean.float()
    var = bn.running_var.float()
    eps = bn.eps
    beta = bn.bias.float()

    scale = g / torch.sqrt(var + eps)  # (out_ch,)

    w_folded = w * scale.reshape(-1, 1, 1, 1)
    b_folded = (b_conv - mu) * scale + beta

    return w_folded.to(torch.bfloat16), b_folded.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# TTNN custom preprocessor
# ---------------------------------------------------------------------------


def custom_preprocessor(torch_model: nn.Module, name: str) -> dict:
    """TTNN parameter preprocessor for DiffusionDrive.

    Called by ttnn.model_preprocessing.preprocess_model_parameters for each
    submodule in the model tree.

    Conv+BN folding is handled at the ResNet / FPN module level by explicitly
    calling fold_bn() before building the TTNN conv weight tensors; this
    function handles standalone Conv2d and Linear layers.

    LayerNorm is intentionally NOT handled here (see module docstring).
    """
    from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

    import ttnn  # lazy import — fold_bn users don't need ttnn

    parameters: dict = {}

    if isinstance(torch_model, nn.Linear):
        parameters["weight"] = preprocess_linear_weight(torch_model.weight, dtype=ttnn.bfloat16)
        if torch_model.bias is not None:
            parameters["bias"] = preprocess_linear_bias(torch_model.bias, dtype=ttnn.bfloat16)

    # Conv2d: return plain tensors (no BN fold here — fold is done upstream
    # when Conv+BN pairs are processed together in ttnn_resnet34.py)
    # LayerNorm: fall through to default handler

    return parameters
