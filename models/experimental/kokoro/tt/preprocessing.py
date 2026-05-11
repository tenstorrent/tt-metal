# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert PyTorch Kokoro weights into TTNN parameter tensors."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import ttnn


def preprocess_bert_encoder_linear(
    linear: nn.Linear,
    *,
    device: ttnn.MeshDevice,
    weights_dtype: Optional[Any] = None,
) -> dict[str, ttnn.Tensor]:
    """
    Pack `bert_encoder` nn.Linear for `ttnn.linear`.

    Reference: `reference/kokoro_plbert.py` — `d_en = bert_encoder(bert_dur).transpose(-1, -2)`.
    """
    if weights_dtype is None:
        weights_dtype = ttnn.bfloat16

    weight = linear.weight.data
    bias: Optional[torch.Tensor] = linear.bias.data if linear.bias is not None else None

    parameters: dict[str, ttnn.Tensor] = {
        "weight": ttnn.from_torch(
            weight,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        ),
    }
    if bias is not None:
        bias_tt = ttnn.from_torch(
            bias.reshape(1, 1, 1, -1),
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        parameters["bias"] = bias_tt
    return parameters
