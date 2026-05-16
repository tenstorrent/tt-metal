# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN LayerNorm wrapper for RVC.

Used in Hubert encoder (~24 layers) and VITS encoder.
Stage 1: Correctness-first, interleaved, bfloat16.
"""

import torch
import ttnn
from typing import Optional


class TTNNLayerNorm:
    """
    LayerNorm wrapper using ttnn.layer_norm.

    Supports optional learnable weight and bias parameters.
    """

    def __init__(
        self,
        device,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        self.device = device
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.weight = None
        self.bias_param = None

    def load_weights(
        self,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Load learnable parameters.

        Args:
            weight: [normalized_shape] gamma parameter.
            bias: [normalized_shape] beta parameter.
        """
        if weight is not None:
            w = weight.unsqueeze(0).unsqueeze(0).float()
            self.weight = ttnn.from_torch(
                w,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if bias is not None:
            b = bias.unsqueeze(0).unsqueeze(0).float()
            self.bias_param = ttnn.from_torch(
                b,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass.

        Args:
            x: TTNN tensor on device, TILE_LAYOUT [..., normalized_shape].

        Returns:
            Layer-normalized TTNN tensor.
        """
        return ttnn.layer_norm(
            x,
            epsilon=self.eps,
            weight=self.weight,
            bias=self.bias_param,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def deallocate(self):
        """Free device memory."""
        if self.weight is not None:
            ttnn.deallocate(self.weight)
            self.weight = None
        if self.bias_param is not None:
            ttnn.deallocate(self.bias_param)
            self.bias_param = None
