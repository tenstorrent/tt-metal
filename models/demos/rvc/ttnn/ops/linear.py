# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Linear wrapper for RVC.

Follows the Whisper implementation pattern:
    weights in DRAM → move activations to L1 → compute → result to DRAM.

Stage 1: Correctness-first, interleaved, bfloat16.
"""

import torch
import ttnn
from typing import Optional


class TTNNLinear:
    """
    Linear layer wrapper using ttnn.linear.

    Maintains weights on device (DRAM) and computes in L1.
    """

    def __init__(
        self,
        device,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        self.weight = None
        self.bias_param = None

    def load_weights(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Load PyTorch linear weights.

        Args:
            weight: [out_features, in_features] — will be transposed for TTNN.
            bias: [out_features] or None.
        """
        # ttnn.linear expects weight as [in_features, out_features]
        # PyTorch stores as [out_features, in_features], so we transpose
        w_t = weight.T.contiguous().float()
        self.weight = ttnn.from_torch(
            w_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        if bias is not None:
            # ttnn.linear expects bias as [1, 1, out_features] or broadcastable
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
            x: TTNN tensor on device, TILE_LAYOUT [*, in_features].

        Returns:
            TTNN tensor on device [*, out_features].
        """
        return ttnn.linear(
            x,
            self.weight,
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
