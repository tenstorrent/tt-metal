# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5-27B RMSNorm implementation for TTNN.

Qwen3_5RMSNorm initializes weight to zeros and applies:
    output = rms_norm(x) * (1.0 + weight)

TTNN's rms_norm applies:
    output = rms_norm(x) * weight

So we pre-adjust the weight to (1.0 + weight) before converting to TTNN.
"""

import torch
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule


class TTNNQwen35RMSNorm(TTNNModule):
    """TTNN-accelerated RMSNorm for Qwen3.5-27B.

    Handles the (1 + weight) adjustment that is specific to Qwen3_5RMSNorm.
    """

    @classmethod
    def from_torch(cls, rms_norm):
        """Create TTNNQwen35RMSNorm from PyTorch Qwen3_5RMSNorm.

        Args:
            rms_norm: PyTorch Qwen3_5RMSNorm layer with .weight and .eps attributes.

        Returns:
            TTNNQwen35RMSNorm instance.
        """
        if rms_norm.weight is None:
            print(f"Warning: RMSNorm layer {rms_norm} has no weight. Returning original layer.")
            return rms_norm
        new_norm = cls()
        new_norm._fallback_torch_layer = rms_norm
        new_norm.eps = rms_norm.eps
        return new_norm

    def preprocess_weights_impl(self):
        """Preprocess RMSNorm weights for TTNN.

        Applies the (1.0 + weight) adjustment that Qwen3_5RMSNorm uses in forward,
        then expands to [32, dim] for TILE_LAYOUT compatibility and converts to ttnn tensor.
        """
        # Qwen3_5RMSNorm: output = rms_norm(x.float()) * (1.0 + weight.float())
        # TTNN rms_norm: output = rms_norm(x) * weight
        # So adjusted_weight = (1.0 + weight.float()).to(bfloat16)
        adjusted_weight = (1.0 + self.torch_layer.weight.float()).to(torch.bfloat16)

        # Expand to [32, dim] for TILE_LAYOUT (32 = tile height)
        self.tt_weight = ttnn.from_torch(
            adjusted_weight.unsqueeze(0).expand(32, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

    def move_weights_to_device_impl(self):
        """Move preprocessed weights to TTNN device."""
        if self.device.get_num_devices() > 1:
            weight_torch = ttnn.to_torch(self.tt_weight)
            self.tt_weight = ttnn.from_torch(
                weight_torch,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        else:
            self.tt_weight = ttnn.to_device(self.tt_weight, self.device)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: rms_norm with pre-adjusted (1+weight).

        Args:
            x: Input tensor in any layout. Will be converted to TILE_LAYOUT if needed.

        Returns:
            Normalized tensor with same shape as input.
        """
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.rms_norm(x, weight=self.tt_weight, epsilon=self.eps)
        return x
