"""
This is the Conv2dPatch of Gemma-3-4b-it
Only difference: 4D weight tensor conversion to 2D for gemma compatibility
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.tt_transformers.tt.multimodal.llama_conv2d_patch import TtLlamaConv2dPatch


def create_gemma_conv2d_wrapper():
    """
    Creates a gemma-specific wrapper that handles 4D→2D weight conversion.
    """

    def gemma_state_dict_preprocessor(state_dict, state_dict_prefix, out_channels):
        modified_state_dict = state_dict.copy()
        weight_key = f"{state_dict_prefix}_linear.weight"

        if weight_key in modified_state_dict:
            weight = modified_state_dict[weight_key]
            # Gemma-specific: Convert 4D weights to 2D if needed
            if weight.ndim == 4:
                modified_state_dict[weight_key] = weight.view(out_channels, -1)

        return modified_state_dict

    return gemma_state_dict_preprocessor


class TtGemmaConv2dPatch:
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias,
    ):
        # Create gemma-specific state dict preprocessor
        preprocessor = create_gemma_conv2d_wrapper()

        # Preprocess state_dict for gemma 4D→2D weight conversion
        gemma_state_dict = preprocessor(state_dict, state_dict_prefix, out_channels)

        self.llama_conv2d = TtLlamaConv2dPatch(
            mesh_device=mesh_device,
            state_dict=gemma_state_dict,
            state_dict_prefix=state_dict_prefix,
            dtype=dtype,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def forward(self, x: torch.Tensor):
        return self.llama_conv2d.forward(x)

    def __call__(self, x: torch.Tensor):
        return self.forward(x)
