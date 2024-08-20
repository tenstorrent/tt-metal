# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import collections.abc
import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


import ttnn
from tt_lib.fallback_ops import fallback_ops


class TtSwinPatchEmbeddings(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.config = config
        self.device = device
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (
            image_size[0] // patch_size[0],
            image_size[1] // patch_size[1],
        )
        self.weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.projection.weight"], self.device)
        self.bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.projection.bias"], self.device)

        self.projection = fallback_ops.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            weights=self.weight,
            biases=self.bias,
            stride=patch_size,
        )

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = fallback_ops.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = fallback_ops.pad(pixel_values, pad_values)
        return pixel_values

    def forward(self, pixel_values: Optional[ttnn.Tensor]) -> Tuple[ttnn.Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.get_legacy_shape()
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        batch, channel, height, width = embeddings.get_legacy_shape()
        output_dimensions = (height, width)
        embeddings = fallback_ops.reshape(embeddings, 1, batch, channel, height * width)
        embeddings = ttnn.transpose(embeddings, -2, -1)

        return embeddings, output_dimensions
