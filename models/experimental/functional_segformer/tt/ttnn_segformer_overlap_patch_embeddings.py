# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_segformer.tt.common import Conv
import torch


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class TtSegformerOverlapPatchEmbeddings:
    """Construct the overlapping patch embeddings."""

    def __init__(self, parameters, model, stride, patch_size):
        super().__init__()
        if patch_size == 7:
            self.proj = model.proj  # parameters.proj
            self.torch_conv = True
        else:
            self.proj = Conv([stride, stride, patch_size // 2, patch_size // 2], parameters=parameters["proj"])
            self.torch_conv = False

    def __call__(
        self,
        pixel_values: ttnn.Tensor,
        parameters,
    ):
        device = pixel_values.device()

        if self.torch_conv:
            pixel_values = ttnn_to_torch(pixel_values)
            pixel_values = pixel_values.to(dtype=torch.float)
            embeddings = self.proj(pixel_values)
            pixel_values = torch_to_ttnn(pixel_values, device=device)
            embeddings = torch_to_ttnn(embeddings, device=device)
        else:
            pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))
            embeddings = self.proj(device, pixel_values)

            embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
            embeddings = ttnn.to_device(embeddings, device=device)

            embeddings = ttnn.permute(embeddings, (0, 3, 1, 2))
        batch_size, _, input_height, input_width = embeddings.shape

        ttnn.deallocate(pixel_values)
        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)

        embeddings = ttnn.reshape(
            embeddings, (embeddings.shape[0], embeddings.shape[1], embeddings.shape[2] * embeddings.shape[3])
        )
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device)
        embeddings = ttnn.permute(embeddings, (0, 2, 1))
        if len(embeddings.shape) == 2:
            embeddings = ttnn.reshape(embeddings, (1, embeddings.shape[0], embeddings.shape[1]))

        embeddings = ttnn.layer_norm(
            embeddings,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi,
            ),
        )
        return embeddings, input_height, input_width
