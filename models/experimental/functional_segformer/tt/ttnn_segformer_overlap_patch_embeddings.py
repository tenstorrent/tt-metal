# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib


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

    def __init__(self, parameters, model):
        super().__init__()
        if model.proj.stride[0] == 4:
            self.proj = model.proj  # parameters.proj
        else:
            self.proj = parameters.proj

    def __call__(self, pixel_values: ttnn.Tensor, parameters, model):
        device = pixel_values.device()
        if model.proj.stride[0] == 4:
            pixel_values = ttnn_to_torch(pixel_values)
            pixel_values = pixel_values.to(torch.float)
            embeddings = self.proj(pixel_values)
            pixel_values = torch_to_ttnn(pixel_values, device)
            embeddings = torch_to_ttnn(embeddings, device)
        else:
            pixel_values = ttnn.permute(pixel_values, (0, 2, 3, 1))
            pixel_values = tt_lib.tensor.interleaved_to_sharded(
                pixel_values, self.proj.conv.input_sharded_memory_config
            )
            embeddings = self.proj(pixel_values)
            embeddings = self.proj.copy_output_from_device(embeddings)
            embeddings = ttnn.to_device(embeddings, device)
            embeddings = ttnn.permute(embeddings, (0, 3, 1, 2))

        embeddings = ttnn.from_device(embeddings)
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.ROW_MAJOR_LAYOUT)

        embeddings_hw = embeddings  # used instead of shape

        embeddings = ttnn.reshape(
            embeddings, (embeddings.shape[0], embeddings.shape[1], embeddings.shape[2] * embeddings.shape[3])
        )
        embeddings = ttnn.to_layout(embeddings, layout=ttnn.TILE_LAYOUT)
        embeddings = ttnn.to_device(embeddings, device)
        embeddings = ttnn.permute(embeddings, (0, 2, 1))
        if len(embeddings.shape) == 2:
            embeddings = ttnn.reshape(embeddings, (1, embeddings.shape[0], embeddings.shape[1]))
        embeddings = ttnn.layer_norm(embeddings, weight=parameters.layer_norm.weight, bias=parameters.layer_norm.bias)
        return embeddings, embeddings_hw
