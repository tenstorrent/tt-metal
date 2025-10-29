# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import torch

import ttnn


def pad_channels_up_to_target(input_tensor, target=16):
    assert len(input_tensor.shape) == 4, "Expected input tensor to rank 4"
    N, C, H, W = input_tensor.shape
    if C < target:
        return torch.nn.functional.pad(input_tensor, (0, 0, 0, 0, 0, target - C), mode="constant", value=0)
    else:
        return input_tensor


def create_random_input_tensor(
    batch: int,
    groups: int,
    input_channels: int = 4,
    input_height: int = 1056,
    input_width: int = 160,
    channel_order: Literal["first", "last"] = "last",
    fold: bool = True,
    pad: bool = True,
    device=None,
    memory_config=None,
    mesh_mapper=None,
):
    torch_input_tensor = torch.randn(batch, input_channels * groups, input_height, input_width)

    # We almost always (unless running full model) want to ensure we have least 16 because conv2d requires it
    ttnn_input_tensor = pad_channels_up_to_target(torch_input_tensor, 16) if pad else torch_input_tensor

    ttnn_input_tensor = ttnn_input_tensor if channel_order == "first" else ttnn_input_tensor.permute(0, 2, 3, 1)

    if fold:
        if channel_order == "first":
            ttnn_input_tensor = ttnn_input_tensor.reshape(batch, 1, input_channels * groups, input_height * input_width)
        else:
            ttnn_input_tensor = ttnn_input_tensor.reshape(batch, 1, input_height * input_width, -1)

    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, device=device, memory_config=memory_config, mesh_mapper=mesh_mapper
    )

    return torch_input_tensor, ttnn_input_tensor
