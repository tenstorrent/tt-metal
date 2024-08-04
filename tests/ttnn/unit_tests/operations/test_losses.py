# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
        [2, 1280, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "loss_mode",
    [
        ["none", ttnn.LossReductionMode.NONE],
        ["mean", ttnn.LossReductionMode.MEAN],
        ["sum", ttnn.LossReductionMode.SUM],
    ],
)
def test_mse_loss(device, input_shapes, loss_mode):
    torch_input_tensor_a = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((input_shapes), dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.mse_loss)
    torch_output_tensor = golden_fn(
        torch_input_tensor_a.to(torch.float32), torch_input_tensor_b.to(torch.float32), reduction=loss_mode[0]
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output_tensor = ttnn.mse_loss(input_tensor_a, input_tensor_b, reduction=loss_mode[1])
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if loss_mode[0] != "none":
        output_tensor = output_tensor[0, 0, 0, 0]

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [2, 1280, 8, 8],
        [2, 640, 16, 16],
        [2, 1280, 8, 8],
        [2, 1280, 16, 16],
        [2, 1280, 16, 16],
        [2, 1280, 32, 32],
    ],
)
@pytest.mark.parametrize(
    "loss_mode",
    [
        ["none", ttnn.LossReductionMode.NONE],
        ["mean", ttnn.LossReductionMode.MEAN],
        ["sum", ttnn.LossReductionMode.SUM],
    ],
)
def test_l1_loss(device, input_shapes, loss_mode):
    torch_input_tensor_a = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((input_shapes), dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.l1_loss)
    torch_output_tensor = golden_fn(
        torch_input_tensor_a.to(torch.float32), torch_input_tensor_b.to(torch.float32), reduction=loss_mode[0]
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)

    output_tensor = ttnn.l1_loss(input_tensor_a, input_tensor_b, reduction=loss_mode[1])
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if loss_mode[0] != "none":
        output_tensor = output_tensor[0, 0, 0, 0]

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
