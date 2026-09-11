# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import torch.nn as nn
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


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

    if loss_mode[0] in ("mean", "sum"):
        # Reduced losses are scalars; PCC is undefined on constant tensors.
        assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=3)
    else:
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

    if loss_mode[0] in ("mean", "sum"):
        # Reduced losses are scalars; PCC is undefined on constant tensors.
        assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=3)
    else:
        assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [2, 64, 32, 32],
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
def test_mse_loss_with_output_tensor(device, input_shapes, loss_mode):
    torch_input_tensor_a = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((input_shapes), dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.mse_loss)
    torch_output_tensor = golden_fn(
        torch_input_tensor_a.to(torch.float32), torch_input_tensor_b.to(torch.float32), reduction=loss_mode[0]
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    out_shape = input_shapes if loss_mode[0] == "none" else [1, 1, 32, 32]
    preallocated_out = ttnn.empty(out_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    returned_tensor = ttnn.mse_loss(
        input_tensor_a, input_tensor_b, reduction=loss_mode[1], output_tensor=preallocated_out
    )

    actual_res = ttnn.to_torch(returned_tensor)
    actual_out = ttnn.to_torch(preallocated_out)

    if loss_mode[0] in ("mean", "sum"):
        assert_with_ulp(torch_output_tensor, actual_res, ulp_threshold=3)
        assert_with_ulp(torch_output_tensor, actual_out, ulp_threshold=3)
    else:
        assert_with_pcc(torch_output_tensor, actual_res, 0.9999)
        assert_with_pcc(torch_output_tensor, actual_out, 0.9999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [1, 1, 32, 32],
        [2, 64, 32, 32],
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
def test_l1_loss_with_output_tensor(device, input_shapes, loss_mode):
    torch_input_tensor_a = torch.randn((input_shapes), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((input_shapes), dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.l1_loss)
    torch_output_tensor = golden_fn(
        torch_input_tensor_a.to(torch.float32), torch_input_tensor_b.to(torch.float32), reduction=loss_mode[0]
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    out_shape = input_shapes if loss_mode[0] == "none" else [1, 1, 32, 32]
    preallocated_out = ttnn.empty(out_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    returned_tensor = ttnn.l1_loss(
        input_tensor_a, input_tensor_b, reduction=loss_mode[1], output_tensor=preallocated_out
    )

    actual_res = ttnn.to_torch(returned_tensor)
    actual_out = ttnn.to_torch(preallocated_out)

    if loss_mode[0] in ("mean", "sum"):
        assert_with_ulp(torch_output_tensor, actual_res, ulp_threshold=3)
        assert_with_ulp(torch_output_tensor, actual_out, ulp_threshold=3)
    else:
        assert_with_pcc(torch_output_tensor, actual_res, 0.9999)
        assert_with_pcc(torch_output_tensor, actual_out, 0.9999)

