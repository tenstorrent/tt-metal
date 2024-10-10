# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_mac(input, tensor1, tensor2):
    return torch.add(torch.mul(input, tensor1), tensor2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mac_all_tensors(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_mac(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.mac(input_tensor, input_tensor1, input_tensor2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar1", [5.5])
@pytest.mark.parametrize("scalar2", [-13.2])
def test_mac_tensor_with_2_scalaras(device, h, w, scalar1, scalar2):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = scalar1
    torch_input_tensor2 = scalar2
    torch_output_tensor = torch.unsqueeze(
        torch.unsqueeze(torch_mac(torch_input_tensor, torch_input_tensor1, torch_input_tensor2), 0), 0
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.mac(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_where_all_tensors(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor.to(torch.bool), torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.where(input_tensor, input_tensor1, input_tensor2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar", [15.5])
def test_where_tts(device, h, w, scalar):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor.to(torch.bool), torch_input_tensor1, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)

    output_tensor = ttnn.where(input_tensor, input_tensor1, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


def run_ternary_test_where_TST(device, h, w, scalar, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor.to(torch.bool), scalar, torch_input_tensor1)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)

    output_tensor = ttnn_function(input_tensor, scalar, input_tensor1)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar", [15.5])
def test_where_tst(device, h, w, scalar):
    run_ternary_test_where_TST(device, h, w, scalar, ttnn.where, torch.where)


def run_ternary_test_where_TSS(device, h, w, scalar1, scalar2, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor.to(torch.bool), scalar1, scalar2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn_function(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("scalar1", [15.5])
@pytest.mark.parametrize("scalar2", [31.2])
def test_where_tss(device, h, w, scalar1, scalar2):
    run_ternary_test_where_TSS(device, h, w, scalar1, scalar2, ttnn.where)


def run_ternary_test_value(device, h, w, value, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16).uniform_(-100, 100)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn_function(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("value", [15.5])
def test_addcmul(device, h, w, value):
    run_ternary_test_value(device, h, w, value, ttnn.addcmul)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("value", [15.5])
def test_addcdiv(device, h, w, value):
    run_ternary_test_value(device, h, w, value, ttnn.addcdiv)
