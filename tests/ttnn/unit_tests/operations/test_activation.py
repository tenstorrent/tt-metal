# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import torch.nn.functional as F

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_activation_unary_test(device, h, w, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardtanh(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardtanh, F.hardtanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardswish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardswish, F.hardswish)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.log_sigmoid, F.logsigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.mish, lambda _x: F.mish(_x.to(torch.float)))


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu6(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.relu6, F.relu6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.gelu, F.gelu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hardsigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.hardsigmoid, F.hardsigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sigmoid(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sigmoid, torch.sigmoid)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.sign, torch.sign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_softsign(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.softsign, F.softsign)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_swish(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.swish, F.hardswish)


def run_activation_softplus_test(device, h, w, beta, threshold, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, beta=beta, threshold=threshold)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, beta, threshold)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("beta", [-1, 1, 2, 0.5, 10])
@pytest.mark.parametrize("threshold", [20, 40, -5, 10, -20])
def test_softplus(device, h, w, beta, threshold):
    run_activation_softplus_test(device, h, w, beta, threshold, ttnn.softplus, F.softplus)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanhshrink(device, h, w):
    run_activation_unary_test(device, h, w, ttnn.tanhshrink, F.tanhshrink)


def run_activation_unary_test_glu(device, batch_size, h, w, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor = torch.randn((batch_size, h, w), dtype=torch.bfloat16).unsqueeze(0)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_glu(device, batch_size, h, w):
    run_activation_unary_test_glu(device, batch_size, h, w, ttnn.glu, F.glu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_reglu(device, batch_size, h, w):
    run_activation_unary_test_glu(device, batch_size, h, w, ttnn.reglu, torch_reglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_swiglu(device, batch_size, h, w):
    run_activation_unary_test_glu(device, batch_size, h, w, ttnn.swiglu, torch_swiglu)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_geglu(device, batch_size, h, w):
    run_activation_unary_test_glu(device, batch_size, h, w, ttnn.geglu, torch_geglu)


def torch_reglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.relu(tensB)


def torch_swiglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.silu(tensB)


def torch_geglu(input_tensor, *args, **kwargs):
    split_size = input_tensor.size(-1) // 2
    split_tensors = torch.split(input_tensor, split_size_or_sections=[split_size, split_size], dim=-1)
    tensA, tensB = split_tensors[0], split_tensors[1]
    return tensA * F.gelu(tensB)


def torch_heaviside(x, *args, **kwargs):
    value = kwargs.pop("scalar")
    result = torch.heaviside(x, torch.tensor(value, dtype=x.dtype))
    return result


def torch_prelu(x, *args, **kwargs):
    weight = kwargs.pop("scalar")
    result = F.prelu(x, torch.tensor(weight, dtype=x.dtype))
    return result


def run_activation_test_scalarB(device, h, w, scalar, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_activation_test_scalarB_key(device, h, w, scalar, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar=scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_elu(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.elu, F.elu)


@pytest.mark.parametrize("scalar", [0.5, 1.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_hardshrink(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.hardshrink, F.hardshrink)


@pytest.mark.parametrize("scalar", [0.88])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_heaviside(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.heaviside, torch_heaviside)


@pytest.mark.parametrize("scalar", [-0.5, 0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_leaky_relu(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.leaky_relu, F.leaky_relu)


@pytest.mark.parametrize("scalar", [-0.5, 1.0, 0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_prelu(device, h, w, scalar):
    run_activation_test_scalarB_key(device, h, w, scalar, ttnn.prelu, torch_prelu)


@pytest.mark.parametrize("scalar", [0.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_softshrink(device, h, w, scalar):
    run_activation_test_scalarB(device, h, w, scalar, ttnn.softshrink, F.softshrink)


def run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar1=scalar1, scalar2=scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def torch_clip(x, *args, **kwargs):
    min = kwargs.pop("scalar1")
    max = kwargs.pop("scalar2")
    return torch.clamp(x, min=min, max=max)


@pytest.mark.parametrize("scalar1", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("scalar2", [0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarBC_clip(device, h, w, scalar1, scalar2):
    run_activation_test_scalarBC_key(device, h, w, scalar1, scalar2, ttnn.clip, torch_clip)


def run_activation_test_threshold(device, h, w, scalar1, scalar2, ttnn_function, torch_function, pcc=0.99):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, value=scalar1, threshold=scalar2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("value", [-0.5, -0.1, -5.5])
@pytest.mark.parametrize("threshold", [-0.5, 1.5, 27.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_threshold(device, h, w, value, threshold):
    run_activation_test_threshold(device, h, w, value, threshold, ttnn.threshold, F.threshold)
