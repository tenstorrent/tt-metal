# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import isnan
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp, assert_equal


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_mac_all_tensors(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor1 = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor2 = torch.rand((h, w), dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

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

    golden_fn = ttnn.get_golden_function(ttnn.mac)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)

    output_tensor = ttnn.mac(input_tensor, scalar1, scalar2)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)


def assert_where_with_pcc(torch_input_tensor, torch_input1, torch_input2, device, pcc=0.9999):
    def from_torch_if_tensor(x):
        if not isinstance(x, torch.Tensor):
            return x

        return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor, input1, input2 = (
        from_torch_if_tensor(arg)
        for arg in ((torch_input_tensor > 0).to(torch_input_tensor.dtype), torch_input1, torch_input2)
    )
    golden_fn = ttnn.get_golden_function(ttnn.where)
    torch_output_tensor = golden_fn(torch_input_tensor > 0, torch_input1, torch_input2)
    output_tensor = ttnn.where(input_tensor, input1, input2)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [64, 1, 64, 128, 128, 1],
        [64, 64, 1, 128, 1, 128],
        [1, 1, 64, 128, 128, 1],
        [64, 1, 64, 1, 1, 128],
    ],
)
def test_where_bcast(device, dtype, hc, ht, hf, wc, wt, wf):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=dtype).uniform_(-100, 100)

    assert_where_with_pcc(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, device)


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
def test_addcdiv(device, h, w, value):
    run_ternary_test_value(device, h, w, value, ttnn.addcdiv)


@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [15.5, 0, 1.0, -5.0])
@pytest.mark.parametrize(
    "hc, ht, hf, wc, wt, wf",
    [
        [64, 64, 64, 128, 128, 128],  # no bcast
        # Row / Col bcast cases
        [64, 64, 64, 128, 128, 1],
        [64, 64, 64, 128, 1, 128],
        [64, 64, 64, 1, 128, 128],
        [64, 64, 1, 128, 128, 128],
        [64, 1, 64, 128, 128, 128],
        [1, 64, 64, 128, 128, 128],
        [64, 64, 1, 128, 128, 1],
        [64, 1, 64, 128, 1, 128],
        [1, 64, 64, 1, 128, 128],
        [64, 1, 1, 128, 1, 1],  # scalar bcast case
    ],
)
def test_addcmul_with_bcast(device, tor_dtype, ttnn_dtype, hc, ht, hf, wc, wt, wf, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((hc, wc), dtype=tor_dtype).uniform_(-100, 500)
    torch_input_tensor1 = torch.rand((ht, wt), dtype=tor_dtype).uniform_(-200, 200)
    torch_input_tensor2 = torch.rand((hf, wf), dtype=tor_dtype).uniform_(-300, 400)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor = ttnn.to_device(input_tensor, device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.to_device(input_tensor1, device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.to_device(input_tensor2, device)
    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat8_b),
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.0,
        -0.5,
    ],
)
@pytest.mark.parametrize(
    "a_shape, b_shape, c_shape",
    [
        ((1, 2, 1088, 1024), (1, 2, 1, 1024), (1, 2, 1088, 1024)),  # Composite
        ((1, 2, 1088, 1024), (1, 2, 1, 1), (1, 2, 1088, 1024)),  # Composite
        ((4, 2, 1088, 1024), (1, 2, 1088, 1024), (1, 1, 1088, 1024)),  # HLK
    ],
)
def test_addcmul_with_bcast_bf8b(device, torch_dtype, ttnn_dtype, a_shape, b_shape, c_shape, value):
    """
    Test addcmul: Block format datatype inputs with subtile broadcast use composite Addcmul implementation.
    """
    torch.manual_seed(0)

    torch_input_tensor = torch.randn(a_shape, dtype=torch_dtype)
    torch_input_tensor1 = torch.randn(b_shape, dtype=torch_dtype)
    torch_input_tensor2 = torch.randn(c_shape, dtype=torch_dtype)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor, input_tensor1, input_tensor2, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    torch_output_tensor = golden_fn(torch_input_tensor, torch_input_tensor1, torch_input_tensor2, value=value)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
@pytest.mark.parametrize("value", [1.0, 0.5])
@pytest.mark.parametrize(
    "in_data1_shape, in_data2_shape, in_data3_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32), (1, 1, 32, 32)),
        ((1, 1, 1, 1024), (1, 1, 1024, 1024), (1, 1, 1, 1024)),
        ((1, 1, 1024, 1), (1, 1, 1024, 1024), (1, 1, 1024, 1)),
        ((1, 1, 1, 1), (1, 1, 1024, 1024), (1, 1, 1, 1)),
    ],
)
def test_addcmul(device, torch_dtype, ttnn_dtype, value, in_data1_shape, in_data2_shape, in_data3_shape):
    in_data1 = torch.full(in_data1_shape, 0.0031, dtype=torch_dtype)
    in_data2 = torch.full(in_data2_shape, 508.0, dtype=torch_dtype)
    in_data3 = torch.full(in_data3_shape, 748.0, dtype=torch_dtype)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_with_ulp(output_tensor, golden_tensor)


def test_addcmul_with_int32_inputs(device):
    in_data1 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    in_data2 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    in_data3 = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch.int32)
    value = 1
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor3 = ttnn.from_torch(in_data3, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.addcmul(input_tensor1, input_tensor2, input_tensor3, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    golden_fn = ttnn.get_golden_function(ttnn.addcmul)
    golden_tensor = golden_fn(in_data1, in_data2, in_data3, value=value)

    assert_equal(golden_tensor, output_tensor)
