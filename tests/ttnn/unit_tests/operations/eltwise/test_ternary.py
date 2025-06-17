# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import isnan
from tests.ttnn.utils_for_testing import assert_with_pcc


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
        from_torch_if_tensor(arg) for arg in (torch_input_tensor, torch_input1, torch_input2)
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
        [64, 64, 64, 128, 128, 128],
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize("scalar", [15.5, float("nan")])
def test_where_tts(device, dtype, h, w, scalar):
    if dtype == torch.float32 and isnan(scalar):
        pytest.xfail("#22308 ttnn.where erroneously propagates NaNs")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)

    assert_where_with_pcc(torch_input_tensor, torch_input_tensor1, scalar, device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize("scalar", [15.5, float("nan")])
def test_where_tst(device, dtype, h, w, scalar):
    if dtype == torch.float32 and isnan(scalar):
        pytest.xfail("#22308 ttnn.where erroneously propagates NaNs")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)
    torch_input_tensor1 = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)

    assert_where_with_pcc(torch_input_tensor, scalar, torch_input_tensor1, device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("h, w", [[64, 128]])
@pytest.mark.parametrize("scalar1, scalar2", [[15.5, 31.2], [15.5, float("nan")], [float("nan"), 31.2]])
def test_where_tss(device, dtype, h, w, scalar1, scalar2):
    if dtype == torch.float32 and (isnan(scalar1) or isnan(scalar2)):
        pytest.xfail("#22308 ttnn.where erroneously propagates NaNs")

    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=dtype).uniform_(-100, 100)

    assert_where_with_pcc(torch_input_tensor, scalar1, scalar2, device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_where_nans(device, dtype):
    if dtype == torch.float32:
        pytest.xfail("#22308 ttnn.where erroneously propagates NaNs")

    torch.manual_seed(0)

    C = torch.ones(1, 4, 1, dtype=dtype)
    T = torch.randn(1, 4, 768, dtype=dtype)
    F = torch.full((1, 4, 768), float("nan"), dtype=dtype)

    assert_where_with_pcc(C, T, F, device)


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
