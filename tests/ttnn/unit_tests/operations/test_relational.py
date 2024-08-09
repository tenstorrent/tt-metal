# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_relational_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_relational_z_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gtz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gtz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gt(device, h, w):
    run_relational_test(device, h, w, ttnn.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ltz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.ltz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ge(device, h, w):
    run_relational_test(device, h, w, ttnn.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lt(device, h, w):
    run_relational_test(device, h, w, ttnn.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.lez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_le(device, h, w):
    run_relational_test(device, h, w, ttnn.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eqz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.eqz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eq(device, h, w):
    run_relational_test(device, h, w, ttnn.eq)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.nez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ne(device, h, w):
    run_relational_test(device, h, w, ttnn.ne)


def run_relational_test_with_scalar(device, h, w, scalar, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.full((h, w), fill_value=scalar, device=device, layout=ttnn.TILE_LAYOUT)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_gt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.gt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_ge(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ge)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_lt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.lt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_le(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.le)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_eq(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.eq)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_ne(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ne)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_gt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.gt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_ge(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ge)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_lt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.lt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_le(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.le)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_eq(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.eq)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_ne(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ne)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.lt)
    torch_output = golden_function(torch_a, torch_b)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.lt(a, b)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast_reversed(device, h, w):
    torch_input_tensor_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.lt)
    torch_output = golden_function(torch_input_tensor_b, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.lt(input_tensor_b, input_tensor_a)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output, output, 0.9999)


@pytest.mark.parametrize("atol", [1e-8, 1e-10])
@pytest.mark.parametrize("rtol", [1e-5, 1e-9])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isclose(device, h, w, atol, rtol):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((1, 1, h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.isclose)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, rtol=rtol, atol=atol)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.isclose(input_tensor_a, input_tensor_b, rtol=rtol, atol=atol)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor)
