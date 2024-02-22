# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def run_relational_test(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


def run_relational_z_test(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor, 0)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gtz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gtz, torch.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gt(device, h, w):
    run_relational_test(device, h, w, ttnn.gt, torch.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ltz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.ltz, torch.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gte(device, h, w):
    run_relational_test(device, h, w, ttnn.gte, torch.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gez, torch.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lt(device, h, w):
    run_relational_test(device, h, w, ttnn.lt, torch.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.lez, torch.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lte(device, h, w):
    run_relational_test(device, h, w, ttnn.lte, torch.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eqz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.eqz, torch.eq)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eq(device, h, w):
    run_relational_test(device, h, w, ttnn.eq, torch.eq)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.nez, torch.ne)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ne(device, h, w):
    run_relational_test(device, h, w, ttnn.ne, torch.ne)


def run_relational_test_scalarB(device, h, w, scalar, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_gt(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.gt, torch.gt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_gte(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.gte, torch.ge)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_lt(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.lt, torch.lt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_lte(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.lte, torch.le)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_eq(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.eq, torch.eq)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_ne(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.ne, torch.ne)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_gt(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.gt, torch.gt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_gte(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.gte, torch.ge)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_lt(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.lt, torch.lt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_lte(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.lte, torch.le)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_eq(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.eq, torch.eq)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_ne(device, h, w, scalar):
    run_relational_test_scalarB(device, h, w, scalar, ttnn.ne, torch.ne)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output = torch.lt(torch_a, torch_b)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.lt(a, b)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast_1(device, h, w):
    torch_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output = torch.lt(torch_b, torch_a)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.lt(b, a)
    tt_output = ttnn.to_torch(tt_output)

    assert_with_pcc(torch_output, tt_output, 0.9999)


def run_relational_isclose_test(device, h, w, atol, rtol, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((h, w), dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b, rtol=rtol, atol=atol)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b, rtol, atol)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("atol", [1e-8, 1e-10])
@pytest.mark.parametrize("rtol", [1e-5, 1e-9])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isclose_a(device, h, w, atol, rtol):
    run_relational_isclose_test(device, h, w, atol, rtol, ttnn.isclose, torch.isclose)
