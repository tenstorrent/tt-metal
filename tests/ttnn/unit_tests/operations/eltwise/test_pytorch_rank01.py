# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

#  mul(Tensor<[]> self = ?, Tensor<[1, 24, 768]> other = ?) // scalar, 3d
with ttnn.manage_device(device_id=0) as device:
    a_py = torch.rand((), dtype=torch.bfloat16)
    a = ttnn.from_torch(a_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(a_py.shape)
    b_py = torch.rand(([1, 24, 768]), dtype=torch.bfloat16)
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape)
    pt_out = torch.mul(a_py, b_py)
    res = ttnn.mul(a, b, use_legacy=False)
    tt_out = ttnn.to_torch(res)

    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 1 : pcc_msg", pcc_msg)
    assert pcc

# mul(Tensor<[2, 1]> self = ?, Tensor<[]> other = ?) // 2d, scalar
with ttnn.manage_device(device_id=0) as device:
    a_py = torch.rand(([2, 1]), dtype=torch.bfloat16)
    a = ttnn.from_torch(a_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(a_py.shape)
    b_py = torch.rand((), dtype=torch.bfloat16)
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape)
    pt_out = torch.mul(a_py, b_py)
    res = ttnn.mul(a, b, use_legacy=False)
    tt_out = ttnn.to_torch(res)

    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 2 : pcc_msg", pcc_msg)
    assert pcc

# exp(Tensor<[]> self = ?) // scalar tensor
with ttnn.manage_device(device_id=0) as device:
    b_py = torch.rand((), dtype=torch.bfloat16)
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape, b_py)
    pt_out = torch.exp(b_py)
    res = ttnn.exp(b)
    tt_out = ttnn.to_torch(res)
    print(pt_out, tt_out)
    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 3 : pcc_msg", pcc_msg)
    assert pcc


# where(Tensor<[1, 1, 7, 7]> condition = ?, Tensor<[1, 12, 7, 7]> self = ?, Tensor<[]> other = ?) // 4d, 4d, scalar
with ttnn.manage_device(device_id=0) as device:
    a_py = torch.ones(([1, 1, 7, 7]), dtype=torch.bfloat16)
    a = ttnn.from_torch(a_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    a_py = a_py.bool()
    print(a_py.shape)
    b_py = torch.rand(([1, 12, 7, 7]), dtype=torch.bfloat16)
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape)
    c_py = torch.rand((), dtype=torch.bfloat16)
    c = ttnn.from_torch(c_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(c_py.shape)
    pt_out = torch.where(a_py, b_py, c_py)
    res = ttnn.where(a, b, c)
    tt_out = ttnn.to_torch(res)

    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 4 : pcc_msg", pcc_msg)
    assert pcc

# rank 0 and 1 mul [], 1D
with ttnn.manage_device(device_id=0) as device:
    a_py = torch.rand((), dtype=torch.bfloat16)
    a = ttnn.from_torch(a_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(a_py.shape)
    b_py = torch.rand(([24]), dtype=torch.bfloat16)
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape)
    pt_out = torch.mul(a_py, b_py)
    res = ttnn.mul(a, b, use_legacy=False)
    tt_out = ttnn.to_torch(res)

    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 5 : pcc_msg", pcc_msg)
    assert pcc


@pytest.mark.parametrize(
    "input_shape_a, input_shape_b",
    [
        ((1, 16, 1, 6), ()),
        ((1, 23, 40, 1), (128,)),
        ((1, 23, 40), (1, 23, 1)),
        ((1, 23, 40), (1, 1, 40)),
        ((1, 512), (1, 1)),
        ((3, 480, 640), (3, 1, 1)),
        ((3, 320, 320), (3, 1, 1)),
        ((96, 80), (80,)),
        ((1, 12, 7, 7), ()),
        ((2, 512), (2, 1)),
    ],
)
def test_div_pytorch2(device, input_shape_a, input_shape_b):
    a_py = torch.rand(input_shape_a, dtype=torch.bfloat16) * (100 + 100) - 100
    a = ttnn.from_torch(a_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(a_py.shape)
    b_py = torch.rand(input_shape_b, dtype=torch.bfloat16) * (80 - 1) + 1
    b = ttnn.from_torch(b_py, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print(b_py.shape)
    pt_out = torch.div(a_py, b_py)
    res = ttnn.divide(a, b, use_legacy=False)
    tt_out = ttnn.to_torch(res)

    pcc, pcc_msg = assert_with_pcc(pt_out, tt_out, 0.999)
    print("test 6 div : pcc_msg", pcc_msg)
    assert pcc
