# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.sub,
    ],
)
def test_sub_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    z_torch = x_torch - y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_sub = ttnn.subtract(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_sub)
    print("inputs a, b", x_torch, y_torch)
    print(z_torch, ttnn.to_torch(z_tt), tt_out)
    print(
        "torch out",
        z_torch,
    )
    print("torch out in ttnn", ttnn.to_torch(z_tt))
    print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
def test_add_fp32(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.float32)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.float32)
    z_torch = x_torch + y_torch
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)
    print("inputs a, b", x_torch, y_torch)
    print(z_torch, ttnn.to_torch(z_tt), tt_out)
    # print("torch out", z_torch, )
    print("torch out in ttnn", ttnn.to_torch(z_tt))
    print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status


@skip_for_grayskull("Unsupported dtype for Grayskull")
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.add,
    ],
)
def test_add_bf16(device, ttnn_function):
    torch.set_printoptions(linewidth=200, threshold=10000, precision=15, sci_mode=False, edgeitems=17)
    x_torch = torch.tensor([[1]], dtype=torch.bfloat16)
    y_torch = torch.tensor([[0.00030171126]], dtype=torch.bfloat16)
    z_torch = torch.add(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.from_torch(z_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_add = ttnn.add(x_tt, y_tt)
    tt_out = ttnn.to_torch(z_tt_add)
    print("inputs a, b", x_torch, y_torch)
    print(z_torch, ttnn.to_torch(z_tt), tt_out)
    # print("torch out", z_torch, )
    print("torch out in ttnn", ttnn.to_torch(z_tt))
    print("tt out in torch", tt_out)
    status = torch.allclose(z_torch, tt_out, atol=1e-10, rtol=1e-5, equal_nan=False)
    assert status
