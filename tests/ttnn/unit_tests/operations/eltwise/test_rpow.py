# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = pytest.mark.use_module_device


def run_rpow_test(device, h, w, base, pcc=0.99):
    torch.manual_seed(0)

    # rpow(x, base) = base^x
    # Use small positive range to avoid overflow/underflow with large exponents
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 4.0 - 1.0  # [-1, 3]

    # Golden: torch.pow(scalar_base, input_tensor)
    torch_output_tensor = torch.pow(torch.tensor(base), torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, base)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("base", [2.0, 3.0, 0.5, 10.0, 1.5])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rpow(device, h, w, base):
    run_rpow_test(device, h, w, base, pcc=0.99)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_rpow_base_one(device, h, w):
    """base=1 should always return 1.0 regardless of input"""
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 10.0 - 5.0
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, 1.0)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_rpow_zero_exponent(device, h, w):
    """x=0 should return 1.0 for any base (base^0 = 1)"""
    torch.manual_seed(0)
    torch_input_tensor = torch.zeros((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.ones_like(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.rpow(input_tensor, 2.0)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
