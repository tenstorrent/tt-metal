# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_lgamma_test(device, h, w, pcc=0.999, rtol=1.6e-2, atol=1e-2):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 9.9 + 0.1
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, pcc)
    logger.info(f"PCC: {pcc_value} (threshold: {pcc})")
    allclose_passing = torch.allclose(torch_output_tensor.float(), output_tensor.float(), rtol=rtol, atol=atol)
    if not allclose_passing:
        max_diff = (torch_output_tensor.float() - output_tensor.float()).abs().max().item()
        logger.warning(f"allclose failed: max_diff={max_diff}, rtol={rtol}, atol={atol}")
    return passing, pcc_value


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
        (128, 128),
        (32, 64),
        (64, 128),
    ],
)
def test_lgamma(device, h, w):
    passing, pcc_value = run_lgamma_test(device, h, w, pcc=0.999)
    assert passing, f"PCC {pcc_value} below threshold 0.999"


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
    ],
)
def test_lgamma_small_inputs(device, h, w):
    torch.manual_seed(42)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 0.9 + 0.1
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Small inputs PCC: {pcc_value}")
    assert passing, f"PCC {pcc_value} below threshold 0.999 for small inputs"


@pytest.mark.parametrize(
    "h, w",
    [
        (32, 32),
        (64, 64),
    ],
)
def test_lgamma_large_inputs(device, h, w):
    torch.manual_seed(123)
    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16) * 5.0 + 5.0
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Large inputs PCC: {pcc_value}")
    assert passing, f"PCC {pcc_value} below threshold 0.999 for large inputs"


def test_lgamma_special_values(device):
    torch_input_tensor = torch.tensor([[1.0, 2.0, 1.0, 2.0]] * 8, dtype=torch.bfloat16).reshape(32, 1).expand(32, 32)
    torch_output_tensor = torch.lgamma(torch_input_tensor)
    input_tensor = ttnn.from_torch(torch_input_tensor.contiguous(), layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.lgamma(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    passing, pcc_value = assert_with_pcc(torch_output_tensor, output_tensor, 0.999)
    logger.info(f"Special values PCC: {pcc_value}")
    max_abs = output_tensor.float().abs().max().item()
    logger.info(f"Special values max abs value: {max_abs}")
    assert max_abs < 0.2, f"lgamma(1) and lgamma(2) should be ~0, got max abs {max_abs}"
