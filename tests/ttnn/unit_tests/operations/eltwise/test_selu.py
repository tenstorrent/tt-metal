# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp

pytestmark = pytest.mark.use_module_device


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_selu_bfloat16(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.bfloat16)
    golden = torch.nn.functional.selu(input_tensor)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.selu(tt_input)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize("h,w", [(64, 128)])
def test_selu_fp32(device, h, w):
    torch.manual_seed(0)
    input_tensor = torch.randn((h, w), dtype=torch.float32)
    golden = torch.nn.functional.selu(input_tensor)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.selu(tt_input)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden, result, 0.99)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 64),
        (2, 4, 64, 128),
    ],
)
def test_selu_shapes_bfloat16(device, shape):
    torch.manual_seed(0)
    input_tensor = torch.randn(shape, dtype=torch.bfloat16) * 3.0
    golden = torch.nn.functional.selu(input_tensor)
    tt_input = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.selu(tt_input)
    result = ttnn.to_torch(tt_output)
    assert_with_pcc(golden, result, 0.99)
