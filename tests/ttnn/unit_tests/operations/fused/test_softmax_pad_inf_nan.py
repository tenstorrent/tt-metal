# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("width", [1, 15, 17, 24, 31, 33, 8193, 8200])
@pytest.mark.parametrize("pad_value", [float("inf"), float("nan"), -1.0])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_float32_pad_inf_nan(device, width, pad_value, numeric_stable):
    torch.manual_seed(0)
    torch_input = torch.randn((1, 1, 32, width), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, pad_value)
    output = ttnn.to_torch(ttnn.softmax(ttnn_input, dim=-1, numeric_stable=numeric_stable))
    reference = torch.softmax(torch_input.double(), dim=-1).float()
    assert torch.isfinite(output).all()
    if width > 1:
        assert_with_pcc(reference, output, 0.9999)
    assert torch.allclose(output, reference, rtol=0.01, atol=1e-6)
