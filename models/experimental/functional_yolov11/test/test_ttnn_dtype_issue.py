# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import ttnn
import torch
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_dtype_issue(device):
    a = torch.randn((1, 256, 1, 49), dtype=torch.bfloat16)
    a_ttnn = ttnn.from_torch(a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
    a_ttnn = ttnn.to_dtype(a_ttnn, ttnn.bfloat16)
    ttnn_output = ttnn.to_torch(a_ttnn)
    assert_with_pcc(a, ttnn_output, 0.99999)
