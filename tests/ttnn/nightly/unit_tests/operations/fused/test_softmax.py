# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [[2, 10, 512, 8192]])
def test_ttnn_softmax_sdxl_attention(device, shape):
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_output = F.softmax(torch_input, dim=-1, dtype=torch.bfloat16)
    tt_output = ttnn.softmax(tt_input, dim=-1, numeric_stable=True)
    tt_output_torch = ttnn.to_torch(tt_output)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.999)

    # test program cache
    torch_output2 = F.softmax(torch_output, dim=-1, dtype=torch.bfloat16)
    tt_output2 = ttnn.softmax(tt_output, dim=-1, numeric_stable=True)
    tt_output_torch2 = ttnn.to_torch(tt_output2)
    assert_with_pcc(torch_output2, tt_output_torch2, pcc=0.999)
