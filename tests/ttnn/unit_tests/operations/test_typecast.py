# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import ttnn.deprecated
from models.utility_functions import is_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32/uint32/uint16 data types")
def test_typecast_output_tensor(device):
    torch.manual_seed(0)

    h = w = 32
    from_dtype = ttnn.bfloat16
    to_dtype = ttnn.uint32
    gold_tensor = ttnn.ones([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    bfloat16_tensor = ttnn.ones([h, w], from_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    uint32_preallocated = ttnn.empty([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    output_ttnn = ttnn.typecast(bfloat16_tensor, ttnn.uint32, memory_config=ttnn.L1_MEMORY_CONFIG)

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.typecast(bfloat16_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=uint32_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

    torch_gold = ttnn.to_torch(gold_tensor)
    torch_output_ttnn = ttnn.to_torch(output_ttnn)
    torch_output_ttnn_preallocated = ttnn.to_torch(uint32_preallocated)
    torch.equal(torch_gold, torch_output_ttnn)
    torch.equal(torch_gold, torch_output_ttnn_preallocated)
