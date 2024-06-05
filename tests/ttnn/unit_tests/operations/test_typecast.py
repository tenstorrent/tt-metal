# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import tt_lib
from models.utility_functions import is_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from loguru import logger


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
def test_typecast_output_tensor(device):
    if is_grayskull() and output_dtype in (ttnn.DataType.UINT32, ttnn.DataType.UINT16):
        pytest.skip("GS does not support fp32/uint32/uint16 data types")

    torch.manual_seed(0)

    h = w = 32
    gold_tensor = ttnn.ones([h, w], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    bfloat16_tensor = ttnn.ones([h, w], ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    uint32_preallocated = ttnn.empty([h, w], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    output_ttlib = tt_lib.tensor.typecast(bfloat16_tensor, ttnn.DataType.UINT32, ttnn.L1_MEMORY_CONFIG)
    logger.warning(output_ttlib)

    output_ttnn = ttnn.typecast(bfloat16_tensor, ttnn.DataType.UINT32, memory_config=ttnn.L1_MEMORY_CONFIG)
    logger.warning(output_ttnn)

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.typecast(
        bfloat16_tensor, ttnn.DataType.UINT32, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=uint32_preallocated
    )
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())
    logger.warning(uint32_preallocated)

    torch_gold = ttnn.to_torch(gold_tensor)
    torch_output_ttlib = ttnn.to_torch(output_ttlib)
    torch_output_ttnn = ttnn.to_torch(output_ttnn)
    assert_with_pcc(torch_gold, torch_output_ttlib, 0.999)
    assert_with_pcc(torch_gold, torch_output_ttnn, 0.999)
