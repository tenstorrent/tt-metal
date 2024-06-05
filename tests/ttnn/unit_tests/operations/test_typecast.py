# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from models.utility_functions import is_grayskull

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random

from loguru import logger


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
def test_typecast_output_tensor(device):
    if is_grayskull() and output_dtype in (ttnn.DataType.UINT32, ttnn.DataType.UINT16):
        pytest.skip("GS does not support fp32/uint32/uint16 data types")

    torch.manual_seed(0)

    h = w = 64
    gold_tensor_ones = ttnn.ones([h, w], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    tensor_ones_bfloat16 = ttnn.ones([h, w], ttnn.DataType.BFLOAT16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    logger.warning(tensor_ones_bfloat16)
    output_tensor_preallocated_uint32 = ttnn.zeros(
        [h, w], ttnn.DataType.UINT32, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    output = ttnn.typecast(
        tensor_ones_bfloat16, ttnn.DataType.UINT16, memory_config=ttnn.L1_MEMORY_CONFIG
    )  # , output_tensor=output_tensor_preallocated_uint32)
    # assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())
    logger.warning(output)
    logger.warning(output_tensor_preallocated_uint32)
    torch_gold_ones = ttnn.to_torch(gold_tensor_ones)
    torch_after_typecast = ttnn.to_torch(output_tensor_preallocated_uint32)
    logger.warning(torch_after_typecast)
    assert_with_pcc(torch_gold_ones, torch_after_typecast, 0.999)
