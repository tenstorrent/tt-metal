# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import is_grayskull, is_blackhole, torch_random, skip_for_grayskull


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("shape", [[1, 1, 1, 50304], [1, 3, 5, 25152], [17, 32, 10944]])
def test_tilize_with_zero_padding(device, in_dtype, use_multicore, shape):
    torch_input = torch.randn(shape, dtype=torch.bfloat16).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output_tt = ttnn.tilize_with_zero_padding(ttnn_input, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing
