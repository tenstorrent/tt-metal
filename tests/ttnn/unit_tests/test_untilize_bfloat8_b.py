# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import is_grayskull, is_blackhole, torch_random, skip_for_grayskull


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("use_pack_untilize", [False, True])
@pytest.mark.parametrize("N", [1, 3, 5])
@pytest.mark.parametrize("C", [1, 2])
@pytest.mark.parametrize("H", [32, 448])
@pytest.mark.parametrize("W", [256, 672])
def test_untilize(device, in_dtype, use_multicore, use_pack_untilize, N, C, H, W):
    torch_input_shape = [N, C, H, W]

    torch_input = torch.randn(torch_input_shape).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize(ttnn_input, use_multicore=use_multicore, use_pack_untilize=use_pack_untilize)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("use_pack_untilize", [False, True])
@pytest.mark.parametrize("N", [1, 2])
@pytest.mark.parametrize("C", [1, 7])
@pytest.mark.parametrize("H", [128, 480])
@pytest.mark.parametrize("W", [32, 416])
@pytest.mark.parametrize("i_h", [10, 20])
@pytest.mark.parametrize("i_w", [2, 3])
def test_untilize_with_unpadding(device, in_dtype, use_multicore, use_pack_untilize, N, C, H, W, i_h, i_w):
    torch_input_shape = [N, C, H, W]

    torch_input = torch.randn(torch_input_shape).bfloat16()

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize_with_unpadding(
        ttnn_input, [N - 1, C - 1, H - i_h, W - i_w], use_multicore=use_multicore, use_pack_untilize=use_pack_untilize
    )
    output_torch = ttnn.to_torch(output_tt)
    torch_input = torch_input[:, :, : H - i_h + 1, : W - i_w + 1]
    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing
