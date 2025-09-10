# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import is_grayskull, is_blackhole, torch_random, skip_for_grayskull


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.uint16:
        return torch.randint(0, 100, shape).to(torch.int16)
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    return torch.rand(shape).bfloat16().float()


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("use_pack_untilize", [False, True])
@pytest.mark.parametrize("H", [32, 512])
@pytest.mark.parametrize("W", [1024, 256])
def test_untilize_2D(device, in_dtype, use_multicore, use_pack_untilize, H, W):
    if in_dtype in [ttnn.uint32, ttnn.int32] and not use_pack_untilize:
        pytest.skip(f"Skipping: dtype {in_dtype} with use_pack_untilize=False is unsupported")
    torch_input_shape = [H, W]

    torch_input = random_torch_tensor(in_dtype, torch_input_shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize(ttnn_input, use_multicore=use_multicore, use_pack_untilize=use_pack_untilize)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("H", [128, 2048])
@pytest.mark.parametrize("W", [32, 1056])
def test_tilize_2D(device, in_dtype, use_multicore, H, W):
    torch_input_shape = [H, W]

    torch_input = random_torch_tensor(in_dtype, torch_input_shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output_tt = ttnn.tilize(ttnn_input, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("use_multicore", [False, True])
# Fails on int32 if use_pack_untilize is False AND dtype is uint32/int32
@pytest.mark.parametrize("use_pack_untilize", [False, True])
@pytest.mark.parametrize("H", [32, 43])
@pytest.mark.parametrize("W", [64, 76])
def test_untilize_with_unpadding_2D(device, in_dtype, use_multicore, use_pack_untilize, H, W):
    if in_dtype in [ttnn.uint32, ttnn.int32] and not use_pack_untilize:
        pytest.skip(f"Skipping: dtype {in_dtype} with use_pack_untilize=False is unsupported")
    torch_input_shape = [H, W]

    torch_input = random_torch_tensor(in_dtype, torch_input_shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.untilize_with_unpadding(
        ttnn_input, [H - 1, W - 1], use_multicore=use_multicore, use_pack_untilize=use_pack_untilize
    )
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("pad_value", [2, 1.3])
@pytest.mark.parametrize("H", [32, 43])
@pytest.mark.parametrize("W", [64, 76])
def test_tilize_with_val_padding_2D(device, in_dtype, use_multicore, H, W, pad_value):
    torch_input_shape = [H, W]

    torch_input = random_torch_tensor(in_dtype, torch_input_shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output_tt = ttnn.tilize_with_val_padding(ttnn_input, [64, 128], pad_value, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("H", [128, 98])
@pytest.mark.parametrize("W", [78, 1024])
def test_tilize_with_zero_padding_2D(device, in_dtype, use_multicore, H, W):
    torch_input_shape = [H, W]

    torch_input = random_torch_tensor(in_dtype, torch_input_shape)

    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    output_tt = ttnn.tilize_with_zero_padding(ttnn_input, use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing
