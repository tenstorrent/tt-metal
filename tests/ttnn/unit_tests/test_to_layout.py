# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.utility_functions import is_grayskull, is_blackhole, torch_random, skip_for_grayskull


@pytest.mark.parametrize("height", [32, 30])
@pytest.mark.parametrize("width", [32, 62])
@pytest.mark.parametrize("on_device", [True, False])
@pytest.mark.parametrize("from_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("to_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("start_with_padding", [False, True])
def test_to_layout_2D(device, height, width, on_device, from_layout, to_layout, start_with_padding):
    torch_input_tensor = torch.rand((height, width), dtype=torch.bfloat16)

    pad_h = (ttnn.TILE_SIZE - height % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    pad_w = (ttnn.TILE_SIZE - width % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    if start_with_padding:
        torch_padded_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0
        )
        input_tensor = ttnn.from_torch(torch_padded_input_tensor)
        input_tensor = ttnn.reshape(input_tensor, shape=ttnn.Shape([height, width], ((0, pad_h), (0, pad_w))))
    else:
        input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT

    input_tensor = ttnn.to_layout(input_tensor, from_layout)
    assert input_tensor.layout == from_layout

    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.is_tensor_storage_on_device(input_tensor)

    output_tensor = ttnn.to_layout(input_tensor, to_layout)
    assert output_tensor.layout == to_layout

    if on_device:
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.is_tensor_storage_on_device(output_tensor)

    if (start_with_padding and from_layout == to_layout) or to_layout == ttnn.TILE_LAYOUT:
        assert output_tensor.shape == (height, width)
        assert output_tensor.shape.with_tile_padding() == (height + pad_h, width + pad_w)
    else:
        assert output_tensor.shape == (height, width)
        assert output_tensor.shape.with_tile_padding() == (height, width)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert torch.allclose(torch_input_tensor, output_tensor)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 128 * 1024), (1, 1, 128, 5120), (1, 1, 512, 5120), (1, 1, 128, 128 * 1024)],
)
@pytest.mark.parametrize("on_device", [True])
@pytest.mark.parametrize("from_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("to_layout", [ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_wide_tensor(device, shape, on_device, from_layout, to_layout):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor)
    assert input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    input_tensor = ttnn.to_layout(input_tensor, from_layout)
    assert input_tensor.layout == from_layout

    if on_device:
        input_tensor = ttnn.to_device(input_tensor, device)
        assert ttnn.is_tensor_storage_on_device(input_tensor)

    output_tensor = ttnn.to_layout(input_tensor, to_layout)
    assert output_tensor.layout == to_layout

    if on_device:
        assert ttnn.is_tensor_storage_on_device(output_tensor)
        output_tensor = ttnn.from_device(output_tensor)
        assert not ttnn.is_tensor_storage_on_device(output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert torch.allclose(torch_input_tensor, output_tensor)


@pytest.mark.parametrize("in_dtype", [ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("use_multicore", [False, True])
@pytest.mark.parametrize("use_pack_untilize", [False, True])
def test_untilize_with_unpadding_W_16(device, in_dtype, use_multicore, use_pack_untilize):
    tile_height = 32
    core_count = 56
    tiles_per_core = 4
    H = tile_height * core_count * tiles_per_core
    W = 16

    torch_input_shape = [1, 1, H, W]

    torch_input = torch.randn(torch_input_shape, dtype=torch.bfloat16).bfloat16()

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [tile_height * tiles_per_core, 2 * W],
        core_grid=ttnn.CoreGrid(y=7, x=8),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_input = ttnn.from_torch(torch_input, device=device, dtype=in_dtype, layout=ttnn.TILE_LAYOUT)
    ttnn_input = ttnn.to_memory_config(ttnn_input, sharded_memory_config)

    output_tt = ttnn.untilize_with_unpadding(
        ttnn_input, [0, 0, H - 1, W - 1], use_multicore=use_multicore, use_pack_untilize=use_pack_untilize
    )
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("h", [1, 18, 65])
@pytest.mark.parametrize("w", [1, 15, 17, 29, 33, 49, 63, 65])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_device(device, h, w, input_layout, output_layout):
    torch.manual_seed(2005)
    torch_input_tensor = torch_random((h, w), -0.1, 0.1, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat16, layout=input_layout)
    new_layout_tensor = ttnn.to_layout(input_tensor, layout=output_layout)
    torch_brought_back = ttnn.to_torch(new_layout_tensor)

    assert_with_pcc(torch_input_tensor, torch_brought_back)


@pytest.mark.parametrize("shape", [[3, 50, 1, 3, 768], [3, 1370, 1, 32, 1280]])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_5D(shape, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[4, 7, 58, 1, 37, 256], [1, 3, 64, 1, 32, 1280]])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_6D(shape, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[3, 50, 1, 1, 768], [3, 50, 1, 1, 1024], [3, 197, 1, 1, 768], [3, 197, 1, 1, 1024]])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_nd_hangs(shape, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[1, 768], [3, 230], [32, 768], [32, 143]])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_to_layout_for_2D(shape, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)
