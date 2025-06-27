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
        pytest.skip("Modifying logical shape with borrowed buffer is not supported!")
        torch_padded_input_tensor = torch.nn.functional.pad(
            torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0
        )
        input_tensor = ttnn.from_torch(torch_padded_input_tensor)
        input_tensor = ttnn.reshape(input_tensor, [height, width], [height + pad_h, width + pad_w])
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
        assert output_tensor.padded_shape == (height + pad_h, width + pad_w)
    else:
        assert output_tensor.shape == (height, width)
        assert output_tensor.padded_shape == (height, width)

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


@pytest.mark.parametrize(
    "shape", [[3, 1370, 1, 1, 1280], [3, 50, 1, 1, 768], [3, 50, 1, 1, 1024], [3, 197, 1, 1, 768], [3, 197, 1, 1, 1024]]
)
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


@pytest.mark.parametrize("shape", [1, 5, 14, 97, 0, ()])
def test_to_from_01d(device, shape):
    torch.manual_seed(2005)
    torch_input = torch.rand(shape)

    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.float32)
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.TILE_LAYOUT)
    ttnn_input = ttnn.to_device(ttnn_input, device)
    ttnn_input = ttnn.from_device(ttnn_input)
    ttnn_input = ttnn.to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn.to_torch(ttnn_input)

    assert_with_pcc(ttnn_input, torch_input)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat16])
def test_to_layout_sharded(dtype, device):
    core_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 5), ttnn.CoreCoord(1, 5)),
        }
    )

    shape1 = [1, 1, 2640, 64]

    shape1_shard_shape = (64, 64)

    shape1_shard_spec = ttnn.ShardSpec(core_grid, shape1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    shape1_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shape1_shard_spec
    )
    torch_input_tensor1 = torch.randn(shape1, dtype=torch.bfloat16)
    ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device, memory_config=shape1_memory_config)

    output = ttnn.to_layout(ttnn_input_tensor1, ttnn.ROW_MAJOR_LAYOUT)

    assert_with_pcc(torch_input_tensor1, ttnn.to_torch(output), 0.9999)


@skip_for_grayskull()
@pytest.mark.parametrize("batch_size", [9, 32])
@pytest.mark.parametrize("sentence_size", [32, 256])
def test_int_untilize(device, batch_size, sentence_size):
    torch_input_tensor = torch.randint(0, 10, (batch_size, sentence_size), dtype=torch.int16)
    ttnn_input = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT)

    output_tt = ttnn.to_layout(ttnn_input, ttnn.ROW_MAJOR_LAYOUT)
    output_torch = ttnn.to_torch(output_tt)

    assert_with_pcc(torch_input_tensor, output_torch)


@pytest.mark.parametrize("shape", [[143], [64], [380]])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_to_layout_for_1D(shape, dtype, input_layout, output_layout, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[30], [64], [2040]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tilize_with_padding_for_1D(shape, dtype, device):
    torch.manual_seed(2005)
    input_a = torch.randn(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.tilize_with_zero_padding(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[32, 32], [32, 128], [512, 512]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_tilize_for_2D(shape, dtype, device):
    torch.manual_seed(2005)
    if dtype == torch.int32:
        input_a = torch.randint(0, 1000, shape, dtype=dtype)
    else:
        input_a = torch.randn(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.tilize(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[10, 10], [100, 100], [1000, 1000]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_tilize_with_zero_padding_for_2D(shape, dtype, device):
    torch.manual_seed(2005)
    if dtype == torch.int32:
        input_a = torch.randint(0, 1000, shape, dtype=dtype)
    else:
        input_a = torch.randn(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.tilize_with_zero_padding(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[10, 10], [50, 50], [300, 300]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_tilize_with_val_padding_for_2D(shape, dtype, device):
    torch.manual_seed(2005)
    if dtype == torch.int32:
        input_a = torch.randint(0, 1000, shape, dtype=dtype)
    else:
        input_a = torch.randn(shape, dtype=dtype)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.tilize_with_val_padding(input_tensor, [512, 512], 70)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[1, 9, 91, 7, 9]])
def test_to_layout_page_error(shape, device):
    torch.manual_seed(2005)

    torch_tensor = torch.rand(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(torch_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.to_torch(output_tensor)

    torch_output = torch_tensor
    assert torch_output.shape == output_tensor.shape
    assert_with_pcc(torch_output, output_tensor, 0.9999)


@pytest.mark.parametrize("shape", [[64, 7680]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_untilize_w1(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, [36, 7667])
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a[:37, :7668], output_tensor)


@pytest.mark.parametrize("shape", [[2, 32, 6144]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_untilize_w2(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, [1, 30, 6140])
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a[:, :31, :6141], output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 1536]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_untilize_w3(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, [0, 0, 31, 1535])
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a[:, :, :32, :1536], output_tensor)


@pytest.mark.parametrize("shape", [[1, 1, 32, 10912]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT])
def test_untilize_w4(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, [0, 0, 0, 10911])
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a[:, :, :1, :10912], output_tensor)


def test_interleaved_to_sharded_block_shareded_unaligned_width(device):
    torch_input_shape = [1, 1, 196, 92]
    torch_input = torch.randn(torch_input_shape, dtype=torch.bfloat16).bfloat16()

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [32, 32],
        core_grid=ttnn.CoreGrid(
            x=7,
            y=3,
        ),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.COL_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    ttnn_input = ttnn.from_torch(torch_input, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_output = ttnn.to_memory_config(ttnn_input, sharded_memory_config)

    output_torch = ttnn.to_torch(ttnn_output)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    assert passing, pcc_msg


@pytest.mark.parametrize("shape", [[12800, 16200]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_wh1(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[11432, 11021]])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("input_layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_to_layout_wh2(shape, input_layout, output_layout, device):
    torch.manual_seed(0)
    input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=input_layout, dtype=ttnn.bfloat16)
    output_tensor = ttnn.to_layout(input_tensor, output_layout)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[32, 128], [2, 4, 96, 256], [1, 160, 64]])
def test_untilize_with_unpad_int32(shape, device):
    torch.manual_seed(2005)
    end_shape = [x - 1 for x in shape]
    input_a = torch.randint(1, 64, shape, dtype=torch.int32)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.int32)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, end_shape)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)
