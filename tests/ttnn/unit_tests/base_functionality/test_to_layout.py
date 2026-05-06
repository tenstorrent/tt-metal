# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import pytest
from models.common.utility_functions import nearest_32

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from models.common.utility_functions import torch_random


@pytest.mark.parametrize("mesh_device", [(2, 4)], ids=["t3k"], indirect=True)
def test_wan22_failure_t3k(mesh_device):
    for _ in range(5):
        torch_input_tensor = torch.rand((1, 6240, 384), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.Layout.ROW_MAJOR,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[None, None, None]
            ),
        )
        output_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        torch_output_tensor = ttnn.to_torch(
            output_tensor, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
        )
        expanded_input_tensor = torch_input_tensor.expand(8, 6240, 384)
        assert_with_pcc(expanded_input_tensor, torch_output_tensor)


def test_wan22_failure():
    for _ in range(5):
        torch_input_tensor = torch.rand((1, 6240, 384), dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            torch_input_tensor,
            layout=ttnn.Layout.ROW_MAJOR,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
        torch_output_tensor = ttnn.to_torch(output_tensor, dtype=torch.bfloat16)
        assert_with_pcc(torch_input_tensor, torch_output_tensor)


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
def test_untilize_with_unpadding_W_16(device, in_dtype, use_multicore):
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

    output_tt = ttnn.untilize_with_unpadding(ttnn_input, [0, 0, H - 1, W - 1], use_multicore=use_multicore)
    output_torch = ttnn.to_torch(output_tt)

    passing, pcc_msg = check_with_pcc_without_tensor_printout(torch_input, output_torch)
    logger.info(pcc_msg)
    assert passing


@pytest.mark.parametrize("h", [1, 18, 65])
@pytest.mark.parametrize("w", [1, 17, 65])
@pytest.mark.parametrize("input_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("output_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
        None,
    ),
)
def test_to_layout_subcore(device, h, w, input_layout, output_layout, sub_core_grids):
    torch.manual_seed(2005)
    for i in range(3):
        # We have found 3 as effective to uncover program cache issues. 2 usually works but given the short runtime of test we are running 3 to be safe
        # Typically run 1 gets hashed and in the case of trace is when things like persistent semaphores are allocated if applicable
        # Typically run 2 is where trace selects the final position of all the tensors (run 1 if no persistents)
        # Therefore run 3 is the first where we truly are in full trace mode (run 2 if no persistents)
        torch_input_tensor = torch_random((h, w), -0.1, 0.1, dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(torch_input_tensor, device=device, dtype=ttnn.bfloat16, layout=input_layout)
        new_layout_tensor = ttnn.to_layout(input_tensor, layout=output_layout, sub_core_grids=sub_core_grids)
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


@pytest.mark.parametrize("shape", [[32, 128 * 1024]])
@pytest.mark.parametrize(
    "sub_core_grids",
    (
        # single core
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
        # multiple disjoint cores
        ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 6)),
            ]
        ),
    ),
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint16])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
        }
    ],
    indirect=True,
)
def test_to_layout_low_perf(shape, device, sub_core_grids, dtype):
    torch.manual_seed(0)
    if dtype == ttnn.int32:
        input_a = torch.randint(-1000, 1000, shape, dtype=torch.int32)
    elif dtype == ttnn.uint16:
        input_a = torch.randint(0, 1000, shape, dtype=torch.int32)
    else:
        input_a = torch.randn(shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)
    output_tensor = ttnn.tilize(input_tensor, sub_core_grids=sub_core_grids, use_low_perf=True)
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


@pytest.mark.parametrize("shape", [[32, 128], [2, 4, 96, 256], [1, 160, 64], [64, 512], [10, 1024, 2048]])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32])
def test_untilize_with_unpad_int32(shape, dtype, device):
    torch.manual_seed(2005)
    end_shape = [x - 1 for x in shape]
    input_a = torch.randint(1, 64, shape, dtype=torch.int32)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    output_tensor = ttnn.untilize_with_unpadding(input_tensor, end_shape)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


@pytest.mark.parametrize("shape", [[3072, 1024], [2, 2048, 512]])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32])
def test_untilize_int32_t(shape, dtype, device):
    torch.manual_seed(2005)
    input_a = torch.randint(1, 64, shape, dtype=torch.int32)
    input_tensor = ttnn.from_torch(input_a, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    output_tensor = ttnn.untilize(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(input_a, output_tensor)


def run_unary_with_aprox_mode_fruit_test(
    device, h, w, memory_type, shard_shape, ttnn_function, vector_mode, approx_mode, pcc=0.9999
):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((1, 1, h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)
    if memory_type == "L1":
        core_grid = device.compute_with_storage_grid_size()
        num_cores = 64

        shard_spec = ttnn.ShardSpec(
            grid=ttnn.num_cores_to_corerangeset(num_cores, device.compute_with_storage_grid_size()),
            shard_shape=shard_shape,
            shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED, buffer_type=ttnn.BufferType.L1, shard_spec=shard_spec
        )
    else:  # memory_type == "DRAM":
        memory_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
        )
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    print(
        f"shape: {input_tensor.shape}, dtype: {input_tensor.dtype}, layout: {input_tensor.layout}, memory_config: {input_tensor.memory_config()}"
    )
    output_tensor = ttnn_function(input_tensor, vector_mode=vector_mode, fast_and_approximate_mode=approx_mode)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [1024 * 128])
@pytest.mark.parametrize("w", [1])
@pytest.mark.parametrize("memory_type", ["L1", "DRAM"])
@pytest.mark.parametrize("shard_shape", [(2048, 32)])
@pytest.mark.parametrize("approx_mode", [True, False])
@pytest.mark.parametrize("vector_mode", [4])
def test_sigmoid_fruit(device, h, w, memory_type, shard_shape, vector_mode, approx_mode):
    class sigmoid_wrap:
        def __init__(self):
            self.golden_function = ttnn.sigmoid.golden_function

        def __call__(self, input_tensor, vector_mode, fast_and_approximate_mode):
            return ttnn.sigmoid(
                input_tensor,
                vector_mode=vector_mode,
                mode=ttnn.SigmoidMode.FastApproximate if fast_and_approximate_mode else ttnn.SigmoidMode.Accurate,
            )

    run_unary_with_aprox_mode_fruit_test(
        device,
        h,
        w,
        memory_type,
        shard_shape,
        sigmoid_wrap(),
        vector_mode=vector_mode,
        approx_mode=approx_mode,
        pcc=0.999,
    )


def test_shard_untilize(device):
    torch.manual_seed(2005)

    torch_tensor = torch.rand(1, 1, 29640, 128, dtype=torch.bfloat16)

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [
            480,
            128,
        ],
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 6),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 7),
                    ttnn.CoreCoord(5, 7),
                ),
            }
        ),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensor = ttnn.from_torch(
        torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_memory_config
    )

    output_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert output_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Memory config is not DRAM"

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_tensor.shape == output_tensor.shape
    assert torch.allclose(torch_tensor, output_tensor, 0.9999)


def test_shard_untilize2(device):
    torch.manual_seed(2005)

    torch_tensor = torch.rand(1, 1, 256, 32768, dtype=torch.bfloat16)

    sharded_memory_config = ttnn.create_sharded_memory_config(
        [
            256,
            1024,
        ],
        core_grid=ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(3, 7),
                ),
            }
        ),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensor = ttnn.from_torch(
        torch_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=sharded_memory_config
    )

    output_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert output_tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG, "Memory config is not DRAM"

    output_tensor = ttnn.to_torch(output_tensor)
    assert torch_tensor.shape == output_tensor.shape
    assert torch.allclose(torch_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param(None, id="empty"),
        pytest.param((1, 1, 9, 79), id="9x79"),
        pytest.param((1, 1, 512, 512), id="512x512"),
        pytest.param((1, 1, 513, 513), id="513x513"),
    ],
)
def test_tensor_to_tile_layout_shape_verification(device, shape):
    """Regression test for issue 19309: Tensor.to(Layout) does not pad tensor and throws"""
    if shape is None:
        pt_tensor = torch.empty(0, dtype=torch.bfloat16, requires_grad=False)
    else:
        pt_tensor = torch.rand(torch.Size(shape), requires_grad=False).bfloat16()

    initial_shape = pt_tensor.shape  # store initial shape

    output_tensor = ttnn.Tensor(pt_tensor, ttnn.bfloat16).to(ttnn.TILE_LAYOUT)  # should not throw
    result_shape = output_tensor.padded_shape  # store result shape

    # Layout verification
    assert ttnn.TILE_LAYOUT == output_tensor.layout
    # Padding verification: shape comparison
    if 4 == len(initial_shape):
        assert result_shape[-2] == nearest_32(initial_shape[-2])
        assert result_shape[-1] == nearest_32(initial_shape[-1])
    else:
        assert 32 == result_shape[0]


@pytest.mark.parametrize("shape", [(30, 62), (17, 47), (65, 33)])
@pytest.mark.parametrize("pad_value", [0, 1, -2, 1.5])
def test_to_layout_pad_value_on_host(shape, pad_value):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    h, w = shape[-2], shape[-1]
    pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE

    torch_output_tensor = torch.nn.functional.pad(
        torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=pad_value
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tiled = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, pad_value=pad_value)
    output_tensor = tiled.to_torch_with_padded_shape()

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("shape", [(30, 62), (17, 47), (65, 33)])
@pytest.mark.parametrize("pad_value", [0, 1, -2, 1.5])
def test_to_layout_pad_value_on_device(device, shape, pad_value):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    h, w = shape[-2], shape[-1]
    pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE

    torch_output_tensor = torch.nn.functional.pad(
        torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=pad_value
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tiled = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, pad_value=pad_value)
    padded_end = [s - 1 for s in tiled.padded_shape]
    output_tensor = ttnn.untilize_with_unpadding(tiled, padded_end)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize(
    "ttnn_dtype, pad_value",
    [
        (ttnn.bfloat16, 0),
        (ttnn.bfloat16, 1.5),
        (ttnn.bfloat16, -2),
        (ttnn.float32, 0),
        (ttnn.float32, 1.5),
        (ttnn.float32, -2),
        (ttnn.int32, 0),
        (ttnn.int32, 5),
        (ttnn.int32, -2),
        (ttnn.uint32, 0),
        (ttnn.uint32, 5),
        (ttnn.uint16, 0),
        (ttnn.uint16, 5),
    ],
)
@pytest.mark.parametrize("shape", [(30, 62)])
def test_to_layout_pad_value_dtype(device, shape, ttnn_dtype, pad_value):
    torch.manual_seed(0)

    if ttnn_dtype == ttnn.int32:
        torch_input_tensor = torch.randint(-100, 100, shape, dtype=torch.int32)
    elif ttnn_dtype in (ttnn.uint32, ttnn.uint16):
        torch_input_tensor = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    h, w = shape[-2], shape[-1]
    pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE

    torch_output_tensor = torch.nn.functional.pad(
        torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=pad_value
    )

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn_dtype)
    tiled = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, pad_value=pad_value)
    padded_end = [s - 1 for s in tiled.padded_shape]
    output_tensor = ttnn.untilize_with_unpadding(tiled, padded_end)
    output_tensor = ttnn.to_torch(output_tensor).to(torch_output_tensor.dtype)

    assert output_tensor.shape == torch_output_tensor.shape
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("shape", [(1, 1, 30, 62)])
def test_to_layout_pad_value_default_is_zero(device, shape):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)

    h, w = shape[-2], shape[-1]
    pad_h = (ttnn.TILE_SIZE - h % ttnn.TILE_SIZE) % ttnn.TILE_SIZE
    pad_w = (ttnn.TILE_SIZE - w % ttnn.TILE_SIZE) % ttnn.TILE_SIZE

    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tiled_default = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
    padded_end = [s - 1 for s in tiled_default.padded_shape]
    output_default = ttnn.to_torch(ttnn.untilize_with_unpadding(tiled_default, padded_end))

    input_tensor = ttnn.from_torch(torch_input_tensor, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
    tiled_explicit = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, pad_value=0.0)
    padded_end = [s - 1 for s in tiled_explicit.padded_shape]
    output_explicit = ttnn.to_torch(ttnn.untilize_with_unpadding(tiled_explicit, padded_end))

    assert torch.equal(output_default, output_explicit)
    assert torch.equal(output_default, torch_output_tensor)


# ---------------------------------------------------------------------------
# to_layout on ND-sharded tensors
#
# Exercises ttnn.to_layout when the input is an ND-sharded tensor and:
#   - the target layout differs from the input layout (TILE <-> ROW_MAJOR)
#   - the requested output memory_config is:
#       * the same ND shard spec as the input   (input/output share shard spec)
#       * a different ND shard spec             (reshard during layout change)
#       * an interleaved memory config          (sharded -> interleaved)
#   - the input and/or output shards are evenly or unevenly divided along the
#     outer (non-tile) dims.
#
# For TILE layout the inner-2 dims of any sharded shape must be 32-aligned, so
# uneven sharding is only applied on the outer dim(s).
# ---------------------------------------------------------------------------


def _make_nd_mem_config(shard_shape, grid, buffer_type=ttnn.BufferType.L1, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    nd_shard_spec = ttnn.NdShardSpec(shard_shape, grid, orientation=orientation)
    return ttnn.MemoryConfig(buffer_type, nd_shard_spec)


def _grid(cols, rows):
    """Contiguous rectangular grid of ``cols * rows`` cores, anchored at (0, 0)."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})


def _cases_nd_sharded():
    """Parametrize cases for ``test_to_layout_nd_sharded_rm_to_tile`` / ``..._tile_to_rm``.

    Each case produces the tuple ``(tensor_shape, input_shard_shape, input_grid,
    output_shard_shape, output_grid)``. Shard shapes use tile-aligned inner-2 dims
    (32-multiples) so the same cases are valid for both TILE and ROW_MAJOR tensors.
    Unevenness can appear on outer dim(s) OR on the inner-2 (tile) dims
    (e.g. 160 / 64 -> shards of 64, 64, 32 along that dim).
    """
    return [
        # Even input & output, same shard spec (1 shard on 1 core).
        pytest.param([1, 1, 64, 64], [1, 1, 64, 64], _grid(1, 1), [1, 1, 64, 64], _grid(1, 1), id="even_same_1shard"),
        # Even input & output, same shard spec, multiple shards along outer dim.
        pytest.param(
            [4, 1, 32, 64], [1, 1, 32, 64], _grid(4, 1), [1, 1, 32, 64], _grid(4, 1), id="even_same_4shards_dim0"
        ),
        # Even input & output, 3-D sharded along dim 0.
        pytest.param([6, 32, 64], [2, 32, 64], _grid(3, 1), [2, 32, 64], _grid(3, 1), id="even_same_3shards_dim0"),
        # Even input & output, different shard spec (reshard along dim 0).
        pytest.param([6, 32, 64], [2, 32, 64], _grid(3, 1), [3, 32, 64], _grid(2, 1), id="even_diff_reshard_dim0"),
        # Even input & output, different shard spec (collapse to 1 shard).
        pytest.param([4, 1, 32, 64], [1, 1, 32, 64], _grid(4, 1), [4, 1, 32, 64], _grid(1, 1), id="even_diff_collapse"),
        # Even input, sharded along dims 0 & 1, reshard to only dim 0.
        pytest.param(
            [4, 6, 32, 32], [1, 2, 32, 32], _grid(6, 1), [2, 6, 32, 32], _grid(2, 1), id="even_diff_reshard_dim01"
        ),
        # Uneven input (dim 0 = 5, shard_shape dim 0 = 2 -> last shard has 1), same shard spec.
        pytest.param([5, 32, 64], [2, 32, 64], _grid(3, 1), [2, 32, 64], _grid(3, 1), id="uneven_in_same"),
        # Uneven output (dim 0 = 6 reshard to shard_shape dim 0 = 4 -> last shard has 2).
        pytest.param([6, 32, 64], [6, 32, 64], _grid(1, 1), [4, 32, 64], _grid(2, 1), id="even_in_uneven_out"),
        # Uneven input & uneven output (both dim 0 unevenly sharded, different specs).
        pytest.param([7, 32, 64], [3, 32, 64], _grid(3, 1), [4, 32, 64], _grid(2, 1), id="uneven_in_uneven_out"),
        # Uneven input, sharded to interleaved-like target (single-shard collapse keeps remainder).
        pytest.param([5, 32, 64], [2, 32, 64], _grid(3, 1), [5, 32, 64], _grid(1, 1), id="uneven_in_collapsed_out"),
        # --- Uneven sharding on the INNER-2 (tile) dims ---
        # Each uneven-inner case is duplicated with two grid sizings:
        #   * one where num_cores >= num_shards  -> 1 shard per core (simpler case)
        #   * one where num_cores <  num_shards  -> multiple shards per core
        #     (ND shards are round-robin distributed across the grid)
        #
        # [2,160,160] sharded [1,64,64]:
        #   dim0 even (2/1=2), dim1 160/64 -> 3 shards (64,64,32), dim2 160/64 -> 3 shards (64,64,32) => 18 shards.
        # Same shard spec across input and output; both last-2 dims unevenly sharded.
        pytest.param(
            [2, 160, 160],
            [1, 64, 64],
            _grid(6, 3),  # 18 cores -> 1 shard/core
            [1, 64, 64],
            _grid(6, 3),
            id="uneven_inner_same_1shard_per_core",
        ),
        # Same tensor/shard shape, but pack 3 shards per core (6 cores, 18 shards).
        pytest.param(
            [2, 160, 160],
            [1, 64, 64],
            _grid(6, 1),  # 6 cores -> 3 shards/core
            [1, 64, 64],
            _grid(6, 1),
            id="uneven_inner_same_multi_shard_per_core",
        ),
        # [3,160,160] sharded [2,64,64]: uneven on ALL dims
        # (dim0 3/2->(2,1), dim1/2 160/64->(64,64,32)) => 2*3*3 = 18 shards.
        pytest.param(
            [3, 160, 160],
            [2, 64, 64],
            _grid(6, 3),  # 18 cores -> 1 shard/core
            [2, 64, 64],
            _grid(6, 3),
            id="uneven_all_dims_same_1shard_per_core",
        ),
        # Same tensor/shard, packed 3 shards per core (6 cores, 18 shards).
        pytest.param(
            [3, 160, 160],
            [2, 64, 64],
            _grid(6, 1),  # 6 cores -> 3 shards/core
            [2, 64, 64],
            _grid(6, 1),
            id="uneven_all_dims_same_multi_shard_per_core",
        ),
        # Same tensor/shard, packed 2 shards per core (9 cores, 18 shards).
        pytest.param(
            [3, 160, 160],
            [2, 64, 64],
            _grid(3, 3),  # 9 cores -> 2 shards/core
            [2, 64, 64],
            _grid(3, 3),
            id="uneven_all_dims_same_2shards_per_core",
        ),
        # Uneven-all-dims input reshard to a DIFFERENT shard spec that is also uneven on the last-2 dims.
        # Input [3,160,160] sharded [2,64,64] (18 shards) packed 3/core over 6 cores ->
        # output [3,96,96] sharded [3,96,96]? dim0 3/3=1, dim1 160/96->(96,64), dim2 160/96->(96,64) => 1*2*2=4 shards.
        # Output uses 2 cores -> 2 shards/core on output.
        pytest.param(
            [3, 160, 160],
            [2, 64, 64],
            _grid(6, 1),  # input: 6 cores -> 3 shards/core
            [3, 96, 96],
            _grid(2, 1),  # output: 2 cores -> 2 shards/core
            id="uneven_all_dims_diff_reshard_multi_shard_per_core",
        ),
        # Inner-2 uneven on input (8 shards) packed 2 shards per core (4 cores), output is a single shard.
        # input [2,96,96] sharded [1,64,64]: dim1/2 96/64->(64,32) => 2*2*2=8 shards.
        pytest.param(
            [2, 96, 96],
            [1, 64, 64],
            _grid(4, 1),  # 4 cores -> 2 shards/core
            [2, 96, 96],
            _grid(1, 1),
            id="uneven_inner_collapsed_out_multi_shard_per_core",
        ),
    ]


def _run_to_layout_nd_sharded(device, tensor_shape, input_mem_config, target_layout, output_mem_config, from_layout):
    torch.manual_seed(0)
    torch_input = torch.randn(tensor_shape, dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=from_layout,
        device=device,
        memory_config=input_mem_config,
    )
    assert input_tensor.layout == from_layout

    output_tensor = ttnn.to_layout(input_tensor, target_layout, memory_config=output_mem_config)
    assert output_tensor.layout == target_layout
    assert tuple(output_tensor.shape) == tuple(tensor_shape)

    output_torch = ttnn.to_torch(output_tensor)
    assert tuple(output_torch.shape) == tuple(tensor_shape)
    assert_with_pcc(torch_input, output_torch, 0.9999)
    assert torch.allclose(torch_input, output_torch)


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, input_grid, output_shard_shape, output_grid",
    _cases_nd_sharded(),
)
@pytest.mark.parametrize(
    "output_memory_mode",
    ["same_shard", "different_shard", "interleaved_dram"],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_layout_nd_sharded_rm_to_tile(
    device,
    tensor_shape,
    input_shard_shape,
    input_grid,
    output_shard_shape,
    output_grid,
    output_memory_mode,
    shard_orientation,
):
    """ROW_MAJOR ND-sharded -> TILE, varying output memory config."""
    input_mem_config = _make_nd_mem_config(input_shard_shape, input_grid, orientation=shard_orientation)

    if output_memory_mode == "same_shard":
        output_mem_config = input_mem_config
    elif output_memory_mode == "different_shard":
        output_mem_config = _make_nd_mem_config(output_shard_shape, output_grid, orientation=shard_orientation)
    elif output_memory_mode == "interleaved_dram":
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        raise ValueError(output_memory_mode)

    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        ttnn.TILE_LAYOUT,
        output_mem_config,
        from_layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize(
    "tensor_shape, input_shard_shape, input_grid, output_shard_shape, output_grid",
    _cases_nd_sharded(),
)
@pytest.mark.parametrize(
    "output_memory_mode",
    ["same_shard", "different_shard", "interleaved_dram"],
)
@pytest.mark.parametrize(
    "shard_orientation",
    [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR],
)
def test_to_layout_nd_sharded_tile_to_rm(
    device,
    tensor_shape,
    input_shard_shape,
    input_grid,
    output_shard_shape,
    output_grid,
    output_memory_mode,
    shard_orientation,
):
    """TILE ND-sharded -> ROW_MAJOR, varying output memory config."""
    input_mem_config = _make_nd_mem_config(input_shard_shape, input_grid, orientation=shard_orientation)

    if output_memory_mode == "same_shard":
        output_mem_config = input_mem_config
    elif output_memory_mode == "different_shard":
        output_mem_config = _make_nd_mem_config(output_shard_shape, output_grid, orientation=shard_orientation)
    elif output_memory_mode == "interleaved_dram":
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        raise ValueError(output_memory_mode)

    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        ttnn.ROW_MAJOR_LAYOUT,
        output_mem_config,
        from_layout=ttnn.TILE_LAYOUT,
    )


# ---------------------------------------------------------------------------
# to_layout between 2D-sharded (HEIGHT/WIDTH/BLOCK_SHARDED) and ND-sharded.
#
# Exercises mixing legacy 2D shard specs with NdShardSpec across the layout
# change. Target layout is parameterized, so each case runs both (RM -> TILE)
# and (TILE -> RM). Shard shapes keep last-2 dims as tile-multiples so the
# same cases are valid for both layouts; unevenness is produced by tensor dims
# that don't divide the corresponding shard dim (e.g. 160 % 64 -> (64,64,32)).
# Multi-shard-per-core packing is exercised on the ND side (ND shards are
# distributed round-robin when num_cores < num_shards).
# ---------------------------------------------------------------------------


def _make_2d_mem_config(
    shard_scheme, shard_shape_2d, grid, buffer_type=ttnn.BufferType.L1, orientation=ttnn.ShardOrientation.ROW_MAJOR
):
    shard_spec = ttnn.ShardSpec(grid, shard_shape_2d, orientation)
    return ttnn.MemoryConfig(shard_scheme, buffer_type, shard_spec)


def _cases_2d_to_nd():
    """Parametrize cases for ``test_to_layout_2d_sharded_to_nd_sharded``.

    Each case produces the tuple ``(tensor_shape, input_2d_shard_layout,
    input_2d_shard_shape, input_2d_grid, output_nd_shard_shape, output_nd_grid)``.
    All 2D shard shapes are (tile-aligned H, tile-aligned W) so valid for both TILE and RM.
    """
    return [
        # HEIGHT_SHARDED even (4 shards on 4 cores) -> ND sharded [1,1,32,64] on 2 cores (2 shards/core).
        pytest.param(
            [1, 1, 128, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            _grid(4, 1),
            [1, 1, 32, 64],
            _grid(2, 1),
            id="h_even_to_nd_multi_per_core",
        ),
        # WIDTH_SHARDED even (4 shards on 4 cores) -> ND sharded, single shard collapse.
        pytest.param(
            [1, 1, 64, 128],
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 32),
            _grid(4, 1),
            [1, 1, 64, 128],
            _grid(1, 1),
            id="w_even_to_nd_collapsed",
        ),
        # BLOCK_SHARDED even (2x2 grid, 4 shards) -> ND sharded [1,1,64,64] on 2 cores (2 shards/core).
        pytest.param(
            [1, 1, 128, 128],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            _grid(2, 2),
            [1, 1, 64, 64],
            _grid(2, 1),
            id="block_even_to_nd_multi_per_core",
        ),
        # BLOCK_SHARDED uneven last-2 dims: tensor 160x160, shard 64x64, 3x3 grid ->
        #   9 shards, last-row/col shards are 32 rows/cols (tile-aligned, but uneven).
        # Output: ND [1,1,64,64] on 3 cores (9 shards / 3 cores = 3 shards/core).
        pytest.param(
            [1, 1, 160, 160],
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            _grid(3, 3),
            [1, 1, 64, 64],
            _grid(3, 1),
            id="block_uneven_last2_to_nd_multi_per_core",
        ),
        # HEIGHT_SHARDED with uneven H: tensor [1,1,160,64], shard (64,64) on 3 cores
        #   -> 3 shards (64, 64, 32 rows). Last-2 dims uneven on H.
        # Output: ND [1,1,64,64] on 1 core (3 shards/core).
        pytest.param(
            [1, 1, 160, 64],
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (64, 64),
            _grid(3, 1),
            [1, 1, 64, 64],
            _grid(1, 1),
            id="h_uneven_last2_to_nd_multi_per_core",
        ),
    ]


def _cases_nd_to_2d():
    """Parametrize cases for ``test_to_layout_nd_sharded_to_2d_sharded``.

    Each case produces the tuple ``(tensor_shape, input_nd_shard_shape, input_nd_grid,
    output_2d_shard_layout, output_2d_shard_shape, output_2d_grid)``.
    """
    return [
        # ND even 2 shards on 2 cores -> HEIGHT_SHARDED (4 shards on 4 cores).
        pytest.param(
            [1, 1, 128, 64],
            [1, 1, 64, 64],
            _grid(2, 1),
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (32, 64),
            _grid(4, 1),
            id="nd_to_h_even",
        ),
        # ND multi-shard-per-core (8 shards packed on 4 cores) -> WIDTH_SHARDED (4 shards on 4 cores).
        pytest.param(
            [1, 1, 64, 256],
            [1, 1, 64, 32],
            _grid(4, 1),  # 8 shards on 4 cores -> 2 shards/core
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            (64, 64),
            _grid(4, 1),
            id="nd_multi_per_core_to_w_even",
        ),
        # ND uneven on last-2 dims with non-trivial outer dim:
        #   tensor [2,1,160,160], ND shard [1,1,64,64] -> dim0 2 shards, dim2 3 shards (160/64 = 64,64,32),
        #   dim3 3 shards (same). Total 2*1*3*3 = 18 shards packed on 3 cores (6 shards/core).
        # Output: BLOCK_SHARDED (64,64). BLOCK_SHARDED flattens outer dims into physical_height, so
        #   physical shape is (2*1*160, 160) = (320, 160). num_shards_H = ceil(320/64) = 5 (even),
        #   num_shards_W = ceil(160/64) = 3 (uneven: 64,64,32). Grid must be 3 cols x 5 rows for
        #   ROW_MAJOR orientation -> 15 cores, 1 shard/core.
        pytest.param(
            [2, 1, 160, 160],
            [1, 1, 64, 64],
            _grid(3, 1),  # 18 ND shards on 3 cores -> 6 shards/core
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            (64, 64),
            _grid(3, 5),  # 3x5 grid = 15 cores (matches num_shards_H=5 x num_shards_W=3 for ROW_MAJOR orientation)
            id="nd_multi_per_core_uneven_to_block_uneven",
        ),
        # ND 3-D tensor uneven on outer dim (tensor [3,128,128] shard [2,64,64] -> 8 shards,
        # dim0 splits 3/2->(2,1), dim1 128/64=2, dim2 128/64=2) packed on 4 cores (2 shards/core).
        # Output: HEIGHT_SHARDED on 4 cores (tensor flattens to (384,128); shard (96,128), 4 even
        # shards). HEIGHT_SHARDED handles 3-D+ tensors by flattening the outer dims into H.
        pytest.param(
            [3, 128, 128],
            [2, 64, 64],
            _grid(4, 1),  # 8 shards on 4 cores -> 2 shards/core
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            (96, 128),
            _grid(4, 1),
            id="nd_3d_multi_per_core_to_h_even",
        ),
    ]


@pytest.mark.parametrize(
    "tensor_shape, input_2d_layout, input_2d_shard, input_2d_grid, output_nd_shard, output_nd_grid",
    _cases_2d_to_nd(),
)
@pytest.mark.parametrize(
    "from_layout, target_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["rm_to_tile", "tile_to_rm"],
)
def test_to_layout_2d_sharded_to_nd_sharded(
    device,
    tensor_shape,
    input_2d_layout,
    input_2d_shard,
    input_2d_grid,
    output_nd_shard,
    output_nd_grid,
    from_layout,
    target_layout,
):
    """2D-sharded (HEIGHT/WIDTH/BLOCK) input -> ND-sharded output via ttnn.to_layout."""
    input_mem_config = _make_2d_mem_config(input_2d_layout, input_2d_shard, input_2d_grid)
    output_mem_config = _make_nd_mem_config(output_nd_shard, output_nd_grid)
    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        target_layout,
        output_mem_config,
        from_layout=from_layout,
    )


@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard, input_nd_grid, output_2d_layout, output_2d_shard, output_2d_grid",
    _cases_nd_to_2d(),
)
@pytest.mark.parametrize(
    "from_layout, target_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["rm_to_tile", "tile_to_rm"],
)
def test_to_layout_nd_sharded_to_2d_sharded(
    device,
    tensor_shape,
    input_nd_shard,
    input_nd_grid,
    output_2d_layout,
    output_2d_shard,
    output_2d_grid,
    from_layout,
    target_layout,
):
    """ND-sharded input -> 2D-sharded (HEIGHT/WIDTH/BLOCK) output via ttnn.to_layout."""
    input_mem_config = _make_nd_mem_config(input_nd_shard, input_nd_grid)
    output_mem_config = _make_2d_mem_config(output_2d_layout, output_2d_shard, output_2d_grid)
    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        target_layout,
        output_mem_config,
        from_layout=from_layout,
    )


# ---------------------------------------------------------------------------
# to_layout from interleaved -> ND-sharded.
#
# Covers RM -> TILE (with tensor shapes that are NOT divisible by tile size on
# the last two dims, so tilize_with_val_padding must pad before writing the
# ND-sharded output) and TILE -> RM.
# ---------------------------------------------------------------------------


def _cases_interleaved_to_nd_rm_to_tile():
    """Parametrize cases for ``test_to_layout_interleaved_to_nd_sharded_rm_to_tile``.

    Each case produces the tuple ``(tensor_shape, output_nd_shard_shape, output_nd_grid)``.
    Output ND shard shape's last-2 dims are tile-multiples (required by TILE output).
    """
    return [
        # Tile-aligned tensor, single-shard ND output.
        pytest.param([1, 1, 64, 64], [1, 1, 64, 64], _grid(1, 1), id="tile_aligned_1shard"),
        # Tile-aligned tensor, multi-shard ND output.
        pytest.param([1, 1, 128, 64], [1, 1, 32, 64], _grid(4, 1), id="tile_aligned_multi_shard"),
        # Tile-aligned tensor, ND output with multi-shard-per-core (8 shards on 4 cores).
        pytest.param([1, 1, 256, 64], [1, 1, 32, 64], _grid(4, 1), id="tile_aligned_multi_per_core"),
        # Non-tile-aligned last dim only (W=62 -> padded to 64). Single-shard output covers padded shape.
        pytest.param([1, 1, 32, 62], [1, 1, 32, 64], _grid(1, 1), id="non_tile_aligned_w_only"),
        # Non-tile-aligned -2 dim only (H=30 -> padded to 32). Single-shard output.
        pytest.param([1, 1, 30, 64], [1, 1, 32, 64], _grid(1, 1), id="non_tile_aligned_h_only"),
        # Non-tile-aligned BOTH last 2 dims (H=30, W=62 -> padded to 32x64). Single-shard output.
        pytest.param([1, 1, 30, 62], [1, 1, 32, 64], _grid(1, 1), id="non_tile_aligned_both_last2_1shard"),
        # Non-tile-aligned both last 2 dims, with multi-shard ND output.
        # [1,1,60,62] -> padded to [1,1,64,64] after tilize; shard [1,1,32,64] on 2 cores.
        pytest.param([1, 1, 60, 62], [1, 1, 32, 64], _grid(2, 1), id="non_tile_aligned_both_last2_multi_shard"),
        # Non-tile-aligned both last 2 dims, outer dim > 1, multi-shard ND output.
        # [2,1,30,62] -> padded to [2,1,32,64] after tilize; shard [1,1,32,64] on 2 cores.
        pytest.param([2, 1, 30, 62], [1, 1, 32, 64], _grid(2, 1), id="non_tile_aligned_with_outer_dim"),
        # Non-tile-aligned, outer dim > 1, multi-shard-per-core on ND output.
        # [4,1,30,62] -> padded [4,1,32,64]; 4 ND shards with [1,1,32,64] on 2 cores -> 2 shards/core.
        pytest.param([4, 1, 30, 62], [1, 1, 32, 64], _grid(2, 1), id="non_tile_aligned_multi_per_core"),
        # Non-tile-aligned, rank-3 tensor.
        # [3,30,62] -> padded [3,32,64]; shard [1,32,64] on 3 cores.
        pytest.param([3, 30, 62], [1, 32, 64], _grid(3, 1), id="non_tile_aligned_rank3"),
        # --- Uneven ND output sharding ---
        # Tile-aligned tensor [3,160,160] with ND shard [2,64,64]: uneven on ALL dims
        #   (dim0 3/2->(2,1), dim1/2 160/64->(64,64,32)) => 2*3*3 = 18 shards.
        # 18 cores, 1 shard/core.
        pytest.param([3, 160, 160], [2, 64, 64], _grid(6, 3), id="uneven_nd_all_dims_1shard_per_core"),
        # Same tensor/shard, multi-shard-per-core on ND output (6 cores, 3 shards/core).
        pytest.param([3, 160, 160], [2, 64, 64], _grid(6, 1), id="uneven_nd_all_dims_multi_shard_per_core"),
        # Tile-aligned tensor, uneven on last 2 dims only (outer dim even).
        #   [2,160,160] sharded [1,64,64] => 2*3*3 = 18 shards on 6 cores (3/core).
        pytest.param([2, 160, 160], [1, 64, 64], _grid(6, 1), id="uneven_nd_inner_multi_shard_per_core"),
        # Non-tile-aligned tensor + uneven ND shard.
        #   [3,158,158] logical -> [3,160,160] padded after tilize; shard [2,64,64] => 18 shards.
        #   Exercises BOTH tilize-with-padding (logical -> padded) AND uneven ND sharding.
        pytest.param([3, 158, 158], [2, 64, 64], _grid(6, 3), id="non_tile_aligned_tensor_uneven_nd_all_dims"),
        # Non-tile-aligned tensor + uneven ND shard + multi-shard-per-core.
        pytest.param([3, 158, 158], [2, 64, 64], _grid(6, 1), id="non_tile_aligned_tensor_uneven_nd_multi_per_core"),
    ]


def _cases_interleaved_to_nd_tile_to_rm():
    """Parametrize cases for ``test_to_layout_interleaved_to_nd_sharded_tile_to_rm``.

    Each case produces the tuple ``(tensor_shape, output_nd_shard_shape, output_nd_grid)``.

    Cases where the TILE input has logical last-2 dims not divisible by 32 take the
    ``untilize_with_unpadding`` path in ``to_layout``'s RM target branch. On main,
    ``untilize_with_unpadding`` rejects interleaved-input + sharded-output with the TT_FATAL
    "Output memory config layout must be INTERLEAVED but got <layout>" (see
    ``untilize_with_unpadding_device_operation.cpp`` final ``else`` branch). Those cases
    are marked xfail(strict=True) so they flip to failure if interleaved->sharded support
    is ever added to ``untilize_with_unpadding``, prompting removal of the xfail.
    """
    untilize_xfail = pytest.mark.xfail(
        raises=RuntimeError,
        reason='TT_FATAL: "Output memory config layout must be INTERLEAVED but got ..." '
        "(untilize_with_unpadding does not support interleaved-input -> sharded-output).",
        strict=True,
    )
    return [
        # Tile-aligned, single-shard ND output.
        pytest.param([1, 1, 64, 64], [1, 1, 64, 64], _grid(1, 1), id="tile_aligned_1shard"),
        # Tile-aligned, multi-shard ND output.
        pytest.param([1, 1, 128, 64], [1, 1, 32, 64], _grid(4, 1), id="tile_aligned_multi_shard"),
        # Tile-aligned, multi-shard-per-core on ND output (8 shards on 4 cores).
        pytest.param([1, 1, 256, 64], [1, 1, 32, 64], _grid(4, 1), id="tile_aligned_multi_per_core"),
        # Rank-3 even.
        pytest.param([4, 32, 64], [1, 32, 64], _grid(4, 1), id="rank3_tile_aligned"),
        # Rank-3 uneven on outer dim, tile-aligned last-2 dims.
        pytest.param([5, 32, 64], [2, 32, 64], _grid(3, 1), id="rank3_uneven_outer"),
        # --- Uneven ND output sharding ---
        # [3,160,160] shard [2,64,64]: uneven on ALL dims, 18 shards on 18 cores (1/core).
        pytest.param([3, 160, 160], [2, 64, 64], _grid(6, 3), id="uneven_nd_all_dims_1shard_per_core"),
        # Same, multi-shard-per-core (6 cores, 3/core).
        pytest.param([3, 160, 160], [2, 64, 64], _grid(6, 1), id="uneven_nd_all_dims_multi_shard_per_core"),
        # Uneven on inner dims only.
        pytest.param([2, 160, 160], [1, 64, 64], _grid(6, 1), id="uneven_nd_inner_multi_shard_per_core"),
        # --- Non-tile-aligned logical shapes: exercise untilize_with_unpadding ---
        # [1,1,30,64] logical -> padded [1,1,32,64] in TILE; shard [1,1,32,64] on 1 core covers padded.
        pytest.param([1, 1, 30, 64], [1, 1, 32, 64], _grid(1, 1), id="non_tile_aligned_h_only", marks=untilize_xfail),
        # [1,1,60,64] -> padded [1,1,64,64]; shard [1,1,32,64] on 2 cores.
        pytest.param(
            [1, 1, 60, 64], [1, 1, 32, 64], _grid(2, 1), id="non_tile_aligned_h_multi_shard", marks=untilize_xfail
        ),
        # [2,1,30,64] -> padded [2,1,32,64]; shard [1,1,32,64] on 2 cores (one per outer slice).
        pytest.param(
            [2, 1, 30, 64], [1, 1, 32, 64], _grid(2, 1), id="non_tile_aligned_with_outer_dim", marks=untilize_xfail
        ),
        # Non-tile-aligned + uneven ND shard: [3,158,158] -> padded [3,160,160]; shard [2,64,64] -> 18 shards
        # on 6 cores (3 shards/core). Exercises untilize_with_unpadding + uneven ND + multi-shard-per-core.
        pytest.param(
            [3, 158, 158],
            [2, 64, 64],
            _grid(6, 1),
            id="non_tile_aligned_tensor_uneven_nd_multi_per_core",
            marks=untilize_xfail,
        ),
    ]


@pytest.mark.parametrize(
    "tensor_shape, output_nd_shard, output_nd_grid",
    _cases_interleaved_to_nd_rm_to_tile(),
)
@pytest.mark.parametrize(
    "input_buffer_type",
    [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
    ids=["dram", "l1"],
)
def test_to_layout_interleaved_to_nd_sharded_rm_to_tile(
    device, tensor_shape, output_nd_shard, output_nd_grid, input_buffer_type
):
    """Interleaved (DRAM or L1) ROW_MAJOR input -> ND-sharded TILE output.

    Includes shapes whose last two dims are not divisible by 32; to_layout must tilize-with-padding
    before writing into the ND-sharded output.
    """
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG if input_buffer_type == ttnn.BufferType.DRAM else ttnn.L1_MEMORY_CONFIG
    output_mem_config = _make_nd_mem_config(output_nd_shard, output_nd_grid)
    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        ttnn.TILE_LAYOUT,
        output_mem_config,
        from_layout=ttnn.ROW_MAJOR_LAYOUT,
    )


@pytest.mark.parametrize(
    "tensor_shape, output_nd_shard, output_nd_grid",
    _cases_interleaved_to_nd_tile_to_rm(),
)
@pytest.mark.parametrize(
    "input_buffer_type",
    [ttnn.BufferType.DRAM, ttnn.BufferType.L1],
    ids=["dram", "l1"],
)
def test_to_layout_interleaved_to_nd_sharded_tile_to_rm(
    device, tensor_shape, output_nd_shard, output_nd_grid, input_buffer_type
):
    """Interleaved (DRAM or L1) TILE input -> ND-sharded ROW_MAJOR output."""
    input_mem_config = ttnn.DRAM_MEMORY_CONFIG if input_buffer_type == ttnn.BufferType.DRAM else ttnn.L1_MEMORY_CONFIG
    output_mem_config = _make_nd_mem_config(output_nd_shard, output_nd_grid)
    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        ttnn.ROW_MAJOR_LAYOUT,
        output_mem_config,
        from_layout=ttnn.TILE_LAYOUT,
    )


# ---------------------------------------------------------------------------
# to_layout TILE -> ROW_MAJOR where the input is ND-sharded and has a logical
# shape whose last-2 dims are NOT divisible by 32 (so padded_shape > logical_shape).
# This exercises the `requires_padding_change=true` branch in to_layout_op.cpp that
# dispatches to `ttnn::untilize_with_unpadding`.
#
# The ND shard shape's last-2 dims must be tile-multiples (TILE layout requirement),
# so the shard covers the *padded* shape; the output RM tensor is unpadded back to
# the original logical shape.
# ---------------------------------------------------------------------------


def _cases_nd_sharded_tile_to_rm_unpadding():
    """Parametrize cases for ``test_to_layout_nd_sharded_tile_to_rm_untilize_with_unpadding``.

    Each case produces the tuple ``(tensor_logical_shape, input_nd_shard_shape, input_nd_grid,
    output_nd_shard_shape, output_nd_grid)``. Input tensor is created with TILE layout;
    its last-2 dims are NOT tile-aligned so logical_shape != padded_shape. Input ND shard
    shape is sized to the padded shape. Output is RM.
    """
    return [
        # Non-tile-aligned + uneven ND sharding + multi-shard-per-core:
        # [3,158,158] logical -> padded [3,160,160]. Input ND shard [2,64,64] on 6 cores (18 shards, 3/core).
        # Output RM ND shard [2,64,64] on 6 cores (uneven on all dims of LOGICAL shape: dim0 3/2, dim1/2 158/64).
        pytest.param(
            [3, 158, 158],
            [2, 64, 64],
            _grid(6, 1),
            [2, 64, 64],
            _grid(6, 1),
            id="uneven_nd_all_dims_multi_per_core",
        ),
    ]


@pytest.mark.parametrize(
    "tensor_shape, input_nd_shard, input_nd_grid, output_nd_shard, output_nd_grid",
    _cases_nd_sharded_tile_to_rm_unpadding(),
)
@pytest.mark.parametrize(
    "output_memory_mode",
    ["nd_sharded", "interleaved_dram"],
)
def test_to_layout_nd_sharded_tile_to_rm_untilize_with_unpadding(
    device, tensor_shape, input_nd_shard, input_nd_grid, output_nd_shard, output_nd_grid, output_memory_mode
):
    """ND-sharded TILE input with non-tile-aligned logical shape -> RM output.

    Exercises `ttnn::untilize_with_unpadding` inside to_layout (triggered when the TILE
    input's logical_shape differs from its padded_shape).
    """
    input_mem_config = _make_nd_mem_config(input_nd_shard, input_nd_grid)
    if output_memory_mode == "nd_sharded":
        output_mem_config = _make_nd_mem_config(output_nd_shard, output_nd_grid)
    elif output_memory_mode == "interleaved_dram":
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
    else:
        raise ValueError(output_memory_mode)

    _run_to_layout_nd_sharded(
        device,
        tensor_shape,
        input_mem_config,
        ttnn.ROW_MAJOR_LAYOUT,
        output_mem_config,
        from_layout=ttnn.TILE_LAYOUT,
    )
