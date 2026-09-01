# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
from models.common.utility_functions import skip_for_slow_dispatch


@pytest.mark.parametrize(
    "shapes",
    [
        [[63, 1, 4], [1, 9, 4]],
        [[13600, 1, 4], [1, 9, 4]],
        [[1, 16, 6, 64, 64], [1, 16, 1, 64, 64]],
        [[63, 1, 4], [1, 1, 1]],
    ],
)
def test_non_4D_channel_bcast(device, shapes):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shapes[0], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shapes[1], dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64, 1, 0])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = input_tensor + scalar
    output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == (size,)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
def test_add_2D_tensors(device, hw):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
def test_add_2D_tensors_with_program_cache(device, hw):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
@pytest.mark.parametrize("scalar", [0.42])
def test_add_scalar(device, hw, scalar):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a + scalar
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
@pytest.mark.parametrize("scalar", [0.42])
def test_reverse_add_scalar(device, hw, scalar):
    torch_input_tensor_a = torch.rand(hw, dtype=torch.bfloat16)
    torch_output_tensor = scalar + torch_input_tensor_a

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    output = scalar + input_tensor_a
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)


@pytest.mark.parametrize("hw", [(32, 64), (1, 1), (0, 0)])
def test_add_4D_tensors(device, hw):
    torch_input_tensor_a = torch.rand((5, 64, hw[0], hw[1]), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((5, 64, hw[0], hw[1]), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.add(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert_with_ulp(torch_output_tensor, output, ulp_threshold=1)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast(device, h, w):
    # See #4005, we basically are using ttnn.repeat to get this to pass.
    torch_input_tensor_a = torch.rand((2, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_input_tensor_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_with_broadcast_on_batch(device, h, w):
    torch_input_tensor_a = torch.rand((1, 16, 1, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((2, 16, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)


@pytest.mark.parametrize("shape", [(8, 16, 384, 384)])
@pytest.mark.parametrize("scalar", [0.125])
def test_add_attention_scores_to_scalar(device, shape, scalar):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor + scalar

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape_a", [(8, 16, 128, 128)])
@pytest.mark.parametrize("shape_b", [(1, 16, 128, 128)])
def test_add_with_batch_broadcast(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape_a", [(4096, 4096)])
@pytest.mark.parametrize("shape_b", [(1, 4096)])
def test_add_dram_and_l1_tensor(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [[], [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    for activation in activations:
        if activation == "relu":
            torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("shape", [(1, 1, 32, 32)])
@pytest.mark.parametrize("activations", [[], [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]])
def test_in_place_add_and_apply_activations(device, shape, activations):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    for activation in activations:
        if activation == "relu":
            torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.add_(input_tensor_a, input_tensor_b, activations=activations)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.skip(reason="#11002/#4005: Bcast does not appear to be doing what we expect.  Leaving test for reference.")
@pytest.mark.parametrize("shape_a", [(1, 1, 8192, 320)])
@pytest.mark.parametrize("shape_b", [(2, 1, 1, 320)])
def test_add_with_different_batch(device, shape_a, shape_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand(shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    compute_with_storage_grid_size = device.compute_with_storage_grid_size()
    device_grid_size = ttnn.CoreGrid(y=compute_with_storage_grid_size.y, x=compute_with_storage_grid_size.x)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=(1024, 64),
        core_grid=device_grid_size,  # ttnn.CoreGrid(y=8, x=5),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Intended to swap code below with: output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # print("here!!!!!!!!!!!!!!!!!!!!!!!")
    # output_tensor = ttnn.bcast(
    #     input_tensor_a,
    #     input_tensor_b,
    #     ttnn.BcastOpMath.ADD,
    #     ttnn.BcastOpDim.H,
    #     memory_config=input_tensor_a.memory_config(),
    # )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_tensor = ttnn.to_torch(output_tensor)

    # We do not support broadcasting as one would expect,
    # our bcast will return a tensor without the batch 2
    # we also get incorrect pcc as well
    torch_output_tensor = torch_output_tensor[:1]

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape_a


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_height_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024 // 8, 1024)
    else:
        shard_shape = (1024, 1024 // 8)

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_width_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024, 1024 // 8)
    else:
        shard_shape = (1024 // 8, 1024)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, width_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_block_sharding(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    shard_shape = (1024 // 2, 1024 // 4)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize(
    "data",
    [
        ([], [], []),
        ([1], [2], [3]),
        ([1], [], []),
        ([], [1], []),
        ([1, 2], [3], [4, 5]),
        ([1], [2, 3], [3, 4]),
        ([1, 2], [3, 4], [4, 6]),
    ],
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_01_volume_tensors(device, data, memory_config):
    (a, b, c_golden) = data
    a = torch.BFloat16Tensor(a)
    b = torch.BFloat16Tensor(b)
    assert torch.add(a, b).tolist() == c_golden

    ttnn_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    ttnn_c = ttnn.add(ttnn_a, ttnn_b)
    c = ttnn.to_torch(ttnn_c).reshape((-1))

    assert c.tolist() == c_golden


@skip_for_slow_dispatch()
@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
def test_add_with_sub_devices(device, input_a_sharded, input_b_sharded, out_sharded, shard_orientation):
    torch.manual_seed(0)
    shape = (1, 1, 1024, 1024)
    torch_input_tensor_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(shape, dtype=torch.bfloat16)

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (1024 // 8, 1024)
    else:
        shard_shape = (1024, 1024 // 8)

    core_range_set = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(2, 2), ttnn.CoreCoord(3, 3)),
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(1, 1)),
            ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 2)),
        ]
    )

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=core_range_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch_input_tensor_a + torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    sub_device = ttnn.SubDevice(
        [
            ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(4, 4)),
                    ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(5, 0)),
                ]
            )
        ]
    )
    sub_device_manager_id = device.create_sub_device_manager([sub_device], 0)
    device.load_sub_device_manager(sub_device_manager_id)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=1)
    assert output_tensor.shape == shape


@pytest.mark.parametrize("ttnn_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_add_block_float_defaults_to_fpu(device, ttnn_dtype):
    """Block-float operands have no SFPU add kernel; block-float add defaults to FPU."""
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((128, 128), dtype=torch.bfloat16) * 100
    torch_input_tensor_b = torch.randn((128, 128), dtype=torch.bfloat16) * 100

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    default = ttnn.to_torch(ttnn.add(input_tensor_a, input_tensor_b))
    fast = ttnn.to_torch(
        ttnn.add(
            input_tensor_a,
            input_tensor_b,
            fast_and_approximate_mode=True,
        )
    )

    assert torch.equal(default, fast)


def test_add_scalar_dram_sharded_input(device):
    """REGRESSION (issue #54138, finding #6): is_native_L1_sharding()'s scalar-operand branch returns
    !is_uneven(a) with NO buffer-type test, so a DRAM-sharded input is accepted as "native L1 sharding"
    and routed to the globally-allocated-CB path. Metal then throws
    "Only L1 buffers can have an associated circular buffer!" -- a message that names neither the op, nor
    sharding, nor the unsupported combination. (The identical-shape branch of the same predicate DOES
    reject DRAM, which is why this only bites the scalar and COL/SCALAR-broadcast branches.)

    The predicate is a router, not a validator: giving it the missing buffer-type check makes it decline,
    which routes the op down the TensorAccessor path that already serves DRAM-sharded tensors elsewhere.
    So the op should simply produce the right answer rather than throwing."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    dram_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_sharded)

    tt_out = ttnn.add(tt_a, 1.5)

    assert_with_pcc(torch_a + 1.5, ttnn.to_torch(tt_out), 0.999)


def test_add_col_bcast_dram_sharded_input(device):
    """Companion to test_add_scalar_dram_sharded_input, covering the OTHER branch that issue #54138
    finding #6 fixed. is_native_L1_sharding gates its subtile-broadcast branch on a flag that previously
    omitted any buffer-type test, so a DRAM-sharded height-sharded `a` under a COL broadcast was also
    accepted as "native L1" and routed to the globally-allocated-CB path. Same Metal circular-buffer
    throw, different branch -- and the scalar test cannot reach it, because that branch requires a
    tensor `b`."""
    torch.manual_seed(0)
    a_shape = [1, 1, 128, 128]
    b_shape = [1, 1, 128, 1]  # single tile column -> COL_B subtile broadcast

    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(core_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR)
    dram_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_a = torch.rand(a_shape, dtype=torch.bfloat16)
    torch_b = torch.rand(b_shape, dtype=torch.bfloat16)

    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=dram_sharded)
    tt_b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    tt_out = ttnn.add(tt_a, tt_b)

    assert_with_pcc(torch_a + torch_b, ttnn.to_torch(tt_out), 0.999)


def test_add_scalar_sharded_output_grid_differs_from_input(device):
    """REGRESSION (https://github.com/tenstorrent/tt-metal/issues/54138): the native path aliases the
    OUTPUT's buffer as a circular buffer while deriving per-core work from A's shard, so A and C must
    agree on their shard spec. is_native_L1_sharding's identical-shape branch rejects a grid mismatch,
    but the scalar and subtile-broadcast branches did not -- an even L1 input with an even L1 output on
    a DIFFERENT grid was accepted as native, and the output then gets addressed on cores where it is not
    allocated.

    Measured on Wormhole before the fix: this HANGS the device (the run had to be killed and the card
    reset), rather than returning a wrong answer. Declining routes it through the TensorAccessor path,
    which handles the mismatch.

    Note the common case is unaffected: when no output memory config is supplied it is derived from A's,
    so the specs match and the op still takes the native path."""
    torch.manual_seed(0)
    shape = [1, 1, 128, 128]

    a_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    a_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(a_grid, [32, 128], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Same tensor, same layout, both even -- only the grid and shard height differ.
    c_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    c_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(c_grid, [64, 128], ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    tt_a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=a_mem)

    tt_out = ttnn.add(tt_a, 1.5, memory_config=c_mem)

    assert_with_pcc(torch_a + 1.5, ttnn.to_torch(tt_out), 0.999)
