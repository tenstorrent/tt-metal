# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from models.utility_functions import get_debug_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc
from enum import Enum


@pytest.mark.parametrize(
    "input_height, input_width, input_memory_layout, input_sharded_memory_config_args, output_sharded_memory_config_args, input_override, output_override",
    [
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            dict(core_grid=ttnn.CoreGrid(y=4, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            128,
            64,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=4, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            128,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=4), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            16,
            ttnn.ROW_MAJOR_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=2, x=1), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            32,
            2304,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        (
            32,
            1792,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=7, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            [32, 32],
            None,
        ),
        (
            32,
            7168,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=7, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            [32, 128],
            None,
        ),
        (
            32,
            320,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=1, x=5), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        (
            8192,
            320,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=8, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=8, x=5), strategy=ttnn.ShardStrategy.BLOCK),
            None,
            None,
        ),
        # (1, 1, 32, 8192) (32 to 8 cores width shardrd)
        (
            32,
            8192,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            (32, 256),
            None,
        ),
        # (1, 1, 32, 8192) (64 to 8 cores width shardrd)
        (
            32,
            8192,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=8, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            (32, 128),
            None,
        ),
        # (1, 1, 32, 1280) (8 to 1 cores width shardrd)
        (
            32,
            1280,
            ttnn.ROW_MAJOR_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=1, x=8), strategy=ttnn.ShardStrategy.WIDTH),
            dict(core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.WIDTH),
            None,
            None,
        ),
        # (1, 1, 128, 1280) (32 cores block sharded to 4 cores height sharded)
        (
            128,
            1280,
            ttnn.TILE_LAYOUT,
            dict(core_grid=ttnn.CoreGrid(y=4, x=8), strategy=ttnn.ShardStrategy.BLOCK),
            dict(core_grid=ttnn.CoreGrid(y=4, x=1), strategy=ttnn.ShardStrategy.HEIGHT),
            None,
            None,
        ),
        (
            160,
            64,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreGrid(y=5, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
            dict(
                core_grid=ttnn.CoreGrid(y=2, x=2),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            (32, 64),
            (32, 96),
        ),
        (
            192,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (96, 128),
            (128, 64),
        ),
        (
            128,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (64, 128),
            (96, 64),
        ),
        (
            96,
            128,
            ttnn.TILE_LAYOUT,
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 2)),
                    }
                ),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1)),
                    }
                ),
                strategy=ttnn.ShardStrategy.BLOCK,
            ),
            (32, 128),
            (64, 64),
        ),
    ],
)
def test_reshard(
    device,
    input_height,
    input_width,
    input_memory_layout,
    input_sharded_memory_config_args,
    output_sharded_memory_config_args,
    input_override,
    output_override,
):
    if isinstance(input_sharded_memory_config_args["core_grid"], (ttnn.CoreGrid)):
        if device.core_grid.y < input_sharded_memory_config_args["core_grid"].y:
            pytest.skip()
        if device.core_grid.y < output_sharded_memory_config_args["core_grid"].y:
            pytest.skip()
    input_shape = [1, 1, input_height, input_width]

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    interleaved_input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=input_memory_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if input_override == None:
        input_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **input_sharded_memory_config_args)
    else:
        input_shard_memory_config = ttnn.create_sharded_memory_config(
            input_override, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
        )

    if output_override == None:
        output_shard_memory_config = ttnn.create_sharded_memory_config(input_shape, **output_sharded_memory_config_args)
    else:
        output_shard_memory_config = ttnn.create_sharded_memory_config(
            output_override, **output_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
        )
    # interleaved_to_sharded
    sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)

    # reshard
    sharded_output_tensor = ttnn.to_memory_config(sharded_input_tensor, output_shard_memory_config)

    # sharded_to_interleaved
    interleaved_output_tensor = ttnn.to_memory_config(sharded_output_tensor, ttnn.DRAM_MEMORY_CONFIG)

    output = ttnn.to_torch(interleaved_output_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


class DirectReadWriteType(Enum):
    READ_ONLY = 0
    WRITE_ONLY = 1
    READ_WRITE = 2
    NONE = 3


@pytest.mark.parametrize(
    "data_transfer_strategy",
    [
        (DirectReadWriteType.READ_ONLY),
        (DirectReadWriteType.WRITE_ONLY),
        (DirectReadWriteType.NONE),
        (DirectReadWriteType.READ_WRITE),
    ],
)
@pytest.mark.parametrize(
    "input_shape, input_shard_shape,  input_sharded_memory_config_args",
    [
        (
            [1, 1, 32, 1024],
            [32, 256],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 3)),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
        (
            [1, 1, 32, 1024],
            [32, 128],
            dict(
                core_grid=ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 2), ttnn.CoreCoord(0, 3)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 4), ttnn.CoreCoord(0, 7)),
                    }
                ),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
        ),
    ],
)
def test_shard_with_corerangeset(
    device, input_shape, input_shard_shape, input_sharded_memory_config_args, data_transfer_strategy
):
    if device.core_grid.y == 7 and input_shard_shape == [32, 128]:
        pytest.skip()
    if (
        ((not (input_shape[2] % input_shard_shape[0] == 0)) or (not (input_shape[3] % input_shard_shape[1] == 0)))
        and (not (data_transfer_strategy == DirectReadWriteType.READ_WRITE))
        and (not (input_sharded_memory_config_args["strategy"] == ttnn.ShardStrategy.HEIGHT))
    ):
        pytest.skip()

    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_shard_memory_config = ttnn.create_sharded_memory_config(
        input_shard_shape, **input_sharded_memory_config_args, use_height_and_width_as_shard_shape=True
    )

    if data_transfer_strategy == DirectReadWriteType.READ_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        interleaved_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        # interleaved_to_sharded
        sharded_input_tensor = ttnn.to_memory_config(interleaved_input_tensor, input_shard_memory_config)
    else:
        sharded_input_tensor = ttnn.from_torch(
            torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device, memory_config=input_shard_memory_config
        )

    if data_transfer_strategy == DirectReadWriteType.WRITE_ONLY or data_transfer_strategy == DirectReadWriteType.NONE:
        # sharded_to_interleaved
        interleaved_output_tensor = ttnn.to_memory_config(sharded_input_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.to_torch(interleaved_output_tensor)
    else:
        output = ttnn.to_torch(sharded_input_tensor)

    assert_with_pcc(torch_input_tensor, output, 1.0)


@pytest.mark.parametrize(
    "shape, strategy, orientation, core_grid",
    [
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 1024, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=2, x=2)),
        ([1, 1, 128, 1024], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=4)),
        ([1, 1, 1024, 128], ttnn.ShardStrategy.BLOCK, ttnn.ShardOrientation.COL_MAJOR, ttnn.CoreGrid(y=4, x=2)),
    ],
)
def test_create_sharded_memory_config(device, shape, strategy, orientation, core_grid):
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )
    shard_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=core_grid,
        strategy=strategy,
        orientation=orientation,
        use_height_and_width_as_shard_shape=False,
    )

    x_t = ttnn.to_memory_config(x, memory_config=shard_config, dtype=ttnn.bfloat16)
    output_data = ttnn.from_device(x_t)
    output_data = ttnn.to_torch(output_data)

    passing = torch.equal(input_data, output_data)
    assert passing


@pytest.mark.parametrize(
    "shape, shard_shape, strategy, orientation, core_grid",
    [
        ([1, 1, 2, 16], None, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=1, x=1)),
        ([1, 1, 2, 16], None, ttnn.ShardStrategy.WIDTH, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=1)),
        ([1, 1, 32, 16], None, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=1)),
        ([1, 1, 64, 16], None, ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=2, x=1)),
        (
            [1, 1, 2, 16],
            [2, 16],
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
        ),
        (
            [1, 1, 5280, 16],
            [5280, 16],
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0)),
                }
            ),
        ),
        # TODO: Add this test back by checking for core grid size and skipping if we can't do it
        #        (
        #            [1, 1, 675840, 16],
        #            [5280, 16],
        #            ttnn.ShardStrategy.HEIGHT,
        #            ttnn.ShardOrientation.ROW_MAJOR,
        #            ttnn.CoreRangeSet(
        #                {
        #                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(11, 9)),  # 120
        #                    ttnn.CoreRange(ttnn.CoreCoord(12, 0), ttnn.CoreCoord(12, 7)),  # 8
        #                }
        #            ),
        #        ),
    ],
)
@pytest.mark.parametrize(
    "input_buffer_type",
    [
        ttnn.L1_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
    ],
)
@pytest.mark.parametrize(
    "output_buffer_type",
    [
        ttnn.L1_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
    ],
)
def test_bh_alignment_i2s(
    device, shape, shard_shape, strategy, orientation, core_grid, input_buffer_type, output_buffer_type
):
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    if shard_shape == None:
        shard_config = ttnn.create_sharded_memory_config(
            shape=shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=False,
        )
    else:
        shard_config = ttnn.create_sharded_memory_config(
            shape=shard_shape,
            core_grid=core_grid,
            strategy=strategy,
            orientation=orientation,
            use_height_and_width_as_shard_shape=True,
        )
    x_t = ttnn.from_torch(
        input_data,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=input_buffer_type,
        dtype=ttnn.bfloat16,
    )
    # So far the sharded tensor alignment is controled by keep_l1_aligned flag, will remove it later after launch
    x_t_sharded = ttnn.interleaved_to_sharded(x_t, shard_config, keep_l1_aligned=True)
    x_t = ttnn.sharded_to_interleaved(x_t_sharded, output_buffer_type, is_l1_aligned=True)
    output_data = ttnn.from_device(x_t)
    output_data = ttnn.to_torch(output_data)
    passing = torch.equal(input_data, output_data)
    assert passing


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "kernel_size",
    ([2, 2],),
)
@pytest.mark.parametrize(
    "padding",
    ([0, 0],),
)
@pytest.mark.parametrize("stride", ([2, 2],))
@pytest.mark.parametrize("dilation", ([1, 1],))
@pytest.mark.parametrize("shape", ([1, 64, 24, 24],))
@pytest.mark.parametrize(
    "output_buffer_type",
    [
        ttnn.L1_MEMORY_CONFIG,
        ttnn.DRAM_MEMORY_CONFIG,
    ],
)
def test_mnist_max_pool_s2i(
    device,
    shape,
    output_buffer_type,
    kernel_size,
    stride,
    padding,
    dilation,
):
    pytest.skip("currently fails due to i2s call, see GH #18425")
    torch.manual_seed(0)
    input_data = torch.randn(shape, dtype=torch.bfloat16)
    in_n, in_c, in_h, in_w = shape
    act_shape = (1, 1, in_n * in_h * in_w, in_c)
    act_permuted = torch.permute(input_data, (0, 2, 3, 1))
    act_reshaped = act_permuted.reshape(act_shape)
    ttact = ttnn.from_torch(act_reshaped, ttnn.bfloat16)

    ttact_device = ttnn.to_device(ttact, device)

    output = ttnn.max_pool2d(
        input_tensor=ttact_device,
        batch_size=in_n,
        input_h=in_h,
        input_w=in_w,
        channels=in_c,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        memory_config=None,
        applied_shard_scheme=None,
        ceil_mode=False,
    )

    x_t = ttnn.sharded_to_interleaved(output, output_buffer_type, is_l1_aligned=True)
    output_data = ttnn.from_device(x_t)
    output_pytorch_padded = torch.Tensor(ttnn.to_torch(output_data))
    output_pytorch = output_pytorch_padded[:, :, :, :in_c]

    ## reference
    golden_pytorch = torch.nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        return_indices=False,
        ceil_mode=False,
    )(input_data)

    ## test for equivalance
    golden_shape = golden_pytorch.shape
    output_pytorch = output_pytorch.reshape(golden_shape[0], golden_shape[2], golden_shape[3], golden_shape[1])

    output_pytorch = torch.permute(output_pytorch, (0, 3, 1, 2))  ## N, C, H, W

    pcc_thresh = 1.0
    assert_with_pcc(output_pytorch, golden_pytorch, pcc_thresh)


@pytest.mark.parametrize(
    "shape, orientation, core_grid_1, core_grid_2",
    [
        ([1, 1, 224, 384], ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=1, x=3), ttnn.CoreGrid(y=1, x=4)),
        ([1, 1, 64, 384], ttnn.ShardOrientation.ROW_MAJOR, ttnn.CoreGrid(y=1, x=3), ttnn.CoreGrid(y=1, x=4)),
    ],
)
def test_reshard_conv(device, shape, orientation, core_grid_1, core_grid_2):
    torch.manual_seed(0)

    debug = False
    if not debug:
        torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    else:
        num_tiles_width = shape[3] // 32
        num_tiles_height = shape[2] // 32
        torch_input_tensor = get_debug_tensor(num_tiles_width, num_tiles_height, dtype=torch.bfloat16)
        torch.set_printoptions(profile="full")
        print("Input Tensor:", torch_input_tensor)
        torch.set_printoptions(profile="default")
    ttnn_tensor_1 = ttnn.from_torch(
        torch_input_tensor,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.create_sharded_memory_config(
            shape=torch_input_tensor.shape,
            core_grid=core_grid_1,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    out_tensor_1 = ttnn.to_torch(ttnn_tensor_1)

    memory_config_2 = ttnn.create_sharded_memory_config(
        shape=torch_input_tensor.shape,
        core_grid=core_grid_2,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    # Interleaved to sharded path
    ttnn_tensor_1_dram = ttnn.to_memory_config(ttnn_tensor_1, ttnn.DRAM_MEMORY_CONFIG)
    ttnn_tensor_2_interleaved_to_sharded = ttnn.to_memory_config(ttnn_tensor_1_dram, memory_config_2)
    out_tensor_2_interleaved_to_sharded = ttnn.to_torch(ttnn_tensor_2_interleaved_to_sharded)

    passing, pcc_msg = assert_with_pcc(out_tensor_1, out_tensor_2_interleaved_to_sharded, pcc=1.0)

    ttnn_tensor_2_resharded = ttnn.reshard(ttnn_tensor_1, memory_config_2)
    out_tensor_2_resharded = ttnn.to_torch(ttnn_tensor_2_resharded)
    if debug:
        torch.set_printoptions(profile="full")
        print("Output Tensor 2 Resharded:", out_tensor_2_resharded)
        torch.set_printoptions(profile="default")

    passing, pcc_msg = assert_with_pcc(out_tensor_1, out_tensor_2_resharded, pcc=1.0)
