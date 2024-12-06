# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv
from tt_lib.fallback_ops import fallback_ops
import math


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


def get_core_grid_from_num_cores(num_cores: int, grid_rows: int = 8, grid_cols: int = 8):
    rows = num_cores // grid_cols
    assert rows <= grid_rows, "Not enough cores for specified core grid"
    ranges = []
    if rows != 0:
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(grid_rows - 1, rows - 1),
            )
        )
    remainder = num_cores % grid_rows
    if remainder != 0:
        assert rows + 1 <= grid_rows, "Not enough cores for specified core grid"
        ranges.append(
            ttnn.CoreRange(
                ttnn.CoreCoord(0, rows),
                ttnn.CoreCoord(remainder - 1, rows),
            )
        )
    return ttnn.CoreRangeSet({*ranges})


class TtNeck:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model,
            "neek.conv1",
            [1, 20, 20, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv2 = Conv(
            torch_model,
            "neek.conv2",
            [1, 20, 20, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv3 = Conv(
            torch_model,
            "neek.conv3",
            [1, 20, 20, 1024],
            (1, 1, 0, 0),
            reshard=False,
        )

        self.conv4 = Conv(
            torch_model,
            "neek.conv4",
            [1, 20, 20, 2048],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv5 = Conv(
            torch_model,
            "neek.conv5",
            [1, 20, 20, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv6 = Conv(
            torch_model,
            "neek.conv6",
            [1, 20, 20, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv7 = Conv(
            torch_model,
            "neek.conv7",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            width_sharding=True,
            deallocate=False,
        )
        self.conv7_2 = Conv(
            torch_model,
            "neek.conv8",
            [1, 40, 40, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv7_3 = Conv(
            torch_model,
            "neek.conv9",
            [1, 40, 40, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv8 = Conv(
            torch_model,
            "neek.conv10",
            [1, 40, 40, 256],
            (1, 1, 1, 1),
        )
        self.conv7_4 = Conv(
            torch_model,
            "neek.conv11",
            [1, 40, 40, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv8_2 = Conv(
            torch_model,
            "neek.conv12",
            [1, 40, 40, 256],
            (1, 1, 1, 1),
        )
        self.conv7_5 = Conv(
            torch_model,
            "neek.conv13",
            [1, 40, 40, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

        self.conv9 = Conv(
            torch_model,
            "neek.conv14",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv9_2 = Conv(
            torch_model,
            "neek.conv15",
            [1, 80, 80, 256],
            (1, 1, 0, 0),
        )
        self.conv9_3 = Conv(
            torch_model,
            "neek.conv16",
            [1, 80, 80, 256],
            (1, 1, 0, 0),
        )
        self.conv10 = Conv(
            torch_model,
            "neek.conv17",
            [1, 80, 80, 128],
            (1, 1, 1, 1),
        )

        self.conv9_4 = Conv(
            torch_model,
            "neek.conv18",
            [1, 80, 80, 256],
            (1, 1, 0, 0),
        )
        self.conv10_2 = Conv(
            torch_model,
            "neek.conv19",
            [1, 80, 80, 128],
            (1, 1, 1, 1),
        )
        self.conv9_5 = Conv(
            torch_model,
            "neek.conv20",
            [1, 80, 80, 256],
            (1, 1, 0, 0),
        )

    def __call__(self, device, input_tensor):
        output_tensor = self.conv1(device, input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor_1 = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        pool_1 = ttnn.max_pool2d(
            input_tensor=output_tensor_1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        pool_2 = ttnn.max_pool2d(
            input_tensor=output_tensor_1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[9, 9],
            stride=[1, 1],
            padding=[4, 4],
            dilation=[1, 1],
        )
        pool_3 = ttnn.max_pool2d(
            input_tensor=output_tensor_1,
            batch_size=1,
            input_h=20,
            input_w=20,
            channels=512,
            kernel_size=[13, 13],
            stride=[1, 1],
            padding=[6, 6],
            dilation=[1, 1],
        )
        ttnn.deallocate(output_tensor_1)

        if pool_1.is_sharded():
            pool_1 = ttnn.sharded_to_interleaved(pool_1, ttnn.L1_MEMORY_CONFIG)
        if pool_2.is_sharded():
            pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
        if pool_3.is_sharded():
            pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)

        if pool_1.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            pool_1 = ttnn.to_layout(pool_1, layout=ttnn.TILE_LAYOUT)
        if pool_2.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            pool_2 = ttnn.to_layout(pool_2, layout=ttnn.TILE_LAYOUT)
        if pool_3.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
            pool_3 = ttnn.to_layout(pool_3, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.concat([pool_3, pool_2, pool_1, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pool_3)
        ttnn.deallocate(pool_2)
        ttnn.deallocate(pool_1)

        output_tensor = self.conv4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv5(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv6(device, output_tensor)
        output_tensor_left_1 = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7(device, output_tensor_left_1)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_shape = output_tensor.shape
        output_tensor = ttnn.untilize_with_unpadding(
            output_tensor,
            output_tensor_end=(
                output_shape[0] - 1,
                output_shape[1] - 1,
                output_shape[2] - 1,
                output_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        output_tensor = ttnn.reshape(output_tensor, (1, 20, 20, 256))

        # upsample optimization
        nhw = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, output_tensor.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            output_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if output_tensor.is_sharded():
            output_tensor = ttnn.reshard(output_tensor, shardspec)
        else:
            output_tensor = ttnn.interleaved_to_sharded(output_tensor, shardspec)

        output_tensor_upsample_1 = ttnn.upsample(
            output_tensor, scale_factor=2, memory_config=output_tensor.memory_config()
        )  # 11

        if output_tensor_upsample_1.is_sharded():
            output_tensor_upsample_1 = ttnn.sharded_to_interleaved(output_tensor_upsample_1, ttnn.L1_MEMORY_CONFIG)

        # output_tensor_upsample_1 = ttnn.upsample(output_tensor, (2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_1 = ttnn.reshape(output_tensor_upsample_1, (1, 1, 1600, 256))
        output_tensor_upsample_1 = ttnn.to_layout(output_tensor_upsample_1, layout=ttnn.TILE_LAYOUT)

        outDowSample5 = input_tensor[1]

        if outDowSample5.is_sharded():
            outDowSample5 = ttnn.sharded_to_interleaved(outDowSample5, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv7_2(device, outDowSample5)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_1)

        output_tensor = self.conv7_3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7_4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8_2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7_5(device, output_tensor)
        output_tensor_left_2 = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if output_tensor_left_2.is_sharded():
            output_tensor_left_2 = ttnn.sharded_to_interleaved(output_tensor_left_2, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv9(device, output_tensor_left_2)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_shape = output_tensor.shape
        output_tensor = ttnn.untilize_with_unpadding(
            output_tensor,
            output_tensor_end=(
                output_shape[0] - 1,
                output_shape[1] - 1,
                output_shape[2] - 1,
                output_shape[3] - 1,
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        output_tensor = ttnn.reshape(output_tensor, (1, 40, 40, 128))

        # upsample optimization
        nhw = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, output_tensor.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            output_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

        if output_tensor.is_sharded():
            output_tensor = ttnn.reshard(output_tensor, shardspec)
        else:
            output_tensor = ttnn.interleaved_to_sharded(output_tensor, shardspec)

        output_tensor_upsample_2 = ttnn.upsample(
            output_tensor, scale_factor=2, memory_config=output_tensor.memory_config()
        )  # 11

        if output_tensor_upsample_2.is_sharded():
            output_tensor_upsample_2 = ttnn.sharded_to_interleaved(output_tensor_upsample_2, ttnn.L1_MEMORY_CONFIG)

        # output_tensor_upsample_2 = ttnn.upsample(output_tensor, (2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_2 = ttnn.reshape(output_tensor_upsample_2, (1, 1, 6400, 128))
        output_tensor_upsample_2 = ttnn.to_layout(output_tensor_upsample_2, ttnn.TILE_LAYOUT)

        outDowSample3 = input_tensor[2]

        output_tensor = self.conv9_2(device, outDowSample3)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_2)

        output_tensor = self.conv9_3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv10(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9_4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv10_2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9_5(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        ttnn.deallocate(input_tensor[0])
        ttnn.deallocate(input_tensor[1])
        ttnn.deallocate(input_tensor[2])

        return output_tensor, output_tensor_left_1, output_tensor_left_2
