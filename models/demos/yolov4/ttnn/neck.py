# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.ttnn.common import Conv, Upsample
from tt_lib.fallback_ops import fallback_ops


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
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv2 = Conv(
            torch_model,
            "neek.conv2",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv3 = Conv(
            torch_model,
            "neek.conv3",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            reshard=False,
        )

        self.conv4 = Conv(
            torch_model,
            "neek.conv4",
            [1, 10, 10, 2048],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv5 = Conv(
            torch_model,
            "neek.conv5",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv6 = Conv(
            torch_model,
            "neek.conv6",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv7 = Conv(
            torch_model,
            "neek.conv7",
            [1, 10, 10, 512],
            (1, 1, 0, 0),
            width_sharding=True,
            deallocate=False,
        )
        self.upsample1 = Upsample(
            [1, 10, 10, 256],
            [2, 2],
        )
        self.conv7_2 = Conv(
            torch_model,
            "neek.conv8",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv7_3 = Conv(
            torch_model,
            "neek.conv9",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv8 = Conv(
            torch_model,
            "neek.conv10",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
        )
        self.conv7_4 = Conv(
            torch_model,
            "neek.conv11",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv8_2 = Conv(
            torch_model,
            "neek.conv12",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
        )
        self.conv7_5 = Conv(
            torch_model,
            "neek.conv13",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

        self.conv9 = Conv(
            torch_model,
            "neek.conv14",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            deallocate=False,
            height_sharding=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.upsample2 = Upsample(
            [1, 20, 20, 128],
            [2, 2],
        )
        self.conv9_2 = Conv(
            torch_model,
            "neek.conv15",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
        )
        self.conv9_3 = Conv(
            torch_model,
            "neek.conv16",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
        )
        self.conv10 = Conv(
            torch_model,
            "neek.conv17",
            [1, 40, 40, 128],
            (1, 1, 1, 1),
        )

        self.conv9_4 = Conv(
            torch_model,
            "neek.conv18",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
        )
        self.conv10_2 = Conv(
            torch_model,
            "neek.conv19",
            [1, 40, 40, 128],
            (1, 1, 1, 1),
        )
        self.conv9_5 = Conv(
            torch_model,
            "neek.conv20",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
        )

    def __call__(self, device, input_tensor):
        output_tensor = self.conv1(device, input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        pool_1 = ttnn.max_pool2d(
            input_tensor=output_tensor,
            batch_size=1,
            input_h=10,
            input_w=10,
            channels=512,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        pool_2 = ttnn.max_pool2d(
            input_tensor=output_tensor,
            batch_size=1,
            input_h=10,
            input_w=10,
            channels=512,
            kernel_size=[9, 9],
            stride=[1, 1],
            padding=[4, 4],
            dilation=[1, 1],
        )
        pool_3 = ttnn.max_pool2d(
            input_tensor=output_tensor,
            batch_size=1,
            input_h=10,
            input_w=10,
            channels=512,
            kernel_size=[13, 13],
            stride=[1, 1],
            padding=[6, 6],
            dilation=[1, 1],
        )

        pool_1 = ttnn.sharded_to_interleaved(pool_1, ttnn.L1_MEMORY_CONFIG)
        pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
        pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)
        pool_1 = ttnn.to_layout(pool_1, layout=ttnn.TILE_LAYOUT)  # This is becauase output_tensor is in TILE_LAYOUT
        pool_2 = ttnn.to_layout(pool_2, layout=ttnn.TILE_LAYOUT)  # This is becauase output_tensor is in TILE_LAYOUT
        pool_3 = ttnn.to_layout(pool_3, layout=ttnn.TILE_LAYOUT)  # This is becauase output_tensor is in TILE_LAYOUT
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

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
        output_tensor = ttnn.reshape(output_tensor, (1, 10, 10, 256))
        output_tensor_upsample_1 = self.upsample1(device, output_tensor)
        output_tensor_upsample_1 = ttnn.sharded_to_interleaved(output_tensor_upsample_1, ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_1 = ttnn.reshape(output_tensor_upsample_1, (1, 1, 400, 256))
        output_tensor_upsample_1 = ttnn.to_layout(output_tensor_upsample_1, layout=ttnn.TILE_LAYOUT)

        outDowSample5 = input_tensor[1]
        output_tensor = self.conv7_2(device, outDowSample5)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

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

        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(6, 3),
                ),
            }
        )
        shard_spec = ttnn.ShardSpec(shard_grid, (64, 64), ttnn.ShardOrientation.COL_MAJOR, False)
        in_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        output_tensor_left_2 = ttnn.to_memory_config(output_tensor_left_2, memory_config=in_sharded_mem_config)
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

        output_tensor = ttnn.reshape(output_tensor, (1, 20, 20, 128))
        output_tensor_upsample_2 = self.upsample2(device, output_tensor)
        output_tensor_upsample_2 = ttnn.sharded_to_interleaved(output_tensor_upsample_2, ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_2 = ttnn.reshape(output_tensor_upsample_2, (1, 1, 1600, 128))
        output_tensor_upsample_2 = ttnn.to_layout(output_tensor_upsample_2, ttnn.TILE_LAYOUT)

        outDowSample3 = input_tensor[2]

        output_tensor = self.conv9_2(device, outDowSample3)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

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

        return output_tensor, output_tensor_left_1, output_tensor_left_2
