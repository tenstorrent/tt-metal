# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.ttnn.common import Conv
from tt_lib.fallback_ops import fallback_ops


class TtNeck:
    def __init__(self, device, parameters, conv_args) -> None:
        self.conv_args = conv_args
        self.parameters = parameters
        self.conv1 = Conv(
            device,
            conv_args.c1,
            parameters.c1,
        )
        self.conv2 = Conv(
            device,
            conv_args.c2,
            parameters.c2,
        )
        self.conv3 = Conv(
            device,
            conv_args.c3,
            parameters.c3,
        )
        self.conv4 = Conv(
            device,
            conv_args.c4,
            parameters.c4,
        )
        self.conv5 = Conv(
            device,
            conv_args.c5,
            parameters.c5,
        )
        self.conv6 = Conv(
            device,
            conv_args.c6,
            parameters.c6,
        )
        self.conv7 = Conv(
            device,
            conv_args.c7,
            parameters.c7,
        )
        self.conv7_2 = Conv(
            device,
            conv_args.c7_2,
            parameters.c7_2,
        )
        self.conv7_3 = Conv(
            device,
            conv_args.c7_3,
            parameters.c7_3,
        )
        self.conv8 = Conv(
            device,
            conv_args.c8,
            parameters.c8,
        )
        self.conv7_4 = Conv(
            device,
            conv_args.c7_4,
            parameters.c7_4,
        )
        self.conv8_2 = Conv(
            device,
            conv_args.c8_2,
            parameters.c8_2,
        )
        self.conv7_5 = Conv(
            device,
            conv_args.c7_5,
            parameters.c7_5,
        )

        self.conv9 = Conv(
            device,
            conv_args.c9,
            parameters.c9,
        )
        self.conv9_2 = Conv(
            device,
            conv_args.c9_2,
            parameters.c9_2,
        )
        self.conv9_3 = Conv(
            device,
            conv_args.c9_3,
            parameters.c9_3,
        )
        self.conv10 = Conv(
            device,
            conv_args.c10,
            parameters.c10,
        )

        self.conv9_4 = Conv(
            device,
            conv_args.c9_4,
            parameters.c9_4,
        )
        self.conv10_2 = Conv(
            device,
            conv_args.c10_2,
            parameters.c10_2,
        )
        self.conv9_5 = Conv(
            device,
            conv_args.c9_5,
            parameters.c9_5,
        )

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor[0])[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv2(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv3(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if not self.parameters.is_320_res:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
            output_tensor_pool_in = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            output_tensor_pool_in = output_tensor

        pool_1 = ttnn.max_pool2d(
            input_tensor=output_tensor_pool_in,
            batch_size=self.conv_args.p1.batch_size,
            input_h=self.conv_args.p1.input_height,
            input_w=self.conv_args.p1.input_width,
            channels=output_tensor.shape[3],
            kernel_size=[self.conv_args.p1.kernel_size, self.conv_args.p1.kernel_size],
            stride=[self.conv_args.p1.stride, self.conv_args.p1.stride],
            padding=[self.conv_args.p1.padding, self.conv_args.p1.padding],
            dilation=[self.conv_args.p1.dilation, self.conv_args.p1.dilation],
        )
        pool_2 = ttnn.max_pool2d(
            input_tensor=output_tensor_pool_in,
            batch_size=self.conv_args.p2.batch_size,
            input_h=self.conv_args.p2.input_height,
            input_w=self.conv_args.p2.input_width,
            channels=output_tensor.shape[3],
            kernel_size=[self.conv_args.p2.kernel_size, self.conv_args.p2.kernel_size],
            stride=[self.conv_args.p2.stride, self.conv_args.p2.stride],
            padding=[self.conv_args.p2.padding, self.conv_args.p2.padding],
            dilation=[self.conv_args.p2.dilation, self.conv_args.p2.dilation],
        )
        pool_3 = ttnn.max_pool2d(
            input_tensor=output_tensor_pool_in,
            batch_size=self.conv_args.p3.batch_size,
            input_h=self.conv_args.p3.input_height,
            input_w=self.conv_args.p3.input_width,
            channels=output_tensor.shape[3],
            kernel_size=[self.conv_args.p3.kernel_size, self.conv_args.p3.kernel_size],
            stride=[self.conv_args.p3.stride, self.conv_args.p3.stride],
            padding=[self.conv_args.p3.padding, self.conv_args.p3.padding],
            dilation=[self.conv_args.p3.dilation, self.conv_args.p3.dilation],
        )

        pool_1 = ttnn.sharded_to_interleaved(pool_1, ttnn.L1_MEMORY_CONFIG)
        pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
        pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)

        pool_all = ttnn.concat([pool_3, pool_2, pool_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        pool_all = ttnn.to_layout(pool_all, layout=ttnn.TILE_LAYOUT)  # This is becauase output_tensor is in TILE_LAYOUT
        ttnn.deallocate(pool_3)
        ttnn.deallocate(pool_2)
        ttnn.deallocate(pool_1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([pool_all, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv4(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv5(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv6(output_tensor)[0]
        output_tensor_left_1 = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7(output_tensor_left_1)[0]
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

        output_tensor = ttnn.reshape(
            output_tensor,
            (
                1,
                self.conv_args.c7.input_height,
                self.conv_args.c7.input_width,
                self.conv_args.c7.out_channels,
            ),
        )
        if self.parameters.is_320_res:
            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 4),
                    ),
                }
            )
            shard_spec = ttnn.ShardSpec(shard_grid, (20, 32), ttnn.ShardOrientation.ROW_MAJOR)
            in_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec
            )
            output_tensor = ttnn.to_memory_config(output_tensor, memory_config=in_sharded_mem_config)
            shard_spec = ttnn.ShardSpec(shard_grid, (80, 32), ttnn.ShardOrientation.ROW_MAJOR)
            out_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )

            output_tensor_upsample_1 = ttnn.upsample(output_tensor, (2, 2), memory_config=out_sharded_mem_config)
            output_tensor_upsample_1 = ttnn.sharded_to_interleaved(output_tensor_upsample_1, ttnn.L1_MEMORY_CONFIG)
        else:
            output_tensor_upsample_1 = ttnn.upsample(output_tensor, (2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor_upsample_1 = ttnn.reshape(
            output_tensor_upsample_1,
            (
                1,
                1,
                output_tensor_upsample_1.shape[1] * output_tensor_upsample_1.shape[2],
                output_tensor_upsample_1.shape[3],
            ),
        )
        output_tensor_upsample_1 = ttnn.to_layout(output_tensor_upsample_1, layout=ttnn.TILE_LAYOUT)

        outDowSample5 = input_tensor[1]

        if not self.parameters.is_320_res:
            if outDowSample5.is_sharded():
                outDowSample5 = ttnn.sharded_to_interleaved(outDowSample5, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv7_2(outDowSample5)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_1)

        output_tensor = self.conv7_3(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7_4(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8_2(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7_5(output_tensor)[0]
        output_tensor_left_2 = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        if self.parameters.is_320_res:
            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(6, 3),
                    ),
                }
            )
            shard_spec = ttnn.ShardSpec(shard_grid, (64, 64), ttnn.ShardOrientation.COL_MAJOR)
            in_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec
            )
            output_tensor_left_2 = ttnn.to_memory_config(output_tensor_left_2, memory_config=in_sharded_mem_config)
        else:
            if output_tensor_left_2.is_sharded():
                output_tensor_left_2 = ttnn.sharded_to_interleaved(output_tensor_left_2, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv9(output_tensor_left_2)[0]
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

        output_tensor = ttnn.reshape(
            output_tensor,
            (
                1,
                self.conv_args.c9.input_height,
                self.conv_args.c9.input_width,
                self.conv_args.c9.out_channels,
            ),
        )

        if self.parameters.is_320_res:
            shard_grid = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(0, 0),
                        ttnn.CoreCoord(7, 4),
                    ),
                }
            )
            shard_spec = ttnn.ShardSpec(shard_grid, (80, 16), ttnn.ShardOrientation.ROW_MAJOR)
            in_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec
            )
            output_tensor = ttnn.to_memory_config(output_tensor, memory_config=in_sharded_mem_config)
            shard_spec = ttnn.ShardSpec(shard_grid, (80 * 4, 16), ttnn.ShardOrientation.ROW_MAJOR)
            out_sharded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
            )

            output_tensor_upsample_2 = ttnn.upsample(output_tensor, (2, 2), memory_config=out_sharded_mem_config)
            output_tensor_upsample_2 = ttnn.sharded_to_interleaved(output_tensor_upsample_2, ttnn.L1_MEMORY_CONFIG)
        else:
            output_tensor_upsample_2 = ttnn.upsample(output_tensor, (2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor_upsample_2 = ttnn.reshape(
            output_tensor_upsample_2,
            (
                1,
                1,
                output_tensor_upsample_2.shape[1] * output_tensor_upsample_2.shape[2],
                output_tensor_upsample_2.shape[3],
            ),
        )
        output_tensor_upsample_2 = ttnn.to_layout(output_tensor_upsample_2, ttnn.TILE_LAYOUT)

        outDowSample3 = input_tensor[2]

        output_tensor = self.conv9_2(outDowSample3)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_2)

        output_tensor = self.conv9_3(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv10(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9_4(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv10_2(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9_5(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        ttnn.deallocate(input_tensor[0])
        ttnn.deallocate(input_tensor[1])
        ttnn.deallocate(input_tensor[2])

        return output_tensor, output_tensor_left_1, output_tensor_left_2
