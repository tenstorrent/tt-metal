# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.yolov4.common import create_conv2d_config
from models.tt_cnn.tt.builder import TtConv2d


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores <= max_cores:
        return cores
    for divisor in range(max_cores, 0, -1):
        if nhw % divisor == 0 and (nhw // divisor) % width == 0:
            return divisor
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
    def __init__(self, device, parameters, conv_args) -> None:
        self.conv_args = conv_args
        self.parameters = parameters
        self.conv1 = TtConv2d(
            create_conv2d_config(
                conv_args.c1,
                parameters.c1.weight,
                parameters.c1.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv2 = TtConv2d(
            create_conv2d_config(
                conv_args.c2,
                parameters.c2.weight,
                parameters.c2.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv3 = TtConv2d(
            create_conv2d_config(
                conv_args.c3,
                parameters.c3.weight,
                parameters.c3.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv4 = TtConv2d(
            create_conv2d_config(
                conv_args.c4,
                parameters.c4.weight,
                parameters.c4.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv5 = TtConv2d(
            create_conv2d_config(
                conv_args.c5,
                parameters.c5.weight,
                parameters.c5.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv6 = TtConv2d(
            create_conv2d_config(
                conv_args.c6,
                parameters.c6.weight,
                parameters.c6.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7 = TtConv2d(
            create_conv2d_config(
                conv_args.c7,
                parameters.c7.weight,
                parameters.c7.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7_2 = TtConv2d(
            create_conv2d_config(
                conv_args.c7_2,
                parameters.c7_2.weight,
                parameters.c7_2.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7_3 = TtConv2d(
            create_conv2d_config(
                conv_args.c7_3,
                parameters.c7_3.weight,
                parameters.c7_3.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv8 = TtConv2d(
            create_conv2d_config(
                conv_args.c8,
                parameters.c8.weight,
                parameters.c8.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7_4 = TtConv2d(
            create_conv2d_config(
                conv_args.c7_4,
                parameters.c7_4.weight,
                parameters.c7_4.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv8_2 = TtConv2d(
            create_conv2d_config(
                conv_args.c8_2,
                parameters.c8_2.weight,
                parameters.c8_2.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7_5 = TtConv2d(
            create_conv2d_config(
                conv_args.c7_5,
                parameters.c7_5.weight,
                parameters.c7_5.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )

        self.conv9 = TtConv2d(
            create_conv2d_config(
                conv_args.c9,
                parameters.c9.weight,
                parameters.c9.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv9_2 = TtConv2d(
            create_conv2d_config(
                conv_args.c9_2,
                parameters.c9_2.weight,
                parameters.c9_2.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv9_3 = TtConv2d(
            create_conv2d_config(
                conv_args.c9_3,
                parameters.c9_3.weight,
                parameters.c9_3.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv10 = TtConv2d(
            create_conv2d_config(
                conv_args.c10,
                parameters.c10.weight,
                parameters.c10.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )

        self.conv9_4 = TtConv2d(
            create_conv2d_config(
                conv_args.c9_4,
                parameters.c9_4.weight,
                parameters.c9_4.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv10_2 = TtConv2d(
            create_conv2d_config(
                conv_args.c10_2,
                parameters.c10_2.weight,
                parameters.c10_2.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv9_5 = TtConv2d(
            create_conv2d_config(
                conv_args.c9_5,
                parameters.c9_5.weight,
                parameters.c9_5.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor[0])

        output_tensor = self.conv2(output_tensor)

        output_tensor = self.conv3(output_tensor)

        output_tensor_pool_in = output_tensor
        if output_tensor_pool_in.memory_config().is_sharded():
            output_tensor_pool_in = ttnn.sharded_to_interleaved(output_tensor_pool_in, ttnn.L1_MEMORY_CONFIG)
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
            applied_shard_scheme=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        if pool_1.memory_config().is_sharded():
            pool_1 = ttnn.sharded_to_interleaved(pool_1, ttnn.L1_MEMORY_CONFIG)
        if pool_2.memory_config().is_sharded():
            pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
        if pool_3.memory_config().is_sharded():
            pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)

        pool_all = ttnn.concat([pool_3, pool_2, pool_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        pool_all = ttnn.to_layout(pool_all, layout=ttnn.TILE_LAYOUT)  # This is becauase output_tensor is in TILE_LAYOUT
        if self.parameters.resolution[0] == 320:
            pool_all = ttnn.add(pool_all, 0.0, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(pool_3)
        ttnn.deallocate(pool_2)
        ttnn.deallocate(pool_1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([pool_all, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv4(output_tensor)

        output_tensor = self.conv5(output_tensor)

        output_tensor_left_1 = self.conv6(output_tensor)

        output_tensor = self.conv7(output_tensor_left_1)

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
        if self.parameters.resolution[0] == 320:
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
            nhw = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2]
            num_cores = determine_num_cores_for_upsample(nhw, output_tensor.shape[2])
            core_grid = get_core_grid_from_num_cores(num_cores)
            shardspec = ttnn.create_sharded_memory_config_(
                output_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
            )

            if is_blackhole():
                output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
            else:
                output_tensor = ttnn.to_memory_config(output_tensor, shardspec)

            output_tensor_upsample_1 = ttnn.upsample(
                output_tensor, scale_factor=2, memory_config=output_tensor.memory_config()
            )
            output_tensor_upsample_1 = ttnn.sharded_to_interleaved(output_tensor_upsample_1, ttnn.L1_MEMORY_CONFIG)

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

        output_tensor = self.conv7_2(outDowSample5)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if self.parameters.resolution[0] == 320:
            output_tensor_upsample_1 = ttnn.add(output_tensor_upsample_1, 0.0, dtype=ttnn.bfloat8_b)
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_1)

        output_tensor = self.conv7_3(output_tensor)

        output_tensor = self.conv8(output_tensor)

        output_tensor = self.conv7_4(output_tensor)

        output_tensor = self.conv8_2(output_tensor)

        output_tensor_left_2 = self.conv7_5(output_tensor)

        if self.parameters.resolution[0] == 320:
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

        output_tensor = self.conv9(output_tensor_left_2)

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

        if self.parameters.resolution[0] == 320:
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
            nhw = output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2]
            num_cores = determine_num_cores_for_upsample(nhw, output_tensor.shape[2])
            core_grid = get_core_grid_from_num_cores(num_cores)
            shardspec = ttnn.create_sharded_memory_config_(
                output_tensor.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
            )

            if is_blackhole():
                output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
            else:
                output_tensor = ttnn.to_memory_config(output_tensor, shardspec)

            output_tensor_upsample_2 = ttnn.upsample(
                output_tensor, scale_factor=2, memory_config=output_tensor.memory_config()
            )
            output_tensor_upsample_2 = ttnn.sharded_to_interleaved(output_tensor_upsample_2, ttnn.L1_MEMORY_CONFIG)

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

        output_tensor = self.conv9_2(outDowSample3)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if self.parameters.resolution[0] == 320:
            output_tensor_upsample_2 = ttnn.add(output_tensor_upsample_2, 0.0, dtype=ttnn.bfloat8_b)
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_2)

        output_tensor = self.conv9_3(output_tensor)

        output_tensor = self.conv10(output_tensor)

        output_tensor = self.conv9_4(output_tensor)

        output_tensor = self.conv10_2(output_tensor)

        output_tensor = self.conv9_5(output_tensor)

        ttnn.deallocate(input_tensor[0])
        ttnn.deallocate(input_tensor[1])
        ttnn.deallocate(input_tensor[2])

        return output_tensor, output_tensor_left_1, output_tensor_left_2
