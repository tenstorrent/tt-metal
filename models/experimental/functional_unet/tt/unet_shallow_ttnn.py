# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch

from typing import List

from models.utility_functions import nearest_32
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, ParameterDict


def nearest_16(x):
    return math.ceil(x / 16) * 16


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


def concatenate(inputs: List, dim=-1, groups=2):
    assert len(inputs) > 0
    assert dim < 0
    assert all(tensor.is_sharded() for tensor in inputs), "All inputs to `ttnn.concat` must be sharded"
    max_idx, output_memory_config = max(
        ((i, t.memory_config()) for i, t in enumerate(inputs)), key=lambda m: m[1].shard_spec.num_cores()
    )
    for i in range(0, len(inputs)):
        if i == max_idx:
            continue
        tensor = inputs[i]
        memory_config = tensor.memory_config()
        shard_shape = memory_config.shard_spec.shape
        output_shard_shape = output_memory_config.shard_spec.shape
        output_shard_shape[dim] += shard_shape[dim]
        output_memory_config.shard_spec.shape = output_shard_shape

        reshard_shape = output_shard_shape
        reshard_shape[dim] = shard_shape[dim]
        if reshard_shape != shard_shape:
            memory_config.shard_spec.shape = reshard_shape
            memory_config.shard_spec.grid = output_memory_config.shard_spec.grid
            memory_config.shard_spec.orientation = output_memory_config.shard_spec.orientation
            inputs[i] = ttnn.reshard(tensor, memory_config)
    return ttnn.concat(inputs, dim=dim, memory_config=output_memory_config, groups=groups)


class UNetConv2D:
    def __init__(
        self,
        conv,
        bn=None,
        device=None,
        cache={},
        activation="relu",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
        reshard_if_not_optimal=False,
        mesh_mapper=None,
    ):
        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = conv.use_1d_systolic_array
        self.deallocate_activation = True
        self.cache = cache
        self.mesh_mapper = mesh_mapper

        shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            if self.use_1d_systolic_array
            else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=(
                conv.use_activation_double_buffer if "use_activation_double_buffer" in conv else False
            ),
            enable_split_reader=conv.use_split_reader if "use_split_reader" in conv else False,
            enable_subblock_padding=False,
            activation=activation,
            output_layout=output_layout,
            input_channels_alignment=conv.input_channels_alignment if "input_channels_alignment" in conv else 32,
            reshard_if_not_optimal=reshard_if_not_optimal,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias

        bias = torch.reshape(bias, (1, 1, 1, -1))

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    def __call__(self, x):
        conv_kwargs = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "batch_size": self.batch_size,
            "input_height": self.input_height,
            "input_width": self.input_width,
            "kernel_size": self.kernel_size,
            "stride": self.stride,
            "padding": self.padding,
            "dilation": [1, 1],
            "groups": 2,
            "device": self.device,
            "conv_config": self.conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.weight):
            self.weight = ttnn.prepare_conv_weights(
                weight_tensor=self.weight,
                weights_format="OIHW",
                input_memory_config=x.memory_config(),
                input_layout=x.get_layout(),
                **conv_kwargs,
            )
            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self.bias,
                input_memory_config=x.memory_config(),
                input_layout=x.get_layout(),
                **conv_kwargs,
            )
            self.weight = ttnn.to_device(self.weight, self.device)
            self.bias = ttnn.to_device(self.bias, self.device) if self.bias else None

        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **conv_kwargs,
            compute_config=self.compute_config,
            conv_op_cache=self.cache,
        )
        return x


class UNetMaxPool2D:
    def __init__(self, pool, channels, device=None):
        self.pool = pool
        self.channels = channels
        self.device = device

    def __call__(self, x):
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.pool.batch_size,
            input_h=self.pool.input_height,
            input_w=self.pool.input_width,
            channels=self.channels,
            kernel_size=[self.pool.kernel_size, self.pool.kernel_size],
            stride=[self.pool.stride, self.pool.stride],
            padding=[self.pool.padding, self.pool.padding],
            dilation=[self.pool.dilation, self.pool.dilation],
        )
        return x


class UNetDownblock:
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        pool,
        device,
        conv_cache={},
        mesh_mapper=None,
    ):
        self.conv1 = UNetConv2D(
            conv1, bn=bn1, device=device, cache=conv_cache, reshard_if_not_optimal=True, mesh_mapper=mesh_mapper
        )
        self.conv2 = UNetConv2D(
            conv2,
            bn=bn2,
            device=device,
            cache=conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.pool1 = UNetMaxPool2D(pool, conv2.out_channels, device=device)

    def __call__(self, x):
        assert list(x.shape) == [
            1,
            1,
            self.conv1.input_height * self.conv1.input_width * self.conv1.batch_size,
            x.shape[-1],  # Channels can be padded
        ], f"Expected downblock input to flattened into [1, 1, BHW, C] but was {list(x.shape)}"
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x = self.pool1(x)
        return x, residual


class UNetUpblock:
    def __init__(
        self,
        conv1,
        bn1,
        conv2,
        bn2,
        conv3,
        bn3,
        device,
        conv_cache={},
        mesh_mapper=None,
    ):
        self.device = device
        self.conv1 = UNetConv2D(conv1, bn1, device, conv_cache, reshard_if_not_optimal=True, mesh_mapper=mesh_mapper)
        self.conv2 = UNetConv2D(conv2, bn2, device, conv_cache, mesh_mapper=mesh_mapper)
        self.conv3 = UNetConv2D(conv3, bn3, device, conv_cache, mesh_mapper=mesh_mapper)

    def upsample(self, x):
        # Need to reshape into (B, H, W, C) to get correct output from ttnn.upsample
        x = ttnn.reshape(
            x, (self.conv1.batch_size, self.conv1.input_height // 2, self.conv1.input_width // 2, x.shape[-1])
        )

        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)

        x = ttnn.upsample(x, (2, 2), memory_config=x.memory_config())
        x = ttnn.reshape(
            x, (1, 1, self.conv1.batch_size * self.conv1.input_height * self.conv1.input_width, x.shape[-1])
        )
        return x

    def __call__(self, x, residual):
        assert list(x.shape)[:2] == [
            1,
            1,
        ], f"Expected upblock input to flattened into [1, 1, BHW, C] but was {list(x.shape)}"

        residual = ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        x = self.upsample(x)

        if not residual.is_sharded():
            core_grid = get_core_grid_from_num_cores(x.memory_config().shard_spec.num_cores())
            memory_config = ttnn.create_sharded_memory_config_(
                residual.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
            )
            residual = ttnn.to_memory_config(residual, memory_config)

        y = concatenate([x, residual], dim=-1)
        ttnn.deallocate(x)
        ttnn.deallocate(residual)

        y = self.conv1(y)
        y = self.conv2(y)
        y = self.conv3(y)

        return y


class UNet:
    def __init__(self, parameters: ParameterDict, device, mesh_mapper=None) -> None:
        self.device = device
        self.conv_cache = {}
        self.downblock1 = UNetDownblock(
            parameters.c1,
            parameters.b1,
            parameters.c1_2,
            parameters.b1_2,
            parameters.p1,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.downblock2 = UNetDownblock(
            parameters.c2,
            parameters.b2,
            parameters.c2_2,
            parameters.b2_2,
            parameters.p2,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.downblock3 = UNetDownblock(
            parameters.c3,
            parameters.b3,
            parameters.c3_2,
            parameters.b3_2,
            parameters.p3,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.downblock4 = UNetDownblock(
            parameters.c4,
            parameters.b4,
            parameters.c4_2,
            parameters.b4_2,
            parameters.p4,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )

        self.bnc = UNetConv2D(
            parameters.bnc,
            parameters.bnb,
            device,
            cache=self.conv_cache,
            reshard_if_not_optimal=True,
            mesh_mapper=mesh_mapper,
        )
        self.bnc2 = UNetConv2D(
            parameters.bnc_2, parameters.bnb_2, device, cache=self.conv_cache, mesh_mapper=mesh_mapper
        )

        self.upblock1 = UNetUpblock(
            parameters.c5,
            parameters.b5,
            parameters.c5_2,
            parameters.b5_2,
            parameters.c5_3,
            parameters.b5_3,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.upblock2 = UNetUpblock(
            parameters.c6,
            parameters.b6,
            parameters.c6_2,
            parameters.b6_2,
            parameters.c6_3,
            parameters.b6_3,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.upblock3 = UNetUpblock(
            parameters.c7,
            parameters.b7,
            parameters.c7_2,
            parameters.b7_2,
            parameters.c7_3,
            parameters.b7_3,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.upblock4 = UNetUpblock(
            parameters.c8,
            parameters.b8,
            parameters.c8_2,
            parameters.b8_2,
            parameters.c8_3,
            parameters.b8_3,
            device,
            conv_cache=self.conv_cache,
            mesh_mapper=mesh_mapper,
        )

        self.output_layer = UNetConv2D(
            parameters.output_layer, device=device, cache=self.conv_cache, activation="", mesh_mapper=mesh_mapper
        )

        self.parallel_config = ttnn._ttnn.operations.conv.determine_parallel_config(
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            batch_size=self.downblock1.conv1.batch_size,
            input_channels=self.downblock1.conv1.in_channels,
            output_height=self.downblock1.conv2.input_height,
            output_width=self.downblock1.conv2.input_width,
            output_channels=self.downblock1.conv1.out_channels,
            compute_grid_size=device.compute_with_storage_grid_size(),
            block_shard_orientation=ttnn.ShardOrientation.ROW_MAJOR,
            is_out_tiled=True,
            enable_channels_padding=True,
        )
        self.input_sharded_memory_config = ttnn._ttnn.operations.conv.create_sharded_memory_config_from_parallel_config(
            tensor_shape=ttnn.Shape(
                [
                    1,
                    1,
                    self.downblock1.conv1.batch_size
                    * self.downblock1.conv1.input_height
                    * self.downblock1.conv1.input_width,
                    nearest_16(self.downblock1.conv1.in_channels),
                ]
            ),
            parallel_config=self.parallel_config,
            tile_size=32,
        )

    def bottleneck(self, x):
        x = self.bnc(x)
        return self.bnc2(x)

    def postprocess_output_tensor(self, x):
        # Convert the output tensor (in TILE layout) to RM to prevent transferring padding back to host.
        return ttnn.to_layout(
            ttnn.reshape(
                x, shape=ttnn.Shape([1, 1, x.shape[2], 16], [1, 1, x.shape[2], 32])
            ),  # At the moment we can only reduce the padding from 32 to 16 because reshape is broken.
            ttnn.ROW_MAJOR_LAYOUT,
        )

    def __call__(self, x, move_input_tensor_to_device=True):
        assert len(x.shape) == 4, f"Expected UNet input tensors to be rank 4 (was {len(x.shape)})"

        if move_input_tensor_to_device:
            x = ttnn.to_device(x, device=self.device, memory_config=self.input_sharded_memory_config)

        x, c1_residual = self.downblock1(x)
        x, c2_residual = self.downblock2(x)
        x, c3_residual = self.downblock3(x)
        x, c4_residual = self.downblock4(x)

        x = self.bottleneck(x)

        x = self.upblock1(x, c4_residual)
        ttnn.deallocate(c4_residual)
        x = self.upblock2(x, c3_residual)
        ttnn.deallocate(c3_residual)
        c2_residual = ttnn.to_memory_config(c2_residual, ttnn.L1_MEMORY_CONFIG)
        x = self.upblock3(x, c2_residual)
        ttnn.deallocate(c2_residual)
        x = self.upblock4(x, c1_residual)
        ttnn.deallocate(c1_residual)

        x = self.output_layer(x)

        return self.postprocess_output_tensor(x)
