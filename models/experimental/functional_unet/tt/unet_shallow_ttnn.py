# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn
import torch

from typing import List

from ttnn.operations.conv2d import determine_parallel_config, create_sharded_memory_config_from_parallel_config

from models.utility_functions import nearest_32
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, ParameterDict


def determine_num_cores_for_upsample(nhw: int, width: int, max_cores=64) -> int:
    gcd_nhw_width = math.gcd(nhw, width)
    cores = nhw // gcd_nhw_width
    if cores > max_cores:
        for divisor in range(max_cores, 0, -1):
            if nhw % divisor == 0 and (nhw // divisor) % width == 0:
                cores = divisor
                break
    return cores


# TODO: Make this valid over any num_cores
def get_core_grid_from_num_cores(num_cores: int):
    if num_cores == 44:
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(7, 4),
                ),
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 5),
                    ttnn.CoreCoord(3, 5),
                ),
            }
        )
    elif num_cores == 48:
        return ttnn.CoreGrid(x=8, y=6)
    raise RuntimeError(f"Could not get core grid given num_cores={num_cores}")


def unet_concat(inputs: List, dim=-1):
    assert len(inputs) > 0
    assert dim < 0
    all_sharded = all(tensor.is_sharded() for tensor in inputs)
    if all_sharded:
        max_idx, memory_config = max(
            ((i, t.memory_config()) for i, t in enumerate(inputs)), key=lambda m: m[1].shard_spec.num_cores()
        )
        for i in range(0, len(inputs)):
            if i == max_idx:
                continue
            t = inputs[i]
            t_mem_config = t.memory_config()
            t_shard_shape = t_mem_config.shard_spec.shape
            output_shard_shape = memory_config.shard_spec.shape
            output_shard_shape[dim] += t_shard_shape[dim]
            memory_config.shard_spec.shape = output_shard_shape

            reshard_shape = output_shard_shape
            reshard_shape[dim] = t_shard_shape[dim]
            if reshard_shape != t_shard_shape:
                t_mem_config.shard_spec.shape = reshard_shape
                t_mem_config.shard_spec.grid = memory_config.shard_spec.grid
                t_mem_config.shard_spec.orientation = memory_config.shard_spec.orientation
                inputs[i] = ttnn.reshard(t, t_mem_config)
    else:
        memory_config = ttnn.DRAM_MEMORY_CONFIG
        for i in range(0, len(inputs)):
            if inputs[i].is_sharded():
                inputs[i] = ttnn.to_memory_config(inputs[i], memory_config)
    return ttnn.concat(inputs, dim=dim, memory_config=memory_config)


class UNetPointwiseConv2D:
    def __init__(
        self,
        conv,
        device=None,
        activation_dtype=ttnn.bfloat16,
        mesh_mapper=None,
    ):
        self.device = device
        self.in_channels = conv.in_channels
        self.mesh_mapper = mesh_mapper
        self.activation_dtype = activation_dtype

        weight, bias = conv.module.weight, conv.module.bias

        assert conv.kernel_size == (1, 1)
        assert conv.stride == (1, 1)
        assert conv.padding == (0, 0)

        weight = weight.reshape(1, 1, self.in_channels, 1)
        bias = torch.reshape(bias, (1, 1, 1, -1))

        # Do this in two steps because tensors are padded differently in multi-device vs. single device
        self.weight = ttnn.from_torch(weight, device=None, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        self.bias = ttnn.from_torch(bias, device=None, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
        self.weight = ttnn.to_layout(self.weight, ttnn.TILE_LAYOUT).to(device)
        self.bias = ttnn.to_layout(self.bias, ttnn.TILE_LAYOUT).to(device)

    def __call__(self, x):
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.linear(
            x,
            self.weight,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            bias=self.bias,
            dtype=self.activation_dtype,
            core_grid=ttnn.CoreGrid(y=8, x=8),
        )
        return x


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
            math_fidelity=ttnn.MathFidelity.LoFi,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            fp32_dest_acc_enabled=True,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=(
                conv.use_activation_double_buffer if "use_activation_double_buffer" in conv else False
            ),
            enable_split_reader=conv.use_split_reader if "use_split_reader" in conv else False,
            enable_subblock_padding=False,
            activation=activation,
            output_layout=ttnn.TILE_LAYOUT,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn is not None:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias

        weight = weight
        bias = torch.reshape(bias, (1, 1, 1, -1))

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    def __call__(self, x):
        x, _, _, self.weight, self.bias = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            #bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
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
            device=self.device,
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
        should_reshard=False,
        mesh_mapper=None,
    ):
        self.conv1 = UNetConv2D(
            conv1,
            bn=bn1,
            device=device,
            cache=conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.conv2 = UNetConv2D(
            conv2,
            bn=bn2,
            device=device,
            cache=conv_cache,
            mesh_mapper=mesh_mapper,
        )
        self.pool1 = UNetMaxPool2D(pool, conv2.out_channels, device=device)

        self.should_reshard = should_reshard

        self.parallel_config = determine_parallel_config(
            is_1d_systolic=True,
            batch_size=self.conv1.batch_size,
            input_channels=self.conv1.in_channels,
            output_height=self.conv2.input_height,
            output_width=self.conv2.input_width,
            output_channels=self.conv1.out_channels,
            device=device,
            is_out_tiled=True,
        )
        self.sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            tensor_shape=[
                1,
                1,
                self.conv1.input_width * self.conv1.input_height * self.conv1.batch_size,
                nearest_32(self.conv1.in_channels),
            ],
            parallel_config=self.parallel_config,
            tile_size=32 if conv1.dtype == ttnn.bfloat8_b else 1,
        )

    def __call__(self, x):
        assert list(x.shape) == [
            1,
            1,
            nearest_32(self.conv1.input_height * self.conv1.input_width * self.conv1.batch_size),
            x.shape[-1],  # Channels can be padded
        ], f"Expected downblock input to flattened into [1, 1, BHW, C] but was {list(x.shape)}"
        if self.should_reshard:
            x = ttnn.to_memory_config(
                x,
                memory_config=self.sharded_memory_config,
            )
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x = self.pool1(x)
        return x, residual


class UNetUpblock:
    def __init__(
        self, conv1, bn1, conv2, bn2, conv3, bn3, device, conv_cache={}, should_reshard=False, mesh_mapper=None
    ):
        self.device = device
        self.conv1 = UNetConv2D(conv1, bn1, device, conv_cache, mesh_mapper=mesh_mapper)
        self.conv2 = UNetConv2D(conv2, bn2, device, conv_cache, mesh_mapper=mesh_mapper)
        self.conv3 = UNetConv2D(conv3, bn3, device, conv_cache, mesh_mapper=mesh_mapper)

        self.should_reshard = should_reshard

        parallel_config = determine_parallel_config(
            is_1d_systolic=True,
            batch_size=self.conv1.batch_size,
            input_channels=self.conv1.in_channels,
            output_height=self.conv2.input_height,
            output_width=self.conv2.input_width,
            output_channels=self.conv1.out_channels,
            device=device,
            is_out_tiled=True,
        )
        self.sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            tensor_shape=[
                1,
                1,
                self.conv1.input_width * self.conv1.input_height * self.conv1.batch_size,
                nearest_32(self.conv1.in_channels),
            ],
            parallel_config=parallel_config,
            tile_size=32 if conv1.dtype == ttnn.bfloat8_b else 1,
        )

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

        x = ttnn.upsample(x, (2, 2, 1), memory_config=x.memory_config())
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

        y = unet_concat([x, residual], dim=-1)
        ttnn.deallocate(x)
        ttnn.deallocate(residual)

        if self.should_reshard:
            y = ttnn.to_memory_config(y, self.sharded_memory_config)

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
            should_reshard=False,
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
            should_reshard=True,
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
            should_reshard=True,
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
            should_reshard=True,
            mesh_mapper=mesh_mapper,
        )

        self.bnc = UNetConv2D(parameters.bnc, parameters.bnb, device, cache=self.conv_cache, mesh_mapper=mesh_mapper)
        self.bnc2 = UNetConv2D(
            parameters.bnc_2, parameters.bnb_2, device, cache=self.conv_cache, mesh_mapper=mesh_mapper
        )
        bnc_parallel_config = determine_parallel_config(
            is_1d_systolic=True,
            batch_size=self.bnc.batch_size,
            input_channels=self.bnc.in_channels,
            output_height=self.bnc2.input_height,
            output_width=self.bnc2.input_width,
            output_channels=self.bnc.out_channels,
            device=device,
            is_out_tiled=True,
        )
        self.bnc_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            tensor_shape=[
                1,
                1,
                self.bnc.input_width * self.bnc.input_height * self.bnc.batch_size,
                self.bnc.in_channels,
            ],
            parallel_config=bnc_parallel_config,
            tile_size=(32 if self.bnc.conv_config.dtype == ttnn.bfloat8_b else 1),
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
            should_reshard=False,
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
            should_reshard=False,
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
            should_reshard=True,
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
            should_reshard=False,
            mesh_mapper=mesh_mapper,
        )

        self.output_layer = UNetPointwiseConv2D(
            parameters.output_layer,
            device=device,
            mesh_mapper=mesh_mapper,
        )

        input_parallel_config = determine_parallel_config(
            is_1d_systolic=True,
            batch_size=self.downblock1.conv1.batch_size,
            input_channels=self.downblock1.conv1.in_channels,
            output_height=self.downblock1.conv2.input_height,
            output_width=self.downblock1.conv2.input_width,
            output_channels=self.downblock1.conv1.out_channels,
            device=device,
            is_out_tiled=False,
        )
        self.input_sharded_memory_config = create_sharded_memory_config_from_parallel_config(
            tensor_shape=[
                1,
                1,
                self.downblock1.conv1.input_height
                * self.downblock1.conv1.input_width
                * self.downblock1.conv1.batch_size,
                nearest_32(self.downblock1.conv1.in_channels),  # Input is padded to 32
            ],
            parallel_config=input_parallel_config,
            tile_size=1,
        )

    def bottleneck(self, x):
        if x.is_sharded():
            x = ttnn.reshard(x, self.bnc_sharded_memory_config)
        else:
            x = ttnn.interleaved_to_sharded(
                x,
                self.bnc_sharded_memory_config,
            )
        x = self.bnc(x)
        return self.bnc2(x)

    def __call__(self, inputs):
        assert len(inputs.shape) == 4, f"Expected UNet input tensors to be rank 4 (was {len(inputs.shape)})"

        x = ttnn.to_device(inputs, self.device, self.downblock1.sharded_memory_config)

        x, c1_residual = self.downblock1(x)
        x, c2_residual = self.downblock2(x)
        x, c3_residual = self.downblock3(x)
        x, c4_residual = self.downblock4(x)

        x = self.bottleneck(x)

        x = self.upblock1(x, c4_residual)
        x = self.upblock2(x, c3_residual)
        x = self.upblock3(x, c2_residual)
        x = self.upblock4(x, c1_residual)

        x = self.output_layer(x)

        return x
