# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from loguru import logger

from ttnn.operations.conv2d import determine_parallel_config, create_sharded_memory_config_from_parallel_config

from models.utility_functions import nearest_32
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, ParameterDict


# Unet concat tensor wrapper
def unet_concat(ttnn_tensors, dim=-1, use_reshard=True, perf_mode=False):
    if not perf_mode:
        return ttnn.concat(ttnn_tensors, dim=3)

    assert len(ttnn_tensors) > 0
    assert dim < 0
    ttlib_tensors = ttnn_tensors
    all_sharded = all(t.is_sharded() for t in ttlib_tensors)
    if all_sharded:
        max_idx, output_mem_config = max(
            ((i, t.memory_config()) for i, t in enumerate(ttlib_tensors)), key=lambda m: m[1].shard_spec.num_cores()
        )
        for i in range(0, len(ttlib_tensors)):
            if i == max_idx:
                continue
            t = ttlib_tensors[i]
            t_mem_config = t.memory_config()
            t_shard_shape = t_mem_config.shard_spec.shape
            output_shard_shape = output_mem_config.shard_spec.shape
            output_shard_shape[dim] += t_shard_shape[dim]
            output_mem_config.shard_spec.shape = output_shard_shape

            reshard_shape = output_shard_shape
            reshard_shape[dim] = t_shard_shape[dim]
            if reshard_shape != t_shard_shape:
                t_mem_config.shard_spec.shape = reshard_shape
                t_mem_config.shard_spec.grid = output_mem_config.shard_spec.grid
                t_mem_config.shard_spec.orientation = output_mem_config.shard_spec.orientation
                ttlib_tensors[i] = ttnn.experimental.tensor.reshard(t, t_mem_config)
    else:
        output_mem_config = ttnn.DRAM_MEMORY_CONFIG
        for i in range(0, len(ttlib_tensors)):
            if ttlib_tensors[i].is_sharded():
                ttlib_tensors[i] = ttnn.to_memory_config(ttlib_tensors[i], output_mem_config)
    return ttnn.concat(ttlib_tensors, dim=dim, memory_config=output_mem_config)


class UNetConv2D:
    def __init__(
        self,
        batch,
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
        self.batch_size = batch
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

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            math_fidelity=ttnn.MathFidelity.LoFi,
            height_sharding=self.use_1d_systolic_array,
            deallocate_activation=self.deallocate_activation,
            fp32_dest_acc_enabled=True,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=True,
            enable_subblock_padding=False,
            activation=activation,
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

        # Required for pointwise convolutions (output layer)
        if bias.shape[-1] == 1:
            bias = bias.repeat((1, 1, 32, 32))

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    def __call__(self, x):
        x, output_height, output_width, self.weight, self.bias = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
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
    def __init__(
        self,
        pool,
        device=None,
        reader_patterns_cache={},
        mesh_mapper=None,
    ):
        self.pool = pool
        self.max_pool = ttnn.MaxPool2d(
            kernel_size=pool.kernel_size,
            stride=pool.stride,
            padding=pool.padding,
            dilation=pool.dilation,
            dtype=pool.dtype,
            batch_size=pool.batch_size,
            input_height=pool.input_height,
            input_width=pool.input_width,
            reader_patterns_cache=reader_patterns_cache,
            parallel_config_override=pool.parallel_config_override,
            deallocate_activation=True,
            device=device,
            mesh_mapper=mesh_mapper,
        )

    def __call__(self, x):
        # For some reason the shard widths don't always match - so don't assert on it
        assert (
            x.memory_config().shard_spec.num_cores()
            == self.max_pool.max_pool.input_sharded_memory_config.shard_spec.num_cores()
            and x.memory_config().shard_spec.shape[0]
            == self.max_pool.max_pool.input_sharded_memory_config.shard_spec.shape[0]
        ), "Expected same input shard to match max pool's shard configuration"
        return self.max_pool(x)


class UNetDownblock:
    def __init__(
        self,
        batch,
        conv1,
        bn1,
        conv2,
        bn2,
        pool,
        device,
        conv_cache={},
        max_pool_cache={},
        should_reshard=False,
        mesh_mapper=None,
    ):
        self.conv1 = UNetConv2D(batch, conv1, bn=bn1, device=device, cache=conv_cache, mesh_mapper=mesh_mapper)
        self.conv2 = UNetConv2D(batch, conv2, bn=bn2, device=device, cache=conv_cache, mesh_mapper=mesh_mapper)
        self.pool1 = UNetMaxPool2D(pool, device=device, reader_patterns_cache=max_pool_cache, mesh_mapper=mesh_mapper)
        self.should_reshard = should_reshard
        if self.should_reshard:
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

    def __call__(self, x):
        assert list(x.shape) == [
            1,
            1,
            nearest_32(self.conv1.input_height * self.conv1.input_width * self.conv1.batch_size),
            x.shape[-1],  # Channels can be padded
        ], f"Expected downblock input to flattened into [1, 1, BHW, C] but was {list(x.shape)}"
        if self.should_reshard:
            sharded_memory_config = create_sharded_memory_config_from_parallel_config(
                tensor_shape=[
                    1,
                    1,
                    x.shape[0] * x.shape[1] * x.shape[2],
                    x.shape[3],
                ],
                parallel_config=self.parallel_config,
                tile_size=32 if x.dtype == ttnn.bfloat8_b else 1,
            )
            x = ttnn.to_memory_config(
                x,
                memory_config=sharded_memory_config,
            )
        print("1")
        x = self.conv1(x)
        print("2")
        x = self.conv2(x)
        print("residual")
        residual = ttnn.sharded_to_interleaved(
            x, memory_config=ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16
        )  # pool deletes its activation - spill to DRAM
        print("pool")
        x = self.pool1(x)
        return x, residual


class UNetUpblock:
    def __init__(
        self, batch, conv1, bn1, conv2, bn2, conv3, bn3, device, conv_cache={}, should_reshard=False, mesh_mapper=None
    ):
        self.device = device
        self.conv1 = UNetConv2D(batch, conv1, bn1, device, conv_cache, mesh_mapper=mesh_mapper)
        self.conv2 = UNetConv2D(batch, conv2, bn2, device, conv_cache, mesh_mapper=mesh_mapper)
        self.conv3 = UNetConv2D(batch, conv3, bn3, device, conv_cache, mesh_mapper=mesh_mapper)

        self.should_reshard = should_reshard
        if self.should_reshard:
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
        x = ttnn.upsample(x, (2, 2, 1))
        x = ttnn.reshape(
            x, (1, 1, self.conv1.batch_size * self.conv1.input_height * self.conv1.input_width, x.shape[-1])
        )
        return x

    def __call__(self, x, residual):
        assert list(x.shape)[:2] == [
            1,
            1,
        ], f"Expected downblock input to flattened into [1, 1, BHW, C] but was {list(x.shape)}"

        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = self.upsample(x)

        residual = ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
        x = unet_concat([x, residual], dim=-1, perf_mode=True)
        ttnn.deallocate(residual)

        if self.should_reshard:
            x = ttnn.to_memory_config(x, self.sharded_memory_config)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        return x


class UNet:
    def __init__(
        self,
        parameters: ParameterDict,
        batch,
        device,
        weights_mesh_mapper=None,
    ) -> None:
        self.device = device
        self.conv_cache = {}
        self.max_pool_cache = {}
        self.downblock1 = UNetDownblock(
            batch,
            parameters.c1,
            parameters.b1,
            parameters.c1_2,
            parameters.b1_2,
            parameters.p1,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )
        self.downblock2 = UNetDownblock(
            batch,
            parameters.c2,
            parameters.b2,
            parameters.c2_2,
            parameters.b2_2,
            parameters.p2,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )
        self.downblock3 = UNetDownblock(
            batch,
            parameters.c3,
            parameters.b3,
            parameters.c3_2,
            parameters.b3_2,
            parameters.p3,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )
        self.downblock4 = UNetDownblock(
            batch,
            parameters.c4,
            parameters.b4,
            parameters.c4_2,
            parameters.b4_2,
            parameters.p4,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )

        self.bnc = UNetConv2D(
            batch,
            parameters.bnc,
            parameters.bnb,
            device,
            cache=self.conv_cache,
            mesh_mapper=weights_mesh_mapper,
        )
        self.bnc2 = UNetConv2D(
            batch,
            parameters.bnc_2,
            parameters.bnb_2,
            device,
            cache=self.conv_cache,
            mesh_mapper=weights_mesh_mapper,
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
            batch,
            parameters.c5,
            parameters.b5,
            parameters.c5_2,
            parameters.b5_2,
            parameters.c5_3,
            parameters.b5_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=False,
            mesh_mapper=weights_mesh_mapper,
        )
        self.upblock2 = UNetUpblock(
            batch,
            parameters.c6,
            parameters.b6,
            parameters.c6_2,
            parameters.b6_2,
            parameters.c6_3,
            parameters.b6_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )
        self.upblock3 = UNetUpblock(
            batch,
            parameters.c7,
            parameters.b7,
            parameters.c7_2,
            parameters.b7_2,
            parameters.c7_3,
            parameters.b7_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )
        self.upblock4 = UNetUpblock(
            batch,
            parameters.c8,
            parameters.b8,
            parameters.c8_2,
            parameters.b8_2,
            parameters.c8_3,
            parameters.b8_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
            mesh_mapper=weights_mesh_mapper,
        )

        self.output_layer = UNetConv2D(
            batch,
            parameters.output_layer,
            bn=None,
            device=device,
            cache=self.conv_cache,
            activation="",
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            mesh_mapper=weights_mesh_mapper,
        )

    def bottleneck(self, x):
        x = ttnn.to_memory_config(
            x,
            memory_config=self.bnc_sharded_memory_config,
        )
        x = self.bnc(x)
        return self.bnc2(x)

    def __call__(self, x, original_shape, perf_mode=False):
        assert len(x.shape) == 4, f"Expected UNet input tensors to be rank 4 (was {len(x.shape)})"

        x = x.to(self.device)
        print("downblock1")
        x, c1_residual = self.downblock1(x)
        print("downblock2")
        x, c2_residual = self.downblock2(x)
        print("downblock3")
        x, c3_residual = self.downblock3(x)
        print("downblock4")
        x, c4_residual = self.downblock4(x)
        print("bottleneck")
        x = self.bottleneck(x)

        print("upblock1")
        x = self.upblock1(x, c4_residual)
        print("upblock2")
        x = self.upblock2(x, c3_residual)
        print("upblock3")
        x = self.upblock3(x, c2_residual)
        print("upblock4")
        x = self.upblock4(x, c1_residual)

        # Pointwise convolutions currently don't handle padded inputs
        print("pad")
        x = x.cpu().pad_to_tile(0)
        print("output_layer")
        x = self.output_layer(x)

        x = ttnn.from_device(x)

        return x
