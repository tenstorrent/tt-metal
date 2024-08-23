# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from loguru import logger
from typing import Optional, Tuple
from tt_lib import profiler

from ttnn.operations.conv2d import determine_parallel_config, create_sharded_memory_config_from_parallel_config

from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, ParameterDict


# Unet reshard wrapper
def unet_reshard(
    ttnn_tensor,
    sharded_memory_config,
    use_reshard=True,
    interleaved_memory_config=ttnn.L1_MEMORY_CONFIG,
    dtype=None,
):
    if use_reshard:
        return ttnn.to_memory_config(
            ttnn_tensor,
            memory_config=sharded_memory_config,
        )
    else:
        ttl_tensor = ttnn_tensor
        ttl_tensor = ttnn.sharded_to_interleaved(ttl_tensor, interleaved_memory_config, dtype)
        ttl_tensor = ttnn.interleaved_to_sharded(
            ttl_tensor,
            sharded_memory_config,
            dtype,
        )
        return ttl_tensor


# Unet concat tensor wrapper
def unet_concat(ttnn_tensors, dim=-1, use_reshard=True, perf_mode=False):
    if not perf_mode:
        return ttnn.concat(ttnn_tensors, dim=3)

    assert len(ttnn_tensors) > 0
    assert dim < 0
    rank = len(ttnn_tensors[0].shape)
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
    def __init__(self, conv, bn=None, device=None, cache={}, activation="relu"):
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

        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat8_b,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            height_sharding=self.use_1d_systolic_array,
            deallocate_activation=self.deallocate_activation,
            fp32_dest_acc_enabled=True,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            activation=activation,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn:
            weight, bias = fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias

        weight = weight
        bias = torch.reshape(bias, (1, 1, 1, -1))

        if bias.shape[-1] == 1:
            bias = bias.repeat((1, 1, 32, 32))
            logger.warning(f"Found a bias we need to replicate {bias.shape}")

        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32)

    def __call__(self, x):
        logger.info(
            f"running conv - input={x.shape} in_h={self.input_height}, in_w={self.input_width}, in_c={self.in_channels}, out=c{self.out_channels}, pad={self.padding}, weight={self.weight.shape}"
        )
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
        logger.info(f"done conv out shape: {output_height} {output_width}")
        return x


class UNetMaxPool2D:
    def __init__(self, pool, device=None, reader_patterns_cache={}):
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
        )

    def __call__(self, x):
        logger.info(f"running max_pool - input={x.shape}, in_h={self.pool.input_height}, in_w={self.pool.input_width}")
        return self.max_pool(x)


class UNetMaxPool2DNew:
    def __init__(self, pool, channels, device=None, reader_patterns_cache={}):
        self.channels = channels
        self.kernel_size = pool.kernel_size
        self.stride = pool.stride
        self.padding = pool.padding
        self.dilation = pool.dilation
        self.batch_size = pool.batch_size
        self.input_height = pool.input_height
        self.input_width = pool.input_width
        self.device = device

    def __call__(self, x):
        return ttnn.max_pool2d_new(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=self.input_height,
            input_w=self.input_width,
            channels=self.channels,
            kernel_size=[self.kernel_size, self.kernel_size],
            stride=[self.stride, self.stride],
            padding=[self.padding, self.padding],
            dilation=[self.dilation, self.dilation],
            device=self.device,
        )


class UNetDownblock:
    def __init__(self, conv1, bn1, conv2, bn2, pool, device, conv_cache={}, max_pool_cache={}, should_reshard=False):
        self.conv1 = UNetConv2D(conv1, bn=bn1, device=device, cache=conv_cache)
        self.conv2 = UNetConv2D(conv2, bn=bn2, device=device, cache=conv_cache)
        self.pool1 = UNetMaxPool2D(pool, device=device, reader_patterns_cache=max_pool_cache)

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
                    self.conv1.in_channels,
                ],
                parallel_config=parallel_config,
                tile_size=32 if conv1.dtype == ttnn.bfloat8_b else 1,
            )
            logger.info(f"Created shardspec: {parallel_config}, {self.sharded_memory_config}")

    def __call__(self, x, perf_mode=False):
        if self.should_reshard:
            x = unet_reshard(x, self.sharded_memory_config, use_reshard=False)
        x = self.conv1(x)
        x = self.conv2(x)
        residual = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        x = self.pool1(x)
        return x, residual


class UNetUpblock:
    def __init__(self, conv1, bn1, conv2, bn2, conv3, bn3, device, conv_cache={}, should_reshard=False):
        self.device = device
        self.conv1 = UNetConv2D(conv1, bn1, device, conv_cache)
        self.conv2 = UNetConv2D(conv2, bn2, device, conv_cache)
        self.conv3 = UNetConv2D(conv3, bn3, device, conv_cache)

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
                    self.conv1.in_channels,
                ],
                parallel_config=parallel_config,
                tile_size=32 if conv1.dtype == ttnn.bfloat8_b else 1,
            )
            logger.info(f"Created shardspec: {parallel_config}, {self.sharded_memory_config}")

    def __call__(self, x, residual, factor, perf_mode=False, use_reshard=False):
        logger.info("to layout")
        if not perf_mode:
            # need to convert into interleaved, then back into sharded due to pcc issues
            # for certain tensor shape sizes, you get pcc issues when trying to convert between data layouts
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)

        logger.info("upsample op and reshape")
        x = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.upsample(x, (2, 2, 1))
        x = ttnn.reshape(x, (1, 1, factor, x.shape[-1]))

        logger.info("concat")
        x = unet_concat([x, residual], dim=-1, perf_mode=perf_mode)
        ttnn.deallocate(residual)

        if self.should_reshard and use_reshard:
            # x = unet_reshard(x, self.sharded_memory_config, use_reshard=use_reshard)
            x = ttnn.to_memory_config(x, self.sharded_memory_config)

        logger.info(f"conv1")
        x = self.conv1(x)
        logger.info(f"conv2")
        x = self.conv2(x)
        logger.info(f"conv3")
        x = self.conv3(x)

        return x


class UNet:
    def __init__(self, parameters: ParameterDict, device) -> None:
        self.conv_cache = {}
        self.max_pool_cache = {}
        self.downblock1 = UNetDownblock(
            parameters.c1,
            parameters.b1,
            parameters.c1_2,
            parameters.b1_2,
            parameters.p1,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=False,
        )
        self.downblock2 = UNetDownblock(
            parameters.c2,
            parameters.b2,
            parameters.c2_2,
            parameters.b2_2,
            parameters.p2,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
        )
        self.downblock3 = UNetDownblock(
            parameters.c3,
            parameters.b3,
            parameters.c3_2,
            parameters.b3_2,
            parameters.p3,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
        )
        self.downblock4 = UNetDownblock(
            parameters.c4,
            parameters.b4,
            parameters.c4_2,
            parameters.b4_2,
            parameters.p4,
            device,
            conv_cache=self.conv_cache,
            max_pool_cache=self.max_pool_cache,
            should_reshard=True,
        )

        self.bnc = UNetConv2D(parameters.bnc, parameters.bnb, device, cache=self.conv_cache)
        self.bnc2 = UNetConv2D(parameters.bnc_2, parameters.bnb_2, device, cache=self.conv_cache)

        self.upsample1 = UNetUpblock(
            parameters.c5,
            parameters.b5,
            parameters.c5_2,
            parameters.b5_2,
            parameters.c5_3,
            parameters.b5_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=False,
        )
        self.upsample2 = UNetUpblock(
            parameters.c6,
            parameters.b6,
            parameters.c6_2,
            parameters.b6_2,
            parameters.c6_3,
            parameters.b6_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
        )
        self.upsample3 = UNetUpblock(
            parameters.c7,
            parameters.b7,
            parameters.c7_2,
            parameters.b7_2,
            parameters.c7_3,
            parameters.b7_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
        )
        self.upsample4 = UNetUpblock(
            parameters.c8,
            parameters.b8,
            parameters.c8_2,
            parameters.b8_2,
            parameters.c8_3,
            parameters.b8_3,
            device,
            conv_cache=self.conv_cache,
            should_reshard=True,
        )

        self.output_layer = UNetConv2D(
            parameters.output_layer, bn=None, device=device, cache=self.conv_cache, activation=""
        )

        self.cache = {}

    def __call__(self, device, input_tensor, original_shape, perf_mode=False):
        nhw = original_shape[-4] * original_shape[-2] * original_shape[-1]

        input_tensor = input_tensor.to(device, ttnn.L1_MEMORY_CONFIG)

        logger.info(f"C1 {input_tensor.shape}")
        x, c1_residual = self.downblock1(input_tensor, perf_mode=perf_mode)

        logger.info(f"C2 {x.shape}")
        x, c2_residual = self.downblock2(x, perf_mode=perf_mode)

        logger.info(f"C3 {x.shape}")
        x, c3_residual = self.downblock3(x, perf_mode=perf_mode)

        logger.info("C4")
        x, c4_residual = self.downblock4(x, perf_mode=perf_mode)

        logger.info("bnc")
        # TODO: Need to reshard here
        x = self.bnc(x)
        x = self.bnc2(x)

        logger.info("upsample1")
        x = self.upsample1(x, c4_residual, nhw // 64, perf_mode=perf_mode)

        logger.info("upsample2")
        x = self.upsample2(x, c3_residual, nhw // 16, perf_mode=perf_mode)

        logger.info("upsample3")
        # TODO: Need to add 'padded_input_channels' in conv1
        x = self.upsample3(x, c2_residual, nhw // 4, perf_mode=perf_mode)

        logger.info(f"upsample4 {x.shape} {c1_residual.shape}")
        x = self.upsample4(x, c1_residual, nhw, perf_mode=perf_mode, use_reshard=True)

        logger.info("output_layer")
        x = x.cpu().pad_to_tile(0)
        x = self.output_layer(x)

        return ttnn.from_device(x)
