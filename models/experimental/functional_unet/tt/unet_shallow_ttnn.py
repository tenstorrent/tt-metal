# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn

import tt_lib.fallback_ops
from tt_lib import profiler

from loguru import logger


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
        ttl_tensor = ttnn.experimental.tensor.sharded_to_interleaved(ttl_tensor, interleaved_memory_config, dtype)
        ttl_tensor = ttnn.experimental.tensor.interleaved_to_sharded(
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


# Unet shallow ttnn implementation
class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.p1 = parameters.p1
        self.c2 = parameters.c2
        self.c2_2 = parameters.c2_2
        self.p2 = parameters.p2
        self.c3 = parameters.c3
        self.c3_2 = parameters.c3_2
        self.p3 = parameters.p3
        self.c4 = parameters.c4
        self.c4_2 = parameters.c4_2
        self.p4 = parameters.p4
        self.bnc = parameters.bnc
        self.bnc_2 = parameters.bnc_2
        self.c5 = parameters.c5
        self.c5_2 = parameters.c5_2
        self.c5_3 = parameters.c5_3
        self.c6 = parameters.c6
        self.c6_2 = parameters.c6_2
        self.c6_3 = parameters.c6_3
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c8_3 = parameters.c8_3
        self.output_layer = parameters.output_layer

    def __call__(self, device, input_tensor, original_shape, perf_mode=False):
        nhw = original_shape[-4] * original_shape[-2] * original_shape[-1]
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        profiler.tracy_message("c1")
        output_tensor = self.c1(input_tensor)
        output_tensor = self.c1_2(output_tensor)
        if perf_mode:
            save_c1_2_out = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        else:
            save_c1_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p1(output_tensor)

        profiler.tracy_message("c2")
        output_tensor = unet_reshard(output_tensor, self.c2.conv.input_sharded_memory_config, use_reshard=False)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2_2(output_tensor)
        if perf_mode:
            save_c2_2_out = output_tensor
        else:
            save_c2_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p2(output_tensor)

        profiler.tracy_message("c3")
        output_tensor = unet_reshard(output_tensor, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c3_2(output_tensor)
        if perf_mode:
            save_c3_2_out = output_tensor
        else:
            save_c3_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p3(output_tensor)

        profiler.tracy_message("c4")
        output_tensor = unet_reshard(output_tensor, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(output_tensor)
        output_tensor = self.c4_2(output_tensor)
        if perf_mode:
            save_c4_2_out = output_tensor
        else:
            save_c4_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p4(output_tensor)

        profiler.tracy_message("bnc")
        output_tensor = unet_reshard(output_tensor, self.bnc.conv.input_sharded_memory_config)
        output_tensor = self.bnc(output_tensor)
        output_tensor = self.bnc_2(output_tensor)

        if not perf_mode:
            # need to convert into interleaved, then back into sharded due to pcc issues
            # for certain tensor shape sizes, you get pcc issues when trying to convert between data layouts
            output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        ## upsample block
        profiler.tracy_message("upsample1")
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1))
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 64, output_tensor.shape[-1]))

        profiler.tracy_message("concat1")
        save_c4_2_out = ttnn.to_layout(save_c4_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c4_2_out], dim=-1, perf_mode=perf_mode)

        profiler.tracy_message("c5")
        output_tensor = unet_reshard(output_tensor, self.c5.conv.input_sharded_memory_config)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.c5_2(output_tensor)
        output_tensor = self.c5_3(output_tensor)

        if not perf_mode:
            # need to convert into interleaved, then back into sharded due to pcc issues
            # for certain tensor shape sizes, you get pcc issues when trying to convert between data layouts
            output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        profiler.tracy_message("upsample2")
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1))
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 16, output_tensor.shape[-1]))

        profiler.tracy_message("concat2")
        save_c3_2_out = ttnn.to_layout(save_c3_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c3_2_out], dim=-1, perf_mode=perf_mode)

        profiler.tracy_message("c6")
        output_tensor = unet_reshard(output_tensor, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.c6_2(output_tensor)
        output_tensor = self.c6_3(output_tensor)

        if not perf_mode:
            # need to convert into interleaved, then back into sharded due to pcc issues
            # for certain tensor shape sizes, you get pcc issues when trying to convert between data layouts
            output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        profiler.tracy_message("upsample3")
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1))
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw // 4, output_tensor.shape[-1]))

        profiler.tracy_message("concat3")
        save_c2_2_out = ttnn.to_layout(save_c2_2_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = unet_concat([output_tensor, save_c2_2_out], dim=-1, perf_mode=perf_mode)

        profiler.tracy_message("c7")
        hacked_shard_shape = self.c7.conv.input_sharded_memory_config.shard_spec.shape
        hacked_shard_shape[1] = output_tensor.shape[-1]
        self.c7.conv.input_sharded_memory_config.shard_spec.shape = hacked_shard_shape
        output_tensor = unet_reshard(output_tensor, self.c7.conv.input_sharded_memory_config)
        output_tensor = self.c7(output_tensor)
        output_tensor = self.c7_2(output_tensor)
        output_tensor = self.c7_3(output_tensor)

        if not perf_mode:
            # need to convert into interleaved, then back into sharded due to pcc issues
            # for certain tensor shape sizes, you get pcc issues when trying to convert between data layouts
            output_tensor = ttnn.experimental.tensor.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        profiler.tracy_message("upsample4")
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1))
        output_tensor = ttnn.reshape(output_tensor, (1, 1, nhw, output_tensor.shape[-1]))

        profiler.tracy_message("concat4")
        output_tensor = unet_concat([output_tensor, save_c1_2_out], dim=-1, perf_mode=perf_mode)

        profiler.tracy_message("c8")
        output_tensor = unet_reshard(output_tensor, self.c8.conv.input_sharded_memory_config)
        output_tensor = self.c8(output_tensor)
        output_tensor = self.c8_2(output_tensor)
        output_tensor = self.c8_3(output_tensor)
        output_tensor = self.output_layer(output_tensor)
        return ttnn.from_device(output_tensor)
