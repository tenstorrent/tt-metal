# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn

import tt_lib.fallback_ops
from tt_lib import profiler
from models.experimental.functional_unet_n300.tt.common import Conv
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
        device,
        parameters,
        batch,
        model,
        mesh_mapper=None,
    ) -> None:
        self.c1 = Conv(
            [batch, 1056, 160, 4],
            (1, 1, 1, 1),
            model.c1.weight,
            model.b1.bias,
            bn_weights=model.b1.weight,
            bn_running_var=model.b1.running_var,
            bn_running_mean=model.b1.running_mean,
            act_block_h=5 * 32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c1_2 = Conv(
            [batch, 1056, 160, 16],
            (1, 1, 1, 1),
            model.c1_2.weight,
            model.b1_2.bias,
            bn_weights=model.b1_2.weight,
            bn_running_var=model.b1_2.running_var,
            bn_running_mean=model.b1_2.running_mean,
            act_block_h=5 * 32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c2 = Conv(
            [batch, 528, 80, 16],
            (1, 1, 1, 1),
            model.c2.weight,
            model.b2.bias,
            bn_weights=model.b2.weight,
            bn_running_var=model.b2.running_var,
            bn_running_mean=model.b2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c2_2 = Conv(
            [batch, 528, 80, 16],
            (1, 1, 1, 1),
            model.c2_2.weight,
            model.b2_2.bias,
            bn_weights=model.b2_2.weight,
            bn_running_var=model.b2_2.running_var,
            bn_running_mean=model.b2_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c3 = Conv(
            [batch, 264, 40, 16],
            (1, 1, 1, 1),
            model.c3.weight,
            model.b3.bias,
            bn_weights=model.b3.weight,
            bn_running_var=model.b3.running_var,
            bn_running_mean=model.b3.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c3_2 = Conv(
            [batch, 264, 40, 32],
            (1, 1, 1, 1),
            model.c3_2.weight,
            model.b3_2.bias,
            bn_weights=model.b3_2.weight,
            bn_running_var=model.b3_2.running_var,
            bn_running_mean=model.b3_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c4 = Conv(
            [batch, 132, 20, 32],
            (1, 1, 1, 1),
            model.c4.weight,
            model.b4.bias,
            bn_weights=model.b4.weight,
            bn_running_var=model.b4.running_var,
            bn_running_mean=model.b4.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c4_2 = Conv(
            [batch, 132, 20, 32],
            (1, 1, 1, 1),
            model.c4_2.weight,
            model.b4_2.bias,
            bn_weights=model.b4_2.weight,
            bn_running_var=model.b4_2.running_var,
            bn_running_mean=model.b4_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.bnc = Conv(
            [batch, 66, 10, 32],
            (1, 1, 1, 1),
            model.bnc.weight,
            model.bnb.bias,
            bn_weights=model.bnb.weight,
            bn_running_var=model.bnb.running_var,
            bn_running_mean=model.bnb.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.bnc_2 = Conv(
            [batch, 66, 10, 64],
            (1, 1, 1, 1),
            model.bnc_2.weight,
            model.bnb_2.bias,
            bn_weights=model.bnb_2.weight,
            bn_running_var=model.bnb_2.running_var,
            bn_running_mean=model.bnb_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c5 = Conv(
            [batch, 132, 20, 96],
            (1, 1, 1, 1),
            model.c5.weight,
            model.b5.bias,
            bn_weights=model.b5.weight,
            bn_running_var=model.b5.running_var,
            bn_running_mean=model.b5.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c5_2 = Conv(
            [batch, 132, 20, 32],
            (1, 1, 1, 1),
            model.c5_2.weight,
            model.b5_2.bias,
            bn_weights=model.b5_2.weight,
            bn_running_var=model.b5_2.running_var,
            bn_running_mean=model.b5_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c5_3 = Conv(
            [batch, 132, 20, 32],
            (1, 1, 1, 1),
            model.c5_3.weight,
            model.b5_3.bias,
            bn_weights=model.b5_3.weight,
            bn_running_var=model.b5_3.running_var,
            bn_running_mean=model.b5_3.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c6 = Conv(
            [batch, 264, 40, 64],
            (1, 1, 1, 1),
            model.c6.weight,
            model.b6.bias,
            bn_weights=model.b6.weight,
            bn_running_var=model.b6.running_var,
            bn_running_mean=model.b6.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c6_2 = Conv(
            [batch, 264, 40, 32],
            (1, 1, 1, 1),
            model.c6_2.weight,
            model.b6_2.bias,
            bn_weights=model.b6_2.weight,
            bn_running_var=model.b6_2.running_var,
            bn_running_mean=model.b6_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c6_3 = Conv(
            [batch, 264, 40, 32],
            (1, 1, 1, 1),
            model.c6_3.weight,
            model.b6_3.bias,
            bn_weights=model.b6_3.weight,
            bn_running_var=model.b6_3.running_var,
            bn_running_mean=model.b6_3.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c7 = Conv(
            [batch, 528, 80, 48],
            (1, 1, 1, 1),
            model.c7.weight,
            model.b7.bias,
            bn_weights=model.b7.weight,
            bn_running_var=model.b7.running_var,
            bn_running_mean=model.b7.running_mean,
            act_block_h=32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c7_2 = Conv(
            [batch, 528, 80, 16],
            (1, 1, 1, 1),
            model.c7_2.weight,
            model.b7_2.bias,
            bn_weights=model.b7_2.weight,
            bn_running_var=model.b7_2.running_var,
            bn_running_mean=model.b7_2.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c7_3 = Conv(
            [batch, 528, 80, 16],
            (1, 1, 1, 1),
            model.c7_3.weight,
            model.b7_3.bias,
            bn_weights=model.b7_3.weight,
            bn_running_var=model.b7_3.running_var,
            bn_running_mean=model.b7_3.running_mean,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c8 = Conv(
            [batch, 1056, 160, 32],
            (1, 1, 1, 1),
            model.c8.weight,
            model.b8.bias,
            bn_weights=model.b8.weight,
            bn_running_var=model.b8.running_var,
            bn_running_mean=model.b8.running_mean,
            act_block_h=5 * 32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c8_2 = Conv(
            [batch, 1056, 160, 16],
            (1, 1, 1, 1),
            model.c8_2.weight,
            model.b8_2.bias,
            bn_weights=model.b8_2.weight,
            bn_running_var=model.b8_2.running_var,
            bn_running_mean=model.b8_2.running_mean,
            act_block_h=5 * 32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.c8_3 = Conv(
            [batch, 1056, 160, 16],
            (1, 1, 1, 1),
            model.c8_3.weight,
            model.b8_3.bias,
            bn_weights=model.b8_3.weight,
            bn_running_var=model.b8_3.running_var,
            bn_running_mean=model.b8_3.running_mean,
            act_block_h=5 * 32,
            height_sharding=True,
            mesh_mapper=mesh_mapper,
        )
        self.output_layer = Conv(
            [batch, 1056, 160, 16],
            (1, 1, 1, 1),
            model.output_layer.weight,
            model.output_layer.bias,
            act_block_h=5 * 32,
            height_sharding=True,
            fused_op=False,
            mesh_mapper=mesh_mapper,
        )

        self.p1 = parameters.p1
        self.p2 = parameters.p2
        self.p3 = parameters.p3
        self.p4 = parameters.p4

    def __call__(self, device, input_tensor, original_shape, perf_mode=False):
        torch_input_tensor_nchw = torch.randn([2, 32, 1056, 160], dtype=torch.bfloat16).float()
        # torch_input_tensor_nchw = torch.ones(conv_input_shape, dtype=torch.bfloat16).float()
        torch_input_tensor = torch.permute(torch_input_tensor_nchw, (0, 2, 3, 1))
        torch_input_tensor = torch_input_tensor.reshape(
            torch_input_tensor.shape[0],
            1,
            torch_input_tensor.shape[1] * torch_input_tensor.shape[2],
            torch_input_tensor.shape[3],
        )
        tt_input_tensor = ttnn.from_torch(torch_input_tensor, ttnn.bfloat16)
        out = self.c8(device, tt_input_tensor)
        ttnn.deallocate(out)
        nhw = original_shape[-4] * original_shape[-2] * original_shape[-1]

        profiler.tracy_message("c1")
        output_tensor = self.c1(device, input_tensor)
        output_tensor = self.c1_2(device, output_tensor)
        save_c1_2_out = output_tensor
        # if perf_mode:
        #     save_c1_2_out = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        # else:
        #     save_c1_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
        #         output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
        #     )
        output_tensor = self.p1(output_tensor)

        profiler.tracy_message("c2")
        # output_tensor = unet_reshard(output_tensor, self.c2.conv.input_sharded_memory_config, use_reshard=False)
        output_tensor = self.c2(device, output_tensor)
        output_tensor = self.c2_2(device, output_tensor)
        if perf_mode:
            save_c2_2_out = output_tensor
        else:
            save_c2_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p2(output_tensor)

        profiler.tracy_message("c3")
        # output_tensor = unet_reshard(output_tensor, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c3(device, output_tensor)
        output_tensor = self.c3_2(device, output_tensor)
        if perf_mode:
            save_c3_2_out = output_tensor
        else:
            save_c3_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p3(output_tensor)

        profiler.tracy_message("c4")
        # output_tensor = unet_reshard(output_tensor, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(device, output_tensor)
        output_tensor = self.c4_2(device, output_tensor)
        if perf_mode:
            save_c4_2_out = output_tensor
        else:
            save_c4_2_out = ttnn.experimental.tensor.sharded_to_interleaved(
                output_tensor, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.experimental.tensor.DataType.BFLOAT16
            )
        output_tensor = self.p4(output_tensor)

        profiler.tracy_message("bnc")
        # output_tensor = unet_reshard(output_tensor, self.bnc.conv.input_sharded_memory_config)
        output_tensor = ttnn.from_device(output_tensor)
        # print(output_tensor.memory_config())
        output_tensor = self.bnc(device, output_tensor)
        output_tensor = self.bnc_2(device, output_tensor)

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
        ttnn.deallocate(save_c4_2_out)

        profiler.tracy_message("c5")
        # output_tensor = unet_reshard(output_tensor, self.c5.conv.input_sharded_memory_config)
        output_tensor = self.c5(device, output_tensor)
        output_tensor = self.c5_2(device, output_tensor)
        output_tensor = self.c5_3(device, output_tensor)

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
        ttnn.deallocate(save_c3_2_out)

        profiler.tracy_message("c6")
        # output_tensor = unet_reshard(output_tensor, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(device, output_tensor)
        output_tensor = self.c6_2(device, output_tensor)
        output_tensor = self.c6_3(device, output_tensor)

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
        ttnn.deallocate(save_c2_2_out)

        profiler.tracy_message("c7")
        # hacked_shard_shape = self.c7.conv.input_sharded_memory_config.shard_spec.shape
        # hacked_shard_shape[1] = output_tensor.shape[-1]
        # self.c7.conv.input_sharded_memory_config.shard_spec.shape = hacked_shard_shape
        # output_tensor = unet_reshard(output_tensor, self.c7.conv.input_sharded_memory_config)
        output_tensor = self.c7(device, output_tensor)
        output_tensor = self.c7_2(device, output_tensor)
        output_tensor = self.c7_3(device, output_tensor)

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
        ttnn.deallocate(save_c1_2_out)

        profiler.tracy_message("c8")
        # output_tensor = unet_reshard(output_tensor, self.c8.conv.input_sharded_memory_config)
        output_tensor = self.c8(device, output_tensor)
        output_tensor = self.c8_2(device, output_tensor)
        output_tensor = self.c8_3(device, output_tensor)
        output_tensor = self.output_layer(output_tensor)
        return ttnn.from_device(output_tensor)
