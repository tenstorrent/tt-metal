# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from typing import Tuple
import ttnn


def fold_bn_to_conv_weights_bias(model, path):
    bn_weight = model[path + ".conv.1.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".conv.1.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + ".conv.0.weight"]
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = model[path + ".conv.1.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".conv.1.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    return (
        ttnn.from_torch(
            weight,
        ),
        ttnn.from_torch(bias),
    )


class Conv:
    def __init__(
        self,
        model,
        path,
        input_params,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        fused_op=True,
        width_sharding=False,
        output_layout=ttnn.TILE_LAYOUT,
        enable_split_reader=False,
        enable_act_double_buffer=False,
    ) -> None:
        if fused_op:
            self.weights, self.bias = fold_bn_to_conv_weights_bias(model, path)
        else:
            weight = model[path + ".conv.0.weight"]
            bias = model[path + ".conv.0.bias"]
            self.weights = ttnn.from_torch(weight)
            bias = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias)
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.output_layout = output_layout
        self.enable_split_reader = enable_split_reader
        self.enable_act_double_buffer = enable_act_double_buffer

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            activation=self.activation,
            shard_layout=self.shard_layout,
            act_block_w_div=1,
            input_channels_alignment=16 if self.input_params[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=self.enable_act_double_buffer,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        output_tensor, [self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
            compute_config=compute_config,
            return_output_dim=False,
            return_weights_and_bias=True,
        )
        return output_tensor


class Upsample:
    def __init__(self, input_params, scale_factor, mode="nearest") -> None:
        self.batch_size = input_params[0]
        self.input_height = input_params[1]
        self.input_width = input_params[2]
        self.input_channels = input_params[3]
        self.scale_h = scale_factor[0]
        self.scale_w = scale_factor[1]
        self.mode = mode

    # helper functions for upsample for block sharded inputs
    def determine_num_cores_for_upsample(
        self, batch_size: int, height: int, width: int, num_channels: int, max_grid_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        max_nshards_h = min(
            batch_size * height * width, max_grid_size[0]
        )  ## height along NHW (N: batch size, H: height, W: width)
        max_nshards_w = min(num_channels, max_grid_size[1])  ## width along C (number of channels)
        ## find nshards_h along NHW
        nshards_h = max_nshards_h
        while nshards_h > 0:
            if batch_size * height % nshards_h == 0:
                break
            nshards_h -= 1
        ## find nshards_w along C
        nshards_w = max_nshards_w
        while nshards_w > 0:
            ## make sure: 1. nshards_w divides num_channels, and 2. shard_shape[1] is aligned to 32B
            if num_channels % nshards_w == 0 and math.ceil(num_channels * 2 / nshards_w) % 32 == 0:
                break
            nshards_w -= 1
        if nshards_w == 0 or nshards_h == 0:
            raise ValueError(f"nshards_h or nshards_w is 0: nshards_h={nshards_h}, nshards_w={nshards_w}")
        return [nshards_h, nshards_w]

    def get_core_grid_from_num_cores_for_upsample(self, num_cores: Tuple[int, int], max_grid_size: Tuple[int, int]) -> ttnn.CoreRangeSet:  # type: ignore
        ncores_h, ncores_w = num_cores
        assert ncores_h <= max_grid_size[0]
        assert ncores_w <= max_grid_size[1]
        return ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(ncores_w - 1, ncores_h - 1),
                )
            }
        )

    def __call__(self, device, input_tensor):
        device_grid = device.compute_with_storage_grid_size()
        max_grid_size = [device_grid.y, device_grid.x]
        num_cores = self.determine_num_cores_for_upsample(
            self.batch_size, self.input_height, self.input_width, self.input_channels, max_grid_size
        )
        shard_grid = self.get_core_grid_from_num_cores_for_upsample(num_cores, max_grid_size)
        shard_height = math.ceil(self.input_height * self.input_width / num_cores[0])
        shard_width = math.ceil(self.input_channels / num_cores[1])
        shard_spec = ttnn.ShardSpec(shard_grid, (shard_height, shard_width), ttnn.ShardOrientation.ROW_MAJOR, False)
        in_sharded_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)
        output_tensor = ttnn.to_memory_config(input_tensor, memory_config=in_sharded_mem_config)
        out_shard_spec = ttnn.ShardSpec(
            shard_grid,
            (shard_height * self.scale_h * self.scale_w, shard_width),
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        )
        out_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, out_shard_spec
        )
        return ttnn.upsample(output_tensor, (self.scale_h, self.scale_w), memory_config=out_sharded_mem_config)
