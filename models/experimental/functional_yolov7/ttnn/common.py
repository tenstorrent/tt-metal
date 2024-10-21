# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


class Conv:
    def __init__(
        self,
        input_params,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        activation="",
        groups=1,
        dtype=ttnn.bfloat8_b,
        num_cores_nhw=None,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.dtype = dtype

        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.groups = groups
        self.deallocate = deallocate
        self.activation = activation
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.num_cores_nhw = num_cores_nhw

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=self.shard_layout,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            input_channels_alignment=32,
            transpose_shards=True,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
        )
        if self.num_cores_nhw is not None:
            shard_grid = get_shard_grid_from_num_cores(self.num_cores_nhw, device)
            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        # if config_override and "act_block_h" in config_override:
        # conv_config.act_block_h_override = config_override["act_block_h"]

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=(self.conv_params[0], self.conv_params[1]),
            stride=(self.conv_params[2], self.conv_params[3]),
            padding=(self.conv_params[4], self.conv_params[5]),
            dilation=(self.conv_params[6], self.conv_params[7]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
            groups=self.groups,
        )
        # print("output_tensor shape: ", output_tensor.shape)
        # output_tensor = ttnn.reshape(
        #     output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        # )
        # print("output_tensor shape: ", output_tensor.shape)
        return output_tensor
