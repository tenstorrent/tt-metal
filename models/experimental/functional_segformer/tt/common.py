# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        groups=1,
        dtype=ttnn.bfloat16,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.groups = groups
        self.dtype = dtype
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )

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
            input_channels_alignment=16 if input_tensor.shape[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_split_reader=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_tensor.shape[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=input_tensor.shape[0],
            input_height=input_tensor.shape[1],
            input_width=input_tensor.shape[2],
            conv_config=conv_config,
            groups=self.groups,
        )
        ## TODO: Op | WARNING  | Tensor at index 0 is not allocated
        # print("sr2a", output_tensor.shape)

        # output_tensor = ttnn.from_device(output_tensor)
        # output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

        # output_tensor = ttnn.reshape(
        #     output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        # )
        # del _out_height, _out_width

        return output_tensor, _out_height, _out_width
