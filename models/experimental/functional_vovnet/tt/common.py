# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


class Conv:
    def __init__(
        self,
        device,
        path,
        conv_params,
        *,
        act_block_h=None,
        activation="",
        split_conv=False,
        seperable_conv_norm_act=False,
        fused_op=True,
        debug=False,
        groups=1,
        effective_se=False,
        parameters=None,
        pw=False,
    ) -> None:
        self.fused_op = fused_op
        path = path
        if fused_op:
            if pw:
                self.weights = parameters[f"{path}.conv_pw.weight"]
                self.bias = parameters[f"{path}.conv_pw.bias"]
            else:
                self.weights = parameters[f"{path}.conv.weight"]
                self.bias = parameters[f"{path}.conv.bias"]
            self.groups = groups
        else:
            if effective_se:
                self.weights = parameters[f"{path}.weight"]
                self.bias = parameters[f"{path}.bias"]
                self.groups = groups
            else:
                self.weights = parameters[f"{path}.conv_dw.weight"]
                self.bias = parameters[f"{path}.conv_dw.bias"]
                self.groups = self.weights.shape[0]

        self.conv_params = conv_params

        self.debug = debug

        self.device = device
        self.act_block_h = act_block_h

        if fused_op:
            activation = ""
        self.shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        if (
            self.weights.shape[0] == 768
            or self.weights.shape[0] == 224
            or self.weights.shape[0] == 1024
            or (self.weights.shape[0] == 256 and self.weights.shape[1] == 256)
            or (self.weights.shape[0] == 512 and self.weights.shape[1] == 512)
        ):
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        if self.weights.shape[0] == 192 or (self.weights.shape[0] == 512 and self.weights.shape[1] == 736):
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.out_channels = self.weights.shape[0]
        self.reader_patterns_cache = {}
        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            activation=activation,
            # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            shard_layout=self.shard_layout,
            input_channels_alignment=16,  # if self.input_params[-1] < 16 else 32,
            reshard_if_not_optimal=True,
            deallocate_activation=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            act_block_w_div=1,
            enable_weights_double_buffer=True,
        )
        if self.act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h

        if self.fused_op:
            self.bn = nn.BatchNorm2d(num_features=self.weights.shape[0])

            self.bn.weight.data = ttnn.to_torch(parameters[f"{path}.bn.weight"])  # Example values
            self.bn.bias.data = ttnn.to_torch(parameters[f"{path}.bn.bias"])
            # Assign new values to running mean and variance (used during inference)
            self.bn.running_mean = ttnn.to_torch(parameters[f"{path}.bn.running_mean"])
            self.bn.running_var = ttnn.to_torch(parameters[f"{path}.bn.running_var"])

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, input_tensor):
        N, C, H, W = input_tensor.shape
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.conv_config.input_channels_alignment = 16 if input_tensor.shape[-1] < 16 else 32
        output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            in_channels=input_tensor.shape[-1],
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias if self.bias else None,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[-1]),
            batch_size=input_tensor.shape[0],
            input_height=input_tensor.shape[1],
            input_width=input_tensor.shape[2],
            conv_config=self.conv_config,
            conv_op_cache=self.reader_patterns_cache,
            debug=self.debug,
            groups=self.groups,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.reshape(output_tensor, (N, _out_height, _out_width, output_tensor.shape[-1]))
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        if self.fused_op:
            output_tensor = ttnn.to_torch(output_tensor)
            output_tensor = self.bn(output_tensor)
            output_tensor = ttnn.from_torch(
                output_tensor, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            output_tensor = ttnn.relu(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        return output_tensor, _out_height, _out_width
