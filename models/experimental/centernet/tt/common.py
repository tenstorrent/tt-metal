# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtConv:
    def __init__(
        self,
        device,
        path,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        activation="",
        fused_op=True,
        debug=False,
        groups=1,
        bias=False,
        split_conv=False,
        seperable_conv_norm_act=False,
        effective_se=False,
        parameters=None,
        pw=False,
        conv_transpose=False,
    ) -> None:
        self.fused_op = fused_op
        path = path

        self.weights = parameters[f"{path}.weight"]
        self.bias = parameters[f"{path}.bias"]

        self.conv_params = conv_params
        self.conv_transpose = conv_transpose
        self.debug = debug
        self.groups = groups
        self.device = device
        self.act_block_h = act_block_h

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.out_channels = self.weights.shape[0]
        self.reader_patterns_cache = {}
        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation=activation,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=reshard,
            deallocate_activation=deallocate,
        )
        if self.act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, input_tensor):
        N, C, H, W = input_tensor.shape
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        N, H, W, C = input_tensor.shape
        if C == 512:
            self.conv_config.shard_layout = None

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        if self.conv_transpose:
            output_tensor, [_out_height, _out_width] = ttnn.conv_transpose2d(
                input_tensor=input_tensor,
                weight_tensor=self.weights,
                in_channels=input_tensor.shape[-1],
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self.bias if self.bias else None,
                kernel_size=self.kernel_size,
                stride=(self.conv_params[0], self.conv_params[1]),
                padding=(self.conv_params[2], self.conv_params[-1]),
                output_padding=(0, 0),
                batch_size=input_tensor.shape[0],
                input_height=input_tensor.shape[1],
                input_width=input_tensor.shape[2],
                conv_config=self.conv_config,
                groups=self.groups,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=False,
            )
        else:
            output_tensor, [_out_height, _out_width] = ttnn.conv2d(
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
                groups=self.groups,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=False,
            )

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.reshape(output_tensor, (N, _out_height, _out_width, output_tensor.shape[-1]))
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))

        return output_tensor


class TtBatch_norm:
    def __init__(self, device, path, parameters):
        self.parameters = parameters

        self.weight = parameters[f"{path}.weight"]
        self.bias = parameters[f"{path}.bias"]
        self.running_mean = parameters[f"{path}.running_mean"]
        self.running_var = parameters[f"{path}.running_var"]

    def __call__(self, input):
        input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
        bn = ttnn.batch_norm(
            input,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=self.weight,
            bias=self.bias,
            training=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        return bn
