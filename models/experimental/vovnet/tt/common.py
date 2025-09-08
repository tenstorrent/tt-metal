# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

try:
    pass

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def get_nested(params, path_list):
    d = params
    for key in path_list:
        if key.isdigit():
            key = int(key)
        d = d[key]
    return d


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
        fused_op=True,
        debug=False,
        groups=1,
        effective_se=False,
        parameters=None,
        pw=False,
        use_1d_systolic_array=False,
        deallocate_activation=False,
    ) -> None:
        self.fused_op = fused_op
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = deallocate_activation
        path_parts = path.split(".")

        if fused_op:
            if pw:
                self.weights = get_nested(parameters, path_parts + ["conv_pw", "weight"])
                self.bias = get_nested(parameters, path_parts + ["conv_pw", "bias"])
            else:
                self.weights = get_nested(parameters, path_parts + ["conv", "weight"])
                self.bias = get_nested(parameters, path_parts + ["conv", "bias"])
            self.groups = groups
        else:
            if effective_se:
                self.weights = get_nested(parameters, path_parts + ["fc", "weight"])
                self.bias = get_nested(parameters, path_parts + ["fc", "bias"])
                self.groups = groups
            else:
                self.weights = get_nested(parameters, path_parts + ["conv_dw", "weight"])
                self.bias = None  # get_nested(parameters, path_parts + ["conv_dw", "bias"])
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
            weights_dtype=ttnn.bfloat8_b,
            activation=activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            deallocate_activation=self.deallocate_activation,
            reallocate_halo_output=False,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        if self.act_block_h is not None:
            self.conv_config.act_block_h_override = act_block_h

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, input_tensor):
        if input_tensor.shape[1] != 1:
            N, C, H, W = input_tensor.shape
            input_height = input_tensor.shape[1]
            input_width = input_tensor.shape[2]
        else:
            input_height = int(math.sqrt((input_tensor.shape[2])))
            input_width = int(math.sqrt((input_tensor.shape[2])))

        if input_tensor.shape[-1] == 16:
            in_channel = 3
        else:
            in_channel = input_tensor.shape[-1]
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            in_channels=in_channel,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias if self.bias else None,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[-1]),
            batch_size=input_tensor.shape[0],
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            dtype=ttnn.bfloat16,
            groups=self.groups,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return output_tensor, _out_height, _out_width
