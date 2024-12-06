# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
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


def fold_bn_to_conv_weights_bias_torch(model, path):
    bn_weight = model[path + ".conv.1.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".conv.1.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + ".conv.0.weight"]
    weight = ((weight / torch.sqrt(bn_running_var)) * bn_weight).to(torch.bfloat16)

    bn_running_mean = model[path + ".conv.1.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".conv.1.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(-1).to(torch.bfloat16)
    return (
        weight,
        bias,
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
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            shard_layout=self.shard_layout,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            act_block_w_div=1,
            packer_l1_accum_enabled=False,
            input_channels_alignment=16 if self.input_params[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=self.enable_act_double_buffer,
            output_layout=self.output_layout,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
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
        )
        return output_tensor


class Conv_with_split:
    def __init__(
        self,
        model,
        path,
        input_params,
        conv_params,
        *,
        act_block_h=None,
        fused_op=True,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
        split_factor=2,
    ) -> None:
        self.input_params = input_params
        self.dtype = dtype
        self.split_factor = split_factor
        self.act_block_h = act_block_h
        self.conv_params = conv_params
        self.activation = activation
        self.fused_op = fused_op
        self.model = model
        self.path = path

        if self.fused_op:
            self.weights, self.bias = fold_bn_to_conv_weights_bias(self.model, self.path)
        else:
            weight = self.model[self.path + ".conv.0.weight"]
            bias = self.model[self.path + ".conv.0.bias"]
            self.weights = ttnn.from_torch(weight)
            bias = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias)

        input_channels = self.weights.shape[1]
        self.output_channels = self.weights.shape[0]
        assert input_channels % split_factor == 0
        self.split_input_channels = input_channels // split_factor
        if width_sharding == True:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])

    def __call__(self, device, input_tensor):
        print("*********", input_tensor.shape)
        batch, height, width, channel = self.input_params
        input_tensor = ttnn.to_torch(input_tensor)
        self.weights = ttnn.to_torch(self.weights)
        split_input_tensors = torch.split(input_tensor, self.split_input_channels, 3)
        split_weight_tensors = torch.split(self.weights, self.split_input_channels, 1)
        reader_patterns_cache = {}
        weights_dtype = ttnn.bfloat16
        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            math_fidelity=ttnn.MathFidelity.LoFi,
            shard_layout=self.shard_layout,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            activation=self.activation,
            # input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        for i in range(self.split_factor):
            tt_weight_tensor = ttnn.from_torch(
                split_weight_tensors[i], weights_dtype if weights_dtype != ttnn.bfloat8_b else ttnn.float32
            )
            if i == 0:
                tt_bias_tensor = self.bias
            # torch_input_tensor = torch.permute(split_input_tensors[i], (0, 2, 3, 1))
            tt_input_tensor = ttnn.from_torch(split_input_tensors[i], ttnn.bfloat16)
            [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
                input_tensor=tt_input_tensor,
                weight_tensor=tt_weight_tensor,
                in_channels=self.split_input_channels,
                out_channels=self.output_channels,
                device=device,
                bias_tensor=tt_bias_tensor,
                kernel_size=self.kernel_size,
                stride=(self.conv_params[0], self.conv_params[1]),
                padding=(self.conv_params[2], self.conv_params[3]),
                batch_size=batch,
                input_height=height,
                input_width=width,
                conv_config=conv_config,
                conv_op_cache=reader_patterns_cache,
            )
            tt_conv_output_tensor = ttnn.from_device(tt_output_tensor_on_device)
            torch_conv_output_tensor = ttnn.to_torch(tt_conv_output_tensor)
            if i == 0:
                torch_output_tensor = torch_conv_output_tensor
            else:
                torch_output_tensor = torch.add(torch_output_tensor, torch_conv_output_tensor)
        output_tensor = ttnn.from_torch(torch_output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(
            output_tensor, (input_tensor.shape[0], out_height, out_width, output_tensor.shape[3])
        )
        del out_height, out_width
        print("leaving split conv")
        return output_tensor
