# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    TtConv2d,
    AutoShardedStrategyConfiguration,
)


class TTRegNetBottleneck:
    def __init__(
        self,
        device,
        parameters,
        model_args,
        model_config,
        layer_config,
        stride=1,
        downsample=False,
        groups=1,
        shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        torch_model=None,
        block_name=None,
        stage_name=None,
    ):
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.model_config = model_config
        self.dtype = ttnn.bfloat16
        self.torch_model = torch_model
        self.block_name = block_name
        self.stage_name = stage_name
        # ------------------------- conv1: 1x1 + ReLU -------------------------
        conv1_params = model_args["conv1"]["conv"]
        conv1_config = self._create_conv_config(
            parameters=parameters["conv1"],
            batch_size=conv1_params["batch_size"],
            input_height=conv1_params["input_height"],
            input_width=conv1_params["input_width"],
            in_channels=conv1_params["in_channels"],
            out_channels=conv1_params["out_channels"],
            stride=conv1_params["stride"],
            kernel_size=conv1_params["kernel_size"],
            padding=conv1_params["padding"],
            groups=conv1_params["groups"],
        )
        self.conv1 = TtConv2d(conv1_config, device=device)

        # --------------------- conv2: 3x3 grouped + ReLU ---------------------
        conv2_params = model_args["conv2"]["conv"]
        conv2_config = self._create_conv_config(
            parameters=parameters["conv2"],
            batch_size=conv2_params["batch_size"],
            input_height=conv2_params["input_height"],
            input_width=conv2_params["input_width"],
            in_channels=conv2_params["in_channels"],
            out_channels=conv2_params["out_channels"],
            stride=conv2_params["stride"],
            kernel_size=conv2_params["kernel_size"],
            padding=conv2_params["padding"],
            groups=conv2_params["groups"],
        )
        self.conv2 = TtConv2d(conv2_config, device=device)
        # --------------------------- SE: fc1 (1x1 + ReLU) --------------------
        se_fc1_params = model_args["se"]["fc1"]
        se_fc1_config = self._create_conv_config(
            parameters=parameters["se"]["fc1"],
            batch_size=se_fc1_params["batch_size"],
            input_height=se_fc1_params["input_height"],
            input_width=se_fc1_params["input_width"],
            in_channels=se_fc1_params["in_channels"],
            out_channels=se_fc1_params["out_channels"],
            stride=se_fc1_params["stride"],
            kernel_size=se_fc1_params["kernel_size"],
            padding=se_fc1_params["padding"],
            groups=se_fc1_params["groups"],
        )
        self.se_fc1 = TtConv2d(se_fc1_config, device=device)
        # --------------------------- SE: fc2 (1x1, no act) -------------------
        se_fc2_params = model_args["se"]["fc2"]
        se_fc2_config = self._create_conv_config(
            parameters=parameters["se"]["fc2"],
            batch_size=se_fc2_params["batch_size"],
            input_height=se_fc2_params["input_height"],
            input_width=se_fc2_params["input_width"],
            in_channels=se_fc2_params["in_channels"],
            out_channels=se_fc2_params["out_channels"],
            stride=se_fc2_params["stride"],
            kernel_size=se_fc2_params["kernel_size"],
            padding=se_fc2_params["padding"],
            groups=se_fc2_params["groups"],
            activation=None,
        )

        self.se_fc2 = TtConv2d(se_fc2_config, device=device)
        # ----------------------- conv3: 1x1 projection (no act) --------------
        conv3_params = model_args["conv3"]["conv"]

        conv3_config = self._create_conv_config(
            parameters=parameters["conv3"],
            batch_size=conv3_params["batch_size"],
            input_height=conv3_params["input_height"],
            input_width=conv3_params["input_width"],
            in_channels=conv3_params["in_channels"],
            out_channels=conv3_params["out_channels"],
            stride=conv3_params["stride"],
            kernel_size=conv3_params["kernel_size"],
            padding=conv3_params["padding"],
            groups=conv3_params["groups"],
            activation=None,
        )
        self.conv3 = TtConv2d(conv3_config, device=device)
        # ------------------------------ optional downsample -------------------
        if downsample:
            try:
                downsample_conv_params = model_args["downsample"]["conv"]
            except:
                downsample_conv_params = model_args.downsample["0"]
            downsample_conv_config = self._create_conv_config(
                parameters=parameters["downsample"],
                batch_size=downsample_conv_params["batch_size"],
                input_height=downsample_conv_params["input_height"],
                input_width=downsample_conv_params["input_width"],
                in_channels=downsample_conv_params["in_channels"],
                out_channels=downsample_conv_params["out_channels"],
                stride=downsample_conv_params["stride"],
                kernel_size=downsample_conv_params["kernel_size"],
                padding=downsample_conv_params["padding"],
                groups=downsample_conv_params["groups"],
                activation=None,
            )
            self.downsample_layer = TtConv2d(downsample_conv_config, device=device)
        else:
            self.downsample_layer = None

    def _create_conv_config(
        self,
        parameters,
        batch_size,
        input_height,
        input_width,
        in_channels,
        out_channels,
        stride,
        kernel_size,
        padding,
        groups,
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
    ):
        # Convert weights to float32 format (required by tt_cnn builder)
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor):
            weight = ttnn.from_torch(ttnn.to_torch(weight), dtype=ttnn.float32)

        # Convert bias to shape (1, 1, 1, out_channels) in float32
        bias = None
        if "bias" in parameters and parameters.bias is not None:
            bias_torch = ttnn.to_torch(parameters.bias).reshape(1, 1, 1, -1)
            bias = ttnn.from_torch(bias_torch, dtype=ttnn.float32)

        # Convert stride to list format (required by ttnn.conv2d)
        if isinstance(stride, int):
            stride_list = [stride, stride]
        elif isinstance(stride, tuple) and len(stride) == 2:
            stride_list = list(stride)
        else:
            stride_list = stride

        # Convert padding to list format (required by ttnn.conv2d)
        if isinstance(padding, int):
            padding_list = [padding, padding]
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding_list = list(padding)
        elif isinstance(padding, tuple) and len(padding) == 4:
            padding_list = list(padding)
        else:
            padding_list = padding

        # Select math fidelity based on block (HiFi4 for block 2 for better accuracy)
        math_fidelity = ttnn.MathFidelity.HiFi4

        return Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=kernel_size,
            stride=stride_list,
            padding=padding_list,
            groups=groups,
            weight=weight,
            bias=bias,
            activation=activation,
            activation_dtype=self.dtype,
            weights_dtype=self.dtype,
            output_dtype=self.dtype,
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            enable_act_double_buffer=False,
        )

    def __call__(self, x, device):
        downsample_input = ttnn.clone(x)
        # conv1- 1x1 convolution (using new TtConv2d interface)
        out = self.conv1(x)
        out, (height, width) = self.conv2(out, return_output_dim=True)
        # SE module
        # reduce mean
        out1 = ttnn.reallocate(out)
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.reshape(out, (1, height, width, out.shape[-1]))
        se_out = ttnn.mean(out, dim=[1, 2], keepdim=True)
        # SE fc1
        se_out = self.se_fc1(se_out)
        # SE fc2
        se_out = self.se_fc2(se_out)
        se_out = ttnn.sigmoid(se_out)
        out_4d = ttnn.multiply(out1, se_out)
        # Flatten back to match identity format
        batch, channels = out_4d.shape[0], out_4d.shape[-1]
        out = ttnn.reshape(out_4d, (1, 1, batch * height * width, channels))
        # conv3: 1x1 projection
        out = self.conv3(out)
        if self.downsample_layer is not None:
            # downsample
            downsample_input = self.downsample_layer(downsample_input)
        # Add
        out = ttnn.add(out, downsample_input)
        out = ttnn.relu(out)
        return out
