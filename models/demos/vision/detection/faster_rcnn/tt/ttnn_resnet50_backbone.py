# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
ResNet-50 backbone implementation in TTNN for Faster-RCNN.
Follows the torchvision ResNet-50 structure with FrozenBatchNorm2d folded into Conv2d.
Outputs multi-scale feature maps for FPN consumption.
"""

import math

import ttnn


class TtConv2D:
    """TTNN Conv2D wrapper with fused activation support.

    Stage 2 optimization: ReLU is fused with conv using ttnn.UnaryWithParam.
    Uses bfloat8_b weights and LoFi math fidelity for throughput.
    """

    def __init__(
        self,
        weight,
        bias,
        device,
        batch_size,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        activation=None,
        deallocate_activation=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        enable_act_double_buffer=False,
        reshard_if_not_optimal=True,
        output_layout=ttnn.TILE_LAYOUT,
    ):
        self.device = device
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.groups = groups
        self.weight = weight
        self.bias = bias

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=activation,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            enable_act_double_buffer=enable_act_double_buffer,
            output_layout=output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=reshard_if_not_optimal,
            enable_weights_double_buffer=True,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            spatial = x.shape[2] // self.batch_size
            input_height = int(math.sqrt(spatial))
            input_width = spatial // input_height

        [x, [h, w], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=ttnn.bfloat8_b,
        )
        return x, h, w


class TtBottleneck:
    """ResNet-50 Bottleneck block in TTNN.

    Structure: 1x1 conv(+ReLU) -> 3x3 conv(+ReLU) -> 1x1 conv -> add shortcut -> ReLU
    BatchNorm is pre-folded into convolution weights during preprocessing.
    ReLU is fused with conv1 and conv2 via ttnn.UnaryWithParam (Stage 2 optimization).
    """

    def __init__(
        self,
        parameters,
        device,
        batch_size,
        in_channels,
        mid_channels,
        out_channels,
        stride,
        layer_name,
        has_downsample=False,
    ):
        self.device = device
        self.has_downsample = has_downsample

        relu_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        self.conv1 = TtConv2D(
            weight=parameters[f"backbone.{layer_name}.conv1.weight"],
            bias=parameters[f"backbone.{layer_name}.conv1.bias"],
            device=device,
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=relu_activation,
            deallocate_activation=False,
            enable_act_double_buffer=True,
        )

        self.conv2 = TtConv2D(
            weight=parameters[f"backbone.{layer_name}.conv2.weight"],
            bias=parameters[f"backbone.{layer_name}.conv2.bias"],
            device=device,
            batch_size=batch_size,
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            activation=relu_activation,
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

        self.conv3 = TtConv2D(
            weight=parameters[f"backbone.{layer_name}.conv3.weight"],
            bias=parameters[f"backbone.{layer_name}.conv3.bias"],
            device=device,
            batch_size=batch_size,
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=None,
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

        if has_downsample:
            self.downsample = TtConv2D(
                weight=parameters[f"backbone.{layer_name}.downsample.0.weight"],
                bias=parameters[f"backbone.{layer_name}.downsample.0.bias"],
                device=device,
                batch_size=batch_size,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                activation=None,
                deallocate_activation=False,
                enable_act_double_buffer=True,
            )

    def __call__(self, x):
        identity = x

        out, h, w = self.conv1(x)
        out, h, w = self.conv2(out)
        out, h, w = self.conv3(out)

        if self.has_downsample:
            identity, _, _ = self.downsample(identity)

        if identity.memory_config() != out.memory_config():
            identity = ttnn.to_memory_config(identity, out.memory_config())

        result = ttnn.add(identity, out)
        ttnn.deallocate(identity)
        ttnn.deallocate(out)
        result = ttnn.relu(result)

        return result


class TtResNet50Backbone:
    """ResNet-50 backbone in TTNN that returns multi-scale features for FPN.

    For a 320x320 input:
        - conv1 + maxpool -> 80x80 x 64
        - layer1 (C2)    -> 80x80 x 256
        - layer2 (C3)    -> 40x40 x 512
        - layer3 (C4)    -> 20x20 x 1024
        - layer4 (C5)    -> 10x10 x 2048
    """

    def __init__(self, parameters, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size

        self.conv1 = TtConv2D(
            weight=parameters["backbone.conv1.weight"],
            bias=parameters["backbone.conv1.bias"],
            device=device,
            batch_size=batch_size,
            in_channels=16,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            deallocate_activation=True,
            enable_act_double_buffer=True,
            reshard_if_not_optimal=False,
        )

        self.layer1 = self._make_layer(parameters, device, batch_size, "layer1", 64, 64, 256, 3, stride=1)
        self.layer2 = self._make_layer(parameters, device, batch_size, "layer2", 256, 128, 512, 4, stride=2)
        self.layer3 = self._make_layer(parameters, device, batch_size, "layer3", 512, 256, 1024, 6, stride=2)
        self.layer4 = self._make_layer(parameters, device, batch_size, "layer4", 1024, 512, 2048, 3, stride=2)

    def _make_layer(
        self, parameters, device, batch_size, layer_name, in_channels, mid_channels, out_channels, num_blocks, stride
    ):
        blocks = []
        blocks.append(
            TtBottleneck(
                parameters=parameters,
                device=device,
                batch_size=batch_size,
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                stride=stride,
                layer_name=f"{layer_name}.0",
                has_downsample=True,
            )
        )
        for i in range(1, num_blocks):
            blocks.append(
                TtBottleneck(
                    parameters=parameters,
                    device=device,
                    batch_size=batch_size,
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1,
                    layer_name=f"{layer_name}.{i}",
                    has_downsample=False,
                )
            )
        return blocks

    def __call__(self, x):
        x, h, w = self.conv1(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=h,
            input_w=w,
            channels=64,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dtype=ttnn.bfloat16,
            deallocate_input=True,
        )

        for block in self.layer1:
            x = block(x)
        c2 = x

        for block in self.layer2:
            x = block(x)
        c3 = x

        for block in self.layer3:
            x = block(x)
        c4 = x

        for block in self.layer4:
            x = block(x)
        c5 = x

        return {"0": c2, "1": c3, "2": c4, "3": c5}
