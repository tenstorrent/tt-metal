# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from .utils import create_conv2d_config, create_maxpool_config, post_process_conv_output


@dataclass
class ResNet50Optimizations:
    """TTNN implementation of ResNet50Optimizations"""

    conv1_7x7: dict
    bottleneck_1x1_first: dict
    bottleneck_3x3: dict
    bottleneck_1x1_last: dict
    downsample_1x1: dict


resnet50_optimizations = ResNet50Optimizations(
    conv1_7x7={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_first={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_3x3={
        "activation": ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        "deallocate_activation": False,
        "reallocate_halo_output": True,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    bottleneck_1x1_last={
        "activation": None,
        "deallocate_activation": True,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
    downsample_1x1={
        "activation": None,
        "deallocate_activation": False,
        "reallocate_halo_output": False,
        "packer_l1_acc": False,
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
    },
)


class TtBottleneck:
    """TTNN implementation of Bottleneck"""

    expansion = 4

    def __init__(
        self,
        parameters,
        device,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        style="caffe",
        model_config=None,
        optimizations=None,
    ):
        self.device = device
        self.stride = stride
        self.has_downsample = downsample is not None
        self.style = style
        self.model_config = model_config
        self.optimizations = optimizations or resnet50_optimizations

        self.conv1_weight = parameters.conv1.weight
        self.conv1_bias = parameters.conv1.bias
        self.conv2_weight = parameters.conv2.weight
        self.conv2_bias = parameters.conv2.bias
        self.conv3_weight = parameters.conv3.weight
        self.conv3_bias = parameters.conv3.bias

        if self.has_downsample and hasattr(parameters, "downsample") and parameters.downsample:
            self.downsample_weight = parameters.downsample.weight
            self.downsample_bias = parameters.downsample.bias
        else:
            self.downsample_weight = None
            self.downsample_bias = None

    def _get_conv1(self, batch_size, height, width):
        if self.style == "caffe" and self.stride > 1:
            stride = (self.stride, self.stride)
        else:
            stride = (1, 1)
        opt = self.optimizations.bottleneck_1x1_first
        out_channels, in_channels = self.conv1_weight.shape[0], self.conv1_weight.shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            weight=self.conv1_weight,
            bias=self.conv1_bias,
            stride=stride,
            padding=(0, 0),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_conv2(self, batch_size, height, width):
        if self.style == "caffe":
            stride = (1, 1)
        else:
            stride = (self.stride, self.stride) if self.stride > 1 else (1, 1)
        opt = self.optimizations.bottleneck_3x3
        out_channels, in_channels = self.conv2_weight.shape[0], self.conv2_weight.shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(3, 3),
            weight=self.conv2_weight,
            bias=self.conv2_bias,
            stride=stride,
            padding=(1, 1),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_conv3(self, batch_size, height, width):
        opt = self.optimizations.bottleneck_1x1_last
        out_channels, in_channels = self.conv3_weight.shape[0], self.conv3_weight.shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            weight=self.conv3_weight,
            bias=self.conv3_bias,
            stride=(1, 1),
            padding=(0, 0),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_downsample(self, batch_size, height, width):
        if not self.has_downsample or self.downsample_weight is None:
            return None
        opt = self.optimizations.downsample_1x1
        out_channels, in_channels = self.downsample_weight.shape[0], self.downsample_weight.shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(1, 1),
            weight=self.downsample_weight,
            bias=self.downsample_bias,
            stride=(self.stride, self.stride),
            padding=(0, 0),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def __call__(self, x, batch_size, height, width):
        if x.shape[1] == 1 and x.shape[2] > 1:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            _, _, hw, c = x.shape
            h = int(hw**0.5)
            w = hw // h
            while h * w != hw and h > 0:
                h -= 1
                w = hw // h if h > 0 else hw
            x = ttnn.reshape(x, (batch_size, h, w, c))
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            height, width = h, w

        identity = x
        x_flat = ttnn.reshape(x, (1, 1, batch_size * height * width, x.shape[3]))

        conv1 = self._get_conv1(batch_size, height, width)
        out, (conv1_h, conv1_w) = conv1(x_flat, return_output_dim=True)
        out = post_process_conv_output(
            out, batch_size, conv1_h, conv1_w, self.conv1_weight.shape[0], to_dram=True, reshape_4d=True
        )

        conv2 = self._get_conv2(batch_size, conv1_h, conv1_w)
        out_flat = ttnn.reshape(out, (1, 1, batch_size * conv1_h * conv1_w, out.shape[3]))
        out, (conv2_h, conv2_w) = conv2(out_flat, return_output_dim=True)
        out = post_process_conv_output(
            out, batch_size, conv2_h, conv2_w, self.conv2_weight.shape[0], to_dram=True, reshape_4d=True
        )

        conv3 = self._get_conv3(batch_size, conv2_h, conv2_w)
        out_flat = ttnn.reshape(out, (1, 1, batch_size * conv2_h * conv2_w, out.shape[3]))
        out, (final_h, final_w) = conv3(out_flat, return_output_dim=True)
        out = post_process_conv_output(
            out, batch_size, final_h, final_w, self.conv3_weight.shape[0], to_dram=True, reshape_4d=True
        )

        if self.has_downsample:
            downsample = self._get_downsample(batch_size, height, width)
            identity_flat = ttnn.reshape(identity, (1, 1, batch_size * height * width, identity.shape[3]))
            identity, _ = downsample(identity_flat, return_output_dim=True)
            identity = post_process_conv_output(
                identity, batch_size, final_h, final_w, self.downsample_weight.shape[0], to_dram=True, reshape_4d=True
            )

        if identity.memory_config() != out.memory_config():
            identity = ttnn.to_memory_config(identity, out.memory_config())
        if identity.layout != out.layout:
            identity = ttnn.to_layout(identity, out.layout)

        out = ttnn.add_(out, identity, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)])

        return out, final_h, final_w


class TtResNet50:
    """TTNN implementation of ResNet50"""

    def __init__(self, conv_args, parameters, device, model_config=None, optimizations=None, out_indices=(1, 2, 3)):
        self.device = device
        self.style = "caffe"
        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }
        self.optimizations = optimizations or resnet50_optimizations
        self.out_indices = out_indices
        self.conv_args = conv_args

        self.conv1_weight = parameters.conv1.weight
        self.conv1_bias = parameters.conv1.bias

        self.in_channels = 64
        self.layer1 = self._make_layer(parameters, "layer1", 64, 3, stride=1)
        self.layer2 = self._make_layer(parameters, "layer2", 128, 4, stride=2)
        self.layer3 = self._make_layer(parameters, "layer3", 256, 6, stride=2)
        self.layer4 = self._make_layer(parameters, "layer4", 512, 3, stride=2)

    def _make_layer(self, parameters, layer_name, planes, blocks, stride=1):
        layers = []

        downsample = None
        if stride != 1 or self.in_channels != planes * TtBottleneck.expansion:
            downsample = True

        layer_idx = int(layer_name[-1])
        layer_params = getattr(parameters, f"{layer_name}_0")
        layers.append(
            TtBottleneck(
                parameters=layer_params,
                device=self.device,
                in_channels=self.in_channels,
                out_channels=planes,
                stride=stride,
                downsample=downsample,
                style=self.style,
                model_config=self.model_config,
                optimizations=self.optimizations,
            )
        )
        self.in_channels = planes * TtBottleneck.expansion

        for i in range(1, blocks):
            layer_params = getattr(parameters, f"{layer_name}_{i}")
            layers.append(
                TtBottleneck(
                    parameters=layer_params,
                    device=self.device,
                    in_channels=self.in_channels,
                    out_channels=planes,
                    stride=1,
                    downsample=None,
                    style=self.style,
                    model_config=self.model_config,
                    optimizations=self.optimizations,
                )
            )

        return layers

    def _get_conv1(self, batch_size, height, width):
        opt = self.optimizations.conv1_7x7
        out_channels, in_channels = self.conv1_weight.shape[0], self.conv1_weight.shape[1]
        config = create_conv2d_config(
            input_height=height,
            input_width=width,
            in_channels=in_channels,
            out_channels=out_channels,
            batch_size=batch_size,
            kernel_size=(7, 7),
            weight=self.conv1_weight,
            bias=self.conv1_bias,
            stride=(2, 2),
            padding=(3, 3),
            model_config=self.model_config,
            activation=opt.get("activation"),
            deallocate_activation=opt.get("deallocate_activation", False),
            reallocate_halo_output=opt.get("reallocate_halo_output", False),
            packer_l1_acc=opt.get("packer_l1_acc", False),
            enable_act_double_buffer=opt.get("enable_act_double_buffer", False),
            enable_weights_double_buffer=opt.get("enable_weights_double_buffer", False),
        )
        return TtConv2d(config, self.device)

    def _get_maxpool(self, height, width, batch_size):
        effective_batch_size = 1
        effective_height = height
        effective_width = width * batch_size if batch_size > 1 else width

        config = create_maxpool_config(
            input_height=effective_height,
            input_width=effective_width,
            channels=64,
            batch_size=effective_batch_size,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            dtype=ttnn.bfloat16,
        )
        return TtMaxPool2d(config, self.device)

    def __call__(self, x, batch_size=1, input_height=None, input_width=None, return_intermediates=False):
        if input_height is None or input_width is None:
            if x.shape[1] == 1 and x.shape[2] > 1:
                _, _, hw, c = x.shape
                h = int(hw**0.5)
                w = hw // h
                input_height, input_width = h, w
            else:
                input_height, input_width = x.shape[1], x.shape[2]

        intermediates = {}
        height, width = input_height, input_width

        if x.shape[1] == 1 and x.shape[2] > 1:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, (batch_size, height, width, x.shape[3]))
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        conv1 = self._get_conv1(batch_size, height, width)
        x, (height, width) = conv1(x, return_output_dim=True)
        x = post_process_conv_output(x, batch_size, height, width, 64, to_dram=True, reshape_4d=True)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)

        if batch_size > 1:
            if x.shape[1] == 1 and x.shape[2] > 1:
                _, _, hw, c = x.shape
                x = ttnn.reshape(x, (batch_size, height, width, c))

            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

            maxpool_outputs = []
            maxpool = self._get_maxpool(height, width, 1)

            for i in range(batch_size):
                img_slice = ttnn.slice(x, [i, 0, 0, 0], [i + 1, height, width, 64])

                img_flat = ttnn.reshape(img_slice, (1, 1, height * width, 64))
                img_slice.deallocate(True)

                pooled = maxpool(img_flat)
                img_flat.deallocate(True)

                pooled_h = height // 2
                pooled_w = width // 2
                pooled = ttnn.reshape(pooled, (1, pooled_h, pooled_w, 64))
                pooled = ttnn.to_memory_config(pooled, ttnn.DRAM_MEMORY_CONFIG)
                maxpool_outputs.append(pooled)

            x = ttnn.concat(maxpool_outputs, dim=0)

            for output in maxpool_outputs:
                output.deallocate(True)

            height = height // 2
            width = width // 2
        else:
            if x.shape[1] == 1 and x.shape[2] > 1:
                pool_input = ttnn.reshape(x, (1, 1, batch_size * x.shape[2], 64))
            else:
                pool_input = ttnn.reshape(x, (1, 1, batch_size * height * width, 64))

            try:
                if pool_input.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                    pool_input = ttnn.to_memory_config(pool_input, ttnn.DRAM_MEMORY_CONFIG)
            except:
                pool_input = ttnn.to_memory_config(pool_input, ttnn.DRAM_MEMORY_CONFIG)

            maxpool = self._get_maxpool(height, width, batch_size)
            x = maxpool(pool_input)

            height = height // 2
            width = width // 2

            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, (batch_size, height, width, 64))

        if return_intermediates:
            intermediates["after_maxpool"] = x

        outputs = []

        for i, block in enumerate(self.layer1):
            x, height, width = block(x, batch_size, height, width)
            if return_intermediates:
                intermediates[f"layer1.{i}"] = x
        if 0 in self.out_indices:
            outputs.append(x)

        for i, block in enumerate(self.layer2):
            x, height, width = block(x, batch_size, height, width)
            if return_intermediates:
                intermediates[f"layer2.{i}"] = x
        if 1 in self.out_indices:
            outputs.append(x)

        for i, block in enumerate(self.layer3):
            x, height, width = block(x, batch_size, height, width)
            if return_intermediates:
                intermediates[f"layer3.{i}"] = x
        if 2 in self.out_indices:
            outputs.append(x)

        for i, block in enumerate(self.layer4):
            x, height, width = block(x, batch_size, height, width)
            if return_intermediates:
                intermediates[f"layer4.{i}"] = x
        if 3 in self.out_indices:
            outputs.append(x)

        if return_intermediates:
            return tuple(outputs), intermediates
        return tuple(outputs)
