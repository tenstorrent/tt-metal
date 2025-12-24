# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from typing import Optional
from models.tt_cnn.tt.builder import TtConv2d
import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)


@dataclass
class BottleneckOptimizer:
    conv1: dict
    conv2: dict
    conv3: dict
    downsample: dict


bottleneck_layer_optimisations = {
    "default": BottleneckOptimizer(
        conv1={"act_block_h": 32, "memory_config": ttnn.DRAM_MEMORY_CONFIG},
        conv2={
            "act_block_h": 32,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv3={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
        },
        downsample={
            "memory_config": None,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
    ),
    "layer1": BottleneckOptimizer(
        conv1={
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "reallocate_halo_output": True,
        },
        conv2={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        downsample={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
    "layer2": BottleneckOptimizer(
        conv1={
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv2={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        downsample={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
    "layer3": BottleneckOptimizer(
        conv1={
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv2={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        downsample={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
    "layer4": BottleneckOptimizer(
        conv1={
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv2={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        downsample={
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


def get_bottleneck_optimisation(layer_name):
    for key in ["layer1", "layer2", "layer3", "layer4"]:
        if key in layer_name:
            return bottleneck_layer_optimisations[key]
    return bottleneck_layer_optimisations["default"]


class TTBottleneck:
    expansion: int = 4

    def __init__(
        self,
        parameters,
        device,
        downsample,
        stride,
        model_args,
        model_config,
        dilation: int = 1,
        name: Optional[str] = "bottleneck",
        layer_optimisations=bottleneck_layer_optimisations["default"],
    ) -> None:
        self.name = name
        self.layer_optimisations = layer_optimisations

        self.conv_config_1 = Conv2dConfiguration.from_model_args(
            model_args["conv1"],
            weights=parameters.conv1["weight"],
            bias=parameters.conv1["bias"],
            **layer_optimisations.conv1,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv1 = TtConv2d(self.conv_config_1, device)

        self.conv_config_2 = Conv2dConfiguration.from_model_args(
            model_args["conv2"],
            weights=parameters.conv2["weight"],
            bias=parameters.conv2["bias"],
            **layer_optimisations.conv2,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv2 = TtConv2d(self.conv_config_2, device)

        self.conv_config_3 = Conv2dConfiguration.from_model_args(
            model_args["conv3"],
            weights=parameters.conv3["weight"],
            bias=parameters.conv3["bias"],
            **layer_optimisations.conv3,
            math_fidelity=model_config["MATH_FIDELITY"],
            weights_dtype=model_config["WEIGHTS_DTYPE"],
            activation_dtype=model_config["ACTIVATIONS_DTYPE"],
        )
        self.conv3 = TtConv2d(self.conv_config_3, device)

        self.downsample = downsample
        if downsample:
            self.downsample_config = Conv2dConfiguration.from_model_args(
                model_args["downsample"]["0"],
                weights=getattr(parameters.downsample, "0", None)["weight"],
                bias=getattr(parameters.downsample, "0", None)["bias"],
                **layer_optimisations.downsample,
                math_fidelity=model_config["MATH_FIDELITY"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                activation_dtype=model_config["ACTIVATIONS_DTYPE"],
            )
            self.downsample_conv = TtConv2d(self.downsample_config, device)

        self.model_config = model_config
        return

    def __call__(
        self,
        x,
        device,
        in_shape,
    ):
        # Convert to DRAM interleaved for DRAM sliced conv's
        if self.layer_optimisations.downsample.get("slice_config", False) or self.layer_optimisations.conv1.get(
            "slice_config", False
        ):
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        # conv1 is 1x1 conv
        out, shape = self.conv1(x, return_output_dim=True)

        # FIXME: PCC drop when persistent L1 buffer is used
        if self.downsample:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        # conv2 is 3x3 conv
        out, shape = self.conv2(out, return_output_dim=True)
        # conv3 is 1x1 conv
        out, shape = self.conv3(out, return_output_dim=True)
        # run downsample conv 1x1 if required
        if self.downsample:
            ds_out, _ = self.downsample_conv(x, return_output_dim=True)
            ttnn.deallocate(x)
            ds_out = ttnn.reallocate(ds_out)
        else:
            ds_out = x

        if ds_out.shape != out.shape:
            ds_out = ttnn.reshape(ds_out, (1, 1, ds_out.shape[0] * ds_out.shape[1] * ds_out.shape[2], ds_out.shape[3]))
        if ds_out.layout != out.layout:
            ds_out = ttnn.to_layout(ds_out, out.layout)
        if ds_out.memory_config() != out.memory_config():
            ds_out = ttnn.to_memory_config(ds_out, out.memory_config())

        out = ttnn.add_(
            out,
            ds_out,
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
        )
        out = ttnn.reallocate(out)
        ttnn.deallocate(ds_out)
        return out, shape
