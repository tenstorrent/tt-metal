# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.retinanet.tt.utils import TTConv2D
from dataclasses import dataclass
from typing import Optional


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
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "reshard_if_not_optimal": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
        },
        downsample={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "slice_config": ttnn.Conv2dL1FullSliceConfig,
            # "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2),
        },
    ),
    "layer2": BottleneckOptimizer(
        conv1={
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "reshard_if_not_optimal": True,
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
        },
        downsample={
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2),
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
    "layer3": BottleneckOptimizer(
        conv1={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "reshard_if_not_optimal": True,
        },
        conv2={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
        },
        downsample={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
    "layer4": BottleneckOptimizer(
        conv1={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "reshard_if_not_optimal": True,
            "dtype": ttnn.bfloat16,
        },
        conv2={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reshard_if_not_optimal": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "dtype": ttnn.bfloat16,
        },
        conv3={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reshard_if_not_optimal": True,
        },
        downsample={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "reshard_if_not_optimal": True,
            "deallocate_activation": True,
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
        downsample,
        stride,
        model_config,
        dilation: int = 1,
        name: Optional[str] = "bottleneck",
        layer_optimisations=bottleneck_layer_optimisations["default"],
    ) -> None:
        self.name = name
        self.layer_optimisations = layer_optimisations

        self.conv1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_reshape=True,
            **layer_optimisations.conv1,
        )

        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride if downsample else 1,
            padding=dilation,
            dilation=dilation,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_reshape=True,
            **layer_optimisations.conv2,
        )

        self.conv3 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            activation=None,
            is_reshape=True,
            **layer_optimisations.conv3,
        )

        self.downsample = downsample
        if downsample:
            self.downsample_conv = TTConv2D(
                kernel_size=1,
                stride=stride,
                padding=0,
                dilation=1,
                # parameters=parameters.downsample,
                parameters=getattr(parameters.downsample, "0", None),
                kernel_fidelity=model_config,
                activation=None,
                is_reshape=True,
                **layer_optimisations.downsample,
            )

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

        # conv1 is 1x1 conv
        out, shape = self.conv1(device, x, in_shape)

        # FIXME: PCC drop when persistent L1 buffer is used
        if self.downsample:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        # conv2 is 3x3 conv
        out, shape = self.conv2(device, out, shape)
        # conv3 is 1x1 conv
        out, shape = self.conv3(device, out, shape)
        # run downsample conv 1x1 if required
        if self.downsample:
            ds_out, _ = self.downsample_conv(device, x, in_shape)
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
