# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ResOptimizer:
    conv1: Dict[Any, Any]
    conv2: Dict[Any, Any]
    conv3: Dict[Any, Any]
    shape: tuple


res_layer_optimisations = {
    "default": ResOptimizer(
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
        shape=(0, 0, 0, 0),
    ),
    "instance_Res3": ResOptimizer(
        conv1={
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
        },
        conv2={
            "act_block_h": 512,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "reshard_if_not_optimal": True,
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        shape=(1, 64, 128, 512),
    ),
    "instance_Res2": ResOptimizer(
        conv1={
            "act_block_h": 128,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv2={
            "act_block_h": 32,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv3={
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        shape=(1, 128, 256, 256),
    ),
    "semantics_Res3": ResOptimizer(
        conv1={
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv2={
            "act_block_h": 512,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "deallocate_activation": True,
            "enable_split_reader": True,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        shape=(1, 64, 128, 512),
    ),
    "semantics_Res2": ResOptimizer(
        conv1={
            "act_block_h": 32,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        conv2={
            "act_block_h": 160,
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
        },
        shape=(1, 128, 256, 256),
    ),
}


class TTRes:
    def __init__(
        self,
        parameters,
        model_config,
        layer_optimisations=res_layer_optimisations["default"],
    ) -> None:
        # conv1_upsample
        self.conv1_upsample = TTUpsample(
            scale_factor=(2),
            mode="bilinear",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # conv1
        self.conv1 = TTConv2D(
            kernel_size=parameters.conv_args["conv1"]["0"].kernel_size,
            stride=parameters.conv_args["conv1"]["0"].stride,
            padding=parameters.conv_args["conv1"]["0"].padding,
            dilation=parameters.conv_args["conv1"]["0"].dilation,
            groups=parameters.conv_args["conv1"]["0"].groups,
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            activation="relu",
            **layer_optimisations.conv1,
        )
        # conv2
        self.conv2 = TTConv2D(
            kernel_size=parameters.conv_args["conv2"]["0"].kernel_size,
            stride=parameters.conv_args["conv2"]["0"].stride,
            padding=parameters.conv_args["conv2"]["0"].padding,
            groups=parameters.conv_args["conv2"]["0"].groups,
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            activation="relu",
            **layer_optimisations.conv2,
        )
        # conv3
        self.conv3 = TTConv2D(
            kernel_size=parameters.conv_args["conv3"]["0"].kernel_size,
            stride=parameters.conv_args["conv3"]["0"].stride,
            padding=parameters.conv_args["conv3"]["0"].padding,
            groups=parameters.conv_args["conv3"]["0"].groups,
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            activation="relu",
            **layer_optimisations.conv3,
        )
        self.shape = layer_optimisations.shape

    def __call__(
        self,
        x,
        res,
        upsample_channels,
        device,
    ):
        # Decoder: upsample and fuse with res3
        logger.debug("Running upsample after ASPP project")
        shape = [self.shape[-4], self.shape[-3] // 2, self.shape[-2] // 2, upsample_channels]

        output = self.conv1_upsample(device, x, shape, sent_to_dram=True, reshape_output=True)

        logger.debug("Running conv1")
        output_res, shape = self.conv1(device, res, self.shape)

        logger.debug("Running concat for res and ASPP upsampled")
        output = ttnn.concat([output_res, output], dim=3)

        logger.debug("Running conv2")
        shape = (self.shape[-4], self.shape[-3], self.shape[-2], upsample_channels + shape[-1])
        output, shape = self.conv2(device, output, shape)

        logger.debug("Running conv3")

        output, shape = self.conv3(device, output, shape)
        return output
