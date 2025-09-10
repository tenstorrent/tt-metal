# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.experimental.panoptic_deeplab.tt.common import TTConv2D
from dataclasses import dataclass


@dataclass
class NeckOptimizer:
    conv1: dict()
    conv2: dict()
    conv3: dict()


neck_optimisations = {
    "optimization_full_tensor": NeckOptimizer(
        conv1={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
        conv2={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
        conv3={
            "act_block_h": 32,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=4),
        },
    ),
    "optimization_small_tensor": NeckOptimizer(
        conv1={
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
            "slice_config": ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceHeight, num_slices=2),
        },
        conv2={
            "act_block_h": 512,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        conv3={
            "act_block_h": 128,
            "shard_layout": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            "memory_config": ttnn.DRAM_MEMORY_CONFIG,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
            "reshard_if_not_optimal": True,
            "enable_split_reader": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


class resnet52Stem:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        layer_optimisations=neck_optimisations["optimization_full_tensor"],
    ) -> None:
        self.conv1 = TTConv2D(
            kernel_size=3,
            stride=2,
            padding=1,
            activation="relu",
            parameters=parameters.conv1,
            kernel_fidelity=model_config,
            **layer_optimisations.conv1,
        )
        self.conv2 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            activation="relu",
            parameters=parameters.conv2,
            kernel_fidelity=model_config,
            **layer_optimisations.conv2,
        )
        self.conv3 = TTConv2D(
            kernel_size=3,
            stride=stride,
            padding=1,
            activation="relu",
            parameters=parameters.conv3,
            kernel_fidelity=model_config,
            **layer_optimisations.conv3,
        )

    def __call__(
        self,
        x,
        device,
    ):
        # conv1 is stride 2 conv 3x3
        logger.debug(f"Running 3x3 conv1")
        out, shape = self.conv1(device, x, x.shape)

        # conv2 and 3 are 3x3 conv's with stride 1
        logger.debug(f"Running 3x3 conv2")
        out, shape = self.conv2(device, out, shape)
        logger.debug(f"Running 3x3 conv3")
        out, shape = self.conv3(device, out, shape)

        out = ttnn.max_pool2d(
            input_tensor=out,
            batch_size=shape[-4],
            input_h=shape[-3],
            input_w=shape[-2],
            channels=shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            in_place_halo=True,
            applied_shard_scheme=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ceil_mode=False,
        )
        out = ttnn.reshape(out, (shape[-4], shape[-3] // 2, shape[-2] // 2, shape[-1]))

        return out
