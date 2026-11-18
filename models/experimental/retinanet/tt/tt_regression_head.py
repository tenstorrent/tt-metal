# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.retinanet.tt.utils import BBOX_REG_SLICE_CONFIG, Conv2dNormActivation, group_norm_grid
from typing import List
from dataclasses import dataclass


@dataclass
class RetinaNetHeadOptimizer:
    """Optimization configuration for RetinaNet head conv blocks"""

    fpn0_conv_blocks: dict
    fpn1_conv_blocks: dict
    fpn2_conv_blocks: dict
    fpn3_conv_blocks: dict
    fpn4_conv_blocks: dict

    fpn0_final_conv: dict
    fpn1_final_conv: dict
    fpn2_final_conv: dict
    fpn3_final_conv: dict
    fpn4_final_conv: dict


retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        fpn0_conv_blocks={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn0_final_conv={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_conv_blocks={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_final_conv={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_conv_blocks={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_final_conv={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_conv_blocks={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_final_conv={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_conv_blocks={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_final_conv={
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
    ),
}


# Usage in ttnn_retinanet_regression_head
def ttnn_retinanet_regression_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
    model_config: dict = None,
    optimization_profile: str = "optimized",
) -> ttnn.Tensor:
    opt_config = retinanet_head_optimizations[optimization_profile]

    fpn_conv_configs = [
        opt_config.fpn0_conv_blocks,
        opt_config.fpn1_conv_blocks,
        opt_config.fpn2_conv_blocks,
        opt_config.fpn3_conv_blocks,
        opt_config.fpn4_conv_blocks,
    ]

    fpn_final_configs = [
        opt_config.fpn0_final_conv,
        opt_config.fpn1_final_conv,
        opt_config.fpn2_final_conv,
        opt_config.fpn3_final_conv,
        opt_config.fpn4_final_conv,
    ]

    all_bbox_regression = []

    grid_size = group_norm_grid(device)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
        math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
        fp32_dest_acc_en=model_config.get("FP32_DEST_ACC_EN", True),
        packer_l1_acc=model_config.get("PACKER_L1_ACC", False),
    )

    for fpn_idx, feature_map in enumerate(feature_maps):
        N, H, W, C = feature_map.shape

        conv_blocks_config = fpn_conv_configs[fpn_idx]
        final_conv_config = fpn_final_configs[fpn_idx]

        if conv_blocks_config is not None:
            conv_config = ttnn.Conv2dConfig(**conv_blocks_config)
        else:
            conv_config = None

        conv_blocks = []
        for conv_idx in range(4):
            conv_block = Conv2dNormActivation(
                parameters=parameters["conv"][conv_idx],
                device=device,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                num_groups=32,
                grid_size=grid_size,
                model_config=model_config,
                compute_config=compute_config,
                conv_config=conv_config,
            )
            conv_blocks.append(conv_block)

        # Apply conv blocks to feature map
        x = feature_map
        for conv_idx, conv_block in enumerate(conv_blocks):
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = conv_block(x, batch_size=batch_size, input_height=H, input_width=W)

        # Final bbox_reg conv layer with config
        if final_conv_config is not None:
            bbox_reg_config = ttnn.Conv2dConfig(**final_conv_config)
        else:
            bbox_reg_config = None

        bbox_regression = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=parameters["bbox_reg"]["weight"],
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            device=device,
            bias_tensor=parameters["bbox_reg"]["bias"],
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            batch_size=batch_size,
            input_height=H,
            input_width=W,
            slice_config=BBOX_REG_SLICE_CONFIG,
            compute_config=compute_config,
            conv_config=bbox_reg_config,
        )

        # Reshape to (N, H*W*num_anchors, 4)
        N, H_final, W_final, C_final = bbox_regression.shape
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final, W_final, num_anchors, 4))
        bbox_regression = ttnn.reshape(bbox_regression, (N, H_final * W_final * num_anchors, 4))

        all_bbox_regression.append(bbox_regression)

    output = ttnn.concat(all_bbox_regression, dim=1)
    return output
