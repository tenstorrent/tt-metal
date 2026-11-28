# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.retinanet.tt.utils import (
    Conv2dNormActivation,
    activation_memory_config,
    conv2d_with_level_weights,
    cls_logits_slice_config,
    group_norm_grid,
    head_conv_config,
)
from dataclasses import dataclass


@dataclass
class RetinaNetClassificationHeadOptimizer:
    """Optimization configuration for RetinaNet classification head conv blocks"""

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


# Define optimization configurations
retinanet_classification_head_optimizations = {
    "optimized": RetinaNetClassificationHeadOptimizer(
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


def ttnn_retinanet_classification_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    num_classes: int = 91,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
    model_config: dict = None,
    optimization_profile: str = "optimized",
) -> ttnn.Tensor:
    """The classification head on ``feature_maps``, the five pyramid levels as NHWC TILE tensors.
    ``parameters`` come from preprocess_classification_head_parameters for the same device; the
    head runs as many conv blocks as they hold. Returns the class logits, [1, anchors, 91]."""
    if input_shapes is None:
        input_shapes = [(fm.shape[1], fm.shape[2]) for fm in feature_maps]

    opt_config = retinanet_classification_head_optimizations[optimization_profile]

    grid_size = group_norm_grid(device)

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
        math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
        fp32_dest_acc_en=model_config.get("FP32_DEST_ACC_EN", True),
        packer_l1_acc=model_config.get("PACKER_L1_ACC", False),
    )

    # Process each FPN level
    all_cls_logits = []
    for fpn_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, input_shapes)):
        fpn_conv_config_dict = getattr(opt_config, f"fpn{fpn_idx}_conv_blocks")
        fpn_final_config_dict = getattr(opt_config, f"fpn{fpn_idx}_final_conv")

        fpn_conv_config = head_conv_config(device, fpn_conv_config_dict)
        fpn_final_config = head_conv_config(device, fpn_final_config_dict)

        x = feature_map
        for conv_idx in range(len(parameters["conv"])):
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
                compute_config=compute_config,
                conv_config=fpn_conv_config,
            )
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            x = conv_block(x, batch_size=batch_size, input_height=H, input_width=W, level=fpn_idx)

        cls_logits, _, _ = conv2d_with_level_weights(
            parameters["cls_logits"],
            fpn_idx,
            input_tensor=x,
            in_channels=in_channels,
            out_channels=num_anchors * num_classes,
            device=device,
            kernel_size=[3, 3],
            stride=[1, 1],
            padding=[1, 1],
            batch_size=batch_size,
            input_height=H,
            input_width=W,
            compute_config=compute_config,
            conv_config=fpn_final_config,
            memory_config=activation_memory_config(device),
            slice_config=cls_logits_slice_config(device),
        )

        N, H_out, W_out, _ = cls_logits.shape
        cls_logits = ttnn.to_memory_config(cls_logits, ttnn.DRAM_MEMORY_CONFIG)
        cls_logits = ttnn.reshape(cls_logits, (N, H_out, W_out, num_anchors, num_classes))

        cls_logits = ttnn.reshape(cls_logits, (N, H_out * W_out * num_anchors, num_classes))

        all_cls_logits.append(cls_logits)

    output = ttnn.concat(all_cls_logits, dim=1)
    return output
