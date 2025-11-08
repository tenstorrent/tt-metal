# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.retinanet.tt.tt_regression_head import Conv2dNormActivation
from loguru import logger
from dataclasses import dataclass


@dataclass
class RetinaNetClassificationHeadOptimizer:
    """Optimization configuration for RetinaNet classification head conv blocks"""

    fpn0_conv_blocks: dict
    fpn1_conv_blocks: dict
    fpn2_conv_blocks: dict
    fpn3_conv_blocks: dict
    fpn4_conv_blocks: dict

    # Final cls_logits configs (one per FPN level)
    fpn0_final_conv: dict
    fpn1_final_conv: dict
    fpn2_final_conv: dict
    fpn3_final_conv: dict
    fpn4_final_conv: dict


# Define optimization profiles
retinanet_classification_head_optimizations = {
    "optimized": RetinaNetClassificationHeadOptimizer(
        # FPN Level 0: Largest spatial (64x64)
        fpn0_conv_blocks={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn0_final_conv={
            "act_block_h_override": 1024,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 1: Medium-large spatial (32x32)
        fpn1_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn1_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 2: Medium spatial (16x16)
        fpn2_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn2_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 3: Small spatial (8x8)
        fpn3_conv_blocks={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn3_final_conv={
            "act_block_h_override": 256,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        # FPN Level 4: Smallest spatial (4x4)
        fpn4_conv_blocks={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
            "enable_act_double_buffer": True,
            "enable_weights_double_buffer": True,
        },
        fpn4_final_conv={
            "act_block_h_override": 32,
            "shard_layout": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "reshard_if_not_optimal": True,
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
    """
    TTNN implementation of RetinaNet classification head.

    Args:
        feature_maps: List of FPN feature tensors in NHWC format
        parameters: Dict containing 'conv' list and 'cls_logits' parameters
        device: TTNN device
        in_channels: Number of input channels (256 for RetinaNet)
        num_anchors: Number of anchors per location (9 for RetinaNet)
        num_classes: Number of classes (91 for COCO)
        batch_size: Batch size
        input_shapes: List of (H, W) tuples for each FPN level

    Returns:
        Output tensor of shape (N, total_anchors, num_classes)
    """
    if input_shapes is None:
        input_shapes = [(fm.shape[1], fm.shape[2]) for fm in feature_maps]

    # Get optimization config
    opt_config = retinanet_classification_head_optimizations[optimization_profile]

    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Create input mask for GroupNorm
    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=model_config["ACTIVATIONS_DTYPE"],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

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
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing FPN Level {fpn_idx}: H={H}, W={W}")
        logger.info(f"{'='*60}")

        # Get per-FPN configurations
        fpn_conv_config_dict = getattr(opt_config, f"fpn{fpn_idx}_conv_blocks")
        fpn_final_config_dict = getattr(opt_config, f"fpn{fpn_idx}_final_conv")

        # Create Conv2dConfig objects for this FPN level
        fpn_conv_config = ttnn.Conv2dConfig(**fpn_conv_config_dict) if fpn_conv_config_dict else None
        fpn_final_config = ttnn.Conv2dConfig(**fpn_final_config_dict) if fpn_final_config_dict else None

        # Log applied configurations
        if fpn_conv_config:
            logger.info(
                f"[FPN{fpn_idx}] ✓ Conv blocks config: act_block_h={fpn_conv_config_dict.get('act_block_h_override', 'auto')}, shard={fpn_conv_config_dict.get('shard_layout', 'auto')}"
            )
        else:
            logger.info(f"[FPN{fpn_idx}] ⚠ Conv blocks config: None (using defaults)")

        if fpn_final_config:
            logger.info(
                f"[FPN{fpn_idx}] ✓ Final conv config: act_block_h={fpn_final_config_dict.get('act_block_h_override', 'auto')}, shard={fpn_final_config_dict.get('shard_layout', 'auto')}"
            )
        else:
            logger.info(f"[FPN{fpn_idx}] ⚠ Final conv config: None (using defaults)")

        # Create 4 Conv2dNormActivation blocks for this FPN level
        x = feature_map
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
                input_mask=input_mask_tensor,
                model_config=model_config,
                compute_config=compute_config,
                conv_config=fpn_conv_config,
            )
            x = conv_block(
                x, batch_size=batch_size, input_height=H, input_width=W, fpn_level=fpn_idx, conv_block_idx=conv_idx
            )

        # Apply final cls_logits convolution
        cls_logits = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=parameters["cls_logits"]["weight"],
            bias_tensor=parameters["cls_logits"]["bias"],
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
        )
        logger.info(f"After cls_logits conv: {cls_logits.shape}")

        # Reshape from (N, H, W, 819) to (N, H, W, 9, 91)
        N, H_out, W_out, _ = cls_logits.shape
        cls_logits = ttnn.reshape(cls_logits, (N, H_out, W_out, num_anchors, num_classes))
        logger.info(f"After reshape to (N, H, W, A, K): {cls_logits.shape}")

        # Flatten to (N, H*W*A, K)
        cls_logits = ttnn.reshape(cls_logits, (N, H_out * W_out * num_anchors, num_classes))
        logger.info(f"After flatten to (N, HWA, K): {cls_logits.shape}")

        all_cls_logits.append(cls_logits)

    # Concatenate all FPN levels
    output = ttnn.concat(all_cls_logits, dim=1)
    logger.info(f"\nFinal output shape: {output.shape}")

    return output
