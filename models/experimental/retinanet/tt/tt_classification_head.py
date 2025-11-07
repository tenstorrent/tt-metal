# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.retinanet.tt.tt_regression_head import Conv2dNormActivation
from loguru import logger
from dataclasses import dataclass


@dataclass
class RetinaNetHeadOptimizer:
    """Optimization configuration for RetinaNet head conv blocks"""

    conv_blocks: dict  # Config for 4 conv blocks
    final_conv: dict  # Config for cls_logits


# Define optimization profiles
retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        conv_blocks={
            # "enable_act_double_buffer": True,
            # "enable_weights_double_buffer": True,
            "deallocate_activation": False,
            "reallocate_halo_output": True,
            # "act_block_h_override": 32,  # NEW: Tune this value
            # "act_block_w_div": 1,  # NEW: Tune this value
        },
        final_conv={
            # "enable_act_double_buffer": True,
            # "enable_weights_double_buffer": True,
            "deallocate_activation": True,
            "reallocate_halo_output": True,
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

    # Get optimization config from the optimizer class
    opt_config = retinanet_head_optimizations[optimization_profile]

    # Create Conv2dConfig for conv blocks using optimizer class settings
    conv_blocks_config = ttnn.Conv2dConfig(**opt_config.conv_blocks)  # Unpack double buffering flags from optimizer

    # Create Conv2dConfig for final cls_logits conv using optimizer class settings
    final_conv_config = ttnn.Conv2dConfig(**opt_config.final_conv)  # Unpack double buffering flags from optimizer
    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Create input mask for GroupNorm
    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=model_config["ACTIVATIONS_DTYPE"],  # was ttnn.DataType.BFLOAT8_B
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
    # Create 4 Conv2dNormActivation blocks
    conv_blocks = []
    for i in range(4):
        conv_block = Conv2dNormActivation(
            parameters=parameters["conv"][i],
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
            conv_config=conv_blocks_config,
        )
        conv_blocks.append(conv_block)

    # Process each FPN level
    all_cls_logits = []
    for level_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, input_shapes)):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing FPN Level {level_idx}: H={H}, W={W}")
        logger.info(f"{'='*60}")

        x = feature_map

        # Apply 4 conv blocks sequentially
        for i, conv_block in enumerate(conv_blocks):
            x = conv_block(
                x, batch_size=batch_size, input_height=H, input_width=W, fpn_level=level_idx, conv_block_idx=i
            )

        # Apply final cls_logits convolution
        # Input: (N, H, W, 256) NHWC
        # Output: (N, H, W, 819) NHWC
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
            conv_config=final_conv_config,
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
