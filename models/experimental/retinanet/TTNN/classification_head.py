# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.retinanet.TTNN.regression_head import Conv2dNormActivation
from loguru import logger


def ttnn_retinanet_classification_head(
    feature_maps: List[ttnn.Tensor],
    parameters: dict,
    device: ttnn.Device,
    in_channels: int = 256,
    num_anchors: int = 9,
    num_classes: int = 91,
    batch_size: int = 1,
    input_shapes: List[tuple] = None,
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

    # Grid size for GroupNorm
    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Create input mask for GroupNorm
    input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, grid_size.y)
    input_mask_tensor = ttnn.from_torch(
        input_mask_tensor,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        )
        conv_blocks.append(conv_block)

    # Process each FPN level
    all_cls_logits = []
    for level_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, input_shapes)):
        logger.info(f"\n--- Processing FPN Level {level_idx} ---")
        logger.info(f"Input shape: {feature_map.shape}, H={H}, W={W}")

        x = feature_map

        # Apply 4 conv blocks sequentially
        for i, conv_block in enumerate(conv_blocks):
            x = conv_block(x, batch_size=batch_size, input_height=H, input_width=W)
            logger.info(f"After conv block {i}: {x.shape}")

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
