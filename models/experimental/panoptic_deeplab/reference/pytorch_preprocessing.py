# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from loguru import logger


def fuse_imagenet_normalization(model):
    """
    Fuse ImageNet normalization constants into stem.conv1 weights of a PyTorch model.
    This is the single point where normalization fusion happens.

    Args:
        model: PyTorch PanopticDeepLab model (or any model with backbone.stem.conv1)

    Returns:
        bool: True if fusion was successful, False otherwise
    """
    if not (hasattr(model, "backbone") and hasattr(model.backbone, "stem") and hasattr(model.backbone.stem, "conv1")):
        logger.warning("Could not find backbone.stem.conv1 to fuse normalization")
        return False

    conv1 = model.backbone.stem.conv1
    conv_weight = conv1.weight.data.clone()
    conv_bias = conv1.bias.data.clone() if conv1.bias is not None else None

    logger.info("Fusing ImageNet normalization into PyTorch backbone.stem.conv1...")

    # ImageNet normalization constants (RGB channels)
    mean = torch.tensor([0.485, 0.456, 0.406], device=conv_weight.device, dtype=conv_weight.dtype)
    std = torch.tensor([0.229, 0.224, 0.225], device=conv_weight.device, dtype=conv_weight.dtype)

    # Weight shape: [out_channels, in_channels, kernel_h, kernel_w]
    out_channels, in_channels, kernel_h, kernel_w = conv_weight.shape
    assert in_channels == 3, f"Expected 3 input channels (RGB), got {in_channels}"

    # Store original weight for bias calculation (before scaling)
    weight_original = conv_weight.clone()

    # Scale weights: for each input channel, divide by std[channel]
    for channel_idx in range(3):
        conv_weight[:, channel_idx, :, :] = conv_weight[:, channel_idx, :, :] / std[channel_idx]

    # Adjust bias: new_bias = old_bias - sum_over_input_channels (W_original * mean / std)
    if conv_bias is None or conv_bias.numel() == 0:
        conv_bias = torch.zeros(out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    else:
        conv_bias = conv_bias.to(device=conv_weight.device, dtype=conv_weight.dtype)

    bias_adjustment = torch.zeros(out_channels, device=conv_weight.device, dtype=conv_weight.dtype)
    for out_channel in range(out_channels):
        for in_channel in range(3):
            # Sum over spatial dimensions: mean[in] / std[in] * sum(W_original[out, in, :, :])
            weight_sum = weight_original[out_channel, in_channel, :, :].sum()
            bias_adjustment[out_channel] += weight_sum * mean[in_channel] / std[in_channel]

    fused_bias = conv_bias - bias_adjustment

    # Update the layer weights (note: conv1 originally has bias=False, but we need to add bias)
    conv1.weight.data = conv_weight
    if conv1.bias is None:
        # Add bias parameter if it doesn't exist
        conv1.bias = nn.Parameter(fused_bias)
    else:
        conv1.bias.data = fused_bias

    logger.info(
        f"ImageNet normalization fusion completed: scaled weights per input channel, adjusted bias for {out_channels} output channels"
    )
    return True
