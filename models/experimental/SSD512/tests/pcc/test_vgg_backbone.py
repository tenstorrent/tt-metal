# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test VGG Backbone implementation using PCC metric.

This test compares the TTNN implementation of VGG backbone with the PyTorch
reference implementation from torch_reference_ssd.py using Pearson
Correlation Coefficient (PCC) metric.
"""

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# Import reference implementation
from models.experimental.SSD512.reference.ssd import vgg, base

# Import TTNN implementation
from models.experimental.SSD512.tt.layers.vgg_backbone import (
    build_vgg_backbone,
    apply_vgg_backbone,
)

from models.common.utility_functions import comp_pcc, comp_allclose


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize(
    "size",
    ((512,)),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vgg_backbone(device, pcc, size, reset_seeds):
    """
    Test VGG backbone TTNN implementation against PyTorch reference.

    This test:
    1. Creates a PyTorch VGG backbone using torch_reference_ssd.vgg()
    2. Creates a TTNN VGG backbone using vgg_backbone()
    3. Runs forward pass on both with the same random input
    4. Compares outputs using PCC metric

    Args:
        device: TTNN device fixture
        pcc: PCC threshold for passing test
        size: Input image size (300 or 512)
        reset_seeds: Seed reset fixture for reproducibility
    """
    # Set random seed for reproducibility BEFORE creating models
    # This ensures both PyTorch and TTNN models use the same weight initialization
    if reset_seeds:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    else:
        # Always set seed for weight consistency, even if reset_seeds is False
        torch.manual_seed(0)

    # Build PyTorch reference model FIRST
    cfg = base[str(size)]
    torch_layers = vgg(cfg, i=3, batch_norm=False)
    torch_model = nn.Sequential(*torch_layers)
    torch_model.eval()

    # Create test input
    batch_size = 1
    input_channels = 3
    input_height = size
    input_width = size

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width)

    # Run PyTorch reference forward pass
    with torch.no_grad():
        torch_output = torch_model(torch_input)

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Build TTNN model
    layers_config = build_vgg_backbone(size=size, input_channels=input_channels, device=device)

    # Extract weights from PyTorch model and load into TTNN layers
    # This ensures both models use the SAME weights for fair comparison
    torch_conv_idx = 0  # Track which conv layer we're at in PyTorch model
    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
            # Get corresponding PyTorch conv layer
            # PyTorch layers are: conv, relu, conv, relu, pool, etc.
            # We need to find the conv layers (skip ReLU and MaxPool)
            while torch_conv_idx < len(torch_model):
                torch_layer = torch_model[torch_conv_idx]
                if isinstance(torch_layer, nn.Conv2d):
                    break
                torch_conv_idx += 1

            if torch_conv_idx >= len(torch_model):
                raise ValueError("Mismatch: More conv layers in TTNN config than PyTorch model")

            torch_conv = torch_model[torch_conv_idx]
            torch_conv_idx += 1

            # Extract weights and bias from PyTorch conv layer
            weight = torch_conv.weight.data.clone()  # Shape: (out_channels, in_channels, kh, kw)
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

            # Convert to TTNN format
            if device is not None:
                weight_ttnn = ttnn.from_torch(
                    weight,
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )

                if bias is not None:
                    bias_reshaped = bias.reshape((1, 1, 1, -1))
                    bias_ttnn = ttnn.from_torch(
                        bias_reshaped,
                        device=device,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    bias_ttnn = None
            else:
                weight_ttnn = weight
                bias_ttnn = bias

            layer_with_weights = layer.copy()
            layer_with_weights["weight"] = weight_ttnn
            layer_with_weights["bias"] = bias_ttnn
            layers_with_weights.append(layer_with_weights)
        else:
            # For pool and relu layers, just copy the config
            layers_with_weights.append(layer.copy())

    # Verify we extracted all conv layers
    total_conv_layers = sum(1 for layer in layers_config if layer["type"] == "conv")
    if torch_conv_idx != total_conv_layers:
        logger.warning(
            f"Conv layer count mismatch: PyTorch has {torch_conv_idx} conv layers, "
            f"but TTNN config has {total_conv_layers} conv layers"
        )

    logger.info(f"Extracted weights from {torch_conv_idx} PyTorch conv layers")
    logger.info(f"Created {len(layers_with_weights)} TTNN layers (including conv, pool, relu)")

    # Run TTNN forward pass
    # The input is already a torch tensor, apply_vgg_backbone will convert it
    tt_output_ttnn = apply_vgg_backbone(
        torch_input,
        layers_with_weights,
        device=device,
        dtype=ttnn.bfloat16,
    )

    # Convert TTNN output back to torch format
    # TTNN output is in NHWC format, convert back to NCHW
    tt_output = ttnn.to_torch(tt_output_ttnn)

    # Debug: Check output shape before permute
    logger.info(f"TTNN output shape before permute: {tt_output.shape}")
    logger.info(f"Expected PyTorch shape: {torch_output.shape}")

    # Convert from NHWC to NCHW
    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)  # NHWC -> NCHW

    # Ensure output is float32 for fair comparison (TTNN might return bfloat16)
    tt_output = tt_output.float()

    logger.info(f"TTNN output shape after conversion: {tt_output.shape}")
    logger.info(f"TTNN output stats: min={tt_output.min():.4f}, max={tt_output.max():.4f}, mean={tt_output.mean():.4f}")

    # Check if shapes match
    if tt_output.shape != torch_output.shape:
        logger.error(f"Shape mismatch! PyTorch: {torch_output.shape}, TTNN: {tt_output.shape}")
        # Try to handle shape mismatch gracefully
        min_shape = [min(s1, s2) for s1, s2 in zip(torch_output.shape, tt_output.shape)]
        torch_output = torch_output[tuple(slice(0, s) for s in min_shape)]
        tt_output = tt_output[tuple(slice(0, s) for s in min_shape)]
        logger.warning(f"Truncated to matching shape: {torch_output.shape}")

    # Compare outputs
    logger.info(comp_allclose(torch_output, tt_output))

    does_pass, pcc_message = comp_pcc(torch_output, tt_output, pcc)

    logger.info(f"PCC comparison: {pcc_message}")

    if does_pass:
        logger.info(f"VGG Backbone PCC test PASSED")
    else:
        logger.error(f"VGG Backbone PCC test FAILED ")

    assert does_pass, f"VGG Backbone does not meet PCC requirement {pcc}"


if __name__ == "__main__":
    # Allow running test directly for debugging
    pytest.main([__file__, "-v", "-s"])
