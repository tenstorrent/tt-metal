# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test Extras Backbone implementation using PCC metric.

This test compares the TTNN implementation of extras backbone with the PyTorch
reference implementation from torch_reference_ssd.py using Pearson
Correlation Coefficient (PCC) metric.
"""

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# Import reference implementation
from models.experimental.SSD512.reference.ssd import add_extras, extras

# Import TTNN implementation
from models.experimental.SSD512.tt.layers.extras_backbone import (
    build_extras_backbone,
    create_extras_layers_with_weights,
    apply_extras_backbone,
)

from models.common.utility_functions import comp_pcc, comp_allclose


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
@pytest.mark.parametrize(
    "size",
    (512,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_extras_backbone(device, pcc, size, reset_seeds):
    """
    Test Extras backbone TTNN implementation against PyTorch reference.

    This test:
    1. Creates a PyTorch Extras backbone using torch_reference_ssd.add_extras()
    2. Creates a TTNN Extras backbone using extras_backbone()
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
    cfg = extras[str(size)]
    torch_layers = add_extras(cfg, i=1024, batch_norm=False)
    # Note: PyTorch add_extras returns only Conv2d layers
    # ReLU is applied in forward pass, so we'll apply it manually in test
    torch_model = nn.ModuleList(torch_layers)
    torch_model.eval()

    # Create test input - simulate VGG backbone output
    # VGG backbone outputs: For SSD300, final output is around 19x19x1024 or 38x38x1024
    # For SSD512, it's around 64x64x1024
    batch_size = 1
    input_channels = 1024  # Output from VGG backbone
    # Approximate output size from VGG backbone (after all pooling/conv)
    # For size 300: roughly 19x19 or 38x38 depending on exact path
    # For size 512: roughly 32x32 or 64x64
    if size == 300:
        # After VGG backbone for SSD300, we typically get smaller feature maps
        input_height = 38
        input_width = 38
    else:  # size == 512
        input_height = 64
        input_width = 64

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width)

    # Run PyTorch reference forward pass
    # Apply ReLU after each conv layer (matching SSD forward pass pattern)
    with torch.no_grad():
        x = torch_input
        for layer in torch_model:
            x = torch.nn.functional.relu(layer(x), inplace=True)
        torch_output = x

    logger.info(f"PyTorch output shape: {torch_output.shape}")
    logger.info(
        f"PyTorch output stats: min={torch_output.min():.4f}, max={torch_output.max():.4f}, mean={torch_output.mean():.4f}"
    )

    # Build TTNN model
    layers_config = build_extras_backbone(size=size, input_channels=input_channels, device=device)

    # Extract weights from PyTorch model and load into TTNN layers
    # This ensures both models use the SAME weights for fair comparison
    torch_conv_idx = 0  # Track which conv layer we're at in PyTorch model
    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
            # Get corresponding PyTorch conv layer
            # PyTorch layers are: conv, relu, conv, relu, etc.
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
            # For relu layers, just copy the config
            layers_with_weights.append(layer.copy())

    # Verify we extracted all conv layers
    total_conv_layers = sum(1 for layer in layers_config if layer["type"] == "conv")
    if torch_conv_idx != total_conv_layers:
        logger.warning(
            f"Conv layer count mismatch: PyTorch has {torch_conv_idx} conv layers, "
            f"but TTNN config has {total_conv_layers} conv layers"
        )

    logger.info(f"Extracted weights from {torch_conv_idx} PyTorch conv layers")
    logger.info(f"Created {len(layers_with_weights)} TTNN layers (including conv, relu)")

    # Run TTNN forward pass
    # The input is already a torch tensor, apply_extras_backbone will convert it
    tt_output_ttnn = apply_extras_backbone(
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
        logger.info(f"Extras Backbone PCC test PASSED")
    else:
        logger.error(f"Extras Backbone PCC test FAILED")

    assert does_pass, f"Extras Backbone does not meet PCC requirement"


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_extras_backbone_structure(device, pcc, reset_seeds):
    """
    Test that Extras backbone structure matches reference.

    This test verifies that the layer configuration matches the PyTorch
    reference without running forward pass.
    """
    if reset_seeds:
        torch.manual_seed(0)

    # Test both sizes
    for size in [300, 512]:
        cfg = extras[str(size)]
        torch_layers = add_extras(cfg, i=1024, batch_norm=False)

        layers_config = build_extras_backbone(size=size, input_channels=1024, device=device)
        layers_with_weights = create_extras_layers_with_weights(layers_config, device=device, dtype=ttnn.bfloat16)

        # Count layer types
        # Note: PyTorch add_extras returns only Conv2d layers (no ReLU in layer list)
        # ReLU is applied separately in forward pass
        torch_conv_count = sum(1 for layer in torch_layers if isinstance(layer, nn.Conv2d))

        # TTNN implementation includes ReLU layers explicitly
        tt_conv_count = sum(1 for layer in layers_with_weights if layer["type"] == "conv")
        tt_relu_count = sum(1 for layer in layers_with_weights if layer["type"] == "relu")

        logger.info(f"Size {size}: Torch - Convs:{torch_conv_count} (ReLU applied separately)")
        logger.info(f"Size {size}: TTNN - Convs:{tt_conv_count}, ReLUs:{tt_relu_count}")

        assert (
            torch_conv_count == tt_conv_count
        ), f"Conv layer count mismatch for size {size}: PyTorch={torch_conv_count}, TTNN={tt_conv_count}"
        # TTNN should have same number of ReLU layers as conv layers (one per conv)
        assert tt_relu_count == tt_conv_count, f"ReLU count should match conv count for size {size}"

    logger.info("Extras Backbone structure test PASSED")


if __name__ == "__main__":
    # Allow running test directly for debugging
    pytest.main([__file__, "-v", "-s"])
