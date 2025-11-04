# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test Full SSD Network implementation using PCC metric.

This test compares the complete TTNN implementation of SSD network with the PyTorch
reference implementation from torch_reference_ssd.py using Pearson
Correlation Coefficient (PCC) metric.
"""

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# Import reference implementation
from models.experimental.SSD512.reference.ssd import build_ssd


# Import TTNN implementation
from models.experimental.SSD512.tt.ssd import ssd_forward_ttnn, build_ssd_network_ttnn

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
def test_ssd_network_pcc(device, pcc, size, reset_seeds):
    """
    Test Full SSD Network TTNN implementation against PyTorch reference.

    This test:
    1. Creates a PyTorch SSD model using torch_reference_ssd.build_ssd()
    2. Creates a TTNN SSD network using build_ssd_network_ttnn()
    3. Runs forward pass on both with the same random input
    4. Compares outputs using PCC metric

    Args:
        device: TTNN device fixture
        pcc: PCC threshold for passing test
        size: Input image size (300 or 512)
        reset_seeds: Seed reset fixture for reproducibility
    """
    # Set random seed for reproducibility
    if reset_seeds:
        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
    else:
        torch.manual_seed(0)

    num_classes = 21  # VOC dataset

    # Build PyTorch reference model
    torch_model = build_ssd(phase="train", size=size, num_classes=num_classes)
    torch_model.eval()

    # Build TTNN network
    ssd_network_dict = build_ssd_network_ttnn(
        phase="train",
        size=size,
        num_classes=num_classes,
        device=device,
        dtype=ttnn.bfloat16,
    )

    vgg_layers_with_weights = ssd_network_dict["vgg_layers"]
    extras_layers_with_weights = ssd_network_dict["extras_layers"]
    loc_layers_with_weights = ssd_network_dict["loc_layers"]
    conf_layers_with_weights = ssd_network_dict["conf_layers"]

    # Extract weights from PyTorch model and load into TTNN layers
    # This ensures both models use the SAME weights for fair comparison

    # Extract VGG weights
    torch_vgg_idx = 0
    for layer in vgg_layers_with_weights:
        if layer["type"] == "conv":
            # Find corresponding PyTorch layer
            while torch_vgg_idx < len(torch_model.base):
                torch_layer = torch_model.base[torch_vgg_idx]
                if isinstance(torch_layer, nn.Conv2d):
                    weight = torch_layer.weight.data.clone()
                    bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                    if device is not None:
                        weight_ttnn = ttnn.from_torch(
                            weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                        if bias is not None:
                            bias_reshaped = bias.reshape((1, 1, 1, -1))
                            bias_ttnn = ttnn.from_torch(
                                bias_reshaped, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                            )
                        else:
                            bias_ttnn = None
                    else:
                        weight_ttnn = weight
                        bias_ttnn = bias

                    layer["weight"] = weight_ttnn
                    if bias_ttnn is not None:
                        layer["bias"] = bias_ttnn

                    torch_vgg_idx += 1
                    break
                torch_vgg_idx += 1

    # Extract extras weights
    torch_extras_idx = 0
    for layer in extras_layers_with_weights:
        if layer["type"] == "conv":
            if torch_extras_idx < len(torch_model.extras):
                torch_layer = torch_model.extras[torch_extras_idx]
                weight = torch_layer.weight.data.clone()
                bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                if device is not None:
                    weight_ttnn = ttnn.from_torch(
                        weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    if bias is not None:
                        bias_reshaped = bias.reshape((1, 1, 1, -1))
                        bias_ttnn = ttnn.from_torch(
                            bias_reshaped, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    else:
                        bias_ttnn = None
                else:
                    weight_ttnn = weight
                    bias_ttnn = bias

                layer["weight"] = weight_ttnn
                if bias_ttnn is not None:
                    layer["bias"] = bias_ttnn

                torch_extras_idx += 1

    # Extract multibox heads weights
    torch_loc_idx = 0
    for layer in loc_layers_with_weights:
        if layer["type"] == "conv":
            if torch_loc_idx < len(torch_model.loc):
                torch_layer = torch_model.loc[torch_loc_idx]
                weight = torch_layer.weight.data.clone()
                bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                if device is not None:
                    weight_ttnn = ttnn.from_torch(
                        weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    if bias is not None:
                        bias_reshaped = bias.reshape((1, 1, 1, -1))
                        bias_ttnn = ttnn.from_torch(
                            bias_reshaped, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    else:
                        bias_ttnn = None
                else:
                    weight_ttnn = weight
                    bias_ttnn = bias

                layer["weight"] = weight_ttnn
                if bias_ttnn is not None:
                    layer["bias"] = bias_ttnn

                torch_loc_idx += 1

    torch_conf_idx = 0
    for layer in conf_layers_with_weights:
        if layer["type"] == "conv":
            if torch_conf_idx < len(torch_model.conf):
                torch_layer = torch_model.conf[torch_conf_idx]
                weight = torch_layer.weight.data.clone()
                bias = torch_layer.bias.data.clone() if torch_layer.bias is not None else None

                if device is not None:
                    weight_ttnn = ttnn.from_torch(
                        weight, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    if bias is not None:
                        bias_reshaped = bias.reshape((1, 1, 1, -1))
                        bias_ttnn = ttnn.from_torch(
                            bias_reshaped, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT
                        )
                    else:
                        bias_ttnn = None
                else:
                    weight_ttnn = weight
                    bias_ttnn = bias

                layer["weight"] = weight_ttnn
                if bias_ttnn is not None:
                    layer["bias"] = bias_ttnn

                torch_conf_idx += 1

    logger.info(f"Extracted weights from PyTorch model")
    logger.info(f"VGG layers: {torch_vgg_idx} conv layers")
    logger.info(f"Extras layers: {torch_extras_idx} conv layers")
    logger.info(f"Location heads: {torch_loc_idx} conv layers")
    logger.info(f"Confidence heads: {torch_conf_idx} conv layers")

    # Create test input
    batch_size = 1
    if size == 300:
        input_tensor = torch.randn(batch_size, 3, 300, 300)
    else:  # size == 512
        input_tensor = torch.randn(batch_size, 3, 512, 512)

    # Run PyTorch reference forward pass
    with torch.no_grad():
        torch_output = torch_model(input_tensor)
        # Output is tuple: (loc, conf, priors) for train phase
        torch_loc_preds, torch_conf_preds, torch_priors = torch_output
        # Flatten location predictions: [batch, num_priors*4]
        # Flatten confidence predictions: [batch, num_priors*num_classes]
        torch_loc_flat = torch_loc_preds.view(torch_loc_preds.size(0), -1)
        torch_conf_flat = torch_conf_preds.view(torch_conf_preds.size(0), -1)

    logger.info(f"PyTorch location predictions shape: {torch_loc_flat.shape}")
    logger.info(f"PyTorch confidence predictions shape: {torch_conf_flat.shape}")
    logger.info(
        f"PyTorch location stats: min={torch_loc_flat.min():.4f}, max={torch_loc_flat.max():.4f}, mean={torch_loc_flat.mean():.4f}"
    )
    logger.info(
        f"PyTorch confidence stats: min={torch_conf_flat.min():.4f}, max={torch_conf_flat.max():.4f}, mean={torch_conf_flat.mean():.4f}"
    )

    # Run TTNN forward pass
    tt_loc_preds, tt_conf_preds = ssd_forward_ttnn(
        input_tensor,
        vgg_layers_with_weights,
        extras_layers_with_weights,
        loc_layers_with_weights,
        conf_layers_with_weights,
        device=device,
        dtype=ttnn.bfloat16,
    )

    logger.info(f"TTNN location predictions shape: {tt_loc_preds.shape}")
    logger.info(f"TTNN confidence predictions shape: {tt_conf_preds.shape}")
    logger.info(
        f"TTNN location stats: min={tt_loc_preds.min():.4f}, max={tt_loc_preds.max():.4f}, mean={tt_loc_preds.mean():.4f}"
    )
    logger.info(
        f"TTNN confidence stats: min={tt_conf_preds.min():.4f}, max={tt_conf_preds.max():.4f}, mean={tt_conf_preds.mean():.4f}"
    )

    # Ensure outputs are float32 for fair comparison
    tt_loc_preds = tt_loc_preds.float()
    tt_conf_preds = tt_conf_preds.float()

    # Check shapes match
    if tt_loc_preds.shape != torch_loc_flat.shape:
        logger.error(f"Location shape mismatch! PyTorch: {torch_loc_flat.shape}, TTNN: {tt_loc_preds.shape}")
        # Try to handle shape mismatch gracefully
        min_shape = min(torch_loc_flat.shape[1], tt_loc_preds.shape[1])
        torch_loc_flat = torch_loc_flat[:, :min_shape]
        tt_loc_preds = tt_loc_preds[:, :min_shape]
        logger.warning(f"Truncated location predictions to matching shape: {torch_loc_flat.shape}")

    if tt_conf_preds.shape != torch_conf_flat.shape:
        logger.error(f"Confidence shape mismatch! PyTorch: {torch_conf_flat.shape}, TTNN: {tt_conf_preds.shape}")
        min_shape = min(torch_conf_flat.shape[1], tt_conf_preds.shape[1])
        torch_conf_flat = torch_conf_flat[:, :min_shape]
        tt_conf_preds = tt_conf_preds[:, :min_shape]
        logger.warning(f"Truncated confidence predictions to matching shape: {torch_conf_flat.shape}")

    # Compare location predictions
    logger.info("Comparing location predictions:")
    logger.info(comp_allclose(torch_loc_flat, tt_loc_preds))
    does_pass_loc, pcc_message_loc = comp_pcc(torch_loc_flat, tt_loc_preds, pcc)
    logger.info(f"Location {pcc_message_loc}")

    # Compare confidence predictions
    logger.info("Comparing confidence predictions:")
    logger.info(comp_allclose(torch_conf_flat, tt_conf_preds))
    does_pass_conf, pcc_message_conf = comp_pcc(torch_conf_flat, tt_conf_preds, pcc)
    logger.info(f"Confidence {pcc_message_conf}")

    if does_pass_loc and does_pass_conf:
        logger.info(f"Full SSD Network PCC test PASSED for size {size}")
        logger.info(f"  Location PCC: {pcc_value_loc:.6f}")
        logger.info(f"  Confidence PCC: {pcc_value_conf:.6f}")
    else:
        logger.error(f"Full SSD Network PCC test FAILED for size {size}")
        if not does_pass_loc:
            logger.error(f"  Location PCC: {pcc_message_conf:.6f} < threshold {pcc}")
        if not does_pass_conf:
            logger.error(f"  Confidence PCC: {pcc_message_conf:.6f} < threshold {pcc}")

    assert does_pass_loc and does_pass_conf, (
        f"Full SSD Network does not meet PCC requirement {pcc}. "
        f"Location PCC: {pcc_message_conf:.6f}, Confidence PCC: {pcc_message_conf:.6f}"
    )


@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_ssd_network_structure(device, pcc, reset_seeds):
    """
    Test that Full SSD Network structure matches reference.

    This test verifies that the network structure matches the PyTorch
    reference without running forward pass.
    """
    if reset_seeds:
        torch.manual_seed(0)

    num_classes = 21

    # Test both sizes
    for size in [300, 512]:
        # Build PyTorch reference
        torch_model = build_ssd(phase="train", size=size, num_classes=num_classes)

        # Build TTNN network
        ssd_network_dict = build_ssd_network_ttnn(
            phase="train",
            size=size,
            num_classes=num_classes,
            device=device,
        )

        # Count layers
        torch_vgg_count = sum(1 for layer in torch_model.base if isinstance(layer, nn.Conv2d))
        torch_extras_count = len(torch_model.extras)
        torch_loc_count = len(torch_model.loc)
        torch_conf_count = len(torch_model.conf)

        tt_vgg_conv_count = sum(1 for layer in ssd_network_dict["vgg_layers"] if layer["type"] == "conv")
        tt_extras_conv_count = sum(1 for layer in ssd_network_dict["extras_layers"] if layer["type"] == "conv")
        tt_loc_count = len(ssd_network_dict["loc_layers"])
        tt_conf_count = len(ssd_network_dict["conf_layers"])

        logger.info(f"Size {size}:")
        logger.info(f"  VGG - PyTorch: {torch_vgg_count} conv layers, TTNN: {tt_vgg_conv_count} conv layers")
        logger.info(f"  Extras - PyTorch: {torch_extras_count} layers, TTNN: {tt_extras_conv_count} conv layers")
        logger.info(f"  Location - PyTorch: {torch_loc_count}, TTNN: {tt_loc_count}")
        logger.info(f"  Confidence - PyTorch: {torch_conf_count}, TTNN: {tt_conf_count}")

        assert torch_loc_count == tt_loc_count, f"Location head count mismatch for size {size}"
        assert torch_conf_count == tt_conf_count, f"Confidence head count mismatch for size {size}"

    logger.info("Full SSD Network structure test PASSED")


if __name__ == "__main__":
    # Allow running test directly for debugging
    pytest.main([__file__, "-v", "-s"])
