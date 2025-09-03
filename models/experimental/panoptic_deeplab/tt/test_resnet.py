# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Tests for Panoptic DeepLab with both random and real weights.

This test module provides comprehensive testing for ResNet implementations:

RANDOM WEIGHTS TESTS:
- test_resnet_stem_pcc: Tests stem layer with random weights
- test_resnet_layer_pcc: Tests individual res layers with random weights
- test_resnet_full_pcc: Tests full ResNet pipeline with random weights

REAL WEIGHTS TESTS:
- test_resnet_stem_pcc_real_weights: Tests stem layer with real R-52.pkl weights
- test_resnet_layer_pcc_real_weights: Tests individual res layers with real weights
- test_resnet_full_pcc_real_weights: Tests full ResNet pipeline with real weights
- test_real_vs_random_weights_comparison: Verifies real weights are different from random

Real weight tests will be skipped if R-52.pkl file is not found.
"""

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_resnet import ResNet
from models.experimental.panoptic_deeplab.tt.resnet.tt_resnet import TtResNet
from models.experimental.panoptic_deeplab.tt.common import (
    create_full_resnet_state_dict,
    create_real_resnet_state_dict,
)


def create_resnet_models(device=None, use_real_weights=False):
    """Create both PyTorch and TTNN ResNet models with same weights.

    Args:
        device: TTNN device for model creation
        use_real_weights: If True, use real weights from R-52.pkl, else use random weights
    """

    # Create state dict using either real or random weights
    if use_real_weights:
        state_dict = create_real_resnet_state_dict()
        logger.info("Using real weights from R-52.pkl")
    else:
        state_dict = create_full_resnet_state_dict()
        logger.info("Using random weights")

    # Create PyTorch model
    pytorch_model = ResNet()

    # Load state dict into PyTorch model (need to handle the nested structure)
    pytorch_state_dict = {}
    for key, value in state_dict.items():
        # Skip bias parameters since PyTorch model has bias=False for all conv layers
        if ".bias" in key and not ".norm.bias" in key:
            continue  # Skip conv bias parameters

        # Convert TTNN-style keys to PyTorch-style keys
        if key.startswith("stem."):
            pytorch_key = key  # stem keys are the same
        elif key.startswith("res"):
            # Convert res2.0.conv1.weight to res2.0.conv1.weight (same)
            pytorch_key = key
        else:
            pytorch_key = key
        pytorch_state_dict[pytorch_key] = value

    pytorch_model.load_state_dict(pytorch_state_dict)
    pytorch_model.eval()

    # Convert model to bfloat16 to match input type
    pytorch_model = pytorch_model.to(torch.bfloat16)

    # Create TTNN model
    if device:
        ttnn_model = TtResNet(device=device, state_dict=state_dict, dtype=ttnn.bfloat16)
    else:
        ttnn_model = None

    return pytorch_model, ttnn_model, state_dict


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_stem_pcc(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch Stem implementations produce similar outputs using PCC."""

    torch.manual_seed(15)

    # Create models
    pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN stem only
    ttnn_stem_output = ttnn_model.stem(ttnn_input)

    # Run PyTorch stem only
    with torch.no_grad():
        torch_stem_output = pytorch_model.stem(torch_input)

    # Convert TTNN output back to torch (NHWC -> NCHW) and compare
    ttnn_output_torch = ttnn.to_torch(ttnn_stem_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_stem_output, ttnn_output_torch, 0.95)

    logger.info(f"ResNet Stem PCC test - {pcc_message}")
    logger.info(f"Stem - PyTorch shape: {torch_stem_output.shape}, TTNN shape: {ttnn_output_torch.shape}")

    assert pcc_passed, f"Stem PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
@pytest.mark.parametrize("layer_name", ["res2", "res3", "res4", "res5"])
def test_resnet_layer_pcc(device, batch_size, height, width, layer_name, reset_seeds):
    """Test that TTNN and PyTorch ResNet layer implementations produce similar outputs using PCC."""

    torch.manual_seed(7)

    # Create models
    pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)

    # Define layer input specifications (channels, height_factor, width_factor)
    # These are the expected input dimensions for each layer after stem processing
    layer_specs = {
        "res2": (128, 1 / 4, 1 / 4),  # After stem: 128 channels, 1/4 resolution
        "res3": (256, 1 / 4, 1 / 4),  # After res2: 256 channels, same resolution
        "res4": (512, 1 / 8, 1 / 8),  # After res3: 512 channels, 1/2 resolution (stride=2)
        "res5": (1024, 1 / 16, 1 / 16),  # After res4: 1024 channels, 1/2 resolution (stride=2)
    }

    in_channels, h_factor, w_factor = layer_specs[layer_name]
    layer_height = int(height * h_factor)
    layer_width = int(width * w_factor)

    # Create appropriately sized input tensor for the specific layer (NCHW format)
    torch_layer_input = torch.randn(batch_size, in_channels, layer_height, layer_width, dtype=torch.bfloat16)

    # Convert to TTNN format (NHWC)
    ttnn_layer_input = ttnn.from_torch(
        torch_layer_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN layer directly
    if layer_name == "res2":
        for block in ttnn_model.res2:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res3":
        for block in ttnn_model.res3:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res4":
        for block in ttnn_model.res4:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res5":
        for block in ttnn_model.res5:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input

    # Run PyTorch layer directly
    with torch.no_grad():
        if layer_name == "res2":
            torch_layer_output = pytorch_model.res2(torch_layer_input)
        elif layer_name == "res3":
            torch_layer_output = pytorch_model.res3(torch_layer_input)
        elif layer_name == "res4":
            torch_layer_output = pytorch_model.res4(torch_layer_input)
        elif layer_name == "res5":
            torch_layer_output = pytorch_model.res5(torch_layer_input)

    # Convert TTNN output back to torch (NHWC -> NCHW) and compare
    ttnn_output_torch = ttnn.to_torch(ttnn_layer_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_layer_output, ttnn_output_torch, 0.95)

    logger.info(f"ResNet {layer_name} PCC test - {pcc_message}")
    logger.info(
        f"{layer_name} - PyTorch input shape: {torch_layer_input.shape}, output shape: {torch_layer_output.shape}"
    )
    logger.info(f"{layer_name} - TTNN output shape: {ttnn_output_torch.shape}")

    assert pcc_passed, f"{layer_name} PCC test failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_full_pcc(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch ResNet implementations produce similar outputs using PCC for all layers."""

    torch.manual_seed(0)

    # Create models
    pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device)

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    ttnn_outputs = ttnn_model(ttnn_input)

    # Run PyTorch model
    with torch.no_grad():
        torch_outputs = pytorch_model(torch_input)

    # Convert TTNN outputs back to torch (NHWC -> NCHW) and compare each layer
    test_results = {}
    failed_layers = []

    for layer_name in ["res2", "res3", "res4", "res5"]:
        torch_output = torch_outputs[layer_name]
        ttnn_output = ttnn_outputs[layer_name]
        ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

        # Compare outputs using PCC (more relaxed for full ResNets)
        pcc_passed, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, 0.95)
        test_results[layer_name] = (pcc_passed, pcc_message)

        # Log PCC result for this layer
        logger.info(f"ResNet {layer_name} PCC test - {pcc_message}")
        logger.info(f"{layer_name} - PyTorch shape: {torch_output.shape}, TTNN shape: {ttnn_output_torch.shape}")

        # Track failed layers
        if not pcc_passed:
            failed_layers.append(layer_name)

    # Log overall test details
    logger.info(f"Input shape: {torch_input.shape}")
    logger.info(f"All layers passed PCC test: {len(failed_layers) == 0}")

    if failed_layers:
        logger.info(f"Failed layers: {failed_layers}")

    # Assert only at the end if any layer failed
    assert (
        len(failed_layers) == 0
    ), f"PCC test failed for layers: {failed_layers}. Details: {[test_results[layer][1] for layer in failed_layers]}"


# Tests with Real Weights from R-52.pkl


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_stem_pcc_real_weights(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch Stem implementations produce similar outputs using PCC with real weights."""

    torch.manual_seed(15)

    # Create models with real weights
    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device, use_real_weights=True)
    except FileNotFoundError:
        pytest.skip("R-52.pkl file not found - skipping real weights test")

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN stem only
    ttnn_stem_output = ttnn_model.stem(ttnn_input)

    # Run PyTorch stem only
    with torch.no_grad():
        torch_stem_output = pytorch_model.stem(torch_input)

    # Convert TTNN output back to torch (NHWC -> NCHW) and compare
    ttnn_output_torch = ttnn.to_torch(ttnn_stem_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_stem_output, ttnn_output_torch, 0.95)

    logger.info(f"ResNet Stem PCC test (real weights) - {pcc_message}")
    logger.info(
        f"Stem (real weights) - PyTorch shape: {torch_stem_output.shape}, TTNN shape: {ttnn_output_torch.shape}"
    )

    assert pcc_passed, f"Stem PCC test with real weights failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
@pytest.mark.parametrize("layer_name", ["res2", "res3", "res4", "res5"])
def test_resnet_layer_pcc_real_weights(device, batch_size, height, width, layer_name, reset_seeds):
    """Test that TTNN and PyTorch ResNet layer implementations produce similar outputs using PCC with real weights."""

    torch.manual_seed(7)

    # Create models with real weights
    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device, use_real_weights=True)
    except FileNotFoundError:
        pytest.skip("R-52.pkl file not found - skipping real weights test")

    # Define layer input specifications (channels, height_factor, width_factor)
    # These are the expected input dimensions for each layer after stem processing
    layer_specs = {
        "res2": (128, 1 / 4, 1 / 4),  # After stem: 128 channels, 1/4 resolution
        "res3": (256, 1 / 4, 1 / 4),  # After res2: 256 channels, same resolution
        "res4": (512, 1 / 8, 1 / 8),  # After res3: 512 channels, 1/2 resolution (stride=2)
        "res5": (1024, 1 / 16, 1 / 16),  # After res4: 1024 channels, 1/2 resolution (stride=2)
    }

    in_channels, h_factor, w_factor = layer_specs[layer_name]
    layer_height = int(height * h_factor)
    layer_width = int(width * w_factor)

    # Create appropriately sized input tensor for the specific layer (NCHW format)
    torch_layer_input = torch.randn(batch_size, in_channels, layer_height, layer_width, dtype=torch.bfloat16)

    # Convert to TTNN format (NHWC)
    ttnn_layer_input = ttnn.from_torch(
        torch_layer_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN layer directly
    if layer_name == "res2":
        for block in ttnn_model.res2:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res3":
        for block in ttnn_model.res3:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res4":
        for block in ttnn_model.res4:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input
    elif layer_name == "res5":
        for block in ttnn_model.res5:
            ttnn_layer_input = block(ttnn_layer_input)
        ttnn_layer_output = ttnn_layer_input

    # Run PyTorch layer directly
    with torch.no_grad():
        if layer_name == "res2":
            torch_layer_output = pytorch_model.res2(torch_layer_input)
        elif layer_name == "res3":
            torch_layer_output = pytorch_model.res3(torch_layer_input)
        elif layer_name == "res4":
            torch_layer_output = pytorch_model.res4(torch_layer_input)
        elif layer_name == "res5":
            torch_layer_output = pytorch_model.res5(torch_layer_input)

    # Convert TTNN output back to torch (NHWC -> NCHW) and compare
    ttnn_output_torch = ttnn.to_torch(ttnn_layer_output).permute(0, 3, 1, 2)

    # Compare outputs using PCC
    pcc_passed, pcc_message = assert_with_pcc(torch_layer_output, ttnn_output_torch, 0.95)

    logger.info(f"ResNet {layer_name} PCC test (real weights) - {pcc_message}")
    logger.info(
        f"{layer_name} (real weights) - PyTorch input shape: {torch_layer_input.shape}, output shape: {torch_layer_output.shape}"
    )
    logger.info(f"{layer_name} (real weights) - TTNN output shape: {ttnn_output_torch.shape}")

    assert pcc_passed, f"{layer_name} PCC test with real weights failed: {pcc_message}"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize(
    "height,width",
    [(512, 1024)],
)
def test_resnet_full_pcc_real_weights(device, batch_size, height, width, reset_seeds):
    """Test that TTNN and PyTorch ResNet implementations produce similar outputs using PCC for all layers with real weights."""

    torch.manual_seed(0)

    # Create models with real weights
    try:
        pytorch_model, ttnn_model, state_dict = create_resnet_models(device=device, use_real_weights=True)
    except FileNotFoundError:
        pytest.skip("R-52.pkl file not found - skipping real weights test")

    # Create input tensor (RGB image)
    torch_input = torch.randn(batch_size, 3, height, width, dtype=torch.bfloat16)

    # Convert input to TTNN format (NHWC)
    ttnn_input = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),  # Convert NCHW to NHWC
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run TTNN model
    ttnn_outputs = ttnn_model(ttnn_input)

    # Run PyTorch model
    with torch.no_grad():
        torch_outputs = pytorch_model(torch_input)

    # Convert TTNN outputs back to torch (NHWC -> NCHW) and compare each layer
    test_results = {}
    failed_layers = []

    for layer_name in ["res2", "res3", "res4", "res5"]:
        torch_output = torch_outputs[layer_name]
        ttnn_output = ttnn_outputs[layer_name]
        ttnn_output_torch = ttnn.to_torch(ttnn_output).permute(0, 3, 1, 2)

        # Compare outputs using PCC (more relaxed for full ResNets)
        pcc_passed, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, 0.95)
        test_results[layer_name] = (pcc_passed, pcc_message)

        # Log PCC result for this layer
        logger.info(f"ResNet {layer_name} PCC test (real weights) - {pcc_message}")
        logger.info(
            f"{layer_name} (real weights) - PyTorch shape: {torch_output.shape}, TTNN shape: {ttnn_output_torch.shape}"
        )

        # Track failed layers
        if not pcc_passed:
            failed_layers.append(layer_name)

    # Log overall test details
    logger.info(f"Input shape: {torch_input.shape}")
    logger.info(f"All layers passed PCC test (real weights): {len(failed_layers) == 0}")

    if failed_layers:
        logger.info(f"Failed layers (real weights): {failed_layers}")

    # Assert only at the end if any layer failed
    assert (
        len(failed_layers) == 0
    ), f"PCC test with real weights failed for layers: {failed_layers}. Details: {[test_results[layer][1] for layer in failed_layers]}"


def test_real_vs_random_weights_comparison():
    """Test that real weights are actually different from random weights."""

    try:
        real_weights = create_real_resnet_state_dict()
        random_weights = create_full_resnet_state_dict()

        # Check that key structures match
        assert set(real_weights.keys()) == set(
            random_weights.keys()
        ), "Real and random weights should have same key structure"

        # Check that weights are actually different
        sample_keys = ["stem.conv1.weight", "res2.0.conv1.weight", "res5.2.conv3.weight"]
        differences_found = 0

        for key in sample_keys:
            if key in real_weights and key in random_weights:
                real_tensor = real_weights[key]
                random_tensor = random_weights[key]

                # Check shapes match
                assert real_tensor.shape == random_tensor.shape, f"Shape mismatch for {key}"

                # Check that they're actually different (not identical)
                if not torch.allclose(real_tensor, random_tensor, atol=1e-6):
                    differences_found += 1

                # Check that real weights have reasonable statistics
                real_std = real_tensor.float().std().item()
                assert real_std > 0, f"Real weights for {key} should have non-zero standard deviation"

        assert differences_found == len(sample_keys), "Real weights should be different from random weights"
        logger.info(
            f"✓ Real weights verified to be different from random weights ({differences_found}/{len(sample_keys)} keys checked)"
        )

    except FileNotFoundError:
        pytest.skip("R-52.pkl file not found - skipping real vs random weights comparison")
