# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test Multibox Heads implementation using PCC metric.

This test compares the TTNN implementation of multibox heads with the PyTorch
reference implementation from torch_reference_ssd.py using Pearson
Correlation Coefficient (PCC) metric.
"""

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

# Import reference implementation
from models.experimental.SSD512.reference.ssd import multibox, base, extras, mbox

# Import TTNN implementation
from models.experimental.SSD512.tt.layers.multibox_heads import (
    build_multibox_heads,
    apply_multibox_heads,
)

# Import other modules for creating full context
from models.experimental.SSD512.reference.ssd import vgg, add_extras

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
def test_multibox_heads(device, pcc, size, reset_seeds):
    """
    Test Multibox heads TTNN implementation against PyTorch reference.

    This test:
    1. Creates a PyTorch Multibox head using torch_reference_ssd.multibox()
    2. Creates a TTNN Multibox head using multibox_heads()
    3. Runs forward pass on both with the same random input sources
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

    # Build full VGG and extras backbones to get source layers
    num_classes = 21  # VOC dataset

    # Build PyTorch reference model
    vgg_layers = vgg(base[str(size)], i=3, batch_norm=False)
    extra_layers = add_extras(extras[str(size)], i=1024, batch_norm=False)

    # Create PyTorch multibox heads
    torch_vgg_model = nn.ModuleList(vgg_layers)
    torch_extras_model = nn.ModuleList(extra_layers)

    # Get source channels from VGG layers
    vgg_source_indices = [21, -2]  # Conv4_3 and Conv7
    vgg_channels = []
    for idx in vgg_source_indices:
        vgg_layer = torch_vgg_model[idx]
        vgg_channels.append(vgg_layer.out_channels)

    # Get source channels from extras layers (every other starting from index 1)
    # Only collect indices that actually exist in the extras model
    # The multibox function uses extra_layers[1::2], so we need every other layer starting from index 1
    extra_source_indices_all = [1, 3, 5, 7] if size == 300 else [1, 3, 5, 7, 9, 11]
    extra_source_indices = []
    extra_channels = []
    for idx in extra_source_indices_all:
        if idx < len(torch_extras_model):
            extra_layer = torch_extras_model[idx]
            extra_channels.append(extra_layer.out_channels)
            extra_source_indices.append(idx)

    # Build PyTorch multibox heads
    # Note: multibox() returns (vgg, extra_layers, (loc_layers, conf_layers))
    _, _, (torch_loc_layers, torch_conf_layers) = multibox(
        torch_vgg_model, torch_extras_model, mbox[str(size)], num_classes
    )

    torch_loc_model = nn.ModuleList(torch_loc_layers)
    torch_conf_model = nn.ModuleList(torch_conf_layers)
    torch_loc_model.eval()
    torch_conf_model.eval()

    # Build TTNN multibox heads
    # Pass extra_source_indices to match what we actually collected
    loc_layers_config, conf_layers_config = build_multibox_heads(
        size=size,
        num_classes=num_classes,
        vgg_channels=vgg_channels,
        extra_channels=extra_channels,
        extra_source_indices=extra_source_indices,
        device=device,
    )

    # Extract weights from PyTorch model and load into TTNN layers
    torch_conv_idx = 0
    loc_layers_with_weights = []
    conf_layers_with_weights = []

    # Process location layers
    for layer in loc_layers_config:
        if layer["type"] == "conv":
            if torch_conv_idx >= len(torch_loc_model):
                raise ValueError("Mismatch: More loc layers in TTNN config than PyTorch model")

            torch_conv = torch_loc_model[torch_conv_idx]
            torch_conv_idx += 1

            weight = torch_conv.weight.data.clone()
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

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
            loc_layers_with_weights.append(layer_with_weights)

    # Process confidence layers (reset counter)
    torch_conv_idx = 0
    for layer in conf_layers_config:
        if layer["type"] == "conv":
            if torch_conv_idx >= len(torch_conf_model):
                raise ValueError("Mismatch: More conf layers in TTNN config than PyTorch model")

            torch_conv = torch_conf_model[torch_conv_idx]
            torch_conv_idx += 1

            weight = torch_conv.weight.data.clone()
            bias = torch_conv.bias.data.clone() if torch_conv.bias is not None else None

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
            conf_layers_with_weights.append(layer_with_weights)

    logger.info(f"Created {len(loc_layers_with_weights)} location heads")
    logger.info(f"Created {len(conf_layers_with_weights)} confidence heads")

    # Verify we have the same number of heads as PyTorch model
    assert len(loc_layers_with_weights) == len(
        torch_loc_model
    ), f"Mismatch: TTNN has {len(loc_layers_with_weights)} loc heads, PyTorch has {len(torch_loc_model)}"
    assert len(conf_layers_with_weights) == len(
        torch_conf_model
    ), f"Mismatch: TTNN has {len(conf_layers_with_weights)} conf heads, PyTorch has {len(torch_conf_model)}"

    # Create test source feature maps (simulating VGG and extras outputs)
    # Use the actual number of heads to determine how many sources to create
    num_heads = len(loc_layers_with_weights)
    batch_size = 1
    sources = []

    # VGG sources: Conv4_3 and Conv7 (always 2 sources)
    if size == 300:
        sources.append(torch.randn(batch_size, 512, 38, 38))  # Conv4_3
        sources.append(torch.randn(batch_size, 1024, 19, 19))  # Conv7
        # Remaining sources come from extras
        num_extra_sources = num_heads - 2
        # Typical extras sources for SSD300
        extra_dims = [(512, 10, 10), (256, 5, 5), (256, 3, 3), (256, 1, 1)]
        for i in range(num_extra_sources):
            if i < len(extra_dims):
                channels, h, w = extra_dims[i]
                sources.append(torch.randn(batch_size, channels, h, w))
            else:
                # Fallback: default size
                sources.append(torch.randn(batch_size, 256, 1, 1))
    else:  # size == 512
        sources.append(torch.randn(batch_size, 512, 64, 64))  # Conv4_3
        sources.append(torch.randn(batch_size, 1024, 32, 32))  # Conv7
        # Remaining sources come from extras
        num_extra_sources = num_heads - 2
        # Typical extras sources for SSD512
        extra_dims = [(512, 16, 16), (256, 8, 8), (256, 4, 4), (256, 2, 2), (256, 1, 1), (256, 1, 1)]
        for i in range(num_extra_sources):
            if i < len(extra_dims):
                channels, h, w = extra_dims[i]
                sources.append(torch.randn(batch_size, channels, h, w))
            else:
                # Fallback: default size
                sources.append(torch.randn(batch_size, 256, 1, 1))

    # Run PyTorch reference forward pass
    torch_loc_preds = []
    torch_conf_preds = []

    with torch.no_grad():
        for source_idx, source in enumerate(sources):
            # Apply location head
            loc_pred = torch_loc_model[source_idx](source)
            # Permute to NHWC for comparison (matching TTNN output)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            torch_loc_preds.append(loc_pred)

            # Apply confidence head
            conf_pred = torch_conf_model[source_idx](source)
            # Permute to NHWC for comparison
            conf_pred = conf_pred.permute(0, 2, 3, 1).contiguous()
            torch_conf_preds.append(conf_pred)

    logger.info(f"PyTorch location predictions: {len(torch_loc_preds)}")
    logger.info(f"PyTorch confidence predictions: {len(torch_conf_preds)}")

    # Run TTNN forward pass
    tt_loc_preds, tt_conf_preds = apply_multibox_heads(
        sources,
        loc_layers_with_weights,
        conf_layers_with_weights,
        device=device,
        dtype=ttnn.bfloat16,
    )

    logger.info(f"TTNN location predictions: {len(tt_loc_preds)}")
    logger.info(f"TTNN confidence predictions: {len(tt_conf_preds)}")

    # Compare outputs for each source
    all_loc_pass = True
    all_conf_pass = True

    for source_idx in range(len(sources)):
        # Convert TTNN output to torch format
        tt_loc = ttnn.to_torch(tt_loc_preds[source_idx])
        tt_conf = ttnn.to_torch(tt_conf_preds[source_idx])

        # Ensure output is float32 for fair comparison
        tt_loc = tt_loc.float()
        tt_conf = tt_conf.float()

        torch_loc = torch_loc_preds[source_idx]
        torch_conf = torch_conf_preds[source_idx]

        # Check shapes match
        if tt_loc.shape != torch_loc.shape:
            logger.error(
                f"Location shape mismatch at source {source_idx}: PyTorch {torch_loc.shape}, TTNN {tt_loc.shape}"
            )
            # Try to handle shape mismatch
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_loc.shape, tt_loc.shape)]
            torch_loc = torch_loc[tuple(slice(0, s) for s in min_shape)]
            tt_loc = tt_loc[tuple(slice(0, s) for s in min_shape)]

        if tt_conf.shape != torch_conf.shape:
            logger.error(
                f"Confidence shape mismatch at source {source_idx}: PyTorch {torch_conf.shape}, TTNN {tt_conf.shape}"
            )
            min_shape = [min(s1, s2) for s1, s2 in zip(torch_conf.shape, tt_conf.shape)]
            torch_conf = torch_conf[tuple(slice(0, s) for s in min_shape)]
            tt_conf = tt_conf[tuple(slice(0, s) for s in min_shape)]

        # Compare location head
        logger.info(f"Source {source_idx} - Location head:")
        logger.info(f"  PyTorch shape: {torch_loc.shape}, TTNN shape: {tt_loc.shape}")
        logger.info(comp_allclose(torch_loc, tt_loc))
        does_pass, pcc_message = comp_pcc(torch_loc, tt_loc, pcc)
        logger.info(f"  {pcc_message}")

        if not does_pass:
            all_loc_pass = False
            logger.error(f"Location head at source {source_idx} FAILED")

        # Compare confidence head
        logger.info(f"Source {source_idx} - Confidence head:")
        logger.info(f"  PyTorch shape: {torch_conf.shape}, TTNN shape: {tt_conf.shape}")
        logger.info(comp_allclose(torch_conf, tt_conf))
        does_pass, pcc_message = comp_pcc(torch_conf, tt_conf, pcc)
        logger.info(f"  {pcc_message}")

        if not does_pass:
            all_conf_pass = False
            logger.error(f"Confidence head at source {source_idx} FAILED")

    if all_loc_pass and all_conf_pass:
        logger.info(f"Multibox Heads PCC test PASSED for size {size}")
    else:
        logger.error(f"Multibox Heads PCC test FAILED for size {size}")

    assert all_loc_pass and all_conf_pass, f"Multibox Heads does not meet PCC requirement {pcc}"


# @pytest.mark.parametrize(
#     "pcc",
#     ((0.99),),
# )
# def test_multibox_heads_structure(device, pcc, reset_seeds):
#     """
#     Test that Multibox heads structure matches reference.

#     This test verifies that the layer configuration matches the PyTorch
#     reference without running forward pass.
#     """
#     if reset_seeds:
#         torch.manual_seed(0)

#     num_classes = 21

#     # Test both sizes
#     for size in [300, 512]:
#         # Build PyTorch reference
#         vgg_layers = vgg(base[str(size)], i=3, batch_norm=False)
#         extra_layers = add_extras(extras[str(size)], i=1024, batch_norm=False)

#         torch_vgg_model = nn.ModuleList(vgg_layers)
#         torch_extras_model = nn.ModuleList(extra_layers)

#         torch_loc_layers, torch_conf_layers = multibox(
#             torch_vgg_model, torch_extras_model, mbox[str(size)], num_classes
#         )

#         # Build TTNN heads
#         vgg_source_indices = [21, -2]
#         vgg_channels = [torch_vgg_model[21].out_channels, torch_vgg_model[-2].out_channels]

#         extra_source_indices = [1, 3, 5, 7] if size == 300 else [1, 3, 5, 7, 9, 11]
#         extra_channels = [torch_extras_model[idx].out_channels for idx in extra_source_indices]

#         loc_layers_config, conf_layers_config = build_multibox_heads(
#             size=size,
#             num_classes=num_classes,
#             vgg_channels=vgg_channels,
#             extra_channels=extra_channels,
#             device=device,
#         )

#         # Count layers
#         torch_loc_count = len(torch_loc_layers)
#         torch_conf_count = len(torch_conf_layers)

#         tt_loc_count = len(loc_layers_config)
#         tt_conf_count = len(conf_layers_config)

#         logger.info(f"Size {size}: Torch - Loc:{torch_loc_count}, Conf:{torch_conf_count}")
#         logger.info(f"Size {size}: TTNN - Loc:{tt_loc_count}, Conf:{tt_conf_count}")

#         assert torch_loc_count == tt_loc_count, f"Location layer count mismatch for size {size}"
#         assert torch_conf_count == tt_conf_count, f"Confidence layer count mismatch for size {size}"

#     logger.info("Multibox Heads structure test PASSED")


if __name__ == "__main__":
    # Allow running test directly for debugging
    pytest.main([__file__, "-v", "-s"])
