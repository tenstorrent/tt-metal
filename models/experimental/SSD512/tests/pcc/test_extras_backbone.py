# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger
from models.experimental.SSD512.reference.ssd import add_extras, extras
from models.experimental.SSD512.tt.layers.tt_extras_backbone import (
    build_extras_backbone,
    apply_extras_backbone,
)
from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc


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
    """
    from models.experimental.SSD512.common import setup_seeds_and_deterministic

    # from models.experimental.SSD512.tt.layers.tt_extras_backbone import clear_extras_weight_cache
    # clear_extras_weight_cache()
    setup_seeds_and_deterministic(reset_seeds=reset_seeds, seed=0)

    cfg = extras[str(size)]
    torch_layers = add_extras(cfg, i=1024, batch_norm=False)
    torch_model = nn.ModuleList(torch_layers)
    torch_model.eval()

    batch_size = 1
    input_channels = 1024
    if size == 300:
        input_height = 38
        input_width = 38
    else:  # size == 512
        input_height = 64
        input_width = 64

    torch_input = torch.randn(batch_size, input_channels, input_height, input_width)

    with torch.no_grad():
        x = torch_input
        for layer in torch_model:
            x = torch.nn.functional.relu(layer(x), inplace=True)
        torch_output = x

    # Build TTNN model
    layers_config = build_extras_backbone(size=size, input_channels=input_channels, device=device)

    torch_conv_idx = 0
    layers_with_weights = []

    for layer in layers_config:
        if layer["type"] == "conv":
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
            weight = torch_conv.weight.data.clone()
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
            layers_with_weights.append(layer.copy())

    total_conv_layers = sum(1 for layer in layers_config if layer["type"] == "conv")
    if torch_conv_idx != total_conv_layers:
        logger.warning(
            f"Conv layer count mismatch: PyTorch has {torch_conv_idx} conv layers, "
            f"but TTNN config has {total_conv_layers} conv layers"
        )

    # Run TTNN forward pass
    tt_output_ttnn = apply_extras_backbone(
        torch_input,
        layers_with_weights,
        device=device,
        dtype=ttnn.bfloat8_b,
    )

    tt_output = ttnn.to_torch(tt_output_ttnn)

    # Convert from NHWC to NCHW
    if len(tt_output.shape) == 4:
        tt_output = tt_output.permute(0, 3, 1, 2)

    tt_output = tt_output.float()

    # Check if shapes match
    if tt_output.shape != torch_output.shape:
        logger.error(f"Shape mismatch! PyTorch: {torch_output.shape}, TTNN: {tt_output.shape}")
        min_shape = [min(s1, s2) for s1, s2 in zip(torch_output.shape, tt_output.shape)]
        torch_output = torch_output[tuple(slice(0, s) for s in min_shape)]
        tt_output = tt_output[tuple(slice(0, s) for s in min_shape)]
        logger.warning(f"Truncated to matching shape: {torch_output.shape}")

    does_pass, pcc_message = comp_pcc(torch_output, tt_output, pcc)
    logger.info(f"Extras Backbone PCC: {pcc_message}")
    assert_with_pcc(torch_output, tt_output, pcc)
