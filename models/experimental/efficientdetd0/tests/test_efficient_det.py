# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.efficientdetd0.reference.efficientdet import EfficientDetBackbone
from models.experimental.efficientdetd0.tt.efficient_det import TtEfficientDetBackbone

import pytest
from loguru import logger

from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    infer_torch_module_args,
    create_custom_mesh_preprocessor,
)
from tests.ttnn.utils_for_testing import check_with_pcc

torch.manual_seed(0)

PCC_THRESHOLD = 0.99


@pytest.mark.parametrize(
    "batch, channels, height, width",
    [
        (1, 3, 512, 512),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_efficient_det(batch, channels, height, width, device):
    torch_model = EfficientDetBackbone(
        num_classes=80,
        compound_coef=0,
        load_weights=False,
    )
    torch_model.eval()
    # Run PyTorch forward pass
    torch_inputs = torch.randn(batch, channels, height, width)
    with torch.no_grad():
        torch_outputs = torch_model(torch_inputs)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    # module_args = infer_module_args(model=torch_model, input=torch_inputs)
    module_args = infer_torch_module_args(model=torch_model, input=torch_inputs)

    # Create TTNN BiFPN model
    ttnn_model = TtEfficientDetBackbone(
        device=device,
        parameters=parameters,
        conv_params=module_args,
        num_classes=80,
        compound_coef=0,
    )
    # Convert inputs to TTNN format
    ttnn_input_tensor = ttnn.from_torch(
        torch_inputs,
        # torch_inputs.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_outputs = ttnn_model(ttnn_input_tensor)

    # Compare each output
    all_passed = True
    for i, (torch_output, ttnn_output) in enumerate(zip(torch_outputs, ttnn_outputs)):
        ttnn_output_torch = ttnn.to_torch(ttnn_output)

        # Get expected dimensions from PyTorch output (NCHW format)
        expected_batch, expected_channels, expected_h, expected_w = torch_output.shape

        # TTNN output is in format [1, 1, H*W, C] (NHWC flattened)
        # Reshape to [batch, H, W, C]
        ttnn_output_torch = ttnn_output_torch.reshape(expected_batch, expected_h, expected_w, expected_channels)

        # Permute from NHWC to NCHW to match PyTorch
        ttnn_output_torch = ttnn_output_torch.permute(0, 3, 1, 2)

        passing, pcc_message = check_with_pcc(torch_output, ttnn_output_torch, PCC_THRESHOLD)
        logger.info(f"Output {i} PCC: {pcc_message}")
        all_passed = all_passed and passing

    if all_passed:
        logger.info("Efficient Det Test Passed!")
    else:
        logger.warning("Efficient Det Test Failed!")

    assert all_passed, f"PCC value is lower than {PCC_THRESHOLD}. Check implementation!"
