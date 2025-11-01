"""Tests for TTNN extra layers implementation."""

import pytest
import torch
import torch.nn as nn
from loguru import logger

from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.SSD512.tt.layers.extras import TtExtraLayers
from models.experimental.SSD512.reference.ssd import add_extras as torch_add_extras


@pytest.mark.parametrize("pcc", [0.97])
def test_extra_layers(device, pcc):
    """Test TTNN extra layers against PyTorch implementation."""

    # Create models
    torch_model = nn.ModuleList(
        torch_add_extras(
            [256, "S", 512, 128, "S", 256, 128, "S", 256, 128, "S", 256], 1024  # Output channels from VGG conv7
        )
    )
    torch_model.eval()

    tt_model = TtExtraLayers({}, device=device)

    # Create test input (output shape from VGG conv7)
    batch_size = 1
    torch_input = torch.randn(batch_size, 1024, 19, 19)  # Example feature map size
    ttnn_input = torch_to_tt_tensor_rm(torch_input, device)

    # Run forward passes
    x = torch_input
    torch_features = []
    for i, layer in enumerate(torch_model):
        x = nn.functional.relu(layer(x), inplace=True)
        if i % 2 == 1:
            torch_features.append(x)

    tt_features = tt_model(ttnn_input)

    # Compare outputs for each feature map
    for i, (torch_feat, tt_feat) in enumerate(zip(torch_features, tt_features)):
        tt_feat = tt_to_torch_tensor(tt_feat)

        # Compare outputs
        output_pass, pcc_value = comp_pcc(torch_feat, tt_feat, pcc)
        logger.info(f"Extra layer {i} PCC: {pcc_value}")

        allclose = comp_allclose(torch_feat, tt_feat)
        logger.info(f"Extra layer {i} allclose: {allclose}")

        assert output_pass, f"Extra layer {i} output does not meet PCC requirement {pcc}"
