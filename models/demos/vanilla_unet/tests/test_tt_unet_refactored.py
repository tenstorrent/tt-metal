# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

from models.demos.vanilla_unet.reference.unet import UNet
from models.demos.vanilla_unet.tt.unet_refactored import create_unet_from_params


@pytest.mark.parametrize(
    "batch_size, input_height, input_width",
    [
        (1, 480, 640),  # Small test case
    ],
)
def test_tt_unet_refactored_with_torch_model(device, batch_size, input_height, input_width):
    """Test that refactored TtUNet can be initialized with a PyTorch model"""

    logger.info(
        f"Testing refactored TtUNet with PyTorch model - "
        f"batch_size={batch_size}, input_size=({input_height}, {input_width})"
    )

    torch_model = UNet(in_channels=3, out_channels=1, init_features=32)
    torch_model.eval()

    # Create TT model with PyTorch model
    tt_model = create_unet_from_params(
        device=device,
        in_channels=3,
        out_channels=1,
        init_features=32,
        input_height=input_height,
        input_width=input_width,
        batch_size=batch_size,
        torch_model=torch_model,
    )

    # Check that all components are initialized
    assert tt_model.encoder1 is not None
    assert tt_model.encoder2 is not None
    assert tt_model.encoder3 is not None
    assert tt_model.encoder4 is not None
    assert tt_model.bottleneck_conv1 is not None
    assert tt_model.bottleneck_conv2 is not None
    assert tt_model.decoder4 is not None
    assert tt_model.decoder3 is not None
    assert tt_model.decoder2 is not None
    assert tt_model.decoder1 is not None
    assert tt_model.final_conv is not None

    logger.info("Refactored TtUNet with PyTorch model test passed!")
