"""Tests for reference SSD512 implementation."""

import pytest
import torch
import torch.nn as nn
from loguru import logger

from models.experimental.SSD512.reference.ssd import build_ssd


@pytest.mark.parametrize("pcc", [0.97])
def test_reference_ssd(pcc):
    """Test reference SSD implementation."""

    # Initialize weights helper
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # Create model and initialize weights
    net = build_ssd("test", 512, 21)
    net.apply(weights_init)
    net.eval()

    # Create test input
    dummy_input = torch.randn(1, 3, 512, 512)

    # Get output shapes for various feature maps
    output = net(dummy_input)

    # Check that we get reasonable detection outputs
    assert len(output) == 3, "Expected output to contain loc, conf and prior predictions"

    loc_out, conf_out, priors = output

    # Check output shapes
    # For 512x512 input:
    # - Feature maps: 64x64, 32x32, 16x16, 8x8, 4x4, 2x2, 1x1
    # - Total prior boxes: 24640
    assert loc_out.shape[1] == 24640 * 4, f"Expected loc_out shape to be [batch, 24640*4], got {loc_out.shape}"
    assert conf_out.shape[1] == 24640 * 21, f"Expected conf_out shape to be [batch, 24640*21], got {conf_out.shape}"
    assert priors.shape == (24640, 4), f"Expected priors shape to be [24640, 4], got {priors.shape}"

    # Check that predictions are reasonable
    assert torch.all(priors >= 0) and torch.all(priors <= 1), "Prior box coordinates should be normalized to [0,1]"
    assert torch.all(conf_out <= 1) and torch.all(conf_out >= 0), "Confidence scores should be in [0,1]"

    logger.info("Reference SSD test passed!")
