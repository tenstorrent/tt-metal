# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for RVC model implementation
"""

import pytest
import torch
import sys
import os

# Add the models directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from models.demos.rvc.main import create_rvc_model


def test_rvc_model_creation():
    """Test that RVC model can be created successfully"""
    model = create_rvc_model(input_dim=128, hidden_dim=256, output_dim=128)
    assert model is not None


def test_rvc_model_forward_pass():
    """Test forward pass of RVC model"""
    model = create_rvc_model(input_dim=128, hidden_dim=256, output_dim=128)
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 128)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, 128)


if __name__ == "__main__":
    test_rvc_model_creation()
    test_rvc_model_forward_pass()
    print("All RVC tests passed!")