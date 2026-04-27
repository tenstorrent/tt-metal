#!/usr/bin/env python3

"""
Simple test to verify RVC implementation works
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.demos.rvc.main import create_rvc_model

if __name__ == "__main__":
    print("Testing RVC implementation...")
    
    # Test model creation
    model = create_rvc_model(input_dim=128, hidden_dim=256, output_dim=128)
    print("✓ Model created successfully")
    
    # Test basic functionality
    import torch
    import torch.nn as nn
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 128)
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    # Check output shape
    assert output.shape == (batch_size, 128)
    print("✓ Forward pass completed successfully")
    
    print("RVC implementation test completed successfully!")