# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Retrieval-based Voice Conversion (RVC) using TTNN APIs
This example demonstrates how to implement RVC using Tenstorrent's TTNN library.
"""

import argparse
import os
import sys

import torch
from torch import nn

# Import TTNN components
import ttnn
from ttnn import (
    Tensor,
    from_torch,
    to_torch,
    matmul,
    permute,
    reshape,
    mean,
    sum as ttnn_sum,
    abs as ttnn_abs,
    sqrt,
    reciprocal,
    multiply,
    add,
    sub,
    pow as ttnn_pow,
    where,
    full,
    zeros,
    ones,
    relu,
    transpose,
)

# Import device management
from ttnn.distributed import DeviceGrid

def create_tt_tensor(tensor_data, device=None):
    """Create a TTNN tensor from PyTorch tensor"""
    tt_tensor = from_torch(tensor_data, layout=ttnn.RowMajorLayout())
    if device:
        tt_tensor = ttnn.to_device(tt_tensor, device)
    return tt_tensor

def create_rvc_model_tt(input_dim=128, hidden_dim=256, output_dim=128, device=None):
    """Create RVC model using TTNN operations"""
    # For this example, we'll create a simplified RVC model that uses TTNN operations
    class RVCModelTT:
        def __init__(self, input_dim, hidden_dim, output_dim, device):
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            self.device = device
            
            # We'll use TTNN tensor operations directly instead of PyTorch
            # Create weight tensors
            self.w1 = torch.randn(input_dim, hidden_dim) * 0.1
            self.b1 = torch.zeros(hidden_dim)
            self.w2 = torch.randn(hidden_dim, output_dim) * 0.1
            self.b2 = torch.zeros(output_dim)
            
            # Convert to TTNN tensors
            self.w1_tt = create_tt_tensor(self.w1, device)
            self.b1_tt = create_tt_tensor(self.b1, device)
            self.w2_tt = create_tt_tensor(self.w2, device)
            self.b2_tt = create_tt_tensor(self.b2, device)
            
        def forward(self, x):
            # Convert input to TTNN tensor
            x_tt = create_tt_tensor(x, self.device)
            
            # First layer: matmul + bias + relu
            x1 = matmul(x_tt, self.w1_tt)
            x1 = add(x1, self.b1_tt)
            x1 = relu(x1)
            
            # Second layer: matmul + bias
            x2 = matmul(x1, self.w2_tt)
            x2 = add(x2, self.b2_tt)
            
            # Convert back to PyTorch for output
            return to_torch(x2)
    
    return RVCModelTT(input_dim, hidden_dim, output_dim, device)

def main():
    """Main function to demonstrate RVC model using TTNN"""
    parser = argparse.ArgumentParser(description="RVC Model using TTNN")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--input-dim", type=int, default=128, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--output-dim", type=int, default=128, help="Output dimension")
    
    args = parser.parse_args()
    
    print(f"Creating RVC model with input_dim={args.input_dim}, hidden_dim={args.hidden_dim}, output_dim={args.output_dim}")
    
    # Initialize device
    device = None
    if args.device == "gpu":
        device = ttnn.open_device(DeviceGrid(8, 4))
    elif args.device == "cpu":
        pass  # Use CPU
    
    # Create model using TTNN
    model = create_rvc_model_tt(args.input_dim, args.hidden_dim, args.output_dim, device)
    
    # Create dummy input
    batch_size = args.batch_size
    dummy_input = torch.randn(batch_size, args.input_dim)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass using TTNN operations
    with torch.no_grad():
        output = model.forward(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    if device:
        ttnn.close_device(device)
    
    print("RVC model example completed using TTNN operations")

if __name__ == "__main__":
    main()