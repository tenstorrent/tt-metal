#!/usr/bin/env python

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test script for Qwen2_5_VLMLP implementation."""

import os
import torch
import ttnn
import importlib
import sys


def main():
    # Import both implementations
    functional = importlib.import_module("models.demos.qwen25_vl.reference.functional")
    functional_ttnn = importlib.import_module("models.demos.qwen25_vl.reference.functional_ttnn")

    # Get mesh size from command line args (default to 1x1 if not specified)
    if len(sys.argv) >= 3:
        mesh_height = int(sys.argv[1])
        mesh_width = int(sys.argv[2])
    else:
        mesh_height = 1
        mesh_width = 1

    # Print mesh configuration
    print(f"Using mesh configuration: {mesh_height}x{mesh_width}")

    # Open a mesh device
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(mesh_height, mesh_width))
    functional_ttnn.set_mesh_device(mesh_device)

    # Create sample input [batch, hidden_size]
    hidden_size = 1280
    intermediate_size = 3420
    batch_size = 100

    # Random input
    input_tensor = torch.randn(batch_size, hidden_size, dtype=torch.float32)

    # Create random weights
    state_dict = {
        "gate_proj": {"weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float32)},
        "up_proj": {"weight": torch.randn(intermediate_size, hidden_size, dtype=torch.float32)},
        "down_proj": {"weight": torch.randn(hidden_size, intermediate_size, dtype=torch.float32)},
    }

    print(f"Input tensor shape: {input_tensor.shape}")
    print(f"gate_proj weight shape: {state_dict['gate_proj']['weight'].shape}")
    print(f"up_proj weight shape: {state_dict['up_proj']['weight'].shape}")
    print(f"down_proj weight shape: {state_dict['down_proj']['weight'].shape}")

    # Run the PyTorch implementation
    result_torch = functional.qwen2_5_vl_mlp(input_tensor, state_dict)
    print(f"PyTorch result shape: {result_torch.shape}")

    # Run the TTNN implementation
    result_ttnn = functional_ttnn.qwen2_5_vl_mlp(input_tensor, state_dict)
    print(f"TTNN result shape: {result_ttnn.shape}")

    # Compare results
    torch_flat = result_torch.flatten()
    ttnn_flat = result_ttnn.flatten()
    diff = torch.abs(torch_flat - ttnn_flat).mean()

    print(f"Average absolute difference: {diff.item()}")

    # Compute Pearson correlation
    from scipy.stats import pearsonr

    pcc = pearsonr(torch_flat.numpy(), ttnn_flat.numpy())[0]
    print(f"Pearson correlation: {pcc}")

    if result_torch.shape == result_ttnn.shape:
        print("✅ Output shapes match")
    else:
        print("❌ Output shapes do not match")

    if pcc > 0.99:
        print("✅ Results match closely (PCC > 0.99)")
    else:
        print("❌ Results differ significantly (PCC <= 0.99)")

    # Print first few values to verify
    print("\nFirst few values:")
    print("PyTorch:", result_torch[0, :10])
    print("TTNN:   ", result_ttnn[0, :10])

    # Close the device
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
