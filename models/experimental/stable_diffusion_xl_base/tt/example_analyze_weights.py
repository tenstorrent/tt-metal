#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Example script demonstrating how to analyze state_dict weights.
This script shows multiple ways to load and analyze model weights.
"""

import torch
from pathlib import Path
from state_dict_analyzer import analyze_state_dict, analyze_state_dict_by_type, get_state_dict_stats


def example_analyze_torch_model():
    """Example: Analyze a PyTorch model's state_dict"""
    print("Example 1: Creating and analyzing a simple PyTorch model")

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(128, 10),
    )

    # Analyze the state_dict
    state_dict = model.state_dict()
    analyze_state_dict(state_dict, "Simple CNN Model")
    analyze_state_dict_by_type(state_dict, "Simple CNN Model")


def example_analyze_pretrained_model():
    """Example: Analyze a pretrained model (if available)"""
    print("Example 2: Analyzing a pretrained model")

    try:
        # Try to load a pretrained ResNet
        import torchvision.models as models

        model = models.resnet18(pretrained=False)  # Set to True if you want pretrained weights

        state_dict = model.state_dict()
        analyze_state_dict(state_dict, "ResNet-18")

        # Get detailed stats
        stats = get_state_dict_stats(state_dict)
        print(f"Model statistics:")
        print(f"  Largest layer: {stats['largest_layer'][0]} ({stats['largest_layer'][1]:,} params)")
        print(f"  Smallest layer: {stats['smallest_layer'][0]} ({stats['smallest_layer'][1]:,} params)")
        print(f"  Data types used: {list(stats['dtypes'].keys())}")

    except ImportError:
        print("torchvision not available, skipping pretrained model example")


def example_analyze_saved_weights():
    """Example: Analyze weights loaded from a file"""
    print("Example 3: Analyzing weights from a saved file")

    # Create some dummy weights and save them
    dummy_weights = {
        "encoder.layer1.weight": torch.randn(512, 256),
        "encoder.layer1.bias": torch.randn(512),
        "encoder.layer2.weight": torch.randn(1024, 512),
        "encoder.layer2.bias": torch.randn(1024),
        "decoder.conv1.weight": torch.randn(128, 64, 3, 3),
        "decoder.conv1.bias": torch.randn(128),
        "decoder.conv2.weight": torch.randn(64, 128, 3, 3),
        "decoder.conv2.bias": torch.randn(64),
        "output.weight": torch.randn(10, 64),
        "output.bias": torch.randn(10),
    }

    # Save and reload (simulating loading from file)
    torch.save(dummy_weights, "temp_weights.pth")
    loaded_weights = torch.load("temp_weights.pth", weights_only=True)

    # Analyze the loaded weights
    analyze_state_dict(loaded_weights, "Dummy Encoder-Decoder Model")

    # Clean up
    Path("temp_weights.pth").unlink(missing_ok=True)


def example_analyze_mixed_dtypes():
    """Example: Analyze weights with different data types"""
    print("Example 4: Analyzing weights with mixed data types and None values")

    mixed_weights = {
        "fp32_weight": torch.randn(100, 50).float(),
        "fp16_weight": torch.randn(50, 25).half(),
        "bf16_weight": torch.randn(25, 10).bfloat16(),
        "int8_weight": torch.randint(-128, 127, (10, 5)).to(torch.int8),
        "bool_mask": torch.randint(0, 2, (5, 5)).bool(),
        "none_weight": None,  # Add a None value to test
        "another_none": None,
    }

    analyze_state_dict(mixed_weights, "Mixed Dtype Model")
    analyze_state_dict_by_type(mixed_weights, "Mixed Dtype Model")


def example_print_tensor_info():
    """Example: Using print_tensor_info function"""
    print("Example 5: Using print_tensor_info function")

    # Test various tensor types
    tensor1 = torch.randn(3, 4)
    tensor2 = None
    tensor3 = torch.ones(10, 20).half()

    print_tensor_info(tensor1, "tensor1")
    print_tensor_info(tensor2, "tensor2")
    print_tensor_info(tensor3, "tensor3")
    print_tensor_info("not_a_tensor", "string_value")


def analyze_your_state_dict(state_dict_path: str):
    """
    Function to analyze your own state_dict from a file.

    Args:
        state_dict_path: Path to your .pth or .pt file containing state_dict
    """
    print(f"Analyzing state_dict from: {state_dict_path}")

    try:
        # Load the state_dict
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)

        # If the file contains a model with state_dict method, extract it
        if hasattr(state_dict, "state_dict"):
            state_dict = state_dict.state_dict()
        elif isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Analyze the weights
        model_name = Path(state_dict_path).stem
        analyze_state_dict(state_dict, model_name)
        analyze_state_dict_by_type(state_dict, model_name)

        # Get and print detailed stats
        stats = get_state_dict_stats(state_dict)
        print(f"\nDetailed Statistics for {model_name}:")
        print(f"  Total layers: {stats['total_layers']}")
        print(f"  Total parameters: {stats['total_parameters']:,}")
        print(f"  Total size: {stats['total_size_gb']:.4f} GB")
        if stats["largest_layer"]:
            print(f"  Largest layer: {stats['largest_layer'][0]} ({stats['largest_layer'][1]:,} params)")
        if stats["smallest_layer"]:
            print(f"  Smallest layer: {stats['smallest_layer'][0]} ({stats['smallest_layer'][1]:,} params)")
        print(f"  Data types: {list(stats['dtypes'].keys())}")

    except Exception as e:
        print(f"Error loading state_dict: {e}")


if __name__ == "__main__":
    print("State Dict Weight Analyzer Examples")
    print("=" * 50)

    # Run examples
    example_analyze_torch_model()
    print("\n" + "=" * 80 + "\n")

    example_analyze_pretrained_model()
    print("\n" + "=" * 80 + "\n")

    example_analyze_saved_weights()
    print("\n" + "=" * 80 + "\n")

    example_analyze_mixed_dtypes()
    print("\n" + "=" * 80 + "\n")

    example_print_tensor_info()
    print("\n" + "=" * 80 + "\n")

    print("To analyze your own state_dict, use:")
    print("  python example_analyze_weights.py")
    print("  # Then call: analyze_your_state_dict('path/to/your/model.pth')")
    print("\nOr import the functions directly:")
    print("  from state_dict_analyzer import analyze_state_dict, print_tensor_info")
    print("  analyze_state_dict(your_state_dict, 'Your Model Name')")
    print("  print_tensor_info(your_tensor, 'tensor_name')  # For individual tensors")
