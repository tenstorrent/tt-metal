# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Dict, Any


def analyze_state_dict(state_dict: Dict[str, torch.Tensor], model_name: str = "Model") -> None:
    """
    Analyze and print all weights in a state_dict with their names, sizes, and total size in GB.

    Args:
        state_dict: Dictionary containing model weights (torch.Tensor values)
        model_name: Name of the model for display purposes
    """
    print(f"\n{'='*80}")
    print(f"STATE DICT ANALYSIS: {model_name}")
    print(f"{'='*80}")

    total_params = 0
    total_size_bytes = 0

    # Header
    print(f"{'Layer Name':<50} {'Shape':<25} {'Params':<15} {'Size (MB)':<12} {'dtype':<10}")
    print(f"{'-'*50} {'-'*25} {'-'*15} {'-'*12} {'-'*10}")

    # Sort keys for consistent output
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]

        if tensor is None:
            # Handle None values
            print(f"{key:<50} {'None':<25} {'-':<15} {'-':<12} {'None':<10}")
        elif isinstance(tensor, torch.Tensor):
            # Get tensor info
            shape = tuple(tensor.shape)
            num_params = tensor.numel()
            size_bytes = tensor.numel() * tensor.element_size()
            size_mb = size_bytes / (1024 * 1024)
            dtype = str(tensor.dtype)

            # Accumulate totals
            total_params += num_params
            total_size_bytes += size_bytes

            # Format shape string
            shape_str = str(shape)
            if len(shape_str) > 25:
                shape_str = shape_str[:22] + "..."

            # Print row
            print(f"{key:<50} {shape_str:<25} {num_params:<15,} {size_mb:<12.2f} {dtype:<10}")
        else:
            print(f"{key:<50} {'Non-tensor':<25} {'-':<15} {'-':<12} {type(tensor).__name__:<10}")

    # Summary
    total_size_gb = total_size_bytes / (1024**3)
    print(f"\n{'-'*112}")
    print(f"{'SUMMARY':<50}")
    print(f"{'-'*112}")
    print(f"Total layers: {len(state_dict)}")
    print(f"Total parameters: {total_params:,}")
    print(f"Total size: {total_size_gb:.4f} GB ({total_size_bytes:,} bytes)")
    print(f"Average layer size: {total_size_bytes/len(state_dict)/1024/1024:.2f} MB")
    print(f"{'='*80}\n")


def analyze_state_dict_by_type(state_dict: Dict[str, torch.Tensor], model_name: str = "Model") -> None:
    """
    Analyze state_dict grouped by layer types (e.g., conv, linear, norm, etc.).

    Args:
        state_dict: Dictionary containing model weights
        model_name: Name of the model for display purposes
    """
    print(f"\n{'='*80}")
    print(f"STATE DICT ANALYSIS BY TYPE: {model_name}")
    print(f"{'='*80}")

    # Group by layer type
    layer_types = {}

    for key in state_dict.keys():
        tensor = state_dict[key]
        if tensor is None:
            # Handle None values
            layer_type = "None"
            if layer_type not in layer_types:
                layer_types[layer_type] = {"count": 0, "params": 0, "size_bytes": 0}
            layer_types[layer_type]["count"] += 1
        elif isinstance(tensor, torch.Tensor):
            # Determine layer type from key
            if "conv" in key.lower():
                layer_type = "Convolution"
            elif "linear" in key.lower() or "proj" in key.lower():
                layer_type = "Linear"
            elif "norm" in key.lower() or "bn" in key.lower():
                layer_type = "Normalization"
            elif "embed" in key.lower():
                layer_type = "Embedding"
            elif "bias" in key.lower():
                layer_type = "Bias"
            elif "weight" in key.lower():
                layer_type = "Weight"
            else:
                layer_type = "Other"

            if layer_type not in layer_types:
                layer_types[layer_type] = {"count": 0, "params": 0, "size_bytes": 0}

            layer_types[layer_type]["count"] += 1
            layer_types[layer_type]["params"] += tensor.numel()
            layer_types[layer_type]["size_bytes"] += tensor.numel() * tensor.element_size()

    # Print summary by type
    print(f"{'Layer Type':<20} {'Count':<8} {'Parameters':<15} {'Size (MB)':<12} {'Size (GB)':<12}")
    print(f"{'-'*20} {'-'*8} {'-'*15} {'-'*12} {'-'*12}")

    total_params = 0
    total_size_bytes = 0

    for layer_type, stats in sorted(layer_types.items()):
        size_mb = stats["size_bytes"] / (1024**2)
        size_gb = stats["size_bytes"] / (1024**3)
        total_params += stats["params"]
        total_size_bytes += stats["size_bytes"]

        print(f"{layer_type:<20} {stats['count']:<8} {stats['params']:<15,} {size_mb:<12.2f} {size_gb:<12.4f}")

    print(f"{'-'*69}")
    print(
        f"{'TOTAL':<20} {len(state_dict):<8} {total_params:<15,} {total_size_bytes/1024**2:<12.2f} {total_size_bytes/1024**3:<12.4f}"
    )
    print(f"{'='*80}\n")


def get_state_dict_stats(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Get numerical statistics about the state_dict.

    Args:
        state_dict: Dictionary containing model weights

    Returns:
        Dictionary with statistics
    """
    stats = {
        "total_layers": len(state_dict),
        "total_parameters": 0,
        "total_size_bytes": 0,
        "total_size_gb": 0,
        "layer_shapes": {},
        "dtypes": {},
        "largest_layer": None,
        "smallest_layer": None,
    }

    max_params = 0
    min_params = float("inf")

    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            num_params = tensor.numel()
            size_bytes = num_params * tensor.element_size()

            stats["total_parameters"] += num_params
            stats["total_size_bytes"] += size_bytes
            stats["layer_shapes"][key] = tuple(tensor.shape)

            # Track dtypes
            dtype_str = str(tensor.dtype)
            if dtype_str not in stats["dtypes"]:
                stats["dtypes"][dtype_str] = 0
            stats["dtypes"][dtype_str] += 1

            # Track largest/smallest layers
            if num_params > max_params:
                max_params = num_params
                stats["largest_layer"] = (key, num_params)
            if num_params < min_params:
                min_params = num_params
                stats["smallest_layer"] = (key, num_params)

    stats["total_size_gb"] = stats["total_size_bytes"] / (1024**3)

    return stats


def print_tensor_info(tensor, name: str = "tensor"):
    """
    Print tensor information: shape and dtype, or 'None' if tensor is None.

    Args:
        tensor: PyTorch tensor or None
        name: Name to display for the tensor
    """
    if tensor is None:
        print(f"{name}: None")
    elif isinstance(tensor, torch.Tensor):
        print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
    else:
        print(f"{name}: {type(tensor).__name__} (not a tensor)")


if __name__ == "__main__":
    # Example usage
    print("State Dict Analyzer Utility")
    print("Import this module and use:")
    print("  analyze_state_dict(state_dict, 'Your Model Name')")
    print("  analyze_state_dict_by_type(state_dict, 'Your Model Name')")
    print("  stats = get_state_dict_stats(state_dict)")
