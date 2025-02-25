# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Convert Qwen2.5-VL model state dict from flat to nested format.
The nested format is used by the functional implementations in functional.py.
"""

import os
import torch
from collections import defaultdict
from model import Qwen2_5_VLForConditionalGeneration


def nested_dict():
    """Create an infinitely nestable defaultdict."""
    return defaultdict(nested_dict)


def print_dict_structure(d, prefix="", max_tensor_info=10):
    """Print the structure of a nested dictionary, including tensor shapes.

    Args:
        d: The dictionary to print
        prefix: Prefix for indentation
        max_tensor_info: Maximum number of tensor dimensions to show per level
    """
    tensor_count = 0
    for k, v in sorted(d.items()):
        # Skip layers other than layer 0
        if k.isdigit() and k != "0":
            continue

        if isinstance(v, (dict, defaultdict)):
            print(f"{prefix}{k}/")
            print_dict_structure(v, prefix + "  ", max_tensor_info)
        elif isinstance(v, (torch.Tensor, torch.nn.Parameter)):
            tensor_count += 1
            if tensor_count <= max_tensor_info:
                dtype_str = str(v.dtype).replace("torch.", "")
                print(f"{prefix}{k}: {tuple(v.shape)} ({dtype_str})")
            elif tensor_count == max_tensor_info + 1:
                print(f"{prefix}... and more tensors ...")


def convert_state_dict(flat_dict):
    """Convert a flat state dict to nested format.

    Example:
        Input: {'model.layers.0.self_attn.q_proj.weight': tensor(...)}
        Output: {'model': {'layers': {'0': {'self_attn': {'q_proj': {'weight': tensor(...)}}}}}}
    """
    nested = nested_dict()

    for key, value in flat_dict.items():
        # Skip if not a tensor or buffer
        if not isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            continue

        # Split the key into parts
        parts = key.split(".")

        # Traverse the nested dict, creating the structure
        current = nested
        for part in parts[:-1]:
            current = current[part]

        # Set the leaf value
        current[parts[-1]] = value

    return nested


def extract_vision_state_dict(nested_dict):
    """Extract just the vision-related parts of the state dict.
    This includes patch embedding, rotary embeddings, and vision blocks.
    """
    vision_dict = {}

    if "model" in nested_dict:
        model_dict = nested_dict["model"]

        # Extract patch embedding
        if "patch_embed" in model_dict:
            vision_dict["patch_embed"] = model_dict["patch_embed"]

        # Extract rotary embeddings
        if "rotary_pos_emb" in model_dict:
            vision_dict["rotary_pos_emb"] = model_dict["rotary_pos_emb"]

        # Extract vision blocks
        if "blocks" in model_dict:
            vision_dict["blocks"] = model_dict["blocks"]

        # Extract patch merger
        if "merger" in model_dict:
            vision_dict["merger"] = model_dict["merger"]

    return vision_dict


def main():
    # Create output directory
    os.makedirs("weights", exist_ok=True)

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )

    print("\nConverting state dict...")
    flat_dict = model.state_dict()
    nested = convert_state_dict(flat_dict)

    print("\nFull model structure:")
    print_dict_structure(nested)

    print("\nExtracting vision components...")
    vision_dict = extract_vision_state_dict(nested)

    print("\nVision components structure:")
    print_dict_structure(vision_dict)

    print("\nSaving converted weights...")
    torch.save(vision_dict, "weights/vision_weights.pt")
    print("Done! Saved to weights/vision_weights.pt")


if __name__ == "__main__":
    main()
