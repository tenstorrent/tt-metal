# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch


def set_mesh_device(device):
    """Set the mesh device to use for all operations."""
    global mesh_device
    mesh_device = device


def convert_state_dict(flat_dict):
    """Convert a flat state dict to nested format.

    Example:
        Input: {'model.layers.0.self_attn.q_proj.weight': tensor(...)}
        Output: {'model': {'layers': [{'self_attn': {'wq': {'weight': tensor(...)}}}]}}
    """
    # Map the attention weights to the Meta format (until image attention is generic)
    mappings = {
        "q_proj": "wq",
        "k_proj": "wk",
        "v_proj": "wv",
        "out_proj": "wo",
        # MLP weight mappings to TtLlamaImageFeedForward format
        "fc1": "c_fc",
        "fc2": "c_proj",
    }
    for m, n in mappings.items():
        flat_dict = {k.replace(m, n): v for k, v in flat_dict.items()}

    nested = {}

    for key, value in flat_dict.items():
        # Skip if not a tensor or buffer
        if not isinstance(value, (torch.Tensor, torch.nn.Parameter)):
            continue

        # Split the key into parts
        parts = key.split(".")

        # Traverse the nested dict, creating the structure
        current = nested
        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the leaf value
        current[parts[-1]] = value

    return nested


def flatten_state_dict(state_dict, prefix=""):
    """Flatten a nested state dict into a flat dict.

    Example:
        Input: {'model': {'layers': [{'self_attn': {'q_proj': {'weight': tensor(...)}}}]}}
        Output: {'model.layers.0.self_attn.q_proj.weight': tensor(...)}
    """
    flat_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, dict):
            flat_dict.update(flatten_state_dict(value, prefix=f"{prefix}.{key}" if len(prefix) > 0 else key))
        if isinstance(value, list):
            for i, item in enumerate(value):
                flat_dict.update(
                    flatten_state_dict(item, prefix=f"{prefix}.{key}.{i}" if len(prefix) > 0 else f"{key}.{i}")
                )
        else:
            flat_dict[f"{prefix}.{key}" if len(prefix) > 0 else key] = value
    return flat_dict
