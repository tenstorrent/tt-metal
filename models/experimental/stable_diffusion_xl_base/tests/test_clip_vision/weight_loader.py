# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight loading utilities for converting PyTorch weights to TTNN tensors."""

import json
from pathlib import Path

import torch
import utils

import ttnn

_THIS_DIR = Path(__file__).parent


def load_weights_from_pytorch(state_dict, device):
    """
    Load model weights from PyTorch state_dict instead of tensorbin files.

    Args:
        state_dict: PyTorch model state_dict
        device: TTNN device to load weights onto

    Returns:
        List of TTNN tensors. The input tensor slot (index 390) contains None
        and should be passed separately to forward() as pixel_values.
    """
    # Load tensor configuration
    with open(_THIS_DIR / "tensor_load_config.json", "r") as f:
        tensor_config = json.load(f)
    dram_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

    # Sort by tensor_idx to ensure correct order
    sorted_configs = sorted(tensor_config.items(), key=lambda x: x[1]["tensor_idx"])

    result = []
    for arg_name, config in sorted_configs:
        weight_name = config["weight_name"]
        layout = getattr(ttnn.Layout, config["layout"])
        ttnn_dtype = getattr(ttnn.DataType, config["dtype"])
        on_device = config["on_device"]

        if weight_name == "__INPUT__":
            # Input tensor is passed separately to forward() as pixel_values
            result.append(None)

        elif weight_name == "__POSITION_IDS__":
            # Generate position IDs [0, 1, 2, ..., 256]
            pos_ids = torch.arange(257, dtype=torch.int32).unsqueeze(0)
            ttnn_tensor = ttnn.from_torch(pos_ids, dtype=ttnn_dtype, layout=layout)
            if on_device:
                ttnn_tensor = ttnn.to_device(ttnn_tensor, device, dram_memory_config)
            result.append(ttnn_tensor)

        elif weight_name is None:
            raise ValueError(f"Unknown tensor {arg_name} has no weight mapping")

        else:
            # Regular weight from state_dict
            ttnn_tensor = utils.load_weight_from_pytorch(
                state_dict,
                weight_name,
                layout,
                ttnn_dtype,
                device if on_device else None,
                dram_memory_config if on_device else None,
            )
            result.append(ttnn_tensor)

    return result
