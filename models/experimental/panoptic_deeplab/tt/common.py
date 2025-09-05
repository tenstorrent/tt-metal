# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Common utilities for Panoptic DeepLab.

This module provides functions used across the Panoptic DeepLab model.
"""

import torch
import pickle
import os
import ttnn
from loguru import logger


def from_torch_fast(
    t: torch.Tensor,
    *,
    device: ttnn.Device | ttnn.MeshDevice | None = None,
    layout: ttnn.Layout | None = None,
    dtype: ttnn.DataType | None = None,
    memory_config: ttnn.MemoryConfig | None = None,
    to_host: bool = False,
    mesh_mapper: ttnn.TensorToMesh | None = None,
    # The argument shard_dim is a bit problematic. If set, it creates a mesh mapper with the given
    # device. But for a host tensor, the device is None, so a mesh mapper can not be created.
    shard_dim: int | None = None,
) -> ttnn.Tensor:
    assert shard_dim is None or device is not None, "shard_dim requires device"

    if isinstance(device, ttnn.MeshDevice):
        if shard_dim is not None:
            mesh_mapper = ttnn.ShardTensorToMesh(device, dim=shard_dim)
        if mesh_mapper is None:
            mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    elif isinstance(device, ttnn.Device):
        mesh_mapper = None

    float32_in = t.dtype == torch.float32
    float32_out = dtype == ttnn.float32 or (dtype is None and float32_in)

    # ttnn.to_layout does not support changing the datatype or memory_config if the layout already matches. ttnn.clone
    # does not support changing the datatype if the input is not tiled. An option could be to tilize the input before
    # changing the datatype and then untilize again, but it was not tested if this would be faster than converting the
    # datatype on the host. Also ttnn.to_dtype does not support device tensors. Additionally, `ttnn.to_layout` is lossy
    # for float32.
    if device is None or layout is None or layout == ttnn.ROW_MAJOR_LAYOUT or (float32_in and float32_out):
        return ttnn.from_torch(
            t,
            device=None if to_host else device,
            layout=layout,
            dtype=dtype,
            memory_config=memory_config,
            mesh_mapper=mesh_mapper,
        )

    tensor = ttnn.from_torch(t, device=device, mesh_mapper=mesh_mapper)

    if tensor.shape[-2] == 32 and t.shape[-2] == 1:
        # Work around the fact that the shape is erroneously set to the padded shape under certain conditions.
        assert isinstance(device, ttnn.MeshDevice)
        assert dtype in (ttnn.bfloat4_b, ttnn.bfloat8_b)
        tensor = tensor.reshape(ttnn.Shape(t.shape))

    tensor = ttnn.to_layout(tensor, layout, dtype=dtype, memory_config=memory_config)

    if to_host:
        tensor = tensor.cpu()

    return tensor


def load_resnet_weights_from_pickle(pickle_path: str = None) -> dict:
    """
    Load ResNet weights from R-52.pkl file.

    Args:
        pickle_path: Path to the pickle file. If None, uses default path relative to this file.

    Returns:
        Dictionary containing the ResNet state dict.
    """
    if pickle_path is None:
        # Default path relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pickle_path = os.path.join(current_dir, "..", "weights", "R-52.pkl")

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle file not found at: {pickle_path}")

    logger.debug(f"Loading ResNet weights from: {pickle_path}")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    if "model" not in data:
        raise ValueError("Pickle file does not contain 'model' key")

    model_state = data["model"]
    logger.debug(f"Loaded {len(model_state)} weight tensors from pickle file")

    # Convert to bfloat16 to match the random weight functions
    state_dict = {}
    for key, value in model_state.items():
        if isinstance(value, torch.Tensor):
            # Convert to bfloat16 and ensure it's on CPU
            state_dict[key] = value.to(dtype=torch.bfloat16, device="cpu")
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            # Handle numpy arrays by converting to torch tensor first
            import numpy as np

            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy())
                state_dict[key] = tensor.to(dtype=torch.bfloat16, device="cpu")
            else:
                state_dict[key] = value
        else:
            # Keep non-tensor values as-is (like num_batches_tracked)
            state_dict[key] = value

    logger.debug("Converted all tensors to bfloat16")
    return state_dict


def create_resnet_state_dict(pickle_path: str = None) -> dict:
    """
    Create ResNet state dict using weights from R-52.pkl file.

    Args:
        pickle_path: Path to the pickle file. If None, uses default path.

    Returns:
        Dictionary containing the ResNet state dict.
    """
    state_dict = load_resnet_weights_from_pickle(pickle_path)

    # Add missing bias terms that are expected by the model but not present in weights
    # ResNet typically doesn't use bias in conv layers when using batch normalization,
    # but the TtResNet implementation expects them to be zero

    # Get all weight keys to determine what bias terms we need
    weight_keys = [k for k in state_dict.keys() if k.endswith(".weight") and "norm" not in k]

    for weight_key in weight_keys:
        bias_key = weight_key.replace(".weight", ".bias")
        if bias_key not in state_dict:
            # Create zero bias with the appropriate size (number of output channels)
            weight_tensor = state_dict[weight_key]
            if len(weight_tensor.shape) == 4:  # Conv2d weight: [out_channels, in_channels, H, W]
                out_channels = weight_tensor.shape[0]
                state_dict[bias_key] = torch.zeros(out_channels, dtype=torch.bfloat16)

    # Remove any keys that shouldn't be there
    keys_to_remove = [
        k
        for k in state_dict.keys()
        if k.startswith("stem.fc.") or k.endswith(".num_batches_tracked")  # FC layer not used in our model
    ]  # Not needed by TtResNet
    for key in keys_to_remove:
        del state_dict[key]

    logger.debug(f"Added missing bias terms and cleaned up state dict. Final keys: {len(state_dict)}")
    return state_dict
