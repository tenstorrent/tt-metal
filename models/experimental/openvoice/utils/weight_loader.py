# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for converting PyTorch checkpoints to TTNN format.
Handles weight normalization fusion, tensor format conversion, and reshaping.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# TTNN import - will fail without TT hardware/SDK, but we can still develop
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    print("Warning: TTNN not available. Running in development mode.")


def fuse_weight_norm(weight: torch.Tensor, weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """
    Fuse weight normalization into the weight tensor.

    Weight norm decomposes W = g * (v / ||v||)
    We compute the fused weight for inference.

    Args:
        weight: Original weight tensor (may be placeholder)
        weight_g: Magnitude parameter g
        weight_v: Direction parameter v

    Returns:
        Fused weight tensor W = g * (v / ||v||)
    """
    # Compute norm of v along output channel dimension
    # Use reshape + vector_norm to handle PyTorch 2.9+ API changes
    dims = tuple(range(1, weight_v.dim()))

    # Flatten all dims except first, compute norm, then reshape back
    v_flat = weight_v.reshape(weight_v.shape[0], -1)
    norm_v = torch.linalg.vector_norm(v_flat, dim=1, keepdim=True)

    # Reshape norm to broadcast correctly
    norm_shape = [weight_v.shape[0]] + [1] * (weight_v.dim() - 1)
    norm_v = norm_v.reshape(norm_shape)

    # Fused weight: g * (v / ||v||)
    fused = weight_g * (weight_v / norm_v)
    return fused


def remove_weight_norm_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove weight normalization from a state dict by fusing g and v into weight.

    Identifies weight_g and weight_v pairs and replaces with fused weights.

    Args:
        state_dict: PyTorch state dict with weight_g/weight_v parameters

    Returns:
        New state dict with fused weights (no weight norm parameters)
    """
    new_state_dict = {}
    processed_keys = set()

    for key in state_dict.keys():
        if key in processed_keys:
            continue

        # Check if this is a weight_v parameter (indicates weight norm)
        if key.endswith(".weight_v"):
            base_key = key[:-9]  # Remove '.weight_v'
            weight_g_key = base_key + ".weight_g"

            if weight_g_key in state_dict:
                # Fuse weight norm
                weight_v = state_dict[key]
                weight_g = state_dict[weight_g_key]
                fused_weight = fuse_weight_norm(None, weight_g, weight_v)

                # Store as regular weight
                new_state_dict[base_key + ".weight"] = fused_weight
                processed_keys.add(key)
                processed_keys.add(weight_g_key)
            else:
                # No weight_g found, keep as-is
                new_state_dict[key] = state_dict[key]
        elif key.endswith(".weight_g"):
            # Skip - will be processed with weight_v
            continue
        else:
            # Regular parameter, copy as-is
            new_state_dict[key] = state_dict[key]

    return new_state_dict


def reshape_conv1d_to_conv2d_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Reshape Conv1d weight [C_out, C_in, K] to Conv2d format [C_out, C_in, 1, K].

    TTNN conv2d expects 4D weights, so we add a height dimension of 1.

    Args:
        weight: Conv1d weight tensor [C_out, C_in, K]

    Returns:
        Reshaped weight [C_out, C_in, 1, K]
    """
    if weight.dim() == 3:
        return weight.unsqueeze(2)  # [C_out, C_in, K] -> [C_out, C_in, 1, K]
    return weight


def reshape_conv_transpose1d_weight(weight: torch.Tensor) -> torch.Tensor:
    """
    Reshape ConvTranspose1d weight [C_in, C_out, K] to ConvTranspose2d format.

    Note: ConvTranspose has C_in and C_out swapped compared to Conv.

    Args:
        weight: ConvTranspose1d weight tensor [C_in, C_out, K]

    Returns:
        Reshaped weight [C_in, C_out, 1, K]
    """
    if weight.dim() == 3:
        return weight.unsqueeze(2)  # [C_in, C_out, K] -> [C_in, C_out, 1, K]
    return weight


def convert_to_ttnn_tensor(
    tensor: torch.Tensor,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
    layout: Optional[Any] = None,
    keep_pytorch: bool = False,
    is_conv_weight: bool = False,
) -> Any:
    """
    Convert PyTorch tensor to TTNN tensor format.

    Args:
        tensor: PyTorch tensor
        device: TTNN device (optional, can be None for host tensors)
        dtype: TTNN dtype (default: bfloat16)
        layout: TTNN layout (default: TILE_LAYOUT, but ROW_MAJOR for conv weights)
        keep_pytorch: If True, keep as PyTorch tensor (for CPU mode)
        is_conv_weight: If True, use ROW_MAJOR_LAYOUT (required for ttnn.conv2d)

    Returns:
        TTNN tensor or PyTorch tensor
    """
    if not TTNN_AVAILABLE or keep_pytorch:
        # Return PyTorch tensor in development/CPU mode
        return tensor

    dtype = dtype or ttnn.bfloat16

    # Conv weights MUST be ROW_MAJOR for ttnn.conv2d
    # See: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md
    if is_conv_weight:
        layout = ttnn.ROW_MAJOR_LAYOUT
    else:
        layout = layout or ttnn.TILE_LAYOUT

    tt_tensor = ttnn.from_torch(tensor, dtype=dtype, layout=layout)

    # Don't put conv weights on device - they need prepare_conv_weights() first
    if device is not None and not is_conv_weight:
        tt_tensor = ttnn.to_device(tt_tensor, device)

    return tt_tensor


def load_openvoice_checkpoint(
    checkpoint_path: str,
    device: Optional[Any] = None,
    config_path: Optional[str] = None,
    keep_pytorch: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load OpenVoice checkpoint and convert weights to TTNN format.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        device: TTNN device for placing weights
        config_path: Path to config.json (optional)
        keep_pytorch: If True, keep weights as PyTorch tensors (for CPU mode)

    Returns:
        Tuple of (converted_weights, config)
    """
    checkpoint_path = Path(checkpoint_path)

    # Load checkpoint - handle different PyTorch versions and checkpoint formats
    checkpoint = None

    # Try loading with mmap=True first (helps with weight_norm issues in PyTorch 2.9+)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        # Older PyTorch versions don't have mmap parameter
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            # Even older PyTorch versions don't have weights_only
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        # Fallback: try without mmap
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model state dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        # If checkpoint is a model object, get its state dict
        state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

    # Fuse weight normalization
    state_dict = remove_weight_norm_from_state_dict(state_dict)

    # Convert weights to TTNN format
    converted_weights = {}

    for key, tensor in state_dict.items():
        # Detect conv weights - Conv1d (3D) and Conv2d (4D) both need ROW_MAJOR layout
        # Conv weights need ROW_MAJOR layout and should NOT go on device yet
        # (they need to go through prepare_conv_weights() first)
        is_3d_weight = ".weight" in key and tensor.dim() == 3
        is_4d_weight = ".weight" in key and tensor.dim() == 4  # Conv2d weights (e.g., ref_enc.convs)
        is_conv_bias = ".bias" in key and any(
            pattern in key
            for pattern in [".conv", ".ups.", ".pre.", ".proj.", "cond", ".in_layers.", ".res_skip", "ref_enc"]
        )
        is_conv_param = is_3d_weight or is_4d_weight or is_conv_bias

        # Only reshape for TTNN (Conv2d needs 4D, Conv1d uses 3D)
        if not keep_pytorch:
            # Reshape Conv1d weights to Conv2d format for TTNN
            # All 3D weights are Conv1d in this model
            if ".weight" in key and tensor.dim() == 3:
                tensor = reshape_conv1d_to_conv2d_weight(tensor)

            # Note: Conv biases are kept as 1D [C_out] - prepare_conv_bias handles reshaping
            # (Previously we reshaped to 4D here, but that causes issues with prepare_conv_bias)

        # Convert to TTNN tensor (or keep as PyTorch if keep_pytorch=True)
        # Conv weights/biases use ROW_MAJOR_LAYOUT (required by ttnn.conv2d)
        converted_weights[key] = convert_to_ttnn_tensor(
            tensor, device, keep_pytorch=keep_pytorch, is_conv_weight=is_conv_param
        )

    # Load config if provided
    config = {}
    if config_path:
        import json

        with open(config_path, "r") as f:
            config = json.load(f)

    return converted_weights, config


class TTNNParameterDict:
    """
    Hierarchical parameter dictionary for organizing TTNN weights.

    Provides attribute-style access to nested parameters:
        params.encoder.attention.q_proj.weight
    """

    def __init__(self, state_dict: Dict[str, Any], prefix: str = ""):
        self._params = {}
        self._prefix = prefix

        # Organize parameters by module hierarchy
        for key, value in state_dict.items():
            if prefix and not key.startswith(prefix):
                continue

            # Remove prefix
            rel_key = key[len(prefix) :].lstrip(".") if prefix else key

            # Split into parts
            parts = rel_key.split(".")

            if len(parts) == 1:
                # Leaf parameter
                self._params[parts[0]] = value
            else:
                # Nested parameter - create sub-dict if needed
                first = parts[0]
                rest = ".".join(parts[1:])

                if first not in self._params:
                    self._params[first] = {}

                if isinstance(self._params[first], dict):
                    self._params[first][rest] = value

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            return super().__getattribute__(name)

        if name in self._params:
            value = self._params[name]
            if isinstance(value, dict):
                return TTNNParameterDict(value)
            return value

        raise AttributeError(f"No parameter '{name}'")

    def __getitem__(self, key: str) -> Any:
        return self._params[key]

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    def items(self):
        return self._params.items()
