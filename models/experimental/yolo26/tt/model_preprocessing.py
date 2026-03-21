# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model preprocessing utilities for YOLO26.

Handles:
- Loading weights from Ultralytics YOLO26 models
- BatchNorm folding into Conv layers
- Weight format conversion for TTNN
"""

import torch
import ttnn
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

from models.experimental.yolo26.common import fold_bn_to_conv_weights_bias


def load_yolo26_from_ultralytics(variant: str = "yolo26n", weights_path: Optional[str] = None):
    """
    Load YOLO26 model from Ultralytics.

    Args:
        variant: Model variant ('yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x')
        weights_path: Optional path to pre-downloaded weights

    Returns:
        Tuple of (model, state_dict)
    """
    try:
        from ultralytics import YOLO

        if weights_path and Path(weights_path).exists():
            model = YOLO(weights_path)
        else:
            model = YOLO(f"{variant}.pt")

        return model, model.model.state_dict()
    except ImportError:
        raise ImportError("Please install ultralytics: pip install ultralytics")


def load_state_dict_from_file(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Load state dict from a .pth file.

    Args:
        weights_path: Path to weights file

    Returns:
        State dictionary
    """
    return torch.load(weights_path, map_location="cpu")


def get_conv_bn_weights(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract Conv + BatchNorm weights from state dict.

    YOLO26 naming conventions:
    - Simple Conv+BN: model.0.conv.weight, model.0.bn.weight
    - C2f Conv+BN: model.2.cv1.conv.weight, model.2.cv1.bn.weight

    Args:
        state_dict: Model state dictionary
        prefix: Layer prefix (e.g., 'model.0' or 'model.2.cv1')

    Returns:
        Tuple of (conv_weight, bn_weight, bn_bias, bn_mean, bn_var)
    """
    # Try different naming conventions for YOLO26
    possible_conv_keys = [
        f"{prefix}.conv.weight",  # model.0.conv.weight
        f"{prefix}.weight",  # direct weight
    ]

    conv_key = None
    for key in possible_conv_keys:
        if key in state_dict:
            conv_key = key
            break

    if conv_key is None:
        raise KeyError(f"Conv weight not found for prefix: {prefix}. Tried: {possible_conv_keys}")

    conv_weight = state_dict[conv_key]

    # Find corresponding BN weights
    bn_prefix = conv_key.replace(".conv.weight", ".bn").replace(".weight", "")
    if not bn_prefix.endswith(".bn"):
        bn_prefix = f"{prefix}.bn"

    possible_bn_prefixes = [
        bn_prefix,
        f"{prefix}.bn",
        prefix.replace(".conv", ".bn"),
    ]

    bn_weight_key = None
    for bp in possible_bn_prefixes:
        if f"{bp}.weight" in state_dict:
            bn_weight_key = f"{bp}.weight"
            bn_bias_key = f"{bp}.bias"
            bn_mean_key = f"{bp}.running_mean"
            bn_var_key = f"{bp}.running_var"
            break

    if bn_weight_key and bn_weight_key in state_dict:
        bn_weight = state_dict[bn_weight_key]
        bn_bias = state_dict[bn_bias_key]
        bn_mean = state_dict[bn_mean_key]
        bn_var = state_dict[bn_var_key]
    else:
        # No BatchNorm, return identity values
        out_channels = conv_weight.shape[0]
        bn_weight = torch.ones(out_channels)
        bn_bias = torch.zeros(out_channels)
        bn_mean = torch.zeros(out_channels)
        bn_var = torch.ones(out_channels)

    return conv_weight, bn_weight, bn_bias, bn_mean, bn_var


def get_folded_conv_weights(
    state_dict: Dict[str, torch.Tensor], prefix: str, eps: float = 1e-5
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Get folded Conv+BN weights ready for TTNN.

    Args:
        state_dict: Model state dictionary
        prefix: Layer prefix
        eps: BatchNorm epsilon

    Returns:
        Tuple of (weight, bias) as ttnn tensors
    """
    conv_weight, bn_weight, bn_bias, bn_mean, bn_var = get_conv_bn_weights(state_dict, prefix)
    return fold_bn_to_conv_weights_bias(conv_weight, bn_weight, bn_bias, bn_mean, bn_var, eps)


def get_conv_only_weights(
    state_dict: Dict[str, torch.Tensor], conv_prefix: str, bias_prefix: Optional[str] = None
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """
    Get Conv weights without BatchNorm (for final detection layers).

    Args:
        state_dict: Model state dictionary
        conv_prefix: Conv weight key prefix
        bias_prefix: Optional bias key prefix

    Returns:
        Tuple of (weight, bias) as ttnn tensors
    """
    conv_weight = state_dict[f"{conv_prefix}.weight"]

    if bias_prefix and f"{bias_prefix}.bias" in state_dict:
        bias = state_dict[f"{bias_prefix}.bias"]
    elif f"{conv_prefix}.bias" in state_dict:
        bias = state_dict[f"{conv_prefix}.bias"]
    else:
        bias = torch.zeros(conv_weight.shape[0])

    bias = bias.reshape(1, 1, 1, -1)

    return ttnn.from_torch(conv_weight), ttnn.from_torch(bias)


def analyze_model_structure(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Analyze YOLO26 model structure from state dict.

    Args:
        state_dict: Model state dictionary

    Returns:
        Dictionary with model structure information
    """
    structure = {
        "layers": [],
        "backbone_layers": [],
        "neck_layers": [],
        "head_layers": [],
        "total_params": 0,
    }

    for key, value in state_dict.items():
        structure["total_params"] += value.numel()

        parts = key.split(".")
        if len(parts) >= 2:
            layer_idx = parts[1] if parts[0] == "model" else parts[0]

            layer_info = {
                "key": key,
                "shape": list(value.shape),
                "layer_idx": layer_idx,
            }
            structure["layers"].append(layer_info)

    print(f"Total parameters: {structure['total_params']:,}")
    return structure


def convert_weight_to_ttnn_format(weight: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """
    Convert PyTorch weight tensor to TTNN format.

    Args:
        weight: PyTorch tensor
        dtype: Target TTNN dtype

    Returns:
        TTNN tensor
    """
    return ttnn.from_torch(weight, dtype=dtype)


class YOLO26WeightLoader:
    """
    Weight loader for YOLO26 model.

    Handles loading and preprocessing weights from Ultralytics format
    to TTNN-compatible format with BatchNorm folding.
    """

    def __init__(self, state_dict: Dict[str, torch.Tensor]):
        """
        Initialize weight loader.

        Args:
            state_dict: Model state dictionary from Ultralytics
        """
        self.state_dict = state_dict
        self._cache = {}

    def get_conv_bn(self, prefix: str) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get folded Conv+BN weights."""
        if prefix not in self._cache:
            self._cache[prefix] = get_folded_conv_weights(self.state_dict, prefix)
        return self._cache[prefix]

    def get_conv_only(self, prefix: str) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Get Conv-only weights (no BN)."""
        if prefix not in self._cache:
            self._cache[prefix] = get_conv_only_weights(self.state_dict, prefix)
        return self._cache[prefix]

    def has_key(self, key: str) -> bool:
        """Check if key exists in state dict."""
        return key in self.state_dict

    def get_keys_with_prefix(self, prefix: str) -> list:
        """Get all keys starting with prefix."""
        return [k for k in self.state_dict.keys() if k.startswith(prefix)]

    def print_structure(self):
        """Print model structure for debugging."""
        analyze_model_structure(self.state_dict)
