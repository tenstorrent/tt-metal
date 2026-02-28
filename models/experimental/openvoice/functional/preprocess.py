# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Model Parameter Preprocessing for TTNN.

Follows the official TTNN pattern using preprocess_model_parameters
for weight loading and conversion.

Reference: https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/converting_torch_model_to_ttnn.html
"""

from typing import Any, Callable, Dict, Optional

import torch

try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False


def fuse_weight_normalization(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Fuse weight normalization (weight_g, weight_v) into regular weights.

    OpenVoice/MeloTTS checkpoints use PyTorch's weight_norm which stores
    weight_g (magnitude) and weight_v (direction) separately.

    Actual weight = g * (v / ||v||)
    """
    fused = {}
    processed_prefixes = set()

    for key, value in state_dict.items():
        # Check if this is a weight_g tensor
        if key.endswith(".weight_g"):
            prefix = key[:-9]  # Remove '.weight_g'
            processed_prefixes.add(prefix)

            g = value
            v_key = f"{prefix}.weight_v"
            v = state_dict.get(v_key)

            if v is not None:
                # Compute fused weight: w = g * (v / ||v||)
                # v shape varies: [out, in, kernel] for conv, etc.
                dims = tuple(range(1, v.dim()))
                v_norm = torch.norm(v, dim=dims, keepdim=True)
                weight = g * (v / (v_norm + 1e-7))
                fused[f"{prefix}.weight"] = weight
            else:
                # Just g without v - shouldn't happen but handle gracefully
                fused[key] = value

        elif key.endswith(".weight_v"):
            prefix = key[:-9]
            if prefix not in processed_prefixes:
                # v without g - also shouldn't happen
                fused[key] = value

        else:
            # Regular parameter - keep as is
            fused[key] = value

    return fused


def custom_preprocessor(
    model_state_dict: Dict[str, torch.Tensor],
    *,
    dtype: Optional[Any] = None,
    device: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Custom preprocessor for OpenVoice/MeloTTS models.

    Handles:
    1. Weight normalization fusion
    2. Dtype conversion (bfloat16 for activations, fp32 for precision-critical)
    3. Layout conversion (TILE_LAYOUT for efficiency)
    4. Device placement

    Args:
        model_state_dict: PyTorch state dict
        dtype: Target dtype (default: ttnn.bfloat16)
        device: TTNN device for parameter placement

    Returns:
        Dict of preprocessed parameters ready for TTNN inference
    """
    if not TTNN_AVAILABLE:
        return model_state_dict

    if dtype is None:
        dtype = ttnn.bfloat16

    # Fuse weight normalization
    fused_dict = fuse_weight_normalization(model_state_dict)

    parameters = {}
    for name, param in fused_dict.items():
        if param is None:
            parameters[name] = None
            continue

        # Determine appropriate dtype
        param_dtype = dtype

        # Use higher precision for certain critical operations
        if any(x in name for x in ["layer_norm", "ln", "norm", "gamma", "beta"]):
            param_dtype = ttnn.bfloat16  # LN needs precision

        # Convert to TTNN tensor
        if device is not None:
            try:
                # Try to place on device with TILE_LAYOUT
                tt_param = ttnn.from_torch(
                    param.float().contiguous(),
                    dtype=param_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            except Exception:
                # Fallback: keep on host
                tt_param = ttnn.from_torch(
                    param.float().contiguous(),
                    dtype=param_dtype,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
        else:
            tt_param = ttnn.from_torch(
                param.float().contiguous(),
                dtype=param_dtype,
            )

        parameters[name] = tt_param

    return parameters


def preprocess_model_parameters(
    *,
    initialize_model: Optional[Callable] = None,
    model_state_dict: Optional[Dict[str, torch.Tensor]] = None,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
    custom_preprocessor: Optional[Callable] = None,
    convert_to_ttnn: bool = True,
) -> Dict[str, Any]:
    """
    Preprocess model parameters for TTNN inference.

    This follows the official TTNN pattern for model conversion.

    Usage:
        # Option 1: From PyTorch model
        parameters = preprocess_model_parameters(
            initialize_model=lambda: MyModel(),
            device=device,
            convert_to_ttnn=True,
        )

        # Option 2: From state dict
        parameters = preprocess_model_parameters(
            model_state_dict=torch.load("checkpoint.pth"),
            device=device,
            convert_to_ttnn=True,
        )

    Args:
        initialize_model: Lambda that creates and returns PyTorch model
        model_state_dict: Pre-loaded state dict (alternative to initialize_model)
        device: TTNN device for placement
        dtype: Target dtype (default: bfloat16)
        custom_preprocessor: Optional custom preprocessing function
        convert_to_ttnn: Whether to convert to TTNN tensors

    Returns:
        Dict of preprocessed parameters
    """
    # Get state dict
    if model_state_dict is not None:
        state_dict = model_state_dict
    elif initialize_model is not None:
        model = initialize_model()
        if hasattr(model, "state_dict"):
            state_dict = model.state_dict()
        else:
            state_dict = model
    else:
        raise ValueError("Must provide either initialize_model or model_state_dict")

    # Handle nested model key
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]

    if not convert_to_ttnn:
        # Just fuse weight norm, keep as PyTorch
        return fuse_weight_normalization(state_dict)

    # Use custom preprocessor if provided
    if custom_preprocessor is not None:
        return custom_preprocessor(state_dict, dtype=dtype, device=device)

    # Use default preprocessor
    from . import custom_preprocessor as default_preprocessor

    return default_preprocessor(state_dict, dtype=dtype, device=device)


class PreprocessedParameters:
    """
    Container for preprocessed model parameters.

    Provides dict-like access and tracks parameter statistics.
    """

    def __init__(self, parameters: Dict[str, Any], device: Optional[Any] = None):
        self._parameters = parameters
        self._device = device
        self._stats = self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute parameter statistics."""
        total_params = 0
        total_bytes = 0

        for name, param in self._parameters.items():
            if param is None:
                continue

            if isinstance(param, torch.Tensor):
                total_params += param.numel()
                total_bytes += param.numel() * param.element_size()
            elif TTNN_AVAILABLE and hasattr(param, "shape"):
                # TTNN tensor
                numel = 1
                for dim in param.shape:
                    numel *= dim
                total_params += numel
                # Approximate bytes (bfloat16 = 2 bytes)
                total_bytes += numel * 2

        return {
            "total_parameters": total_params,
            "total_bytes": total_bytes,
            "num_tensors": len(self._parameters),
        }

    def __getitem__(self, key: str) -> Any:
        return self._parameters.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def get(self, key: str, default: Any = None) -> Any:
        return self._parameters.get(key, default)

    def keys(self):
        return self._parameters.keys()

    def items(self):
        return self._parameters.items()

    def values(self):
        return self._parameters.values()

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    def to_device(self, device: Any) -> "PreprocessedParameters":
        """Move parameters to device."""
        if not TTNN_AVAILABLE:
            return self

        new_params = {}
        for name, param in self._parameters.items():
            if param is None:
                new_params[name] = None
            elif hasattr(param, "to_device"):
                new_params[name] = ttnn.to_device(param, device)
            else:
                new_params[name] = param

        return PreprocessedParameters(new_params, device)
