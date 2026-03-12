# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
Tensor format conversion utilities for ComfyUI Bridge.

Provides bidirectional conversion between PyTorch standard format and TT-Metal format:
- PyTorch standard: [B, C, H, W]
- TT-Metal format: [B, 1, H*W, C]

These utilities are model-agnostic and support any channel count, making them
reusable across different models (SDXL, SD1.5, VAE, etc.).

Key Functions:
    torch_to_tt_format: Convert PyTorch tensors to TT-Metal format
    tt_to_torch_format: Convert TT-Metal tensors to PyTorch format
    validate_tensor_format: Validate tensor shape and dtype
    detect_format: Auto-detect tensor format

Usage Example:
    >>> tensor = torch.randn(1, 4, 128, 128)  # Standard PyTorch format
    >>> tt_tensor = torch_to_tt_format(tensor, expected_channels=4)
    >>> print(tt_tensor.shape)  # [1, 1, 16384, 4]
    >>> restored = tt_to_torch_format(tt_tensor, expected_channels=4)
    >>> print(restored.shape)  # [1, 4, 128, 128]
"""

import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def torch_to_tt_format(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    """
    Convert PyTorch standard format [B, C, H, W] to TT-Metal format [B, 1, H*W, C].

    TT-Metal's ttnn operations expect tensors in the format [B, 1, H*W, C] where:
    - B: Batch size
    - 1: Sentinel dimension (always 1)
    - H*W: Flattened spatial dimensions
    - C: Channel count

    This function is model-agnostic and works with any channel count.

    Args:
        tensor: Input tensor in PyTorch format [B, C, H, W]
        expected_channels: Expected number of channels for validation
            Examples: 4 (SDXL latents), 3 (RGB images), 1 (grayscale)

    Returns:
        Tensor in TT-Metal format [B, 1, H*W, C]

    Raises:
        ValueError: If tensor is not 4D, channels don't match, or shape is invalid

    Example:
        >>> latents = torch.randn(2, 4, 128, 128)  # SDXL latents
        >>> tt_latents = torch_to_tt_format(latents, expected_channels=4)
        >>> print(tt_latents.shape)  # torch.Size([2, 1, 16384, 4])
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor in format [B, C, H, W], got {tensor.dim()}D tensor: {tensor.shape}")

    B, C, H, W = tensor.shape

    if C != expected_channels:
        raise ValueError(f"Channel mismatch: expected {expected_channels}, got {C}. " f"Tensor shape: {tensor.shape}")

    logger.debug(f"Converting torch format [B={B}, C={C}, H={H}, W={W}] to TT format")

    # [B, C, H, W] -> [B, H, W, C] -> [B, 1, H*W, C]
    tensor = tensor.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    tensor = tensor.reshape(B, 1, H * W, C)  # [B, H, W, C] -> [B, 1, H*W, C]

    logger.debug(f"Converted to TT format: {tensor.shape}")
    return tensor


def tt_to_torch_format(tensor: torch.Tensor, expected_channels: int) -> torch.Tensor:
    """
    Convert TT-Metal format [B, 1, H*W, C] to PyTorch standard format [B, C, H, W].

    Reverses the conversion performed by torch_to_tt_format. This function
    automatically computes H and W from the flattened spatial dimension H*W.

    This function is model-agnostic and works with any channel count.

    Args:
        tensor: Input tensor in TT-Metal format [B, 1, H*W, C]
        expected_channels: Expected number of channels for validation
            Examples: 4 (SDXL latents), 3 (RGB images), 1 (grayscale)

    Returns:
        Tensor in PyTorch format [B, C, H, W]

    Raises:
        ValueError: If tensor is not 4D, channels don't match, dimension[1] is not 1,
            or spatial dimensions cannot be computed as perfect square

    Example:
        >>> tt_latents = torch.randn(2, 1, 16384, 4)  # TT format
        >>> latents = tt_to_torch_format(tt_latents, expected_channels=4)
        >>> print(latents.shape)  # torch.Size([2, 4, 128, 128])
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor in format [B, 1, H*W, C], got {tensor.dim()}D tensor: {tensor.shape}")

    B, dim1, HW, C = tensor.shape

    if dim1 != 1:
        raise ValueError(f"Expected dimension[1]=1 for TT format, got {dim1}. " f"Tensor shape: {tensor.shape}")

    if C != expected_channels:
        raise ValueError(f"Channel mismatch: expected {expected_channels}, got {C}. " f"Tensor shape: {tensor.shape}")

    # Compute H and W from flattened spatial dimension
    # Assumes square spatial dimensions (H == W)
    H = int(HW**0.5)
    W = HW // H

    if H * W != HW:
        raise ValueError(
            f"Cannot compute square spatial dimensions from H*W={HW}. "
            f"Expected perfect square (e.g., 16384 = 128 * 128). "
            f"Computed H={H}, W={W}, H*W={H*W}"
        )

    logger.debug(f"Converting TT format [B={B}, 1, H*W={HW}, C={C}] to torch format " f"[B={B}, C={C}, H={H}, W={W}]")

    # [B, 1, H*W, C] -> [B, H, W, C] -> [B, C, H, W]
    tensor = tensor.reshape(B, H, W, C)  # [B, 1, H*W, C] -> [B, H, W, C]
    tensor = tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

    logger.debug(f"Converted to torch format: {tensor.shape}")
    return tensor


def validate_tensor_format(
    tensor: torch.Tensor,
    expected_channels: int,
    model_type: str = "sdxl",
    expected_dtype: Optional[torch.dtype] = None,
) -> None:
    """
    Validate tensor format, shape, channels, and dtype.

    Performs comprehensive validation to ensure tensor meets expected requirements.
    Raises clear, actionable error messages on validation failures.

    Args:
        tensor: Tensor to validate (either PyTorch or TT format)
        expected_channels: Expected number of channels
            Examples: 4 (SDXL latents), 3 (RGB), 1 (grayscale)
        model_type: Model type for error messages (default: "sdxl")
            Used to provide context-specific error messages
        expected_dtype: Expected tensor dtype (default: None, no dtype check)
            Examples: torch.float32, torch.float16

    Raises:
        ValueError: If tensor fails validation with detailed error message

    Example:
        >>> tensor = torch.randn(1, 4, 128, 128)
        >>> validate_tensor_format(tensor, expected_channels=4, model_type="sdxl")
        # Passes silently
        >>> validate_tensor_format(tensor, expected_channels=3, model_type="sdxl")
        # Raises ValueError with clear message
    """
    # Validate tensor is a torch.Tensor
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected torch.Tensor, got {type(tensor).__name__}. " f"Model: {model_type}")

    # Validate 4D tensor
    if tensor.dim() != 4:
        raise ValueError(
            f"Expected 4D tensor for {model_type}, got {tensor.dim()}D tensor: {tensor.shape}. "
            f"Valid formats are [B, C, H, W] (PyTorch) or [B, 1, H*W, C] (TT-Metal)"
        )

    # Detect format
    detected_format = detect_format(tensor)
    logger.debug(f"Detected format: {detected_format} for tensor shape {tensor.shape}")

    # Validate channels based on format
    if detected_format == "torch":
        B, C, H, W = tensor.shape
        if C != expected_channels:
            raise ValueError(
                f"Channel mismatch for {model_type} (PyTorch format): "
                f"expected {expected_channels}, got {C}. "
                f"Tensor shape: {tensor.shape}"
            )
    elif detected_format == "tt":
        B, _, HW, C = tensor.shape
        if C != expected_channels:
            raise ValueError(
                f"Channel mismatch for {model_type} (TT format): "
                f"expected {expected_channels}, got {C}. "
                f"Tensor shape: {tensor.shape}"
            )
    else:
        raise ValueError(
            f"Ambiguous tensor format for {model_type}: {tensor.shape}. "
            f"Cannot determine if PyTorch [B, C, H, W] or TT [B, 1, H*W, C] format. "
            f"Expected channels: {expected_channels}"
        )

    # Validate dtype if specified
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise ValueError(
            f"Dtype mismatch for {model_type}: expected {expected_dtype}, got {tensor.dtype}. "
            f"Tensor shape: {tensor.shape}"
        )

    logger.debug(
        f"Validation passed for {model_type}: shape={tensor.shape}, "
        f"format={detected_format}, channels={expected_channels}"
    )


def detect_format(tensor: torch.Tensor) -> str:
    """
    Auto-detect tensor format based on shape heuristics.

    Distinguishes between PyTorch standard format [B, C, H, W] and TT-Metal
    format [B, 1, H*W, C] by analyzing tensor dimensions.

    Detection Logic:
        1. If dim[1] == 1 and dim[2] >> dim[3]: Likely TT format [B, 1, H*W, C]
        2. If dim[1] << dim[2] and dim[1] << dim[3]: Likely PyTorch format [B, C, H, W]
        3. Otherwise: Ambiguous format

    Args:
        tensor: 4D tensor to analyze

    Returns:
        str: Format identifier
            - "torch": PyTorch standard format [B, C, H, W]
            - "tt": TT-Metal format [B, 1, H*W, C]
            - "ambiguous": Cannot determine format

    Example:
        >>> torch_tensor = torch.randn(1, 4, 128, 128)
        >>> detect_format(torch_tensor)
        'torch'
        >>> tt_tensor = torch.randn(1, 1, 16384, 4)
        >>> detect_format(tt_tensor)
        'tt'
    """
    if tensor.dim() != 4:
        logger.warning(f"detect_format expects 4D tensor, got {tensor.dim()}D: {tensor.shape}")
        return "ambiguous"

    B, dim1, dim2, dim3 = tensor.shape

    # TT format: [B, 1, H*W, C]
    # Characteristics: dim1 == 1, dim2 is large (H*W), dim3 is small (C)
    if dim1 == 1 and dim2 > dim3 * 10:  # H*W >> C (heuristic: at least 10x)
        logger.debug(f"Detected TT format: [B={B}, 1, H*W={dim2}, C={dim3}]")
        return "tt"

    # PyTorch format: [B, C, H, W]
    # Characteristics: dim1 is small (C), dim2 and dim3 are larger (H, W)
    if dim1 < dim2 and dim1 < dim3:
        logger.debug(f"Detected PyTorch format: [B={B}, C={dim1}, H={dim2}, W={dim3}]")
        return "torch"

    # Ambiguous: Cannot determine format with confidence
    logger.warning(
        f"Ambiguous tensor format: {tensor.shape}. " f"Cannot determine if PyTorch [B, C, H, W] or TT [B, 1, H*W, C]"
    )
    return "ambiguous"


def infer_spatial_dims(HW: int) -> Tuple[int, int]:
    """
    Infer spatial dimensions (H, W) from flattened spatial size H*W.

    Assumes square spatial dimensions (H == W) and computes them from H*W.
    This is a common assumption for models like SDXL (128x128 latents).

    Args:
        HW: Flattened spatial size (H * W)

    Returns:
        Tuple[int, int]: (H, W) spatial dimensions

    Raises:
        ValueError: If H*W is not a perfect square

    Example:
        >>> infer_spatial_dims(16384)
        (128, 128)
        >>> infer_spatial_dims(65536)
        (256, 256)
    """
    H = int(HW**0.5)
    W = HW // H

    if H * W != HW:
        raise ValueError(
            f"Cannot infer square spatial dimensions from H*W={HW}. "
            f"Expected perfect square (e.g., 16384 = 128 * 128). "
            f"Computed H={H}, W={W}, H*W={H*W}"
        )

    return H, W
