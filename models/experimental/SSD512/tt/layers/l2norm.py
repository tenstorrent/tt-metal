# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


"""TTNN implementation of L2 normalization with learned scale, matching the reference L2Norm module."""

import ttnn
import torch
from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtL2Norm:
    def __init__(self, n_channels, scale=20, eps=1e-10, device=None):
        """Create an L2Norm module with learnable per-channel scale.

        Args:
            n_channels: Number of input channels to normalize.
            scale: Initial scale value for all channels.
            eps: Small value added to norm for numerical stability.
            device: Device to place tensors on.
        """
        self.n_channels = n_channels
        self.eps = eps
        self.device = device

        # Create learnable scale parameter initialized to scale
        # Shape: [1, C, 1, 1] for broadcasting across batch/spatial dims
        self.weight = ttnn.full([1, n_channels, 1, 1], scale, device=device)

    def __call__(self, x):
        """Apply L2 normalization and learned scale.

        Args:
            x: Input tensor (ttnn.Tensor) in NHWC format [N, H, W, C] from VGG backbone

        Returns:
            Normalized and scaled tensor in NCHW format [N, C, H, W]
        """
        # Input x is in NHWC format from VGG backbone: [N, H, W, C]
        # Convert to torch to check format and convert to NCHW
        x_torch = tt_to_torch_tensor(x)

        # Check if already in NCHW format (shape[1] == channels)
        if x_torch.shape[1] == self.n_channels:
            # Already in NCHW format
            x_nchw = x_torch
        else:
            # Assume NHWC format: [N, H, W, C] -> [N, C, H, W]
            x_nchw = x_torch.permute(0, 3, 1, 2)

        # Convert back to TTNN in NCHW format for L2Norm computation
        x_nchw_ttnn = torch_to_tt_tensor_rm(x_nchw, device=self.device)

        # Compute L2 norm across channel dimension (dim=1 in NCHW)
        # Square -> sum across channels -> sqrt -> add eps
        squared = ttnn.mul(x_nchw_ttnn, x_nchw_ttnn)
        squared = ttnn.to_layout(squared, layout=ttnn.TILE_LAYOUT)
        norm = ttnn.sqrt(ttnn.sum(squared, dim=1, keepdim=True)) + self.eps  # Sum along C dimension in NCHW

        # Convert norm back to same layout as input for division
        norm = ttnn.to_layout(norm, layout=ttnn.TILE_LAYOUT)

        # Normalize by dividing by norm (broadcasting [N,1,H,W] across [N,C,H,W])
        x_norm = ttnn.div(x_nchw_ttnn, norm)

        # Scale each channel by learned weight (weight is [1, C, 1, 1] in NCHW)
        # This broadcasts [1,C,1,1] across [N,C,H,W]
        out = ttnn.mul(x_norm, self.weight)

        return out


def l2norm(input_tensor, num_channels=512, scale=20.0, device=None):
    """
    Function wrapper for TtL2Norm for convenience.

    Args:
        input_tensor: Input tensor (torch.Tensor in NCHW format)
        num_channels: Number of channels (default: 512)
        scale: Scale parameter (default: 20.0)
        device: TTNN device object

    Returns:
        Normalized and scaled tensor (ttnn.Tensor in NCHW format)
    """
    # Create L2Norm instance
    l2norm_module = TtL2Norm(n_channels=num_channels, scale=scale, device=device)

    # Apply L2Norm
    # Note: TtL2Norm expects input in NCHW format and returns TTNN tensor
    # If input is torch tensor, convert to TTNN first
    if isinstance(input_tensor, torch.Tensor):
        # Convert torch tensor to TTNN tensor in NCHW format
        # TtL2Norm expects NCHW format
        input_ttnn = torch_to_tt_tensor_rm(input_tensor, device=device)
        output = l2norm_module(input_ttnn)
        return output
    else:
        # Already TTNN tensor, apply directly
        output = l2norm_module(input_tensor)
        return output
