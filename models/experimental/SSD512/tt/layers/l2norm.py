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
            x: Input tensor of shape [N, C, H, W]

        Returns:
            Normalized and scaled tensor of same shape as input.
        """
        # Compute L2 norm across channel dimension (dim=1)
        # Square -> sum across channels -> sqrt -> add eps
        squared = ttnn.mul(x, x)  # Element-wise square
        norm = ttnn.sqrt(ttnn.sum(squared, dim=1, keepdim=True)) + self.eps

        # Convert norm back to same layout as input for division
        norm = ttnn.to_layout(norm, layout=ttnn.TILE_LAYOUT)

        # Normalize by dividing by norm (broadcasting 1 across channel dim)
        x_norm = ttnn.div(x, norm)  # norm broadcasts from [N,1,H,W] to [N,C,H,W]

        x_norm = tt_to_torch_tensor(x_norm)
        x_norm = torch.permute(x_norm, (0, 3, 1, 2))
        x_norm = torch_to_tt_tensor_rm(x_norm, device=self.device)
        # Scale each channel by learned weight (broadcasts [1,C,1,1] across N,H,W)
        out = ttnn.mul(x_norm, self.weight)  # weight: [1,C,1,1], x_norm: [N,C,H,W]

        return out
