# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch
from models.common.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor


class TtL2Norm:
    def __init__(self, n_channels, scale=20, eps=1e-10, device=None):
        """Create an L2Norm module with learnable per-channel scale."""
        self.n_channels = n_channels
        self.eps = eps
        self.device = device
        self.weight = ttnn.full([1, n_channels, 1, 1], scale, device=device)

    def __call__(self, x):
        """Apply L2 normalization and learned scale."""
        x_torch = tt_to_torch_tensor(x)
        if x_torch.shape[1] == self.n_channels:
            x_nchw = x_torch
        else:
            x_nchw = x_torch.permute(0, 3, 1, 2)

        x_nchw_ttnn = torch_to_tt_tensor_rm(x_nchw, device=self.device)

        squared = ttnn.mul(x_nchw_ttnn, x_nchw_ttnn)
        squared = ttnn.to_layout(squared, layout=ttnn.TILE_LAYOUT)
        norm = ttnn.sqrt(ttnn.sum(squared, dim=1, keepdim=True)) + self.eps
        norm = ttnn.to_layout(norm, layout=ttnn.TILE_LAYOUT)
        x_norm = ttnn.div(x_nchw_ttnn, norm)

        out = ttnn.mul(x_norm, self.weight)

        return out


def l2norm(input_tensor, num_channels=512, scale=20.0, device=None):
    """Function wrapper for TtL2Norm for convenience."""
    l2norm_module = TtL2Norm(n_channels=num_channels, scale=scale, device=device)

    if isinstance(input_tensor, torch.Tensor):
        input_ttnn = torch_to_tt_tensor_rm(input_tensor, device=device)
        output = l2norm_module(input_ttnn)
        return output
    else:
        output = l2norm_module(input_tensor)
        return output
