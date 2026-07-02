# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI
# SPDX-License-Identifier: Apache-2.0

"""
Functional TTNN Operations for OpenVoice.

Implements core operations in functional style following official TTNN patterns.
Each function takes tensors and parameters, returns output tensors.
"""

from typing import Any

import torch
import torch.nn.functional as F

import ttnn

# ============================================================================
# Shared helpers (canonical implementations — import from here, don't copy)
# ============================================================================


def to_torch_tensor(t, dtype=torch.float32):
    """Convert a TTNN or PyTorch tensor to PyTorch with the given dtype."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.to(dtype) if t.dtype != dtype else t
    return ttnn.to_torch(t).to(dtype)


def ensure_conv1d_weight(w):
    """Ensure weight tensor has correct shape for F.conv1d [out, in, kernel]."""
    if w is None:
        return None
    if w.dim() == 2:
        return w.unsqueeze(2)
    return w


class LayerNorm1d:
    """Layer normalization for 1D sequences (channel-first)."""

    def __init__(self, channels: int, weight: Any = None, bias: Any = None, eps: float = 1e-5):
        self.channels = channels
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def __call__(self, x: Any) -> Any:
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            x = x.transpose(1, -1)
            x = F.layer_norm(x, (self.channels,), self.weight, self.bias, self.eps)
            return x.transpose(1, -1)

        x = ttnn.permute(x, (0, 2, 1))
        x = ttnn.layer_norm(x, weight=self.weight, bias=self.bias, epsilon=self.eps)
        x = ttnn.permute(x, (0, 2, 1))
        return x


class Flip:
    """
    Flip operation for normalizing flows.

    Reverses the channel dimension to alternate which half of channels
    is transformed in coupling layers.

    Note: Uses CPU roundtrip because TTNN lacks native flip operation.
    Impact is minimal (~0.01ms) as this is a simple memory copy.
    """

    def __call__(self, x: Any, *args, reverse: bool = False, **kwargs):
        is_torch = isinstance(x, torch.Tensor)

        if is_torch:
            x = torch.flip(x, [1])
            if not reverse:
                logdet = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
                return x, logdet
            return x

        # CPU roundtrip required - TTNN has no native flip operation
        was_on_device = ttnn.is_tensor_storage_on_device(x)
        device = x.device() if was_on_device else None
        orig_layout = x.get_layout()

        x_torch = ttnn.to_torch(x)
        x_flipped = torch.flip(x_torch, [1])
        x = ttnn.from_torch(x_flipped, dtype=ttnn.bfloat16, layout=orig_layout)

        if was_on_device and device is not None:
            x = ttnn.to_device(x, device)

        if not reverse:
            batch = x.shape[0]
            logdet = ttnn.zeros((batch,), dtype=x.dtype)
            return x, logdet
        return x
