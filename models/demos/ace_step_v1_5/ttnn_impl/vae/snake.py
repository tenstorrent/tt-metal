# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of the Oobleck Snake1d activation.

Reference (diffusers / torch_ref):

    alpha = exp(alpha_log)
    beta  = exp(beta_log)
    y = x + 1/(beta + 1e-9) * sin(alpha * x)**2

The torch parameters live in shape ``[1, C, 1]`` (Conv1d-friendly). For TTNN
we keep tensors in ``[B, T, C]`` layout, so we reshape both parameters to
``[1, 1, C]`` once at init and let TTNN broadcast.
"""

from __future__ import annotations

import numpy as np

from .._ttnn import get_ttnn


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


def _snake_param_to_btc(arr) -> np.ndarray:
    """Reshape a Snake1d parameter from ``[1, C, 1]`` (or ``[C]``) to ``[1, 1, 1, C]``."""
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == 1:
        a = a.reshape(1, 1, 1, -1)
    elif a.ndim == 1:
        a = a.reshape(1, 1, 1, -1)
    else:
        a = a.reshape(1, 1, 1, -1)
    return np.ascontiguousarray(a)


class TtSnake1d:
    """Channel-wise Snake activation operating on ``[B, T, C]`` row-major tensors."""

    def __init__(
        self,
        *,
        alpha_host,
        beta_host,
        device,
        dtype=None,
        memory_config=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.dtype = dtype or getattr(ttnn, "bfloat16", None) or getattr(ttnn, "float16", None)
        if self.dtype is None:
            raise RuntimeError("TTNN build missing a usable dtype (bfloat16/float16)")
        self.memory_config = memory_config if memory_config is not None else getattr(ttnn, "DRAM_MEMORY_CONFIG", None)

        alpha = _snake_param_to_btc(alpha_host)
        beta = _snake_param_to_btc(beta_host)
        self.hidden_dim = int(alpha.shape[-1])

        # Pre-compute exp(alpha) and exp(beta) on host (logscale=True for Oobleck) so device-time
        # arithmetic is one mul + sin + square + add per channel instead of also paying for exp.
        alpha_exp = np.exp(alpha).astype(np.float32)
        # beta_recip = 1 / (exp(beta) + 1e-9)
        beta_recip = (1.0 / (np.exp(beta).astype(np.float32) + 1e-9)).astype(np.float32)

        self._alpha_exp = ttnn.as_tensor(
            alpha_exp, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config
        )
        self._beta_recip = ttnn.as_tensor(
            beta_recip, device=device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, memory_config=self.memory_config
        )

    def __call__(self, x):
        """Apply Snake activation in-place-ish.

        Args:
            x: TTNN tensor of shape ``[B, T, C]`` (row-major) or ``[B, 1, T, C]`` (tile).

        Returns:
            TTNN tensor with the same shape and layout as the input.
        """
        ttnn = self.ttnn
        # Move to a tile-friendly rank-4 layout for elementwise ops.
        orig_shape = tuple(x.shape)
        if len(orig_shape) == 3:
            x4 = ttnn.unsqueeze(x, 1)
            squeeze_back = True
        elif len(orig_shape) == 4:
            x4 = x
            squeeze_back = False
        else:
            raise ValueError(f"TtSnake1d expects rank-3 [B,T,C] or rank-4 [B,1,T,C]; got {orig_shape}")

        x4 = ttnn.to_layout(x4, ttnn.TILE_LAYOUT)

        ax = ttnn.multiply(x4, self._alpha_exp)
        s = ttnn.sin(ax)
        ttnn.deallocate(ax)
        s2 = ttnn.square(s)
        ttnn.deallocate(s)
        term = ttnn.multiply(s2, self._beta_recip)
        ttnn.deallocate(s2)
        y4 = ttnn.add(x4, term)
        ttnn.deallocate(term)

        if squeeze_back:
            y = ttnn.squeeze(y4, 1)
            return y
        return y4
