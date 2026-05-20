# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of ``OobleckResidualUnit``.

Reference forward (diffusers / torch_ref):

    out = conv2(snake2(conv1(snake1(x))))
    if out is shorter than x:  trim x on both sides
    return x + out
"""

from __future__ import annotations

from .._ttnn import get_ttnn
from .conv1d import TtConv1d
from .snake import TtSnake1d


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


class TtOobleckResidualUnit:
    """``Snake -> Conv1d(k=7, dilated) -> Snake -> Conv1d(k=1)`` residual."""

    def __init__(
        self,
        *,
        weights: dict,
        dimension: int,
        dilation: int,
        device,
        activation_dtype=None,
        weights_dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.dimension = int(dimension)
        self.dilation = int(dilation)

        pad = ((7 - 1) * self.dilation) // 2
        self.snake1 = TtSnake1d(
            alpha_host=weights["snake1.alpha"],
            beta_host=weights["snake1.beta"],
            device=device,
            dtype=activation_dtype,
        )
        self.conv1 = TtConv1d(
            weight_host=weights["conv1.weight"],
            bias_host=weights.get("conv1.bias"),
            in_channels=self.dimension,
            out_channels=self.dimension,
            kernel_size=7,
            stride=1,
            padding=pad,
            dilation=self.dilation,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        self.snake2 = TtSnake1d(
            alpha_host=weights["snake2.alpha"],
            beta_host=weights["snake2.beta"],
            device=device,
            dtype=activation_dtype,
        )
        self.conv2 = TtConv1d(
            weight_host=weights["conv2.weight"],
            bias_host=weights.get("conv2.bias"),
            in_channels=self.dimension,
            out_channels=self.dimension,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

    def __call__(self, x):
        """Forward pass on ``[B, T, C]`` row-major TTNN tensor."""
        ttnn = self.ttnn
        if len(x.shape) != 3:
            raise ValueError(f"TtOobleckResidualUnit expects rank-3 [B,T,C], got {x.shape}")

        # Skip-connection trim uses ``x[:, pad:pad+y_T, :]`` → ``ttnn.slice``. TILE slices require
        # 32-aligned starts/sizes on the last two dims; activations here are often TILE after convs.
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # Keep the residual skip tensor in DRAM so it does not occupy L1 during conv1 (k=7).
        # The k=7 conv program's static CB region extends to 139328; any live L1 buffer below
        # that address causes a "CB clashes with L1 buffer" fatal error at program compile time.
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)

        x_T = int(x.shape[1])
        y_T = int(y.shape[1])
        if y_T < x_T:
            pad = (x_T - y_T) // 2
            if pad > 0:
                x = x[:, pad : pad + y_T, :]

        # Both branches are ROW_MAJOR [B,T,C] after snake/conv; elementwise add needs no TILE rank-4 path.
        return ttnn.add(x, y)
