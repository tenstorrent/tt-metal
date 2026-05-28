# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of ``OobleckDecoderBlock``.

One upsampling stage:
    ``snake1 -> conv_t1 -> res_unit1 -> res_unit2 -> res_unit3``
"""

from __future__ import annotations

import math

from .._ttnn import get_ttnn
from ..math_perf_env import ace_step_vae_ensure_interleaved, ace_step_vae_synchronize
from .conv1d import TtConvTranspose1d
from .residual import TtOobleckResidualUnit
from .snake import TtSnake1d


def _require_ttnn():
    ttnn = get_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is required for ace_step_v1_5.ttnn_impl.vae")
    return ttnn


def _strip_prefix(d: dict, prefix: str) -> dict:
    """Return a new dict with keys that started with ``prefix`` re-keyed without it."""
    plen = len(prefix)
    return {k[plen:]: v for k, v in d.items() if k.startswith(prefix)}


class TtOobleckDecoderBlock:
    """One Oobleck decoder upsampling stage."""

    def __init__(
        self,
        *,
        weights: dict,
        input_dim: int,
        output_dim: int,
        stride: int,
        device,
        activation_dtype=None,
        weights_dtype=None,
    ) -> None:
        ttnn = _require_ttnn()
        self.ttnn = ttnn
        self.device = device
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.stride = int(stride)

        self.snake1 = TtSnake1d(
            alpha_host=weights["snake1.alpha"],
            beta_host=weights["snake1.beta"],
            device=device,
            dtype=activation_dtype,
        )
        self.conv_t1 = TtConvTranspose1d(
            weight_host=weights["conv_t1.weight"],
            bias_host=weights.get("conv_t1.bias"),
            in_channels=self.input_dim,
            out_channels=self.output_dim,
            kernel_size=2 * self.stride,
            stride=self.stride,
            padding=math.ceil(self.stride / 2),
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        self.res_unit1 = TtOobleckResidualUnit(
            weights=_strip_prefix(weights, "res_unit1."),
            dimension=self.output_dim,
            dilation=1,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        self.res_unit2 = TtOobleckResidualUnit(
            weights=_strip_prefix(weights, "res_unit2."),
            dimension=self.output_dim,
            dilation=3,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )
        self.res_unit3 = TtOobleckResidualUnit(
            weights=_strip_prefix(weights, "res_unit3."),
            dimension=self.output_dim,
            dilation=9,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

    def __call__(self, x):
        """Forward pass on ``[B, T, C]`` row-major TTNN tensor.

        Output: ``[B, T*stride, output_dim]`` row-major.
        """
        ttnn = self.ttnn
        dram_mc = ttnn.DRAM_MEMORY_CONFIG
        x = ace_step_vae_ensure_interleaved(ttnn, x, memory_config=dram_mc)
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_mc)
        x = self.snake1(x)
        x = self.conv_t1(x)
        ace_step_vae_synchronize(ttnn, self.device)
        x = self.res_unit1(x)
        x = self.res_unit2(x)
        x = self.res_unit3(x)
        return x
