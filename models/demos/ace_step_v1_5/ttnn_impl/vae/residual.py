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
from ..math_perf_env import ace_step_linear_l1_memory_config
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
        # snake2 feeds conv2 (k=1) — no k>7 static CB after this point, so L1 output is safe.
        # This eliminates the snake DRAM write and conv2's _maybe_l1 DRAM→L1 copy.
        self.snake2 = TtSnake1d(
            alpha_host=weights["snake2.alpha"],
            beta_host=weights["snake2.beta"],
            device=device,
            dtype=activation_dtype,
            output_memory_config=ace_step_linear_l1_memory_config(ttnn),
        )
        # Keep 1×1 as ``TtConv1d`` (not ``ttnn.linear``): mcast L1 matmul CBs exceed Blackhole budget
        # at production ``T×C`` (e.g. 512 ch × long audio frames); conv1d 1×1 L1 path is validated E2E.
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

        dram_mc = ttnn.DRAM_MEMORY_CONFIG

        # Keep the residual skip in DRAM during k=7 conv1 (static CB region on Blackhole).
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if x.memory_config() != dram_mc:
            x = ttnn.to_memory_config(x, dram_mc)

        y = self.snake1(x)
        # conv1→snake2 TILE contract: return_sharded tries HEIGHT_SHARDED L1 when
        # ACE_STEP_VAE_K7_SHARDED_OUTPUT=1; otherwise falls back to return_tile (DRAM TILE).
        # snake2 accepts TILE (DRAM→L1 DMA or L1 passthrough) and skips Tilize on ROW_MAJOR.
        y = self.conv1(y, return_sharded=True)
        # snake2→conv2 TILE contract: return_tile keeps L1 TILE out (no Untilize) so conv2
        # k=1 uses TILE in0 on the linear path or a single Untilize before conv1d.
        y = self.snake2(y, return_tile=True)
        y = self.conv2(y)

        x_T = int(x.shape[1])
        y_T = int(y.shape[1])
        if y_T < x_T:
            pad = (x_T - y_T) // 2
            if pad > 0:
                x = x[:, pad : pad + y_T, :]

        # conv1 static CB is now freed; move x from DRAM to L1 so both add operands are L1.
        # add output goes to DRAM so the next residual unit's DRAM check is a no-op.
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        x = ttnn.to_memory_config(x, l1_mc)
        return ttnn.add(x, y, memory_config=dram_mc)
