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
from ..math_perf_env import ace_step_linear_l1_memory_config, ace_step_safe_deallocate, ace_step_vae_synchronize
from .conv1d import TtConv1d
from .snake import TtSnake1d

# Minimum time dimension to engage the HEIGHT_SHARDED add path.
# Below this threshold the Tilize + InterleavedToSharded conversion overhead
# outweighs the benefit of sharded compute (Block 0: T=256 is too small).
# Blocks 1-4 (T >= 2048) benefit from the sharded path.
_SHARDED_ADD_MIN_T = 512


def _height_sharded_add(ttnn, device, x, y, *, l1_mc, dram_mc):
    """Add two L1-interleaved ROW_MAJOR tensors via HEIGHT_SHARDED TILE compute.

    Steps: tilize both → interleaved_to_sharded (same spec) → add → sharded_to_interleaved
    → untilize to DRAM ROW_MAJOR.  Falls back to a plain L1 add on any error so correctness
    is never compromised by an unexpected shard-spec or L1 budget failure.
    """
    try:
        b, t, c = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        grid = device.compute_with_storage_grid_size()
        sharded_mc = ttnn.create_sharded_memory_config(
            shape=(b * t, c),
            core_grid=ttnn.CoreGrid(y=int(grid.y), x=int(grid.x)),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=False,
        )
        # Tilize both operands to L1 TILE (reads from L1, avoids DRAM).
        _til_kw = {"memory_config": l1_mc} if l1_mc is not None else {}
        x_tile = ttnn.to_layout(x, ttnn.TILE_LAYOUT, **_til_kw)
        y_tile = ttnn.to_layout(y, ttnn.TILE_LAYOUT, **_til_kw)
        # Shard both with the same HEIGHT_SHARDED spec.
        x_sh = ttnn.interleaved_to_sharded(x_tile, sharded_mc)
        ttnn.deallocate(x_tile)
        y_sh = ttnn.interleaved_to_sharded(y_tile, sharded_mc)
        ttnn.deallocate(y_tile)
        # Add entirely in sharded L1 — each core operates on its local shard.
        result_sh = ttnn.add(x_sh, y_sh, memory_config=sharded_mc)
        ttnn.deallocate(x_sh)
        ttnn.deallocate(y_sh)
        # De-shard to L1 interleaved TILE, then untilize to DRAM ROW_MAJOR.
        # Two-step avoids an UntilizeDeviceOperation reading from DRAM.
        result = ttnn.sharded_to_interleaved(result_sh, l1_mc)
        ttnn.deallocate(result_sh)
        out = ttnn.to_layout(result, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram_mc)
        ace_step_safe_deallocate(ttnn, result, x, y)
        return out
    except Exception:
        # Fallback: L1 interleaved add (still better than the original DRAM path).
        out = ttnn.add(x, y, memory_config=dram_mc)
        ace_step_safe_deallocate(ttnn, x, y)
        return out


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

        # Skip-connection trim uses ``x[:, pad:pad+y_T, :]`` → ``ttnn.slice``. TILE slices require
        # 32-aligned starts/sizes on the last two dims; upstream conv/snake already return ROW_MAJOR.
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        # Keep the residual skip in DRAM during k=7 conv1 (static CB region on Blackhole). Skip when
        # already DRAM (typical after snake / conv_t / prior residual — avoids ~25 μs dispatch).
        dram_mc = ttnn.DRAM_MEMORY_CONFIG
        if x.memory_config() != dram_mc:
            x_l1 = x
            x = ttnn.to_memory_config(x, dram_mc)
            if x is not x_l1:
                ace_step_safe_deallocate(ttnn, x_l1)

        if self.conv1.kernel_size > 1:
            ace_step_vae_synchronize(ttnn, self.device)

        y = self.snake1(x)
        if self.conv1.kernel_size > 1:
            ace_step_vae_synchronize(ttnn, self.device)
        # conv1→snake2 TILE contract: DRAM TILE out (no HEIGHT_SHARDED L1 conv output — clashes
        # with static CBs on Blackhole when prior residual L1 buffers are still live).
        y = self.conv1(y, return_tile=True)
        ace_step_vae_synchronize(ttnn, self.device)
        # snake2→conv2 TILE contract: return_tile keeps L1 TILE out (no Untilize) so conv2
        # k=1 uses TILE in0 on the linear path or a single Untilize before conv1d.
        y = self.snake2(y, return_tile=True)
        y = self.conv2(y)
        ace_step_vae_synchronize(ttnn, self.device)

        x_T = int(x.shape[1])
        y_T = int(y.shape[1])
        if y_T < x_T:
            pad = (x_T - y_T) // 2
            if pad > 0:
                x = x[:, pad : pad + y_T, :]

        # conv1 static CB is now freed; move x from DRAM to L1 so both add operands are L1.
        # add output goes to DRAM so the next residual unit's DRAM check (line ~102) is a no-op.
        l1_mc = ace_step_linear_l1_memory_config(ttnn)
        x_dram = x
        x = ttnn.to_memory_config(x, l1_mc)
        if x is not x_dram:
            ace_step_safe_deallocate(ttnn, x_dram)
        # For large tensors use HEIGHT_SHARDED add (each core works on its local L1 shard).
        # Small tensors fall back to L1 interleaved add (conversion overhead would dominate).
        t = int(x.shape[1])
        y_add = y
        if t >= _SHARDED_ADD_MIN_T:
            out = _height_sharded_add(ttnn, self.device, x, y_add, l1_mc=l1_mc, dram_mc=dram_mc)
        else:
            out = ttnn.add(x, y_add, memory_config=dram_mc)
            ace_step_safe_deallocate(ttnn, x, y_add)
        ace_step_vae_synchronize(ttnn, self.device)
        return out
