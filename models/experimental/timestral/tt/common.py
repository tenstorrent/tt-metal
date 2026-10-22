# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Timestral: precision, weight upload, per-token linears.

Per-token activations are kept flat, ``(1, 1, rows * padded_seq, D)``, so every linear
folds the whole batch into M. They are viewed as
``(rows, padded_seq, D)`` only around attention; ``padded_seq`` is tile-aligned, so the
two views share one buffer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from models.experimental.timestral.tt import program_configs

TILE = 32


def pad_to_tile(n: int) -> int:
    return -(-n // TILE) * TILE


@dataclass(frozen=True)
class TtTimestralPrecision:
    """bf16 weights and HiFi2 matmuls by default; ``performance()`` stores weights as bfloat8_b at LoFi."""

    bf8_weights: bool = False
    lofi: bool = False
    fp32_dest_acc_en: bool = False

    @classmethod
    def performance(cls) -> "TtTimestralPrecision":
        return cls(bf8_weights=True, lofi=True)

    def weight_dtype(self):
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights else ttnn.bfloat16

    def compute_kernel_config(self, *, math_fidelity=None):
        import ttnn

        from models.common.utility_functions import is_blackhole

        if math_fidelity is None:
            math_fidelity = ttnn.MathFidelity.LoFi if self.lofi else ttnn.MathFidelity.HiFi2
        cls = ttnn.types.BlackholeComputeKernelConfig if is_blackhole() else ttnn.WormholeComputeKernelConfig
        return cls(
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.fp32_dest_acc_en,
            packer_l1_acc=True,
        )

    def norm_kernel_config(self):
        import ttnn

        return self.compute_kernel_config(math_fidelity=ttnn.MathFidelity.HiFi4)


def mesh_mapper(device):
    """Copy to every chip of a multi-chip mesh; ``None`` on one chip."""
    import ttnn

    if hasattr(device, "get_num_devices") and device.get_num_devices() > 1:
        return ttnn.replicate_tensor_to_mesh_mapper(device)
    return None


def upload(device, t: torch.Tensor, *, dtype=None, layout=None):
    """Host tensor -> device DRAM (bf16 tiles by default)."""
    import ttnn

    return ttnn.from_torch(
        t.detach().to(torch.float32).contiguous(),
        dtype=ttnn.bfloat16 if dtype is None else dtype,
        layout=ttnn.TILE_LAYOUT if layout is None else layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(device),
    )


def upload_linear(device, weight: torch.Tensor, dtype):
    """``nn.Linear`` weight ``(out, in)`` -> device ``(in, out)``, the layout ``ttnn.linear`` reads."""
    return upload(device, weight.t(), dtype=dtype)


def upload_bias(device, bias: torch.Tensor | None):
    return None if bias is None else upload(device, bias.reshape(1, -1))


def upload_swiglu(device, w_silu: torch.Tensor, w_mul: torch.Tensor, dtype):
    """Two ``nn.Linear`` weights ``(n, in)`` -> device ``(in, 2n)`` whose column tiles alternate
    ``[silu_t0, mul_t0, silu_t1, ...]``, the pairing ``minimal_matmul(fuse_swiglu=True)`` reads."""
    n, k = w_silu.shape
    pairs = torch.stack([w_silu.t().reshape(k, n // TILE, TILE), w_mul.t().reshape(k, n // TILE, TILE)], dim=2)
    return upload(device, pairs.reshape(k, 2 * n), dtype=dtype)


def linear(x, weight, precision: TtTimestralPrecision, *, bias=None, activation=None):
    """``x @ weight (+ bias)``, then ``activation`` ("silu" / "sigmoid"), with an explicit config
    from ``program_configs`` (DRAM-interleaved, tile-aligned K), else the auto config."""
    import ttnn

    kernel = precision.compute_kernel_config()
    x_shape, (k, n) = tuple(x.padded_shape), tuple(weight.padded_shape)[-2:]
    if x.layout != ttnn.TILE_LAYOUT or x.memory_config().is_sharded() or x.shape[-1] % TILE or x_shape[-1] != k:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )
    m_tiles, k_tiles, n_tiles = math.prod(x_shape[:-1]) // TILE, k // TILE, n // TILE
    grid = x.device().compute_with_storage_grid_size()
    grid = (grid.x, grid.y)
    max_dest = 4 if precision.fp32_dest_acc_en else 8
    if activation is None and program_configs.grid_columns(n_tiles, grid[0]) < grid[0]:
        config = program_configs.minimal_matmul_config(grid, m_tiles, k_tiles, n_tiles, max_dest=max_dest)
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            bias_tensor=bias,
            config=config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )
    config = program_configs.matmul_2d_config(
        grid,
        m_tiles,
        k_tiles,
        n_tiles,
        fused_activation=program_configs.fused_activation(activation),
        max_dest=max_dest,
    )
    return ttnn.linear(
        x, weight, bias=bias, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel
    )


def swiglu(x, weight, precision: TtTimestralPrecision):
    """``(x @ W_mul) * silu(x @ W_silu)`` in one matmul on an ``upload_swiglu`` weight."""
    import ttnn

    x_shape, (k, n) = tuple(x.padded_shape), tuple(weight.padded_shape)[-2:]
    grid = x.device().compute_with_storage_grid_size()
    config = program_configs.swiglu_config(
        (grid.x, grid.y),
        math.prod(x_shape[:-1]) // TILE,
        k // TILE,
        n // (2 * TILE),
        max_dest=4 if precision.fp32_dest_acc_en else 8,
    )
    return ttnn.experimental.minimal_matmul(
        x,
        weight,
        config=config,
        fuse_swiglu=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(),
    )


def rms_norm(x, eps: float, precision: TtTimestralPrecision):
    """Gain-free RMSNorm (the gain is folded into the next linears)."""
    import ttnn

    return ttnn.rms_norm(
        x, epsilon=eps, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=precision.norm_kernel_config()
    )


def add_residual(x, out):
    """``x + out``; consumes both."""
    import ttnn

    y = ttnn.add(x, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    ttnn.deallocate(out)
    return y
