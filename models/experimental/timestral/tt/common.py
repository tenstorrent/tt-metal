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
    """bf16 weights and HiFi2 matmuls by default; ``performance()`` stores weights as bfloat8_b at LoFi,
    except wo / w_down, whose matmuls fold in the bf16 residual add."""

    bf8_weights: bool = False
    lofi: bool = False
    fp32_dest_acc_en: bool = False

    @classmethod
    def performance(cls) -> "TtTimestralPrecision":
        return cls(bf8_weights=True, lofi=True)

    def weight_dtype(self, *, residual: bool = False):
        """``residual``: the matmul folds in the residual add, which needs weight dtype == residual dtype (bf16)."""
        import ttnn

        return ttnn.bfloat8_b if self.bf8_weights and not residual else ttnn.bfloat16

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

    dtype = ttnn.bfloat16 if dtype is None else dtype
    # torch rounds fp32 -> bf16 bit-identically to from_torch, about 10x faster.
    host_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    return ttnn.from_torch(
        t.detach().to(host_dtype).contiguous(),
        dtype=dtype,
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


def _geometry(x, weight):
    """``(m_tiles, k_tiles, n_tiles, grid)`` when an explicit config applies (DRAM-interleaved input,
    tile-aligned K), else None."""
    import ttnn

    x_shape, (k, n) = tuple(x.padded_shape), tuple(weight.padded_shape)[-2:]
    if x.layout != ttnn.TILE_LAYOUT or x.memory_config().is_sharded() or x.shape[-1] % TILE or x_shape[-1] != k:
        return None
    grid = x.device().compute_with_storage_grid_size()
    return math.prod(x_shape[:-1]) // TILE, k // TILE, n // TILE, (grid.x, grid.y)


def _minimal_config(geometry, precision: TtTimestralPrecision, activation=None):
    """``minimal_matmul`` config when N leaves 2D grid columns idle and nothing is fused, else None."""
    if geometry is None or activation is not None:
        return None
    m_tiles, k_tiles, n_tiles, grid = geometry
    if program_configs.grid_columns(n_tiles, grid[0]) == grid[0]:
        return None
    return program_configs.minimal_matmul_config(grid, m_tiles, k_tiles, n_tiles, max_dest=_max_dest(precision))


def _max_dest(precision: TtTimestralPrecision) -> int:
    return 4 if precision.fp32_dest_acc_en else 8


def linear(x, weight, precision: TtTimestralPrecision, *, bias=None, activation=None):
    """``x @ weight (+ bias)``, then ``activation`` ("silu" / "sigmoid"), with an explicit config
    from ``program_configs``, else the auto config."""
    import ttnn

    kernel = precision.compute_kernel_config()
    geometry = _geometry(x, weight)
    if geometry is None:
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            activation=activation,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )
    config = _minimal_config(geometry, precision, activation)
    if config is not None:
        return ttnn.experimental.minimal_matmul(
            x,
            weight,
            bias_tensor=bias,
            config=config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=kernel,
        )
    m_tiles, k_tiles, n_tiles, grid = geometry
    config = program_configs.matmul_2d_config(
        grid,
        m_tiles,
        k_tiles,
        n_tiles,
        fused_activation=program_configs.fused_activation(activation),
        max_dest=_max_dest(precision),
    )
    return ttnn.linear(
        x, weight, bias=bias, program_config=config, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=kernel
    )


def linear_residual(x, weight, precision: TtTimestralPrecision, residual, ones):
    """``residual + x @ weight``; ``ones`` is a device ``(1, N)`` of ones. The add runs in the
    matmul epilogue when the fused kernel allows it (residual dtype == weight dtype), else after it."""
    import ttnn

    config = _minimal_config(_geometry(x, weight), precision)
    if config is not None and residual.dtype == weight.dtype:
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            weight,
            1.0,
            residual,
            ones,
            config=config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=precision.compute_kernel_config(),
        )
    out = linear(x, weight, precision)
    y = ttnn.add(residual, out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(out)
    return y


def linear_concat(xs, weight, precision: TtTimestralPrecision, *, bias=None):
    """``[x_0 | x_1] @ weight (+ bias)``, the concatenation over K read by ``minimal_matmul`` itself;
    ``weight`` stacks the two inputs' row blocks (``upload_concat_linear``)."""
    import ttnn

    x_shape, (k, n) = tuple(xs[0].padded_shape), tuple(weight.padded_shape)[-2:]
    grid = xs[0].device().compute_with_storage_grid_size()
    config = program_configs.minimal_matmul_config(
        (grid.x, grid.y), math.prod(x_shape[:-1]) // TILE, k // TILE, n // TILE, max_dest=_max_dest(precision)
    )
    return ttnn.experimental.minimal_matmul(
        list(xs),
        weight,
        bias_tensor=bias,
        config=config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(),
    )


def upload_concat_linear(device, weights: tuple[torch.Tensor, torch.Tensor], dtype):
    """Two ``nn.Linear`` weights ``(out, in_i)`` -> device ``(in_0' + in_1', out)``, each input block padded
    to whole tiles (``in_i'``), the layout ``linear_concat`` reads."""
    blocks = [torch.nn.functional.pad(w.t(), (0, 0, 0, pad_to_tile(w.shape[1]) - w.shape[1])) for w in weights]
    return upload(device, torch.cat(blocks), dtype=dtype)


def linear_split(x, weight, precision: TtTimestralPrecision, chunks: int):
    """``x @ weight`` as ``chunks`` equal column blocks, each its own ``(1, 1, M, N / chunks)`` tensor."""
    import ttnn

    x_shape, (k, n) = tuple(x.padded_shape), tuple(weight.padded_shape)[-2:]
    grid = x.device().compute_with_storage_grid_size()
    config = program_configs.split_config(
        (grid.x, grid.y),
        math.prod(x_shape[:-1]) // TILE,
        k // TILE,
        n // (chunks * TILE),
        chunks,
        max_dest=_max_dest(precision),
    )
    return ttnn.experimental.minimal_matmul_split(
        x,
        weight,
        chunks=chunks,
        config=config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=precision.compute_kernel_config(),
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
        max_dest=_max_dest(precision),
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


def take_slots(x, rows: int, seq: int, start: int, count: int):
    """Flat ``(1, 1, rows * seq, D)`` -> ``(1, 1, rows * count, D)``: token slots ``[start, start + count)``
    of every row; ``x`` is borrowed."""
    import ttnn

    d = x.shape[-1]
    picked = ttnn.slice(ttnn.reshape(x, (rows, seq, d)), (0, start, 0), (rows, start + count, d))
    return ttnn.reshape(picked, (1, 1, rows * count, d))
