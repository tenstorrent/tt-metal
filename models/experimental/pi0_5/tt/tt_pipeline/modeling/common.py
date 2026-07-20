# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared TTNN helpers for the pi0.5 streamed-denoise port.

VENDORED from ``tt_symbiote.models.pi05.modeling_pi05_common`` with all env switches
BAKED to the measured-best deterministic values (no ``LADDER_*`` branches):
  * SDPA math fidelity   -> HiFi4
  * SDPA compute kernel  -> HiFi4, math_approx=False, fp32_dest_acc_en=False, packer_l1_acc=True
ZERO tt_symbiote imports, no os.environ reads.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import ttnn

from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT, from_torch_pi05
from models.experimental.pi0_5.tt.tile_config import TILE_HEIGHT, TILE_WIDTH

TT_METAL_COMMIT = "58672b47cfd304195798bcf34d44f5dbcbcf5189"

__all__ = [
    "get_sdpa_math_fidelity",
    "get_sdpa_compute_kernel_config",
    "sdpa_prefill_chunk_sizes",
    "precompute_freqs_cis_meta",
    "create_sinusoidal_pos_embedding",
]


def rectangular_core_grid(num_cores: int, device) -> ttnn.CoreGrid:
    """A rectangular ``x x y`` core grid of exactly ``num_cores`` cores on ``device``.

    Finds the widest ``x`` that divides ``num_cores`` and fits ``grid.x``, giving a
    ``(x, num_cores // x)`` rectangle. Raises if no such rectangle fits the device grid.
    """
    grid = device.compute_with_storage_grid_size()
    x = grid.x
    while x > 0 and num_cores % x != 0:
        x -= 1
    y = num_cores // x if x > 0 else 0
    if x == 0 or y > grid.y:
        raise ValueError(f"cannot form a rectangular grid of {num_cores} cores within a {grid.x}x{grid.y} device grid")
    return ttnn.CoreGrid(y=y, x=x)


def rectangular_core_range_set(num_cores: int, device) -> ttnn.CoreRangeSet:
    """``CoreRangeSet`` for :func:`rectangular_core_grid`."""
    core_grid = rectangular_core_grid(num_cores, device)
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))})


def width_sharded_l1_config(height: int, width: int, device, num_cores: int | None = None) -> ttnn.MemoryConfig:
    """Width-sharded L1 config: one tile-width (32 cols) per core over ``width // TILE_SIZE`` cores."""

    assert width % TILE_WIDTH == 0, f"width {width} must be tile-aligned"
    if num_cores is None:
        num_cores = width // TILE_WIDTH
    device_grid_size = device.compute_with_storage_grid_size()
    device_cores = device_grid_size.x * device_grid_size.y
    while num_cores > device_cores:
        num_cores //= 2
    shard_width = width // num_cores
    grid = rectangular_core_range_set(num_cores, device)
    height_padded = ((height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
    shard_spec = ttnn.ShardSpec(grid, [height_padded, shard_width], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def get_sdpa_math_fidelity() -> "ttnn.MathFidelity":
    """SDPA math fidelity -- BAKED HiFi4 (the deterministic, V-raising, E2E-propagating value)."""
    return ttnn.MathFidelity.HiFi4


def get_sdpa_compute_kernel_config() -> "ttnn.WormholeComputeKernelConfig":
    """SDPA compute-kernel config -- BAKED to the deterministic main-path values.

    fp32_dest_acc_en=False (bf16 dest accumulation reduces over a fixed order -> bit
    deterministic on Blackhole); packer_l1_acc=True.
    """
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=get_sdpa_math_fidelity(),
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def sdpa_prefill_chunk_sizes(seq_len_q: int, seq_len_kv: int, *, tile: int = TILE_HEIGHT) -> Tuple[int, int]:
    """q_chunk / k_chunk sizes for ttnn SDPA, mirroring the tt_transformers baseline."""
    longest = max(seq_len_q, seq_len_kv)
    if longest >= 2048:
        base_q, base_k = 256, 256
    elif longest >= 512:
        base_q, base_k = 64, 128
    else:
        base_q, base_k = 64, 64
    q_aligned = ((seq_len_q + tile - 1) // tile) * tile if seq_len_q > 0 else tile
    k_aligned = ((seq_len_kv + tile - 1) // tile) * tile if seq_len_kv > 0 else tile
    return max(min(base_q, q_aligned), tile), max(min(base_k, k_aligned), tile)


def precompute_freqs_cis_meta(
    head_dim: int,
    max_seq_len: int,
    device: "ttnn.Device",
    base: float = 10000.0,
) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
    """Precompute RoPE cos/sin in meta format ``[1, 1, max_seq_len, head_dim]`` (cat([h, h], -1))."""
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    outer = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(outer), torch.cos(outer)], dim=-1)
    sin = torch.cat([torch.sin(outer), torch.sin(outer)], dim=-1)
    cos = cos.reshape(1, 1, max_seq_len, head_dim).contiguous()
    sin = sin.reshape(1, 1, max_seq_len, head_dim).contiguous()
    cos_tt = from_torch_pi05(cos, dtype=ttnn.bfloat16, device=device)
    sin_tt = from_torch_pi05(sin, dtype=ttnn.bfloat16, device=device)
    return cos_tt, sin_tt


def create_sinusoidal_pos_embedding(
    time: torch.Tensor,
    dimension: int,
    device: "ttnn.Device",
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> ttnn.Tensor:
    """Sinusoidal flow-matching timestep embedding ``[batch, dimension]`` (cat([sin, cos]))."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be even")
    half = dimension // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    scaling = (1.0 / period) * 2.0 * math.pi
    time = time.reshape(-1, 1).to(torch.float32)
    sin_input = time * scaling.reshape(1, half)
    emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)
    return from_torch_pi05(emb.contiguous(), dtype=ttnn.bfloat16, device=device)
