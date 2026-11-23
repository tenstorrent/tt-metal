# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Matmul block sizes for the MiniMax-H3 all-gather-matmul shapes.

Keyed on `(K, N, per_core_M)`, since the best block depends on the M-tiles per core, which varies with
video duration. Passed to the linear layers as `default_block_size`, which takes precedence in `get_agmm_config`.

Two hard constraints the op asserts on:

`N_block` must be divisible by `subblock_w`, which `get_matmul_config` leaves at 2 -- so N_block must
be even. (For a fused-SwiGLU ff1 it must be even anyway: gate/up tile pairs interleave along N and a
block must never split a pair.)

`K_block` must divide `K_tiles_per_device` (`K / 32 / tp_factor`). The ring delivers the gathered K
in `K_block`-sized chunks and a partial final chunk is unsupported. With TP=4 that means K_block
divides 42 for K=5376 and 56 for K=7168 -- 8 (the generic default) divides neither, which is why
these shapes need an entry rather than falling back.
"""

from __future__ import annotations

import enum
import math
import os
from typing import NamedTuple

import ttnn

from ....utils.device_constants import BLACKHOLE_COMPUTE_GRID, WORMHOLE_COMPUTE_GRID, compute_grid_key
from ....utils.matmul import agmm_worker_grid
from .length_classes import LENGTH_CLASSES, length_class

# M parallelizes over 12 cores when transposed (M > N) and 10 otherwise (the op reserves the mux axis).
_TILE = 32
_M_CORES_TRANSPOSED = 12
_M_CORES_NON_TRANSPOSED = 10


def _per_core_m(m: int, n: int) -> int:
    """M-tiles each core walks, for an AGMM of M rows / N cols on the op's worker grid."""
    m_tiles = -(-m // _TILE)
    cores = _M_CORES_TRANSPOSED if m > n else _M_CORES_NON_TRANSPOSED
    return -(-m_tiles // cores)


# (K, N, per_core_M) -> (M_block, K_block, N_block). N is the *per-device* output width. Best block at
# subblock (2, 2), swept with models/tt_dit/utils/sweep_mm_block_sizes.py on 4x8 Blackhole Galaxy.
#   (5376, 5376)  attention to_qkv   K_tiles_per_device = 42
#   (7168, 1344)  attention to_out   K_tiles_per_device = 56
#   (5376, 7168)  feed-forward ff1   K_tiles_per_device = 42, fused SwiGLU so N_block must be even
AGMM_BLOCK_SIZES: dict[tuple[int, int, int], tuple[int, int, int]] = {
    (5376, 5376, 1): (2, 6, 16),
    (5376, 5376, 2): (2, 7, 16),
    (5376, 5376, 3): (4, 6, 16),
    (5376, 5376, 4): (6, 7, 16),
    (5376, 5376, 5): (6, 7, 16),
    (5376, 5376, 6): (6, 3, 16),
    (5376, 5376, 7): (4, 6, 16),
    (5376, 5376, 8): (4, 6, 16),
    (5376, 5376, 9): (6, 6, 16),
    (5376, 5376, 10): (6, 6, 16),
    (5376, 5376, 11): (4, 6, 16),
    (5376, 5376, 12): (4, 6, 16),
    (5376, 7168, 1): (2, 21, 6),
    (5376, 7168, 2): (2, 6, 12),
    (5376, 7168, 3): (4, 3, 14),
    (5376, 7168, 4): (4, 3, 16),
    (5376, 7168, 5): (6, 3, 16),
    (5376, 7168, 6): (6, 3, 16),
    (5376, 7168, 7): (4, 6, 14),
    (5376, 7168, 8): (4, 3, 16),
    (5376, 7168, 9): (6, 3, 14),
    (5376, 7168, 10): (6, 3, 16),
    (5376, 7168, 11): (6, 3, 16),
    (5376, 7168, 12): (4, 3, 16),
    (7168, 1344, 1): (2, 14, 6),
    (7168, 1344, 2): (2, 8, 6),
    (7168, 1344, 3): (6, 7, 4),
    (7168, 1344, 4): (8, 8, 6),
    (7168, 1344, 5): (6, 8, 6),
    (7168, 1344, 6): (10, 8, 6),
    (7168, 1344, 7): (8, 8, 6),
    (7168, 1344, 8): (8, 8, 6),
    (7168, 1344, 9): (10, 8, 6),
    (7168, 1344, 10): (10, 8, 6),
    (7168, 1344, 11): (6, 8, 6),
    (7168, 1344, 12): (6, 8, 8),
}


class AgmmMode(enum.Enum):
    """Accuracy / speed trade-off of the dynamic-length path's pinned all-gather matmuls."""

    ACCURATE = "accurate"  # fp32 dest accumulation on every layer
    BALANCED = "balanced"  # fp32 dest accumulation off on the layers with a faster pin that way


DEFAULT_AGMM_MODE = AgmmMode.BALANCED


def resolve_agmm_mode(mode: AgmmMode | str | None = None) -> AgmmMode:
    """`mode` if given, else MINIMAX_H3_AGMM_MODE (`accurate` / `balanced`), else `DEFAULT_AGMM_MODE`."""
    if mode is None:
        mode = os.environ.get("MINIMAX_H3_AGMM_MODE") or DEFAULT_AGMM_MODE
    return AgmmMode(mode)


class AgmmLayerPin(NamedTuple):
    """One layer's pinned program: the (M, K, N) blocking per length class, the worker-grid orientation, the
    subblock and whether Dst accumulates in fp32."""

    blockings: dict[str, tuple[int, int, int]]
    transpose_core_grid: bool
    subblock: tuple[int, int] = (2, 2)
    fp32_dest_acc_en: bool = True


# Pins of the dynamic-length path per device compute grid, mode and (K, N) layer. A mode other than ACCURATE lists
# only the layers it changes. Blackhole's to_qkv and ff1 run non-transposed, against the op's `M > N` rule.
AGMM_PINS: dict[tuple[int, int], dict[AgmmMode, dict[tuple[int, int], AgmmLayerPin]]] = {
    BLACKHOLE_COMPUTE_GRID: {
        AgmmMode.ACCURATE: {
            (5376, 5376): AgmmLayerPin({"short": (4, 6, 16), "long": (4, 6, 16)}, transpose_core_grid=False),
            (5376, 7168): AgmmLayerPin({"short": (6, 3, 16), "long": (4, 3, 16)}, transpose_core_grid=False),
            (7168, 1344): AgmmLayerPin({"short": (6, 8, 6), "long": (6, 8, 6)}, transpose_core_grid=True),
        },
    },
    WORMHOLE_COMPUTE_GRID: {
        AgmmMode.ACCURATE: {
            (5376, 5376): AgmmLayerPin({"short": (10, 7, 8), "long": (8, 7, 12)}, transpose_core_grid=True),
            (5376, 7168): AgmmLayerPin({"short": (10, 7, 10), "long": (10, 7, 10)}, transpose_core_grid=True),
            (7168, 1344): AgmmLayerPin({"short": (10, 8, 6), "long": (12, 8, 6)}, transpose_core_grid=True),
        },
        AgmmMode.BALANCED: {
            (5376, 5376): AgmmLayerPin(
                {"short": (12, 7, 12), "long": (12, 7, 12)},
                transpose_core_grid=True,
                subblock=(4, 2),
                fp32_dest_acc_en=False,
            ),
            (5376, 7168): AgmmLayerPin(
                {"short": (10, 7, 16), "long": (12, 7, 12)},
                transpose_core_grid=True,
                subblock=(2, 4),
                fp32_dest_acc_en=False,
            ),
        },
    },
}


class AgmmPin(NamedTuple):
    """Frozen all-gather-matmul program of one layer at one length class, as the linear's kwargs."""

    matmul_config: ttnn.MinimalMatmulConfig
    num_workers_per_link: int
    transpose_core_grid: bool
    compute_kernel_config: ttnn.WormholeComputeKernelConfig


def _pin_override(k: int, n: int, pin: AgmmLayerPin) -> AgmmLayerPin:
    """For sweeping: `MINIMAX_H3_AGMM_PIN_<K>_<N>="M,K,N"` replaces the layer's blocking at every length class and
    keeps its subblock and Dst accumulation; `"M,K,N,subblock_h,subblock_w"` also sets the subblock, with fp32 dest
    accumulation off above 4 tiles."""
    name = f"MINIMAX_H3_AGMM_PIN_{k}_{n}"
    value = os.environ.get(name)
    if not value:
        return pin
    fields = tuple(int(v) for v in value.split(","))
    if len(fields) not in (3, 5):
        raise ValueError(f"{name}={value!r}: expected M,K,N or M,K,N,subblock_h,subblock_w")
    blockings = {cls.name: fields[:3] for cls in LENGTH_CLASSES}
    if len(fields) == 3:
        return pin._replace(blockings=blockings)
    subblock_h, subblock_w = fields[3:]
    return pin._replace(
        blockings=blockings, subblock=(subblock_h, subblock_w), fp32_dest_acc_en=subblock_h * subblock_w <= 4
    )


def _with_fp32_dest_acc(
    config: ttnn.WormholeComputeKernelConfig, fp32_dest_acc_en: bool
) -> ttnn.WormholeComputeKernelConfig:
    """`config` with only its Dst accumulation set to `fp32_dest_acc_en`."""
    if config.fp32_dest_acc_en == fp32_dest_acc_en:
        return config
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=config.math_fidelity,
        math_approx_mode=config.math_approx_mode,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=config.packer_l1_acc,
        dst_full_sync_en=config.dst_full_sync_en,
        throttle_level=config.throttle_level,
    )


def agmm_pins(
    k: int,
    n: int,
    *,
    full_grid: ttnn.CoreCoord,
    num_links: int,
    compute_kernel_config: ttnn.WormholeComputeKernelConfig,
    mode: AgmmMode | str | None = None,
) -> dict[str, AgmmPin]:
    """The AGMM program of a `(K, N)` layer per length class in `mode` (`resolve_agmm_mode`): the mode's pin for
    the layer, else its ACCURATE pin. `compute_kernel_config` is the caller's, with the pin's Dst accumulation.
    Classes with the same blocking share one pin."""
    device_grid = compute_grid_key(full_grid)
    if device_grid not in AGMM_PINS:
        raise ValueError(f"dynamic_length has no pinned AGMM blockings for the {device_grid} compute grid")
    pins_by_mode = AGMM_PINS[device_grid]
    layer = pins_by_mode.get(resolve_agmm_mode(mode), {}).get((k, n)) or pins_by_mode[AgmmMode.ACCURATE].get((k, n))
    if layer is None:
        raise ValueError(f"dynamic_length has no pinned AGMM blocking for (K, N)={(k, n)}; TP=4 layouts only")
    layer = _pin_override(k, n, layer)
    grid = agmm_worker_grid(full_grid, layer.transpose_core_grid, num_links)
    # in0 senders group along the M-parallel axis (grid.x transposed, grid.y not) into `num_links` groups.
    in0_axis = grid.x if layer.transpose_core_grid else grid.y
    workers = math.ceil(in0_axis / num_links)
    compute_kernel_config = _with_fp32_dest_acc(compute_kernel_config, layer.fp32_dest_acc_en)
    subblock_h, subblock_w = layer.subblock
    pins: dict[tuple[int, int, int], AgmmPin] = {}
    for blocking in layer.blockings.values():
        if blocking not in pins:
            m_block, k_block, n_block = blocking
            if m_block % subblock_h or n_block % subblock_w:
                raise ValueError(
                    f"AGMM blocking {blocking} of (K, N)={(k, n)} does not tile by {subblock_h}x{subblock_w}"
                )
            config = ttnn.MinimalMatmulConfig(
                M_block_size=m_block,
                K_block_size=k_block,
                N_block_size=n_block,
                subblock_h=subblock_h,
                subblock_w=subblock_w,
                compute_with_storage_grid_size=grid,
            )
            pins[blocking] = AgmmPin(config, workers, layer.transpose_core_grid, compute_kernel_config)
    return {name: pins[blocking] for name, blocking in layer.blockings.items()}


def agmm_kwargs(
    pins: dict[str, AgmmPin] | None, k: int, n: int, rows: int, compute_kernel_config: ttnn.WormholeComputeKernelConfig
) -> dict:
    """The linear's blocking and compute kwargs for a `(K, N)` layer at `rows` per-device rows: the length class's
    pin, else (legacy path) the length-keyed swept table with `compute_kernel_config`."""
    if pins is None:
        return {"compute_kernel_config": compute_kernel_config, "default_block_size": agmm_block_size(k, n, rows)}
    return pins[length_class(rows).name]._asdict()


def agmm_block_size(k: int, n: int, m: int) -> tuple[int, int, int] | None:
    """Block sizes for an all-gather matmul of this `(K, N)` at sequence length `M`, or None to let
    the generic path decide. Uses the largest swept `per_core_M` that divides the runtime one.
    """
    per_core_m = _per_core_m(m, n)
    swept = [pcm for (kk, nn, pcm) in AGMM_BLOCK_SIZES if kk == k and nn == n]
    divisors = [pcm for pcm in swept if per_core_m % pcm == 0]
    if not divisors:
        return None
    return AGMM_BLOCK_SIZES[(k, n, max(divisors))]
