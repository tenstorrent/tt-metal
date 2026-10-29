# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Shared host-side helpers for the MiniMax-M3-VL ttnn modules.

Small utilities that every submodule needs: the mesh mapper, the
torch->ttnn Linear weight/bias converters, the head-dim tile rounding, the
standard and weight-matmul compute-kernel configs, the weight-matmul program
config and the SDPA program config.
Centralised here so the modules don't each copy them.
"""
from __future__ import annotations

import functools
import math

import torch

import ttnn

# Circular-buffer bytes per core, below the roughly 1.39 MiB of L1 a Blackhole core has free while the encoder keeps
# its tensors in DRAM. The largest K block that fits is taken, so this also sets K blocks: at 672x672 fc1 and QKV get
# 4, which ran faster than 8 in a matmul sweep on a P150.
LINEAR_L1_BYTES = 1_000_000
# The largest K block (in0_block_w, in tiles): in that sweep fc2 and the out projection ran slower with 16.
LINEAR_MAX_IN0_BLOCK_W = 8
# The fewest tiles of M per core for transpose_mcast: in a size sweep on a P150, a limit of 2 made the weight matmuls
# slower at 768 patches and one of 5 at 1296 patches; one of 3 made them faster at 1024 patches but has not been
# timed in a whole forward.
LINEAR_MIN_TRANSPOSED_PER_CORE_M = 4
# Output subblocks (h, w) in the order ttnn's own matmul configs try them.
_SUBBLOCKS = (
    (4, 2), (2, 4), (8, 1), (1, 8), (7, 1), (1, 7), (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5), (2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1),
)  # fmt: skip


def next_tile_multiple(x: int, tile: int = ttnn.TILE_SIZE) -> int:
    """Round up to the next multiple of the tile size (32)."""
    return ((x + tile - 1) // tile) * tile


def mesh_mapper(device):
    """Replicate-to-mesh mapper for a MeshDevice, else None (single device)."""
    return ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None


def mesh_composer(device, dim: int = 0):
    """Concat-from-mesh composer for a MeshDevice, else None."""
    return ttnn.ConcatMeshToTensor(device, dim=dim) if isinstance(device, ttnn.MeshDevice) else None


def as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    """torch nn.Linear weight [out, in] -> ttnn tile tensor [in, out] (operand-b convention)."""
    w = torch_w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()
    return ttnn.as_tensor(
        w,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(mesh_device),
    )


def as_linear_bias(mesh_device, torch_b: torch.Tensor, dtype) -> ttnn.Tensor:
    """torch bias [out] -> ttnn tile tensor [1, 1, 1, out], replicated."""
    b = torch_b.detach().to(torch.bfloat16).view(1, 1, 1, -1).contiguous()
    return ttnn.as_tensor(
        b,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper(mesh_device),
    )


def hifi4_compute_config(fp32_dest_acc: bool = True, packer_l1_acc: bool = False):
    """Standard HiFi4 compute-kernel config. Matmul-bearing ops use fp32_dest_acc=True;
    LayerNorm uses False (matches the original per-module settings)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc,
        packer_l1_acc=packer_l1_acc,
    )


def linear_compute_config():
    """HiFi4 config for the weight matmuls (ttnn.linear): fp32 accumulation, and packer_l1_acc so the partial sums
    of each output block accumulate in L1 instead of being reloaded for every K block."""
    return hifi4_compute_config(packer_l1_acc=True)


def linear_program_config(mesh_device, rows: int, in_features: int, out_features: int, compute_kernel_config):
    """Program config for ttnn.linear of a (rows, in_features) input and an (in_features, out_features) weight, with
    bf16 (or bfloat8_b) operands and output in DRAM, as the vision encoder runs its weight matmuls. None leaves the
    choice to ttnn.

    A 2D multicast with one output block per core, after a sweep of the encoder's linears at 672x672 on a P150:
    - M over the grid's rows and N over its columns, or, with at least LINEAR_MIN_TRANSPOSED_PER_CORE_M tiles of M
      per core, the other way round (transpose_mcast), whichever leaves fewer output tiles per core;
    - K blocks of at most LINEAR_MAX_IN0_BLOCK_W tiles: the largest that divides K and keeps the circular buffers
      within LINEAR_L1_BYTES;
    - output subblocks: the first in ttnn's order that divides the block and fits the dest register (4 tiles with
      fp32 accumulation, else 8).
    None where a 1D multicast (every row on each core and the columns spread over all of them, or the other way round)
    would give each core no more output tiles, as for small images, and where even K blocks of one tile overflow
    LINEAR_L1_BYTES.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    blocking = _linear_blocking(
        math.ceil(rows / ttnn.TILE_SIZE),
        math.ceil(in_features / ttnn.TILE_SIZE),
        math.ceil(out_features / ttnn.TILE_SIZE),
        grid.x,
        grid.y,
        bool(compute_kernel_config.fp32_dest_acc_en),
    )
    if blocking is None:
        return None
    per_core_m, per_core_n, in0_block_w, subblock_h, subblock_w, transpose_mcast = blocking
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=in0_block_w,
        out_subblock_h=subblock_h,
        out_subblock_w=subblock_w,
        out_block_h=per_core_m,
        out_block_w=per_core_n,
        per_core_M=per_core_m,
        per_core_N=per_core_n,
        transpose_mcast=transpose_mcast,
        fuse_batch=False,
    )


@functools.lru_cache(maxsize=None)
def _linear_blocking(
    m_tiles: int,
    k_tiles: int,
    n_tiles: int,
    grid_x: int,
    grid_y: int,
    fp32_dest_acc: bool,
    *,
    l1_bytes: int = LINEAR_L1_BYTES,
    max_in0_block_w: int = LINEAR_MAX_IN0_BLOCK_W,
    min_transposed_per_core_m: int = LINEAR_MIN_TRANSPOSED_PER_CORE_M,
):
    """linear_program_config's choice, in tiles: (per_core_M, per_core_N, in0_block_w, out_subblock_h, out_subblock_w,
    transpose_mcast), or None. The limits default to the LINEAR_* constants' values at import and, as arguments, are
    part of the cache key."""
    cores = grid_x * grid_y
    # Output tiles per core of a 1D multicast: all the rows and a slice of the columns on each core, or vice versa.
    one_d = min(m_tiles * math.ceil(n_tiles / cores), math.ceil(m_tiles / cores) * n_tiles)
    # Without transpose_mcast the grid's rows (y) take M and its columns (x) take N; with it, the other way round.
    splits = [(math.ceil(m_tiles / grid_y), math.ceil(n_tiles / grid_x), False)]
    if math.ceil(m_tiles / grid_x) >= min_transposed_per_core_m:
        splits.append((math.ceil(m_tiles / grid_x), math.ceil(n_tiles / grid_y), True))
    per_core_m, per_core_n, transpose = min(splits, key=lambda s: (s[0] * s[1], s[2]))
    if per_core_m * per_core_n >= one_d:
        return None

    def cb_bytes(in0_block_w):
        # The 2D multicast factory's buffers: double-buffered in0 and in1 K blocks, the output block and a bias row
        # in bf16 tiles (2 KB), and the output block's partial sums (fp32 tiles with fp32 accumulation).
        block = per_core_m * per_core_n
        partials = block * (4096 if fp32_dest_acc else 2048)
        return 2048 * (2 * in0_block_w * (per_core_m + per_core_n) + block + per_core_n) + partials

    fitting = [w for w in range(1, min(k_tiles, max_in0_block_w) + 1) if k_tiles % w == 0 and cb_bytes(w) <= l1_bytes]
    if not fitting:
        return None
    dest_tiles = 4 if fp32_dest_acc else 8
    subblock_h, subblock_w = next(
        (h, w) for h, w in _SUBBLOCKS if h * w <= dest_tiles and per_core_m % h == 0 and per_core_n % w == 0
    )
    return per_core_m, per_core_n, fitting[-1], subblock_h, subblock_w, transpose


def sdpa_program_config(mesh_device, chunk_size: int = 128):
    """SDPA program config: q and k chunks of chunk_size on the device's whole compute grid.

    Without a program config the op takes chunks of 32; chunks of 128 run a 672x672 image (2304 patches) faster.
    exp_approx_mode stays unset, so SDPA keeps the op's default (approximate) exp either way.
    """
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
        q_chunk_size=chunk_size,
        k_chunk_size=chunk_size,
    )
