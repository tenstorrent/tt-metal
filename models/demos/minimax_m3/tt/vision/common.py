# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the MiniMax-M3 vision modules: uploads, compute configs, the weight matmul and program configs."""
from __future__ import annotations

import functools
import math
from typing import Callable, Optional

import torch

import ttnn

# Circular-buffer bytes per core for a weight matmul. It stays under a Blackhole core's free L1 (about 1.39 MiB)
# and leaves room for the RoPE tables that the blocks read from L1. It also sets the K block: the largest that fits.
LINEAR_L1_BYTES = 1_000_000
# The largest K block (in0_block_w), in tiles.
LINEAR_MAX_IN0_BLOCK_W = 8
# The fewest tiles of M per core with which a weight matmul may use transpose_mcast.
LINEAR_MIN_TRANSPOSED_PER_CORE_M = 4
# Output subblocks (h, w) in the order ttnn's own matmul configs try them.
_SUBBLOCKS = (
    (4, 2), (2, 4), (8, 1), (1, 8), (7, 1), (1, 7), (3, 2), (2, 3), (6, 1), (1, 6),
    (5, 1), (1, 5), (2, 2), (4, 1), (1, 4), (3, 1), (1, 3), (2, 1), (1, 2), (1, 1),
)  # fmt: skip


def to_device(tensor: torch.Tensor, mesh_device, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Upload a host tensor to DRAM as a tile tensor of dtype, replicated over the mesh.

    The tensor is rounded to bf16 on the host first.
    """
    return ttnn.from_torch(
        tensor.to(torch.bfloat16).contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def as_linear_weight(mesh_device, torch_w: torch.Tensor, dtype) -> ttnn.Tensor:
    """Upload an nn.Linear weight [out, in] as the [in, out] tile tensor that ttnn.linear takes."""
    return to_device(torch_w.transpose(-2, -1), mesh_device, dtype)


def as_linear_bias(mesh_device, torch_b: torch.Tensor, dtype) -> ttnn.Tensor:
    """Upload a bias [out] as a [1, 1, 1, out] tile tensor, replicated over the mesh."""
    return to_device(torch_b.view(1, 1, 1, -1), mesh_device, dtype)


def hifi4_compute_config(packer_l1_acc: bool = False):
    """HiFi4 compute config with exact math and fp32 accumulation, as LayerNorm, RoPE and SDPA use it."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
    )


def linear_compute_config():
    """hifi4_compute_config for the weight matmuls, with partial sums accumulated in L1 (packer_l1_acc)."""
    return hifi4_compute_config(packer_l1_acc=True)


def linear(
    x: ttnn.Tensor,
    w: ttnn.Tensor,
    b: Optional[ttnn.Tensor],
    compute_kernel_config,
    activation: Optional[Callable[[ttnn.Tensor], ttnn.Tensor]] = None,
    fused_activation: Optional[ttnn.UnaryWithParam] = None,
) -> ttnn.Tensor:
    """Run the weight matmul x @ w + b into DRAM, then activation, if given, as a separate op.

    w is [in, out] and b is [1, 1, 1, out] or None, as as_linear_weight and as_linear_bias upload them. The program
    config is linear_program_config's for the padded shapes. fused_activation is the same activation in a program
    config's form, and needs activation. Where linear_program_config gives a config, the matmul applies
    fused_activation and activation does not run.
    """
    assert fused_activation is None or activation is not None, "fused_activation needs activation"
    program_config = linear_program_config(
        x.device(), x.padded_shape[-2], w.padded_shape[-2], w.padded_shape[-1], compute_kernel_config
    )
    fused = fused_activation is not None and program_config is not None
    if fused:
        # linear_program_config builds a new config at each call, so no other matmul gets the activation.
        program_config.fused_activation = fused_activation
    out = ttnn.linear(
        x,
        w,
        bias=b,
        compute_kernel_config=compute_kernel_config,
        program_config=program_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return out if activation is None or fused else activation(out)


def linear_program_config(mesh_device, rows: int, in_features: int, out_features: int, compute_kernel_config):
    """Return the program config for a weight matmul of (rows, in_features) by (in_features, out_features).

    Operands and output are bf16 or bfloat8_b, in DRAM. The config is a 2D multicast with one output block per core.
    _linear_blocking picks the split and the blocks within the LINEAR_* limits.
    Returns None to leave the choice to ttnn in two cases. Either a 1D multicast would give each core no more output
    tiles, as for small images, or no K block fits in LINEAR_L1_BYTES.
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
    """Return linear_program_config's choice in tiles, or None.

    The choice is (per_core_M, per_core_N, in0_block_w, out_subblock_h, out_subblock_w, transpose_mcast). The limits
    default to the LINEAR_* values at import time, and are part of the cache key.
    """
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
        # The 2D multicast factory's circular buffers, in 2 KB bf16 tiles: double-buffered in0 and in1 K blocks, the
        # output block and a bias row. With fp32 accumulation the partial sums get their own 4 KB fp32 tiles.
        block = per_core_m * per_core_n
        partials = block * 4096 if fp32_dest_acc else 0
        return 2048 * (2 * in0_block_w * (per_core_m + per_core_n) + block + per_core_n) + partials

    fitting = [w for w in range(1, min(k_tiles, max_in0_block_w) + 1) if k_tiles % w == 0 and cb_bytes(w) <= l1_bytes]
    if not fitting:
        return None
    dest_tiles = 4 if fp32_dest_acc else 8
    subblock_h, subblock_w = next(
        (h, w) for h, w in _SUBBLOCKS if h * w <= dest_tiles and per_core_m % h == 0 and per_core_n % w == 0
    )
    return per_core_m, per_core_n, fitting[-1], subblock_h, subblock_w, transpose


# The q chunks sdpa_program_config picks from. Chunks below 128 are left out: on a P150 a q chunk of 64 saved 0.4 ms
# of the 3.7 ms its query rows predict at 672x672, and none has been timed at a size where the rule would pick it.
SDPA_Q_CHUNK_SIZES = (128, 160, 192, 256)
# On a P150 k chunks of 256 ran a 672x672 image about as fast as q chunks of 160, but change the output's rounding.
SDPA_K_CHUNK_SIZE = 128


def sdpa_program_config(mesh_device, seq_len: int, num_heads: int):
    """SDPA program config for seq_len queries and keys in each of num_heads heads, on the whole compute grid.

    k chunks are SDPA_K_CHUNK_SIZE; _sdpa_q_chunk_size picks the q chunk. Each call returns a new config, so a caller
    may change its own. exp_approx_mode is left unset, so SDPA keeps the op's default, approximate exp.
    """
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=_sdpa_q_chunk_size(seq_len, num_heads, grid.x * grid.y),
        k_chunk_size=SDPA_K_CHUNK_SIZE,
    )


# Unbounded: one int per patch count and grid, and ttnn keeps a compiled SDPA program for each anyway.
@functools.lru_cache(maxsize=None)
def _sdpa_q_chunk_size(seq_len: int, num_heads: int, cores: int) -> int:
    """sdpa_program_config's q chunk, from SDPA_Q_CHUNK_SIZES, on a grid of `cores` cores.

    Takes the chunk that leaves the slowest core the fewest query rows. On a tie it takes the larger chunk, which makes
    fewer passes over k and v.
    """
    # The op deals the num_heads x ceil(seq_len / q) q chunks evenly over the cores, so the slowest core runs
    # ceil(num_heads * ceil(seq_len / q) / cores) chunks of q rows.
    return min(SDPA_Q_CHUNK_SIZES, key=lambda q: (math.ceil(num_heads * math.ceil(seq_len / q) / cores) * q, -q))
