# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-width-sharded matmul helpers for Gemma4 tensor-parallel decode.

Decode is weight-read-bound (M<=32, one activation tile), so spreading each
per-device weight shard across all DRAM banks and running the DRAM-sharded
matmul kernel cuts the per-token weight-read time. Prefill (M>32) reuses the
same width-sharded weight through a 2D matmul program config.

Ported/adapted from the Qwen3.6 Blackhole TP path (tp_common.py).
"""

import math

import ttnn
from models.common.utility_functions import is_blackhole

TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    """Pick a core count dividing n_tiles closest to `target`, factored into <=8x8."""
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


def prefill_grid_default():
    return (8, 10) if is_blackhole() else (8, 8)


# Prefill activation-row cutoff. The 2D matmul's circular buffers scale with
# per_core_M (= ceil(M/TILE/grid_y)), so a single-shot matmul at long context
# (M = seq_len, up to 256k) overflows L1. Following tt_transformers, we reshape
# [1, 1, M, K] -> [1, M/cutoff, cutoff, K] and run ONE batched matmul sized to
# ``cutoff`` rows (the extra batch dim is iterated by the kernel, reusing CBs).
# This keeps per_core_M tiny AND avoids the memory blow-up of a chunk+concat
# (which would need source chunks + a full-size destination simultaneously).
_PREFILL_CUTOFF = 512 if is_blackhole() else 1024
# Fallback per-call row cap for the (rare) M not divisible by the cutoff.
_PREFILL_M_CHUNK = prefill_grid_default()[1] * 8 * TILE_SIZE


def weight_memcfg(k, n):
    """WIDTH_SHARDED DRAM memory config for a per-device weight shard [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_GRID, (k, padded_n // DRAM_CORES), ttnn.ShardOrientation.ROW_MAJOR),
    )


def decode_progcfg(m, k, n):
    """DRAM-sharded matmul program config for decode (small M)."""
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1
    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=math.ceil(m / TILE_SIZE),
        per_core_N=per_core_N,
        fused_activation=None,
    )


def activation_memcfg(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, k // num_cores),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def can_dram_shard(k, n):
    """True if a [k, n] weight shard factors cleanly for the DRAM-sharded path.

    Guards against grid/divisibility asserts so callers can safely fall back to
    an interleaved weight for shapes that don't fit.
    """
    if k % TILE_SIZE != 0 or n <= 0:
        return False
    try:
        rows, cols = _find_grid(k // TILE_SIZE)
    except ValueError:
        return False
    num_cores = rows * cols
    # Activation width-shard needs k evenly split across the core grid.
    return (k // TILE_SIZE) % num_cores == 0 and (k // num_cores) % TILE_SIZE == 0


def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def prefill_progcfg(m, k, n, grid_size=None):
    """2D matmul program config for prefill on a DRAM-width-sharded weight."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))
    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)
    k_tiles = math.ceil(k / TILE_SIZE)
    in0_block_w = min(4, max(1, k_tiles // grid_size[0]))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


class DramShardedLinear:
    """A single DRAM-width-sharded weight served for both decode and prefill.

    Decode (M<=32): width-shard the activation to L1, run the DRAM-sharded
    kernel, return a DRAM-interleaved result. Prefill (M>32): plain matmul with
    an explicit 2D program config (auto-selection overflows L1 for these shapes).
    """

    def __init__(self, weight_torch, mesh_device, mesh_mapper, k, n, dtype, cache_file_name):
        self.k = k
        self.n = n
        self.weight = ttnn.as_tensor(
            weight_torch,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_file_name,
            memory_config=weight_memcfg(k, n),
        )
        self._act_memcfg = activation_memcfg(k)
        self._decode_pc = decode_progcfg(TILE_SIZE, k, n)

    def __call__(self, x, compute_kernel_config=None, out_memory_config=None):
        out_mc = out_memory_config if out_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        if x.shape[-2] <= TILE_SIZE:
            x_sh = ttnn.to_memory_config(x, self._act_memcfg)
            out = ttnn.linear(
                x_sh,
                self.weight,
                program_config=self._decode_pc,
                compute_kernel_config=compute_kernel_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            x_sh.deallocate(True)
            return ttnn.to_memory_config(out, out_mc)

        # Prefill on the width-sharded weight via the 2D matmul kernel.
        M = x.shape[-2]
        if M <= _PREFILL_CUTOFF:
            pc = prefill_progcfg(M, self.k, self.n)
            return ttnn.linear(
                x, self.weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=out_mc
            )

        if M % _PREFILL_CUTOFF == 0:
            # Reshape M into (batch, cutoff) so per_core_M is sized to the cutoff
            # (tiny CBs) and the batch dim is iterated by the kernel. Both
            # reshapes are metadata-only (cutoff is tile-aligned). Single matmul,
            # single-size output — no concat, no memory doubling.
            batch = M // _PREFILL_CUTOFF
            n_dim = x.shape[-1]
            x_r = ttnn.reshape(x, (1, batch, _PREFILL_CUTOFF, n_dim))
            pc = prefill_progcfg(_PREFILL_CUTOFF, self.k, self.n)
            out_r = ttnn.linear(
                x_r, self.weight, program_config=pc, compute_kernel_config=compute_kernel_config, memory_config=out_mc
            )
            out = ttnn.reshape(out_r, (1, 1, M, out_r.shape[-1]))
            return out

        # Fallback for M not divisible by the cutoff (rare; small M in practice):
        # chunk + concat. Only reached for shapes that don't hit the long-context
        # memory pressure, so the concat's transient extra buffer is affordable.
        outs = []
        for start in range(0, M, _PREFILL_M_CHUNK):
            end = min(start + _PREFILL_M_CHUNK, M)
            x_c = ttnn.slice(x, [0, 0, start, 0], [x.shape[0], x.shape[1], end, self.k])
            pc = prefill_progcfg(end - start, self.k, self.n)
            outs.append(
                ttnn.linear(
                    x_c,
                    self.weight,
                    program_config=pc,
                    compute_kernel_config=compute_kernel_config,
                    memory_config=out_mc,
                )
            )
            x_c.deallocate(True)
        out = ttnn.concat(outs, dim=-2, memory_config=out_mc)
        for o in outs:
            o.deallocate(True)
        return out
