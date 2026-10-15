# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM-sharded decode matmul: weight width-sharded over the DRAM banks, activation width-sharded in L1.

The default interleaved matmul reads weights at ~110 GB/s per chip; the DRAM-sharded program streams each bank's
shard straight into the cores that own the matching output columns. Decode only (M = one tile row); prefill keeps the
interleaved copy of the weight.
"""

import math

import ttnn

TILE = ttnn.TILE_SIZE


def _find_cores(k_tiles, n_tiles, max_rows=8, max_cols=8):
    """Largest core count that divides both K and N tiles, as (rows, cols) within the compute grid."""
    for cores in range(max_rows * max_cols, 0, -1):
        if k_tiles % cores or n_tiles % cores:
            continue
        for rows in range(1, max_rows + 1):
            if cores % rows == 0 and cores // rows <= max_cols:
                return rows, cores // rows
    raise AssertionError(f"no core grid for K={k_tiles} N={n_tiles} tiles")


def _largest_divisor(n, cap=8):
    return next(d for d in range(cap, 0, -1) if n % d == 0)


class DramShardedLinear:
    """y = x @ W for a per-device weight of logical shape [k, n]; W is stored in a second, DRAM-sharded copy."""

    def __init__(self, mesh_device, weight, k, n, dtype, compute_kernel_config, mesh_mapper, cache_file_name=None):
        """weight: torch [1, 1, K_total, N_total] (same tensor/mapper as the interleaved copy); k, n: per-device dims."""
        self.k, self.n = k, n
        self.compute_kernel_config = compute_kernel_config
        dram_x = mesh_device.dram_grid_size().x
        padded_n = math.ceil(n / (TILE * dram_x)) * (TILE * dram_x)
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_x - 1, 0))})
        weight_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, padded_n // dram_x), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weight = ttnn.as_tensor(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=weight_cfg,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_file_name,
        )
        rows, cols = _find_cores(k // TILE, n // TILE)
        self.num_cores = rows * cols
        cores = ttnn.num_cores_to_corerangeset(self.num_cores, ttnn.CoreCoord(cols, rows), row_wise=True)
        self.act_cfg = self._l1_width_sharded(cores, k // self.num_cores)
        self.out_cfg = self._l1_width_sharded(cores, n // self.num_cores)
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=_largest_divisor(k // TILE // self.num_cores),
            per_core_M=1,
            per_core_N=n // TILE // self.num_cores,
            fused_activation=None,
        )

    @staticmethod
    def _l1_width_sharded(cores, shard_w):
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (TILE, shard_w), ttnn.ShardOrientation.ROW_MAJOR),
        )

    def __call__(self, x, dtype=ttnn.bfloat16):
        """x: [1, 1, 32, k] (any memory config) -> [1, 1, 32, n] L1 width-sharded (self.out_cfg)."""
        if x.memory_config() != self.act_cfg:
            x = ttnn.to_memory_config(x, self.act_cfg)
        return ttnn.linear(
            x,
            self.weight,
            program_config=self.program_config,
            memory_config=self.out_cfg,
            compute_kernel_config=self.compute_kernel_config,
            dtype=dtype,
        )
