# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Vocabulary-parallel DRAM-sharded terminal projection for one tile of rows.

The caller retains the selected FP32 RMSNorm and BF16 cast. Weight dtype,
HiFi2/FP32 accumulation, logical vocabulary width and sampler indexing are
unchanged. Storage padding belongs only to DRAM bank shards.
"""

import math

import ttnn


class DramHead:
    def __init__(
        self, weight, mesh, compute, *, readers=3, block=4, cores=16, per_core_n=32, output_dtype=ttnn.bfloat16
    ):
        k, n = tuple(weight.shape)[-2:]
        banks = mesh.dram_grid_size().x
        grid = mesh.compute_with_storage_grid_size()
        bank_width = math.ceil(n / (32 * banks * readers)) * 32 * readers
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(dram_grid, (k, bank_width), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.weight = ttnn.to_memory_config(weight, memory)
        self.compute = compute
        self.output_dtype = output_dtype
        self.input_memory = ttnn.create_sharded_memory_config(
            (32, k // cores),
            ttnn.num_cores_to_corerangeset(cores, grid, row_wise=True),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.output_memory = ttnn.create_sharded_memory_config(
            (32, 32 * per_core_n),
            ttnn.num_cores_to_corerangeset(math.ceil(n / (32 * per_core_n)), grid, row_wise=True),
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=block,
            per_core_M=1,
            per_core_N=per_core_n,
            fused_activation=None,
            num_workers_per_dram_bank=readers,
            use_multicore_pipeline=readers > 1,
        )
        self.geometry = dict(
            readers=readers,
            block=block,
            cores=cores,
            per_core_n=per_core_n,
            local_k=k,
            local_n=n,
            bank_storage_width=bank_width,
        )

    def __call__(self, x):
        output = ttnn.linear(
            ttnn.to_memory_config(x, self.input_memory),
            self.weight,
            compute_kernel_config=self.compute,
            program_config=self.program,
            dtype=self.output_dtype,
            memory_config=self.output_memory,
        )
        return ttnn.to_memory_config(output, ttnn.DRAM_MEMORY_CONFIG)
