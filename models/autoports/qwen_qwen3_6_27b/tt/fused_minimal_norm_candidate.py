# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded native minimal RMSNorm adapter after the source-contract repair.

No production/default changes. Native gamma stays FP32 row-major; activation
conversion into8x1 L1 shards is inside the measured boundary. Two independent
stats/semaphore/output sets are owned before tracing. Imports remain deferred.
"""

from .fused_distributed_norm_candidate import DistributedNormBoundary, geometry as _base_geometry


def geometry(batch, storage_row=0):
    assert isinstance(storage_row, int) and 0 <= storage_row < 10
    result = _base_geometry(batch)
    result.update(
        storage_row=storage_row,
        input_core_start=[0, storage_row],
        input_core_end=[7, storage_row],
        stats_core=[0, storage_row],
    )
    return result


class MinimalDistributedNormBoundary(DistributedNormBoundary):
    def __init__(self, model, example, gamma, *, key, projection_key, family, storage_row=0):
        import ttnn
        from .multichip_decoder import _BlackholeCCLManager

        assert family in ("control", "minimal")
        # Reuse only the control's common projection metadata; no DiT allocation
        # or dispatch is performed by this adapter.
        super().__init__(model, example, gamma, key=key, projection_key=projection_key, family="control")
        self.family = family
        geometry(self.batch, storage_row)
        self.storage_row = storage_row
        assert gamma.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(gamma.shape) == (1, 1, 40, 32)
        self.norm_outputs = []
        if family == "control":
            return
        self.input_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, storage_row), ttnn.CoreCoord(7, storage_row))}
        )
        stats_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, storage_row), ttnn.CoreCoord(0, storage_row))}
        )
        self.input_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(self.input_cores, (32, 160), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.stats_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(stats_cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1), subblock_w=1, block_h=1, block_w=5, inplace=False
        )
        self.sems = [ttnn.create_global_semaphore(self.mesh, self.input_cores, 0) for _ in range(2)]
        self.stats = [
            ttnn.empty(
                (1, 1, 32, 128),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.stats_memory,
            )
            for _ in range(2)
        ]
        self.norm_outputs = [
            ttnn.empty(
                (1, 1, self.batch, 1280),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.input_memory,
            )
            for _ in range(2)
        ]
        for stats in self.stats:
            assert tuple(stats.padded_shape) == (1, 1, 32, 128)
        self.ag = _BlackholeCCLManager(self.mesh, model.ccl.num_links, ttnn.Topology.Linear)
        self.ag_sems = [self.ag.get_ag_ping_pong_semaphore() for _ in range(2)]
        self.ag_barriers = [self.ag.get_barrier_semaphore() for _ in range(2)]
        self.ag_memory = (
            ttnn.DRAM_MEMORY_CONFIG if model.policy.get("ccl_decode_memory") == "DRAM" else ttnn.L1_MEMORY_CONFIG
        )
        self.ag_outputs = [
            ttnn.empty(
                (1, 1, self.batch, 5120),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=self.ag_memory,
            )
            for _ in range(2)
        ]

    def prepare(self, x, phase):
        import ttnn

        if self.family == "control":
            return super().prepare(x, phase)
        assert phase in (0, 1)
        local = ttnn.to_memory_config(ttnn.reshape(x, (1, 1, self.batch, 1280)), self.input_memory)
        normalized = ttnn.fused_rms_minimal(
            local,
            self.program,
            1,
            self.mesh,
            self.sems[phase],
            persistent_output_tensor=self.norm_outputs[phase],
            num_links=1,
            topology=ttnn.Topology.Linear,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model.ckc,
            memory_config=self.input_memory,
            epsilon=self.model.config.rms_norm_eps,
            weight=self.gamma,
            stats=self.stats[phase],
        )
        normalized = ttnn.to_memory_config(normalized, self.ag_memory)
        gathered = ttnn.experimental.all_gather_async(
            normalized,
            dim=3,
            cluster_axis=1,
            mesh_device=self.mesh,
            topology=ttnn.Topology.Linear,
            num_links=self.ag.num_links,
            multi_device_global_semaphore=self.ag_sems[phase],
            barrier_semaphore=self.ag_barriers[phase],
            memory_config=self.ag_memory,
            persistent_output_tensor=self.ag_outputs[phase],
        )
        return ttnn.to_memory_config(ttnn.reshape(gathered, (1, self.batch, 5120)), self.canonical_memory)

    def ownership(self):
        result = super().ownership()
        result.update(
            native_norm_outputs=[int(t.buffer_address()) for t in self.norm_outputs],
            native_links=1,
            stats_cores=1,
            stats_tiles=4,
            input_grid=[8, 1],
            input_shard=[32, 160],
            gamma_layout="ROW_MAJOR",
            gamma_shape=[1, 1, 40, 32],
            storage_row=self.storage_row,
            input_core_start=[0, self.storage_row],
            input_core_end=[7, self.storage_row],
            stats_core=[0, self.storage_row],
            physical_rows=32,
        )
        return result
