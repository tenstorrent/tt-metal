# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collectives for a 1xN tensor-parallel mesh (replicated-residual contract: one all-reduce per row-parallel matmul).

NB: tt_transformers' ``tt_all_reduce`` returns only the reduce-scattered shard on 1xN meshes (its callers keep a hidden-sharded
residual); we need the fully replicated sum, so all_reduce = reduce_scatter_minimal_async (dim 3) + all_gather_async (dim 3).
"""

from __future__ import annotations

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL


class KimiCCL:
    """1x1: identity. 1x2 (submesh of the QB2 parent): Linear. 1x4 / 1x8: Ring."""

    def __init__(self, mesh_device, topology: ttnn.Topology | None = None):
        self.mesh_device = mesh_device
        self.shape = tuple(mesh_device.shape)
        self.num_devices = self.shape[0] * self.shape[1]
        self.tp = self.shape[1]
        self.tt_ccl = TT_CCL(mesh_device) if self.num_devices > 1 else None
        if topology is None:
            topology = ttnn.Topology.Ring if self.tp >= 4 else ttnn.Topology.Linear
        self.topology = topology
        self.num_links = self.tt_ccl.get_num_links() if self.tt_ccl else 1

    def _as_4d(self, x):
        shape = tuple(x.shape)
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 1:
            rows = 1
            for d in shape[:-1]:
                rows *= d
            return ttnn.reshape(x, (1, 1, rows, shape[-1])), shape
        return x, None

    def reduce_scatter(self, x: ttnn.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """Sum across the mesh and leave each chip its 1/tp slice of the last dim: [1,1,S,H] -> [1,1,S,H/tp]."""
        if self.num_devices == 1:
            return x
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def all_gather(self, x: ttnn.Tensor, dim: int = 3, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """Concatenate the per-chip shards along ``dim`` on every chip."""
        if self.num_devices == 1:
            return x
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=self.num_links,
            topology=self.topology,
            memory_config=memory_config,
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def all_reduce(self, x: ttnn.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """Sum ``x`` (identical shape on every chip) across the mesh; returns the replicated result with x's shape."""
        if self.num_devices == 1:
            return x
        x4, orig = self._as_4d(x)
        rs = self.reduce_scatter(x4, memory_config)
        if x4 is not x:
            ttnn.deallocate(x4)
        out = self.all_gather(rs, dim=3, memory_config=memory_config)
        ttnn.deallocate(rs)
        if orig is not None:
            out = ttnn.reshape(out, orig)
        return out
