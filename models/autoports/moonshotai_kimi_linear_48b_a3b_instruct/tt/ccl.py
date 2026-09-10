# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collectives for a 1xN tensor-parallel mesh (replicated-residual contract: one all-reduce per row-parallel matmul)."""

from __future__ import annotations

import ttnn
from models.tt_transformers.tt.ccl import TT_CCL, tt_all_gather, tt_all_reduce


class KimiCCL:
    """Wraps tt_transformers' TT_CCL semaphores with the mesh-shape-derived topology.

    * 1x1: every collective is the identity.
    * 1x2 (submesh of the QB2 parent): Linear.
    * 1x4 / 1x8: Ring (P300x2 / P150x8 wiring has the wrap-around links).
    """

    def __init__(self, mesh_device, topology: ttnn.Topology | None = None):
        self.mesh_device = mesh_device
        self.shape = tuple(mesh_device.shape)
        self.num_devices = self.shape[0] * self.shape[1]
        self.tp = self.shape[1]
        self.tt_ccl = TT_CCL(mesh_device) if self.num_devices > 1 else None
        if topology is None:
            topology = ttnn.Topology.Ring if self.tp >= 4 else ttnn.Topology.Linear
        self.topology = topology

    def all_reduce(self, x: ttnn.Tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        """Sum ``x`` (identical shape on every chip) across the mesh; returns the replicated result."""
        if self.num_devices == 1:
            return x
        return tt_all_reduce(
            x,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,  # 1xN meshes: tt_all_reduce treats the whole mesh as the group
            dim=3,
            topology=self.topology,
            memory_config=memory_config,
        )

    def all_gather(self, x: ttnn.Tensor, dim: int = 3, memory_config=ttnn.DRAM_MEMORY_CONFIG) -> ttnn.Tensor:
        if self.num_devices == 1:
            return x
        return tt_all_gather(
            x,
            self.mesh_device,
            dim=dim,
            cluster_axis=1,
            num_links=self.tt_ccl.get_num_links(),
            topology=self.topology,
            memory_config=memory_config,
            tt_ccl=self.tt_ccl,
        )
