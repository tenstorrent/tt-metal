# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Ring setup for sequence parallelism.

`ParallelFactor` names how many devices a dimension is split over and along which mesh axis.
`TT_CCL` owns what ring attention needs across calls: `num_links`, the topology, the global
semaphores and the persistent K/V gather buffers. Semaphores are hardware resources, so one
`TT_CCL` is built per mesh and shared by every layer.
"""

from __future__ import annotations

from typing import NamedTuple

import ttnn


class ParallelFactor(NamedTuple):
    factor: int
    mesh_axis: int


class TT_CCL:
    def __init__(self, mesh_device: ttnn.MeshDevice, *, num_links: int, topology: ttnn.Topology) -> None:
        self.mesh_device = mesh_device
        self.num_links = num_links
        self.topology = topology

    def ring_attention_semaphores(self):
        """Global semaphores for `ring_joint_scaled_dot_product_attention`, created once."""
        raise NotImplementedError("TT_CCL.ring_attention_semaphores")

    def ring_attention_buffers(self, k_shape, v_shape):
        """Persistent K and V gather buffers for one sequence length, shared across layers."""
        raise NotImplementedError("TT_CCL.ring_attention_buffers")
