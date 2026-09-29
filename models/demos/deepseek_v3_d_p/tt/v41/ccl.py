# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Collectives shared by the DeepSeek-V4.1 modules (SP = mesh axis 0, TP = mesh axis 1)."""

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

TP_AXIS = 1


class V41Collectives:
    def __init__(self, mesh_device, topology=ttnn.Topology.Linear):
        self.mesh_device = mesh_device
        self.tp = mesh_device.shape[TP_AXIS]
        self.topology = topology
        self.num_links = 2 if is_blackhole() else 1
        self.tt_ccl = get_tt_ccl(mesh_device) if max(mesh_device.shape) > 1 else None

    def tp_reduce_scatter(self, t, dim=3):
        """Sum TP partials and keep this chip's slice of ``dim``."""
        if self.tp == 1:
            return t
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=TP_AXIS),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            cluster_axis=TP_AXIS,
        )

    def tp_all_gather(self, t, dim=3):
        if self.tp == 1:
            return t
        return ttnn.experimental.all_gather_async(
            t,
            dim=dim,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=TP_AXIS),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.topology,
            cluster_axis=TP_AXIS,
        )

    def tp_all_reduce(self, t):
        """Row-parallel partial sums -> the full result replicated across TP (reduce-scatter + gather)."""
        return self.tp_all_gather(self.tp_reduce_scatter(t))
