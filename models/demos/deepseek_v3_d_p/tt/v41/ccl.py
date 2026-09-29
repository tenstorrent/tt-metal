# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Collectives shared by the DeepSeek-V4.1 modules (SP = mesh axis 0, TP = mesh axis 1)."""

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl, per_axis_topology
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS, TP_AXIS


def fabric_num_links() -> int:
    """Fabric links per chip-to-chip connection V4.1 collectives use: 2 on Blackhole (LoudBox / Galaxy), else 1.

    The one source for every V4.1 collective, including the ones the shared ``TtMoe`` (gate, dispatch, combine,
    shared expert, routed reduce) and ``TtDistributedRmsNorm`` issue, so no module runs on fewer links than the
    fabric has (G2 profile: the MoE and norms at 1 link reached 7-11 % of the 2-link bound)."""
    return 2 if is_blackhole() else 1


class V41Collectives:
    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.tp = mesh_device.shape[TP_AXIS]
        # per mesh axis from the opened fabric: Ring only where it wraps the axis (Galaxy torus), else Linear
        topologies = per_axis_topology()
        self.sp_topology, self.tp_topology = topologies[SP_AXIS], topologies[TP_AXIS]
        self.num_links = fabric_num_links()
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
            topology=self.tp_topology,
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
            topology=self.tp_topology,
            cluster_axis=TP_AXIS,
        )

    def tp_all_reduce(self, t):
        """Row-parallel partial sums -> the full result replicated across TP (reduce-scatter + gather)."""
        return self.tp_all_gather(self.tp_reduce_scatter(t))

    def tp_all_to_all(self, t, in_dim, out_dim):
        """Reshard over TP: concatenate ``in_dim`` across the TP chips and keep this chip's slice of ``out_dim``.

        TP=2 (LoudBox 4x2) gathers ``in_dim`` and keeps this chip's ``out_dim`` slice (``mesh_partition``, local)
        instead of calling ``all_to_all_async_generic``: that op's Fabric2D multicast initialization assumes mesh
        axis 1 runs physically east-west, and on the 4x2 mesh TP runs north-south, so its writer stops in
        ``fail_stop_invalid_fabric_route`` and the collective hangs (bead 8y7.9.1). The gather moves twice the
        all-to-all bytes, only on this 2-chip axis. Remove once the shared op takes the axis's physical direction.
        """
        if self.tp == 1:
            return t
        if self.tp == 2:
            return ttnn.mesh_partition(self.tp_all_gather(t, dim=in_dim), dim=out_dim, cluster_axis=TP_AXIS)
        return ttnn.experimental.all_to_all_async_generic(
            t,
            in_dim=in_dim,
            out_dim=out_dim,
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self.tp_topology,
            cluster_axis=TP_AXIS,
        )
