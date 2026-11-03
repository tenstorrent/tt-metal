# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent contiguous RS staging and outputs under the same residual contract."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _prepare_all_reduce(self):
        super()._prepare_all_reduce()
        self.persistent_rs = {}
        self.persistent_ag = {}
        for rows in self.options.get("persistent_rows", [32, 2048]):
            for slot in (0, 1):
                dummy = ttnn.empty(
                    (1, 1, rows, 2560),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                staging = ttnn.experimental.reduce_scatter_minimal_async_create_intermediate_buffer(
                    dummy, dim=3, cluster_axis=1, topology=ttnn.Topology.Ring
                )
                output = ttnn.empty(
                    (1, 1, rows, 640),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                self.persistent_rs[(rows, slot)] = [staging[0], output, staging[1]]
                self.persistent_ag[(rows, slot)] = dummy

    def _reduce(self, x, slot=0):
        key = (x.shape[-2], slot)
        if key not in self.persistent_rs:
            return super()._reduce(x, slot=slot)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=self.persistent_rs[key],
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
        )
        if self.residual_sharded:
            return reduced
        return ttnn.experimental.all_gather_async(
            reduced,
            persistent_output_buffer=self.persistent_ag[key],
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(1),
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
        )
