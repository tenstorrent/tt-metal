# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-op all-reduce family adapted from GPT-OSS DecodeAllReduce.

GPT-OSS uses 30 residual cores for width2880. K2 uses 8 (width2560,
80 tiles), keeping grouped-norm/projection producer and consumer layouts legal.
"""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.multichip_candidate_base import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        self.ar_dtype = getattr(ttnn, self.options.get("ccl_dtype", "bfloat8_b"))
        cores = self.options.get("ccl_cores", 8)
        grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)

        def mem(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, width // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.ar_mem = mem(2560)
        self.ar_slots = [
            (
                ttnn.empty(
                    (1, 1, 32, 2560 * 4),
                    dtype=self.ar_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    memory_config=mem(2560 * 4),
                ),
                ttnn.create_global_semaphore(self.mesh_device, grid, 0),
            )
            for _ in range(2)
        ]
        self.ar_idx = 0
        return self

    def _reduce(self, x):
        if x.shape[-2] != 32:
            return super()._reduce(x)
        if self.residual_sharded:
            raise ValueError("One-op allreduce uses replicated residual")
        partial = ttnn.to_memory_config(ttnn.typecast(x, self.ar_dtype), self.ar_mem)
        scratch, semaphore = self.ar_slots[self.ar_idx]
        self.ar_idx = 1 - self.ar_idx
        return ttnn.experimental.all_reduce_async(
            partial,
            scratch,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=semaphore,
            memory_config=self.ar_mem,
            dtype=ttnn.bfloat16,
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
        )
