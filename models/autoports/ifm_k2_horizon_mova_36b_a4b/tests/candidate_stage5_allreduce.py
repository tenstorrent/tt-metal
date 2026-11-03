# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-op all-reduce family adapted from GPT-OSS DecodeAllReduce.

GPT-OSS uses 30 residual cores for width2880. K2 uses 8 (width2560,
80 tiles), keeping grouped-norm/projection producer and consumer layouts legal.
"""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _prepare_all_reduce(self):
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
