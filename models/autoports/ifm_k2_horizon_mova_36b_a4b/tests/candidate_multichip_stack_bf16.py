# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic control: BF16 collective transport, identical local arithmetic."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _prepare_all_reduce(self):
        self.ar_dtype = ttnn.bfloat16
        grid = ttnn.num_cores_to_corerangeset(8, self.mesh_device.compute_with_storage_grid_size(), True)

        def mem(width):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, width // 8], ttnn.ShardOrientation.ROW_MAJOR),
            )

        self.ar_mem = mem(2560)
        if not hasattr(self.ccl, "k2_bf16_slots"):
            self.ccl.k2_bf16_slots = [
                (
                    ttnn.empty(
                        (1, 1, 32, 10240),
                        dtype=self.ar_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        memory_config=mem(10240),
                    ),
                    ttnn.create_global_semaphore(self.mesh_device, grid, 0),
                )
                for _ in range(2)
            ]
        self.ar_slots = self.ccl.k2_bf16_slots
