# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Follow the DRAM matmul's computed output grid through norm and residuals."""
import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_alias import AliasCandidate


class NativeCandidate(AliasCandidate):
    native_shards = False

    @classmethod
    def from_state_dict(cls, sd, **kw):
        self = super().from_state_dict(sd, **kw)
        if not self.native_shards:
            return self
        cores = self.residual_cores

        def layout(width):
            sw = math.ceil(width / (32 * cores))
            nc = math.ceil(width / (32 * sw))
            cg = ttnn.num_cores_to_corerangeset(nc, self.device.compute_with_storage_grid_size(), True)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cg, (32, sw * 32), ttnn.ShardOrientation.ROW_MAJOR),
            )

        for name, w in self.dram_weights.items():
            _, _, cfg = self.dram_layouts[name]
            self.dram_layouts[name] = (layout(w.shape[-2]), layout(w.shape[-1]), cfg)
        self.residual_mem = self.dram_layouts["mlp.down_proj"][1]
        grid = self.device.compute_with_storage_grid_size()
        self.norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid.x, grid.y),
            subblock_w=min(4, 128 // cores),
            block_h=1,
            block_w=128 // cores,
            inplace=False,
        )
        return self
