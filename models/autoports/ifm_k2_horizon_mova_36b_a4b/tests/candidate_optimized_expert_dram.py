# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test native sparse TensorAccessor with DRAM-width-sharded expert weights."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        grid = self.mesh_device.dram_grid_size()
        banks = grid.x * grid.y
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        for name in self.options.get("sharded_expert_weights", []):
            weight = self.weights[name]
            shape = tuple(weight.shape)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(cores, [math.prod(shape[:-1]), shape[-1] // banks], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.weights[name] = ttnn.to_memory_config(weight, mem)
        return self
