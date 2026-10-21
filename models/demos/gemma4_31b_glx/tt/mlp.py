# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 dense GeGLU MLP on the SP8 x TP4 Galaxy, one layer: down(gelu_tanh(gate(x)) * up(x)).

Adapted from models/demos/gemma4_d_p/tt/mlp.py (tensor-parallel dense MLP), on this package's residual layout.

Per chunk (160 rows per chip between blocks, 640 per SP row):
    all_gather rows over TP -> [1, 1, 640, 5376]
    gate linear (column-parallel, intermediate 21504 / TP = 5376 per chip, fused gelu_tanh) and up linear
    gate * up
    down linear (row-parallel) -> partial sums [1, 1, 640, 5376]
    reduce_scatter rows over TP (sums the partials) -> [1, 1, 160, 5376]
"""

import ttnn

from .attention import matmul_grid
from .settings import settings

TP_AXIS = 1


class MLP:
    def __init__(self, mesh, w: dict):
        """w: checkpoint tensors gate_proj, up_proj [intermediate, hidden] and down_proj [hidden, intermediate]."""
        self.mesh = mesh
        sp, tp = tuple(mesh.shape)

        def weight(x, dim):
            return ttnn.from_torch(
                x.transpose(0, 1).contiguous(),
                device=mesh,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, (sp, tp), dims=(None, dim)),
            )

        # gate / up^T [hidden, intermediate]: TP column t holds intermediate columns [t * I/TP, ...).
        self.w_gate, self.w_up = weight(w["gate_proj"], -1), weight(w["up_proj"], -1)
        # down^T [intermediate, hidden]: TP column t holds the matching intermediate rows.
        self.w_down = weight(w["down_proj"], -2)

        self.cfg = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, settings.get("MLP_FIDELITY")),
            math_approx_mode=settings.get("MATH_APPROX_MODE"),
            fp32_dest_acc_en=settings.get("MLP_FP32_ACC"),
            packer_l1_acc=settings.get("PACKER_L1_ACC"),
        )
        self.grid = matmul_grid()
        self.act_dtype = ttnn.float32 if settings.get("MLP_FP32_INTERMEDIATE") else ttnn.bfloat16

    def __call__(self, x):
        """x: the residual chunk [1, 1, L / TP, hidden] bf16 per chip. Returns the same layout."""
        xg = ttnn.all_gather(x, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.linear(
            xg,
            self.w_gate,
            activation="gelu_tanh",
            dtype=self.act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.cfg,
            core_grid=self.grid,
        )
        up = ttnn.linear(
            xg,
            self.w_up,
            dtype=self.act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(xg)
        hidden = ttnn.multiply(gate, up, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        partial = ttnn.linear(
            hidden,
            self.w_down,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.cfg,
            core_grid=self.grid,
        )
        ttnn.deallocate(hidden)
        out = ttnn.reduce_scatter(partial, dim=2, cluster_axis=TP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(partial)
        return out
