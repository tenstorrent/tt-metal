# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Restore optimized large-prefill packed minimal SwiGLU at local TP4 width."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_multichip_allreduce import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _mlp(self, x, *, valid_tokens=None):
        if self.sparse or x.shape[-2] == 32:
            return super()._mlp(x, valid_tokens=valid_tokens)
        hidden = ttnn.experimental.minimal_matmul(
            x,
            self.weights["mlp.swiglu.weight"],
            fuse_swiglu=True,
            config=ttnn.MinimalMatmulConfig(
                M_block_size=4,
                K_block_size=16,
                N_block_size=8,
                subblock_h=2,
                subblock_w=4,
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.matmul_compute["mlp"],
        )
        return self._reduce(self._linear(hidden, "mlp.down_proj"))
