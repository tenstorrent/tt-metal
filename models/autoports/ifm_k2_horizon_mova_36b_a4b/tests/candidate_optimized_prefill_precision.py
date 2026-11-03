# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Phase-specific sparse intermediate precision candidate."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _sparse_mm(self, x, w, sparsity, *, sparse_input=False):
        n = w.shape[-1]
        role = "down" if sparse_input else ("value" if n == 1024 else "gate_up")
        groups = x.shape[0] if sparse_input else x.shape[1]

        def option(key, default):
            return self.options.get(
                ("prefill_" if groups > 1 else "") + role + "_" + key, self.options.get(role + "_" + key, default)
            )

        grid = option("grid", [8, 1])
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=option("block", 2),
            out_subblock_h=1,
            out_subblock_w=option("subblock", 1),
            out_block_h=1,
            out_block_w=option("subblock", 1),
            per_core_M=x.shape[-2] // 32,
            per_core_N=math.ceil(n / (32 * math.prod(grid))),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        return ttnn.sparse_matmul(
            x,
            w,
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=sparse_input,
            is_input_b_sparse=not sparse_input,
            program_config=program,
            compute_kernel_config=self.matmul_compute["value" if role == "value" else "experts"],
            dtype=(
                ttnn.bfloat8_b
                if self.options.get("prefill_value8" if role == "value" else "prefill_expert8")
                else ttnn.bfloat16
            ),
            memory_config=(
                ttnn.L1_MEMORY_CONFIG
                if self.options.get("sparse_l1") and groups <= self.options.get("l1_groups", 1)
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )

    def _experts_expanded(self, x, **kwargs):
        out = super()._experts_expanded(x, **kwargs)
        return ttnn.typecast(out, ttnn.bfloat16) if kwargs.get("value") and out.dtype != ttnn.bfloat16 else out
