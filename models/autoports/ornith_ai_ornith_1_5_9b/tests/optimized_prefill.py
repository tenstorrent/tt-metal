# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill kernel-family candidates using the same real-weight whole-layer tests."""
import math

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.optimized_topology import TopologyCandidate


class PrefillCandidate(TopologyCandidate):
    prefill_family = "default"
    prefill_block = 4
    prefill_grid = (8, 8)
    prefill_subblock = 1
    prefill_mlp = "minimal"
    prefill_output_block = 0
    mlp_program = None
    mlp_l1 = False
    dtype_overrides = {}
    fidelity_overrides = {}

    def weight_dtype(self, name):
        return getattr(ttnn, self.dtype_overrides[name]) if name in self.dtype_overrides else super().weight_dtype(name)

    def matmul_compute(self, name):
        if name not in self.fidelity_overrides:
            return super().matmul_compute(name)
        return ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, self.fidelity_overrides[name]),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def program(self, x, n):
        gx, gy = self.prefill_grid
        m = math.ceil(x.shape[2] / 32)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=self.prefill_block,
            out_subblock_h=1,
            out_subblock_w=max(d for d in range(1, self.prefill_subblock + 1) if math.ceil(n / (32 * gx)) % d == 0),
            per_core_M=math.ceil(m / gy),
            per_core_N=math.ceil(n / (32 * gx)),
            out_block_h=1 if self.prefill_output_block else None,
            out_block_w=max(d for d in range(1, self.prefill_output_block + 1) if math.ceil(n / (32 * gx)) % d == 0)
            if self.prefill_output_block
            else None,
            transpose_mcast=False,
            fused_activation=None,
        )

    def linear(self, x, name):
        if x.shape[2] == 1 or self.prefill_family == "default":
            return super().linear(x, name)
        w = self.weights[name + ".weight"]
        if self.prefill_family == "minimal":
            return ttnn.experimental.minimal_matmul(
                x,
                w,
                compute_kernel_config=self.matmul_compute(name),
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=self.prefill_block,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=self.prefill_subblock,
                    compute_with_storage_grid_size=ttnn.CoreCoord(*self.prefill_grid),
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.linear(
            x,
            w,
            program_config=self.program(x, w.shape[-1]),
            compute_kernel_config=self.matmul_compute(name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def finish(self, x, mixed):
        if x.shape[2] == 1 or (self.mlp_program is None and not self.mlp_l1):
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        if self.mlp_l1:
            h = ttnn.to_memory_config(h, ttnn.L1_MEMORY_CONFIG)
        h = ttnn.experimental.minimal_matmul(
            h,
            self.weights["packed_mlp.weight"],
            fuse_swiglu=True,
            compute_kernel_config=self.mlp_compute,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            config=ttnn.MinimalMatmulConfig(
                **self.mlp_program, compute_with_storage_grid_size=ttnn.CoreCoord(*self.prefill_grid)
            )
            if self.mlp_program
            else None,
        )
        return ttnn.add(residual, self.linear(h, "mlp.down_proj"))
