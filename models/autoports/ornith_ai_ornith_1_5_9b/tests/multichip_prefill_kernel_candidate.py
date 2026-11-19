# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Measured short-prefill MLP kernel family; logical rows remain caller-owned."""

import math
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_pre_fusion_reference import MultichipDecoder


class PrefillKernelCandidate(MultichipDecoder):
    def finish(self, x, mixed):
        n = x.shape[2]
        if n == 1 or n > 256:
            return super().finish(x, mixed)
        residual = ttnn.add(x, mixed)
        h = self.norm(residual, "post_attention_layernorm")
        if n <= 64:
            cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=2,
                per_core_M=math.ceil(n / 32),
                per_core_N=2,
                fused_activation=None,
                fuse_batch=True,
                mcast_in0=True,
            )
        else:
            gy = 8 if n <= 128 else 10
            cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(11, gy),
                in0_block_w=16,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=math.ceil(n / (32 * gy)),
                per_core_N=9,
                fused_activation=None,
                transpose_mcast=False,
            )

        def mm(name):
            return ttnn.linear(
                h,
                self.weights[name + ".weight"],
                program_config=cfg,
                compute_kernel_config=self.mlp_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        h = ttnn.multiply(mm("mlp.gate_proj"), mm("mlp.up_proj"), input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        return ttnn.add(residual, self.reduce(self.linear(h, "mlp.down_proj")))
