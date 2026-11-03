# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked EP prefill geometry sweep."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _experts_ep(self, x, logits):
        tokens = x.shape[-2]
        groups = tokens // 32
        routes = self._route(logits, "mlp.gate", 8, 100)
        routes = ttnn.pad(routes, [(0, 0), (0, 0), (0, 0), (0, 28)], 0)
        local_routes = ttnn.linear(
            routes, self.weights["experts.ep_selector"], dtype=ttnn.bfloat16, compute_kernel_config=self.compute
        )
        local_routes = ttnn.reshape(local_routes, (1, groups, 32, 32))
        sparsity = ttnn.to_layout(ttnn.sum(local_routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)

        def mm(inp, w, sparse, down=False):
            grid = self.options.get("ep_down_grid" if down else "ep_up_grid", [8, 5] if down else [8, 6])
            n = w.shape[-1]
            sub = self.options.get("ep_down_subblock" if down else "ep_up_subblock", 2 if down else 1)
            prog = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=tuple(grid),
                in0_block_w=self.options.get("ep_down_block" if down else "ep_up_block", 24 if down else 40),
                out_subblock_h=1,
                out_subblock_w=sub,
                out_block_h=1,
                out_block_w=sub,
                per_core_M=1,
                per_core_N=math.ceil(n / 32 / math.prod(grid)),
                fuse_batch=False,
                fused_activation=None,
                mcast_in0=True,
            )
            return ttnn.sparse_matmul(
                inp,
                w,
                sparsity=sparse,
                nnz=None,
                is_input_a_sparse=down,
                is_input_b_sparse=not down,
                program_config=prog,
                compute_kernel_config=self.matmul_compute["experts"],
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.L1_MEMORY_CONFIG if groups == 1 else ttnn.DRAM_MEMORY_CONFIG,
            )

        x = ttnn.reshape(x, (1, groups, 32, 2560))
        gu = mm(x, self.weights["experts.ep_gate_up"], sparsity)
        product = ttnn.reshape(
            ttnn.mul(gu[..., :768], gu[..., 768:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU]),
            (groups, 32, 32, 768),
        )
        down = mm(product, self.weights["experts.ep_down"], ttnn.reshape(sparsity, (1, 1, groups, 32)), True)
        down = ttnn.reshape(down, (groups, 32, 32, 2560))
        scores = ttnn.reshape(ttnn.permute(local_routes, (0, 1, 3, 2)), (groups, 32, 32, 1))
        result = ttnn.experimental.fast_reduce_nc(ttnn.mul(down, scores), dims=[1])
        return ttnn.reshape(result, (1, 1, tokens, 2560))
