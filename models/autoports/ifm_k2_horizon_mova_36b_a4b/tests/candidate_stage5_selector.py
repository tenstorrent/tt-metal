# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Explicit geometry/fidelity for one-hot local expert selectors."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    def _selector(self, x, w):
        if self.options.get("selector_l1", False):
            x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        per_m = math.ceil(x.shape[-2] / 32 / 8)
        return ttnn.linear(
            x,
            w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG if self.options.get("selector_l1", False) else ttnn.DRAM_MEMORY_CONFIG,
            program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                in0_block_w=w.shape[-2] // 32,
                out_subblock_h=2 if per_m % 2 == 0 else 1,
                out_subblock_w=1,
                per_core_M=per_m,
                per_core_N=1,
                fuse_batch=True,
                mcast_in0=False,
            ),
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=True, packer_l1_acc=True
            ),
        )

    def _experts_ep_value(self, x, logits):
        groups = x.shape[-2] // 32
        routes = self._route(logits, "self_attn.v_router", 4, 64)
        local = self._selector(routes, self.weights["value_ep_selector"])[..., :16]
        local = ttnn.reshape(local, (1, groups, 32, 16))
        sparsity = ttnn.to_layout(ttnn.sum(local, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)
        grid = self.options.get("ep_value_grid", [8, 4])
        sub = self.options.get("ep_value_subblock", 1)
        program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=self.options.get("ep_value_block", 40),
            out_subblock_h=1,
            out_subblock_w=sub,
            out_block_h=1,
            out_block_w=sub,
            per_core_M=1,
            per_core_N=math.ceil(32 / math.prod(grid)),
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
        )
        out = ttnn.sparse_matmul(
            ttnn.reshape(x, (1, groups, 32, 2560)),
            self.weights["value_ep"],
            sparsity=sparsity,
            nnz=None,
            is_input_a_sparse=False,
            is_input_b_sparse=True,
            program_config=program,
            compute_kernel_config=self.matmul_compute["value"],
            dtype=getattr(ttnn, self.options.get("ep_value_activation", "bfloat8_b")),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.reshape(out, (groups, 16, 32, 1024))
        scores = ttnn.reshape(ttnn.permute(local, (0, 1, 3, 2)), (groups, 16, 32, 1))
        out = ttnn.experimental.fast_reduce_nc(
            ttnn.mul(out, scores, input_tensor_a_activations=[ttnn.UnaryOpType.SILU]), dims=[1]
        )
        out = ttnn.typecast(ttnn.reshape(out, (1, 1, groups * 32, 1024)), ttnn.bfloat16)
        return ttnn.experimental.reduce_scatter_minimal_async(
            out,
            dim=3,
            cluster_axis=1,
            multi_device_global_semaphore=self.ccl.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(1),
            topology=ttnn.Topology.Ring,
            num_links=self.options["num_links"],
        )

    def _experts_ep(self, x, logits):
        tokens = x.shape[-2]
        groups = tokens // 32
        routes = self._route(logits, "mlp.gate", 8, 100)
        routes = ttnn.pad(routes, [(0, 0), (0, 0), (0, 0), (0, 28)], 0)
        local_routes = self._selector(routes, self.weights["experts.ep_selector"])
        local_routes = ttnn.reshape(local_routes, (1, groups, 32, 32))
        sparsity = ttnn.to_layout(ttnn.sum(local_routes, dim=2, keepdim=True), ttnn.ROW_MAJOR_LAYOUT)

        def mm(inp, w, sparse, down=False):
            grid = self.options.get("ep_down_grid" if down else "ep_up_grid", [8, 5] if down else [8, 3])
            n = w.shape[-1]
            sub = self.options.get("ep_down_subblock" if down else "ep_up_subblock", 2)
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
