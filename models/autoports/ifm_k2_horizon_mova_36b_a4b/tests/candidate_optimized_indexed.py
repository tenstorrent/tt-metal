# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compact active-expert decode using the checkout's indexed sparse kernel."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_base import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _indexed_mm(self, x, w, ids, placeholder, *, down=False, value=False):
        role = "down" if down else ("value" if value else "gate_up")
        grid = self.options.get(role + "_grid", [8, 2])
        n = w.shape[-1]
        sub = self.options.get(role + "_subblock", 5 if down else (2 if value else 3))
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=tuple(grid),
            in0_block_w=self.options.get(role + "_block", 24 if down else 40),
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
        out = ttnn.sparse_matmul(
            x,
            w,
            sparsity=placeholder,
            indices=ids,
            is_input_a_sparse=down,
            is_input_b_sparse=True,
            program_config=pc,
            compute_kernel_config=self.matmul_compute["value" if value else "experts"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.reshape(out, (1, ids.shape[-1], 32, n))

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if valid_tokens != 1:
            return super()._experts(x, logits=logits, value=value, valid_tokens=valid_tokens)
        e = self.config.mova_num_experts if value else self.config.num_experts
        k = self.config.mova_num_experts_per_tok if value else self.config.num_experts_per_tok
        name = "self_attn.v_router" if value else "mlp.gate"
        logits = logits[:, :, :1, :]
        scores = ttnn.sigmoid(ttnn.typecast(logits, ttnn.float32))
        _, indices = ttnn.topk(ttnn.add(scores, self.weights[name + ".bias"]), k=k, dim=-1)
        selected = ttnn.gather(scores, 3, index=indices)
        weights = ttnn.mul(
            selected,
            ttnn.sum(selected, dim=-1, keepdim=True),
            input_tensor_b_activations=[ttnn.UnaryOpType.RECIP],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.5)],
            dtype=ttnn.bfloat16,
        )
        ids = ttnn.typecast(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint16)
        # The indexed kernels do not read sparsity; it remains a shape contract.
        placeholder = ttnn.to_layout(scores[..., :e], ttnn.ROW_MAJOR_LAYOUT)
        if value:
            out = self._indexed_mm(x, self.weights["value_experts"], ids, placeholder, value=True)
        else:
            gu = self._indexed_mm(x, self.weights["experts.gate_up"], ids, placeholder)
            product = ttnn.mul(gu[..., :768], gu[..., 768:], input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
            out = self._indexed_mm(product, self.weights["experts.down_proj"], ids, placeholder, down=True)
        weights = ttnn.reshape(weights, (1, k, 1, 1))
        weighted = ttnn.mul(out, weights, input_tensor_a_activations=[ttnn.UnaryOpType.SILU] if value else [])
        result = ttnn.experimental.fast_reduce_nc(weighted, dims=[1])
        return ttnn.typecast(result, ttnn.bfloat16) if value and result.dtype != ttnn.bfloat16 else result
