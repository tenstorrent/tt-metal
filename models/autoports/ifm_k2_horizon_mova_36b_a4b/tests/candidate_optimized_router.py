# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused single-group sigmoid/bias/top-k/normalize routing control."""

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _experts_indexed(self, x, *, logits, value=False, valid_tokens=None):
        if valid_tokens != 1:
            return self._experts_expanded(x, logits=logits, value=value, valid_tokens=valid_tokens)
        e = self.config.mova_num_experts if value else self.config.num_experts
        k = self.config.mova_num_experts_per_tok if value else self.config.num_experts_per_tok
        name = "self_attn.v_router" if value else "mlp.gate"
        logits = logits[:, :, :1, :]
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            self.weights[name + ".bias"],
            n_groups=1,
            topk_groups=1,
            n_activated_experts=k,
            route_scale=2.5,
            epsilon=0.0,
            score_func="sigmoid",
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ids = ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT)
        placeholder = self.placeholders[e]
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

    def _route(self, logits, name, top_k, experts):
        if not self.options.get("prefill_composite"):
            return super()._route(logits, name, top_k, experts)
        bias = ttnn.repeat(self.weights[name + ".bias"], (1, 1, logits.shape[-2], 1))
        weights, indices = ttnn.experimental.deepseek_prefill.moe_grouped_topk(
            logits,
            bias,
            n_groups=1,
            topk_groups=1,
            n_activated_experts=top_k,
            route_scale=2.5,
            epsilon=0.0,
            score_func="sigmoid",
            memory_config=ttnn.L1_MEMORY_CONFIG if logits.shape[-2] == 32 else ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.scatter(ttnn.zeros_like(logits), dim=3, index=indices, src=weights)[..., :experts]
