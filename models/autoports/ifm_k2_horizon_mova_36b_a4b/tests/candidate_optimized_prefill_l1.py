# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill producers write matmul inputs directly to L1; no restore/copy control."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    def _norm_base(self, x, name):
        shape = tuple(x.shape)
        grouped = ttnn.reshape(ttnn.typecast(x, ttnn.float32), (1, 1, math.prod(shape) // 1280, 1280))
        y = ttnn.rms_norm(grouped, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute)
        y = ttnn.reshape(y, shape)
        return ttnn.mul(
            y,
            self.weights[name + ".weight"],
            dtype=getattr(
                ttnn, self.options["attention_activation" if name == "input_layernorm" else "mlp_activation"]
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _mlp_base(self, x, *, valid_tokens=None):
        prefix = "mlp.shared_experts" if self.sparse else "mlp"
        router_logits = None
        if x.shape[-2] > 32 and (not self.sparse):
            hidden = ttnn.experimental.minimal_matmul(
                x,
                self.weights[prefix + ".swiglu.weight"],
                fuse_swiglu=True,
                config=ttnn.MinimalMatmulConfig(
                    M_block_size=4,
                    K_block_size=16,
                    N_block_size=8,
                    subblock_h=2,
                    subblock_w=4,
                    compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
                ),
                dtype=ttnn.bfloat8_b,
                compute_kernel_config=self.matmul_compute["mlp"],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        elif self.sparse:
            gu = self._linear(x, "mlp.shared_router")
            router_logits = gu[..., 1536:]
            hidden = ttnn.mul(
                gu[..., :768],
                gu[..., 768:1536],
                input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        else:
            gate = ttnn.linear(
                x,
                self.weights[prefix + ".gate_proj.weight"],
                activation="silu",
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.matmul_compute["mlp"],
            )
            hidden = ttnn.mul(gate, self._linear(x, prefix + ".up_proj"), memory_config=ttnn.L1_MEMORY_CONFIG)
        dense = self._linear(hidden, prefix + ".down_proj")
        return (
            ttnn.add(self._experts(x, valid_tokens=valid_tokens, logits=router_logits), dense) if self.sparse else dense
        )

    def _finish_base(self, residual, gate_logits, attention, *, valid_tokens=None):
        gated = ttnn.mul(
            attention,
            gate_logits,
            input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, math.log(2), 20.0)],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        x = ttnn.add(residual, self._linear(gated, "self_attn.o_proj"))
        return ttnn.add(x, self._mlp(self._norm(x, "post_attention_layernorm"), valid_tokens=valid_tokens))
