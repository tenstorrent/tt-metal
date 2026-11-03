# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Final-policy activation and constant routing-placeholder experiments."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_preclosure import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        if self.sparse:
            import torch

            self.placeholders = {
                e: ttnn.from_torch(
                    torch.zeros(1, 1, 1, e), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
                )
                for e in (64, 100)
            }
        return self

    def _norm(self, x, name):
        if x.shape[-2] != 32 or not self.options.get("sharded_norm"):
            return self._norm_base(x, name)
        # Grouping moves each token's two1280-wide groups into separate rows.
        shape = tuple(x.shape)
        x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        if not self.options.get("bf16_norm"):
            x = ttnn.typecast(x, ttnn.float32)
        grouped = ttnn.reshape(x, (1, 1, 64, 1280))
        cores = self.options.get("norm_cores", 8)
        grid = ttnn.num_cores_to_corerangeset(cores, self.mesh_device.compute_with_storage_grid_size(), True)
        mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(grid, [64, 1280 // cores], ttnn.ShardOrientation.ROW_MAJOR),
        )
        grouped = ttnn.to_memory_config(grouped, mem)
        width = 40 // cores
        sub = next(v for v in range(min(4, width), 0, -1) if width % v == 0)
        prog = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            block_h=2,
            block_w=width,
            subblock_w=sub,
            inplace=False,
        )
        y = ttnn.rms_norm(
            grouped,
            epsilon=self.config.rms_norm_eps,
            compute_kernel_config=self.compute,
            program_config=prog,
            memory_config=mem,
        )
        y = ttnn.reshape(ttnn.to_memory_config(y, ttnn.L1_MEMORY_CONFIG), shape)
        return ttnn.mul(
            y,
            self.weights[name + ".weight"],
            dtype=getattr(
                ttnn,
                self.options.get("attention_activation" if name == "input_layernorm" else "mlp_activation", "bfloat16"),
            ),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def _norm_base(self, x, name):
        shape = tuple(x.shape)
        grouped = ttnn.reshape(ttnn.typecast(x, ttnn.float32), (1, 1, math.prod(shape) // 1280, 1280))
        y = ttnn.rms_norm(grouped, epsilon=self.config.rms_norm_eps, compute_kernel_config=self.compute)
        y = ttnn.reshape(y, shape)
        return ttnn.mul(
            y,
            self.weights[name + ".weight"],
            dtype=getattr(
                ttnn,
                self.options.get("attention_activation" if name == "input_layernorm" else "mlp_activation", "bfloat16"),
            ),
        )

    def _experts_indexed(self, x, *, logits, value=False, valid_tokens=None):
        if valid_tokens != 1:
            return self._experts_expanded(x, logits=logits, value=value, valid_tokens=valid_tokens)
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
        placeholder = (
            self.placeholders[e]
            if self.options.get("constant_placeholder")
            else ttnn.to_layout(scores[..., :e], ttnn.ROW_MAJOR_LAYOUT)
        )
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
