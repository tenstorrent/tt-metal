# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed/separate indexed expert and shared-router family comparison."""

import math

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_optimized_cache import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        if self.sparse and self.options.get("split_experts"):
            import torch

            prefix = f"model.layers.{self.layer_idx}."
            sd = {k.removeprefix(prefix): v for k, v in args[0].items()}
            for proj in ["gate_proj", "up_proj"]:
                weight = torch.stack([sd[f"mlp.experts.{i}.{proj}.weight"].T for i in range(100)]).unsqueeze(0)
                self.weights["experts." + proj] = ttnn.from_torch(
                    weight,
                    device=self.mesh_device,
                    dtype=getattr(ttnn, self.options.get("experts_dtype", "bfloat8_b")),
                    layout=ttnn.TILE_LAYOUT,
                )
        return self

    def _indexed_mm(self, x, w, ids, placeholder, *, down=False, value=False, gate=False):
        role = "down" if down else ("value" if value else "gate_up")
        split = not down and not value and self.options.get("split_experts")
        grid = self.options.get(("split" if split else role) + "_grid", [8, 1] if split else [8, 2])
        n = w.shape[-1]
        sub = self.options.get(
            ("split" if split else role) + "_subblock", 3 if split else (5 if down else (2 if value else 3))
        )
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
            fused_activation=(
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if gate and self.options.get("fused_gate") else None
            ),
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
            dtype=getattr(ttnn, self.options.get("value_activation" if value else "expert_activation", "bfloat16")),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return ttnn.reshape(out, (1, ids.shape[-1], 32, n))

    def _experts(self, x, *, logits, value=False, valid_tokens=None):
        if valid_tokens != 1 or value or not self.options.get("split_experts"):
            return super()._experts(x, logits=logits, value=value, valid_tokens=valid_tokens)
        scores = ttnn.sigmoid(ttnn.typecast(logits[:, :, :1, :], ttnn.float32))
        _, indices = ttnn.topk(ttnn.add(scores, self.weights["mlp.gate.bias"]), k=8, dim=-1)
        selected = ttnn.gather(scores, 3, index=indices)
        weights = ttnn.mul(
            selected,
            ttnn.sum(selected, dim=-1, keepdim=True),
            input_tensor_b_activations=[ttnn.UnaryOpType.RECIP],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.MUL_UNARY_SFPU, 2.5)],
            dtype=ttnn.bfloat16,
        )
        ids = ttnn.typecast(ttnn.to_layout(indices, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint16)
        placeholder = ttnn.to_layout(scores[..., :100], ttnn.ROW_MAJOR_LAYOUT)
        gate = self._indexed_mm(x, self.weights["experts.gate_proj"], ids, placeholder, gate=True)
        up = self._indexed_mm(x, self.weights["experts.up_proj"], ids, placeholder)
        product = ttnn.mul(
            gate, up, input_tensor_a_activations=[] if self.options.get("fused_gate") else [ttnn.UnaryOpType.SILU]
        )
        out = self._indexed_mm(product, self.weights["experts.down_proj"], ids, placeholder, down=True)
        return ttnn.experimental.fast_reduce_nc(ttnn.mul(out, ttnn.reshape(weights, (1, 8, 1, 1))), dims=[1])

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
            dtype=ttnn.bfloat16,
            memory_config=(
                ttnn.L1_MEMORY_CONFIG
                if self.options.get("sparse_l1") and groups <= self.options.get("l1_groups", 1)
                else ttnn.DRAM_MEMORY_CONFIG
            ),
        )
