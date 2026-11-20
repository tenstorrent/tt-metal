# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Preserve close router ranks through attention and router computation."""

import ttnn
from models.demos.gemma4.tt.dram_sharded import DramShardedLinear
from models.demos.gemma4_26b_a4b_qb2.tt.precision_ops import rms_norm


class QKVLinear(DramShardedLinear):
    """Interleaved projection using the imported attention callable interface.

    The shared attention dispatch recognizes callable weights through
    DramShardedLinear. Override its complete call here to keep the original
    interleaved BF16 weight while accumulating QKV products in FP32.
    """

    def __init__(self, weight, mesh_device, *, math_fidelity=ttnn.MathFidelity.HiFi4):
        self.weight = weight
        self.rows = ttnn.transpose(ttnn.typecast(weight, ttnn.float32), -2, -1)
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden_states, compute_kernel_config=None, out_memory_config=None):
        if hidden_states.shape[-2] == 1:
            # Close router ranks amplify decode projection errors. SFPU dots
            # preserve FP32 operands and accumulation; bounded row groups keep
            # the temporary product independent of the complete output width.
            outputs = []
            for start in range(0, self.rows.shape[-2], 256):
                weight = self.rows[:, :, start : start + 256, :]
                repeated = ttnn.repeat(hidden_states, (1, 1, weight.shape[-2], 1))
                products = ttnn.mul(repeated, weight)
                outputs.append(ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1))
            return ttnn.concat(outputs, dim=-1)
        return ttnn.linear(
            hidden_states,
            self.weight,
            dtype=ttnn.float32,
            compute_kernel_config=self.compute,
            memory_config=out_memory_config,
        )


class Router:
    """FP32 routing arithmetic with BF16 expert weights and routing output."""

    def __init__(self, source, epsilon):
        self.source = source
        self.epsilon = epsilon
        self.scale = ttnn.typecast(source.scale, ttnn.float32)
        self.projection_rows = ttnn.transpose(ttnn.typecast(source.proj_weight, ttnn.float32), -2, -1)
        self.compute = ttnn.init_device_compute_kernel_config(
            source.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, hidden_states):
        router = self.source
        normalized = rms_norm(hidden_states, self.epsilon)
        scaled = ttnn.mul(ttnn.mul(normalized, self.scale), router.scalar_root_size)
        if hidden_states.shape[-2] == 1:
            # Float32 matmul inputs use TF32 Src registers on Blackhole. The
            # small router projection uses SFPU products/reduction to preserve
            # the close rank-eight/rank-nine ordering.
            repeated = ttnn.repeat(scaled, (1, 1, router.num_experts, 1))
            products = ttnn.mul(repeated, self.projection_rows)
            scores = ttnn.transpose(ttnn.sum(products, dim=-1, keepdim=True), -2, -1)
        else:
            scores = ttnn.linear(scaled, router.proj_weight, dtype=ttnn.float32, compute_kernel_config=self.compute)
        selected, indices = ttnn.topk(scores, k=router.top_k, dim=-1)
        # Softmax on selected logits equals full softmax followed by top-k
        # sum-normalization, without allowing rounded probabilities to reorder.
        values = ttnn.softmax(selected, dim=-1)
        # Selection has completed before BF16 storage for sparse expert matmul.
        routing = ttnn.scatter(
            ttnn.zeros_like(ttnn.typecast(scores, ttnn.bfloat16)),
            dim=-1,
            index=indices,
            src=ttnn.typecast(values, ttnn.bfloat16),
        )
        return ttnn.mul(routing, router.per_expert_scale)
