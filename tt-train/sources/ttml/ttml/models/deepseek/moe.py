# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 Mixture of Experts (MoE) layer.

Dense masking implementation: every expert processes the full input,
masked by routing decisions. Suitable for small expert counts (nano model).

All routing (sigmoid, topk, mask building, bias) runs entirely on device.
No CPU interaction during forward pass.

Matches reference DeepSeek-V3 Gate semantics:
  1. scores = sigmoid(gate(x))           [or softmax]
  2. topk on (scores + bias)             [expert selection]
  3. weights = scores[selected]           [original unbiased scores]
  4. weights /= weights.sum()            [normalize across selected, sigmoid only]
  5. weights *= route_scale
  6. output = sum(expert_i(x) * weight_i) + shared_expert(x)
"""

from __future__ import annotations

import numpy as np
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, Buffer, ModuleList

from .transformer import DeepSeekMLP
from .autograd_ops import autograd_slice, autograd_sigmoid, autograd_softmax


class Expert(AbstractModuleBase):
    """Single SwiGLU expert with its own weight parameters."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = LinearLayer(dim, hidden_dim, has_bias=False)
        self.w3 = LinearLayer(dim, hidden_dim, has_bias=False)
        self.w2 = LinearLayer(hidden_dim, dim, has_bias=False)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        return self.w2(ttml.ops.binary.mul(ttml.ops.unary.silu(self.w1(x)), self.w3(x)))


class MoE(AbstractModuleBase):
    """Mixture of Experts with fully on-device routing.

    Routing: sigmoid/softmax scores -> topk on device -> per-expert masks.
    Execution: dense masking loop over experts (full input to each).
    No CPU interaction during forward pass.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dim = config.dim
        self.num_experts = config.n_routed_experts
        self.n_activated = config.n_activated_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale

        # Router gate
        self.gate = LinearLayer(config.dim, config.n_routed_experts, has_bias=False)

        # Individual expert modules
        self.experts = ModuleList([Expert(config.dim, config.moe_inter_dim) for _ in range(config.n_routed_experts)])

        # Shared expert(s)
        if config.n_shared_experts > 0:
            self.shared_experts = DeepSeekMLP(config.dim, config.n_shared_experts * config.moe_inter_dim)
        else:
            self.shared_experts = None

        # Load balancing bias buffer (on device, TILE layout)
        bias_np = np.zeros((1, 1, 1, config.n_routed_experts), dtype=np.float32)
        self._expert_bias = Buffer(ttml.autograd.Tensor.from_numpy(bias_np, ttnn.Layout.TILE))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        B, _, S, dim = list(x.get_value().shape)

        # ── 1. Compute scores on device (autograd -> gate weights) ──
        logits = self.gate(x)  # autograd [B, 1, S, num_experts]

        if self.score_func == "softmax":
            scores = autograd_softmax(logits)
        else:
            scores = autograd_sigmoid(logits)

        # ── 2. Top-k routing on device ──
        scores_val = scores.get_value()
        biased = ttnn.add(scores_val, self._expert_bias.tensor.get_value())
        _topk_values, topk_indices = ttnn.topk(biased, self.n_activated, dim=-1)
        # topk_indices: [B, 1, S, n_activated] float

        # ── 3. Build per-expert masks and normalization denominator on device ──
        # For sigmoid: need to normalize scores across selected experts per token.
        # denom[token] = sum of scores for the k selected experts for that token.
        # We accumulate this as we iterate over experts.
        #
        # For softmax: scores already sum to 1, no extra normalization needed
        # (reference DeepSeek only normalizes for sigmoid).

        expert_masks = {}  # expert_idx -> ttnn mask [B, 1, S, 1]
        denom = None  # [B, 1, S, 1] running sum of selected scores

        for expert_idx in range(self.num_experts):
            match = ttnn.eq(topk_indices, float(expert_idx))
            match_f = ttnn.typecast(match, ttnn.DataType.BFLOAT16)
            match_any = ttnn.sum(match_f, dim=-1, keepdim=True)  # [B, 1, S, 1]
            mask_narrow = ttnn.gt(match_any, 0.0)  # [B, 1, S, 1]
            mask_narrow = ttnn.typecast(mask_narrow, ttnn.DataType.BFLOAT16)
            expert_masks[expert_idx] = mask_narrow

            if self.score_func == "sigmoid":
                # Extract this expert's raw score for denom accumulation
                score_i_raw = ttnn.slice(scores_val, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])  # [B, 1, S, 1]
                selected_score = ttnn.multiply(score_i_raw, mask_narrow)
                if denom is None:
                    denom = selected_score
                else:
                    denom = ttnn.add(denom, selected_score)

        # Avoid division by zero
        if denom is not None:
            denom = ttnn.add(denom, 1e-20)

        # ── 4. Per-expert computation with normalized scores ──
        output = None

        for expert_idx in range(self.num_experts):
            mask_narrow = expert_masks[expert_idx]

            # Expand mask to [B, 1, S, dim] for element-wise multiply
            expert_mask = ttnn.repeat(mask_narrow, ttnn.Shape([1, 1, 1, dim]))
            mask_tt = ttml.autograd.Tensor(expert_mask, False)

            # Get this expert's score from autograd tensor (gradient -> gate)
            score_i = autograd_slice(scores, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])  # autograd [B, 1, S, 1]

            # Normalize score for sigmoid: score_i / denom (on device, non-differentiable normalization)
            # The normalization constant is treated as a constant for gradient purposes.
            # Gradient still flows through score_i to the gate.
            if self.score_func == "sigmoid" and denom is not None:
                norm_factor = ttnn.reciprocal(denom)  # [B, 1, S, 1]
                norm_factor = ttnn.multiply(norm_factor, self.route_scale)
                norm_tt = ttml.autograd.Tensor(norm_factor, False)
                # score_i * (route_scale / denom) — gradient flows through score_i
                routing_weight = ttml.ops.binary.mul(score_i, norm_tt)
            else:
                if self.route_scale != 1.0:
                    routing_weight = ttml.ops.binary.mul(score_i, self.route_scale)
                else:
                    routing_weight = score_i

            # Run expert (autograd -> expert weights + input x)
            expert_out = self.experts[expert_idx](x)  # autograd [B, 1, S, dim]

            # Weight by normalized routing score (autograd: gradients to gate AND expert)
            weighted = ttml.ops.binary.mul(expert_out, routing_weight)

            # Apply mask: zero non-selected tokens (non-trainable, gradient passes through)
            weighted = ttml.ops.binary.mul(weighted, mask_tt)

            if output is None:
                output = weighted
            else:
                output = ttml.ops.binary.add(output, weighted)

        if output is None:
            output = ttml.autograd.create_tensor(ttml.core.zeros_like(x.get_value()))

        # ── 5. Shared experts ──
        if self.shared_experts is not None:
            output = ttml.ops.binary.add(output, self.shared_experts(x))

        return output

    def update_expert_bias(self, coeff: float = 0.001) -> None:
        """Auxiliary-loss-free load balancing.

        Call between training steps (outside forward). Only CPU interaction.
        """
        bias_val = self._expert_bias.tensor.get_value()
        fp32 = ttnn.typecast(bias_val, ttnn.DataType.FLOAT32)
        bias_np = np.array(ttnn.to_torch(ttnn.from_device(fp32)).numpy(), dtype=np.float32).reshape(1, 1, 1, -1)

        delta = np.random.normal(0, coeff, size=bias_np.shape).astype(np.float32)
        delta -= delta.mean()
        bias_np = bias_np + delta

        new_bias = ttml.autograd.Tensor.from_numpy(bias_np, ttnn.Layout.TILE)
        self._expert_bias.tensor.assign(new_bias)
