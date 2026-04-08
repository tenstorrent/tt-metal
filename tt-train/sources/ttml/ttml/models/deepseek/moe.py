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
  2. topk on (scores + bias)             [expert selection, with optional group routing]
  3. weights = scores[selected]           [original unbiased scores]
  4. weights /= weights.sum()            [normalize across selected, sigmoid only]
  5. weights *= route_scale
  6. output = sum(expert_i(x) * weight_i) + shared_expert(x)

Load balancing: auxiliary-loss-free expert bias updated between training steps
using on-device token count accumulation.
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

    Routing: sigmoid/softmax scores -> optional group selection -> topk.
    Execution: dense masking loop over experts (full input to each).
    No CPU interaction during forward pass.
    """

    def __init__(self, config) -> None:
        super().__init__()

        # ttnn.topk returns indices as bf16 which represents integers exactly up to 256.
        # Beyond that, ttnn.eq(topk_indices, float(expert_idx)) may match wrong experts.
        if config.n_routed_experts > 256:
            raise ValueError(
                f"n_routed_experts={config.n_routed_experts} exceeds 256, the maximum "
                f"integer exactly representable in bf16. On-device routing via "
                f"ttnn.topk + ttnn.eq would produce incorrect expert masks."
            )

        self.dim = config.dim
        self.num_experts = config.n_routed_experts
        self.n_activated = config.n_activated_experts
        self.n_groups = config.n_expert_groups
        self.n_limited_groups = config.n_limited_groups
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

        # On-device accumulator for token counts per expert [1, 1, 1, num_experts]
        # Each forward adds to this; update_expert_bias reads and resets it.
        counts_np = np.zeros((1, 1, 1, config.n_routed_experts), dtype=np.float32)
        self._token_counts = Buffer(ttml.autograd.Tensor.from_numpy(counts_np, ttnn.Layout.TILE))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        B, _, S, dim = list(x.get_value().shape)

        # ── 1. Compute scores on device (autograd -> gate weights) ──
        logits = self.gate(x)  # autograd [B, 1, S, num_experts]

        if self.score_func == "softmax":
            scores = autograd_softmax(logits)
        else:
            scores = autograd_sigmoid(logits)

        # ── 2. Top-k routing on device (with optional group selection) ──
        scores_val = scores.get_value()
        biased = ttnn.add(scores_val, self._expert_bias.tensor.get_value())

        if self.n_groups > 1:
            experts_per_group = self.num_experts // self.n_groups

            # Reshape to [B, 1, S*n_groups, experts_per_group]
            biased_grouped = ttnn.reshape(biased, [B, 1, S * self.n_groups, experts_per_group])

            # Score each group by its top-2 experts
            top2_vals, _top2_idx = ttnn.topk(biased_grouped, 2, dim=-1)
            group_scores = ttnn.sum(top2_vals, dim=-1, keepdim=True)
            group_scores = ttnn.reshape(group_scores, [B, 1, S, self.n_groups])

            # Select top n_limited_groups groups
            _gv, top_group_indices = ttnn.topk(group_scores, self.n_limited_groups, dim=-1)

            # Build group mask [B, 1, S, num_experts]
            group_mask_parts = []
            for g in range(self.n_groups):
                match = ttnn.eq(top_group_indices, float(g))
                match_f = ttnn.typecast(match, ttnn.DataType.BFLOAT16)
                match_any = ttnn.sum(match_f, dim=-1, keepdim=True)
                group_selected = ttnn.gt(match_any, 0.0)
                group_selected = ttnn.typecast(group_selected, ttnn.DataType.BFLOAT16)
                group_expert_mask = ttnn.repeat(group_selected, ttnn.Shape([1, 1, 1, experts_per_group]))
                group_mask_parts.append(group_expert_mask)

            group_mask = ttnn.concat(group_mask_parts, dim=-1)
            neg_inf = ttnn.multiply(ttnn.subtract(group_mask, 1.0), 1e9)
            biased_masked = ttnn.add(biased, neg_inf)

            _topk_values, topk_indices = ttnn.topk(biased_masked, self.n_activated, dim=-1)
        else:
            _topk_values, topk_indices = ttnn.topk(biased, self.n_activated, dim=-1)
        # topk_indices: [B, 1, S, n_activated]

        # ── 3. Build per-expert masks, denom, and token count accumulator ──
        expert_masks = {}
        denom = None
        # token_counts_batch: [1, 1, 1, num_experts] — count of tokens routed to each expert this batch
        token_counts_batch = None

        for expert_idx in range(self.num_experts):
            match = ttnn.eq(topk_indices, float(expert_idx))
            match_f = ttnn.typecast(match, ttnn.DataType.BFLOAT16)
            match_any = ttnn.sum(match_f, dim=-1, keepdim=True)  # [B, 1, S, 1]
            mask_narrow = ttnn.gt(match_any, 0.0)
            mask_narrow = ttnn.typecast(mask_narrow, ttnn.DataType.BFLOAT16)
            expert_masks[expert_idx] = mask_narrow

            # Sum this expert's mask to get token count: scalar = sum over [B, 1, S, 1]
            # Reshape to [1, 1, 1, B*S] then sum to get a single scalar, but we need
            # per-expert counts in a [1, 1, 1, num_experts] tensor.
            # Simpler: sum the mask to a scalar, place into a one-hot position.
            # We'll accumulate into token_counts_batch by concat at the end.

            if self.score_func == "sigmoid":
                score_i_raw = ttnn.slice(scores_val, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])
                selected_score = ttnn.multiply(score_i_raw, mask_narrow)
                if denom is None:
                    denom = selected_score
                else:
                    denom = ttnn.add(denom, selected_score)

        if denom is not None:
            denom = ttnn.add(denom, 1e-20)

        # Accumulate token counts on device:
        # Stack all expert mask sums into [1, 1, 1, num_experts]
        expert_count_scalars = []
        for expert_idx in range(self.num_experts):
            # Sum mask [B,1,S,1] -> scalar, reshape to [1,1,1,1]
            count = ttnn.sum(expert_masks[expert_idx])  # scalar-ish tensor
            count = ttnn.reshape(count, [1, 1, 1, 1])
            expert_count_scalars.append(count)
        batch_counts = ttnn.concat(expert_count_scalars, dim=-1)  # [1, 1, 1, num_experts]
        # Add to running accumulator
        new_counts = ttnn.add(self._token_counts.tensor.get_value(), batch_counts)
        self._token_counts.tensor.set_value(new_counts)

        # ── 4. Per-expert computation with normalized scores ──
        output = None

        for expert_idx in range(self.num_experts):
            mask_narrow = expert_masks[expert_idx]

            expert_mask = ttnn.repeat(mask_narrow, ttnn.Shape([1, 1, 1, dim]))
            mask_tt = ttml.autograd.Tensor(expert_mask, False)

            score_i = autograd_slice(scores, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])

            if self.score_func == "sigmoid" and denom is not None:
                norm_factor = ttnn.reciprocal(denom)
                norm_factor = ttnn.multiply(norm_factor, self.route_scale)
                norm_tt = ttml.autograd.Tensor(norm_factor, False)
                routing_weight = ttml.ops.binary.mul(score_i, norm_tt)
            else:
                if self.route_scale != 1.0:
                    routing_weight = ttml.ops.binary.mul(score_i, self.route_scale)
                else:
                    routing_weight = score_i

            expert_out = self.experts[expert_idx](x)
            weighted = ttml.ops.binary.mul(expert_out, routing_weight)
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
        """Auxiliary-loss-free load balancing (DeepSeek-V3 style).

        Reads accumulated token counts from device, computes bias adjustment
        using sign(mean_count - expert_count), updates bias on device, and
        resets the counter.

        Call between training steps (e.g., every N steps).
        """
        counts_val = self._token_counts.tensor.get_value()

        # Compute mean token count across experts: [1,1,1,num_experts] -> scalar
        mean_count = ttnn.mean(counts_val, dim=-1, keepdim=True)  # [1,1,1,1]

        # delta = coeff * sign(mean - count_per_expert)
        diff = ttnn.subtract(mean_count, counts_val)  # [1,1,1,num_experts] positive for underused
        delta = ttnn.sign(diff)
        delta = ttnn.multiply(delta, coeff)

        # Zero-center the delta so total bias doesn't drift
        delta_mean = ttnn.mean(delta, dim=-1, keepdim=True)
        delta = ttnn.subtract(delta, delta_mean)

        # Update bias
        new_bias = ttnn.add(self._expert_bias.tensor.get_value(), delta)
        self._expert_bias.tensor.set_value(new_bias)

        # Reset token counts
        zeros = ttnn.zeros_like(counts_val)
        self._token_counts.tensor.set_value(zeros)
