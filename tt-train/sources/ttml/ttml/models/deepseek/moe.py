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
from ttml.common.profiler_utils import profiler_marker
from ttml.modules import AbstractModuleBase, LinearLayer, Buffer, ModuleList

from .transformer import DeepSeekMLP
from .autograd_ops import (
    autograd_slice,
    autograd_sigmoid,
    autograd_softmax,
    moe_routing_normalize,
)


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

        # We need to cast topk output to bf16 for some ops that come next
        # and bf16 represents integers exactly up to 256.
        if config.n_routed_experts > 256:
            raise ValueError(
                f"n_routed_experts={config.n_routed_experts} exceeds 256, the maximum "
                f"integer exactly representable in bf16. On-device routing via "
                f"ttnn.topk + ttnn.eq would produce incorrect expert masks."
            )
        if config.n_expert_groups > 1:
            if config.n_routed_experts % config.n_expert_groups != 0:
                raise ValueError(
                    f"n_routed_experts ({config.n_routed_experts}) must be divisible by "
                    f"n_expert_groups ({config.n_expert_groups}) for grouped routing reshape."
                )
            if not (1 <= config.n_limited_groups <= config.n_expert_groups):
                raise ValueError(
                    f"n_limited_groups ({config.n_limited_groups}) must be in "
                    f"[1, n_expert_groups={config.n_expert_groups}]."
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
        device = ttml.autograd.AutoContext.get_instance().get_device()
        shape = [1, 1, 1, config.n_routed_experts]
        self._expert_bias = Buffer(
            ttml.autograd.create_tensor(
                ttnn.zeros(shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE)
            )
        )

        # On-device accumulator for token counts per expert [1, 1, 1, num_experts]
        # Each forward adds to this; update_expert_bias reads and resets it.
        self._token_counts = Buffer(
            ttml.autograd.create_tensor(
                ttnn.zeros(shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE)
            )
        )

    def compute_routing(self, x: ttml.autograd.Tensor):
        """Compute per-token expert routing decisions.

        Runs: gate -> score (sigmoid/softmax) -> optional group masking -> top-k.

        Exposed as a standalone method so tests can inspect routing without
        triggering the full MoE forward (experts + shared). The return value
        is a tuple ``(scores, topk_values, topk_indices)`` where:
          - ``scores`` is the autograd tensor of per-expert scores (used
            downstream by ``forward`` to build routing weights),
          - ``topk_values`` and ``topk_indices`` are ttnn tensors of shape
            ``[B, 1, S, n_activated]`` describing which experts each token
            was routed to.
        """
        B, _, S, _ = list(x.get_value().shape)

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
            top2_vals, _top2_idx = ttnn.topk(ttnn.typecast(biased_grouped, ttnn.DataType.BFLOAT16), 2, dim=-1)
            group_scores = ttnn.sum(top2_vals, dim=-1, keepdim=True)
            group_scores = ttnn.reshape(group_scores, [B, 1, S, self.n_groups])

            # Select top n_limited_groups groups
            _gv, top_group_indices = ttnn.topk(
                ttnn.typecast(group_scores, ttnn.DataType.BFLOAT16), self.n_limited_groups, dim=-1
            )

            # Build group mask [B, 1, S, num_experts] via scatter + repeat_interleave.
            # Scatter needs UINT32 indices in ROW_MAJOR layout.
            b, _, s, _ = list(biased.shape)
            top_group_indices = ttnn.to_layout(top_group_indices, ttnn.ROW_MAJOR_LAYOUT)
            group_mask = ttnn.zeros(
                [b, 1, s, self.n_groups], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, biased.device()
            )
            group_src = ttnn.ones(
                [b, 1, s, self.n_limited_groups], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, biased.device()
            )
            group_mask = ttnn.scatter(group_mask, -1, top_group_indices, group_src)
            group_mask = ttnn.repeat_interleave(group_mask, experts_per_group, dim=-1)
            group_mask = ttnn.to_layout(group_mask, ttnn.TILE_LAYOUT)
            neg_inf = ttnn.multiply(ttnn.subtract(group_mask, 1.0), 1e9)
            biased_masked = ttnn.add(biased, neg_inf)

            topk_values, topk_indices = ttnn.topk(
                ttnn.typecast(biased_masked, ttnn.DataType.BFLOAT16), self.n_activated, dim=-1
            )
        else:
            topk_values, topk_indices = ttnn.topk(
                ttnn.typecast(biased, ttnn.DataType.BFLOAT16), self.n_activated, dim=-1
            )
        # topk_indices: [B, 1, S, n_activated]

        return scores, topk_values, topk_indices

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        B, _, S, dim = list(x.get_value().shape)

        # Branch subsections from x_in so backward markers are attributed to
        # distinct zones rather than one chained marker path.
        x_in = profiler_marker(x, "[START] [MoE]")

        x_r = profiler_marker(x_in, "[START] [MoE] Routing")
        scores, _topk_values, topk_indices = self.compute_routing(x_r)

        # ── 3. Build dense per-token expert mask via scatter ──
        # Scatter ones at topk indices into a zero tensor to produce a
        # one-hot-per-selected-expert mask [B, 1, S, num_experts] in a single
        # op instead of num_experts independent eq/sum/gt/typecast chains.
        device = x.get_value().device()
        topk_indices = ttnn.to_layout(topk_indices, ttnn.ROW_MAJOR_LAYOUT)
        expert_mask_all = ttnn.zeros([B, 1, S, self.num_experts], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_src = ttnn.ones([B, 1, S, self.n_activated], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_mask_all = ttnn.scatter(expert_mask_all, -1, topk_indices, expert_src)
        expert_mask_all = ttnn.to_layout(expert_mask_all, ttnn.TILE_LAYOUT)

        # ── 4. Routing weights ──
        # For sigmoid: use MoERoutingNormalize which has the full Jacobian
        # (including cross-expert terms via the shared denominator). For
        # softmax: selected scores are used directly (no renorm), so we fall
        # back to the per-expert multiply with a detached mask.
        if self.score_func == "sigmoid":
            routing_weights = moe_routing_normalize(scores, expert_mask_all, self.route_scale, 1e-20)
        else:
            routing_weights = None  # softmax path uses score_i directly below

        # Accumulate token counts on device via a single reduction:
        # expert_mask_all [B, 1, S, num_experts] -> [1, 1, 1, num_experts].
        mask_bs_flat = ttnn.reshape(expert_mask_all, [1, 1, B * S, self.num_experts])
        batch_counts = ttnn.sum(mask_bs_flat, dim=-2, keepdim=True)  # [1, 1, 1, num_experts]
        new_counts = ttnn.add(self._token_counts.tensor.get_value(), batch_counts)
        self._token_counts.tensor.set_value(new_counts)
        scores = profiler_marker(scores, "[END] [MoE] Routing")

        # ── 5. Per-expert weighted outputs ──
        output = None
        x_e = profiler_marker(x_in, "[START] [MoE] Experts")

        for expert_idx in range(self.num_experts):
            x_exp = profiler_marker(x_e, "[START] [MoE] Expert")
            expert_out = self.experts[expert_idx](x_exp)
            expert_out = profiler_marker(expert_out, "[END] [MoE] Expert")

            if self.score_func == "sigmoid":
                # routing_weights is already mask-zero'd for unselected
                # experts, so no separate mask multiply needed.
                rw_i = autograd_slice(
                    routing_weights,
                    [0, 0, 0, expert_idx],
                    [B, 1, S, expert_idx + 1],
                )
                weighted = ttml.ops.binary.mul(expert_out, rw_i)
            else:
                mask_narrow = ttnn.slice(expert_mask_all, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])
                expert_mask = ttnn.repeat(mask_narrow, ttnn.Shape([1, 1, 1, dim]))
                mask_tt = ttml.autograd.Tensor(expert_mask, False)
                score_i = autograd_slice(scores, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])
                if self.route_scale != 1.0:
                    routing_weight = ttml.ops.binary.mul(score_i, self.route_scale)
                else:
                    routing_weight = score_i
                weighted = ttml.ops.binary.mul(expert_out, routing_weight)
                weighted = ttml.ops.binary.mul(weighted, mask_tt)

            if output is None:
                output = weighted
            else:
                output = ttml.ops.binary.add(output, weighted)

        if output is None:
            output = ttml.autograd.create_tensor(ttml.core.zeros_like(x_in.get_value()))

        output = profiler_marker(output, "[END] [MoE] Experts")

        # ── 5. Shared experts ──
        if self.shared_experts is not None:
            x_s = profiler_marker(x_in, "[START] [MoE] SharedExp")
            shared_out = self.shared_experts(x_s)
            shared_out = profiler_marker(shared_out, "[END] [MoE] SharedExp")
            output = ttml.ops.binary.add(output, shared_out)

        output = profiler_marker(output, "[END] [MoE]")
        return output

    def read_activation_probabilities(self) -> np.ndarray:
        """Non-destructive read of the current per-expert activation probabilities.

        Returns the fraction of tokens for which each expert was selected,
        accumulated across all forward passes since the last
        ``update_expert_bias`` (which resets ``_token_counts``). Shape:
        ``(num_experts,)``, dtype ``float32``, values in ``[0, 1]``.

        Uses the self-normalising identity
        ``sum(counts) == n_activated * total_tokens`` so no batch-size /
        grad-accum bookkeeping is required at the call site.
        """
        counts = ttnn.to_torch(self._token_counts.tensor.get_value())
        counts_np = counts.float().cpu().numpy().flatten()
        total = float(counts_np.sum())
        if total <= 0.0:
            return np.zeros_like(counts_np, dtype=np.float32)
        return (counts_np * float(self.n_activated) / total).astype(np.float32)

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
