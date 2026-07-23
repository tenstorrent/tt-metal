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

import math
import numpy as np
import torch
import ttnn
import ttml
from ttml.common.memory_utils import memory_snapshot
from ttml.common.profiler_utils import profiler_marker
from ttml.modules import AbstractModuleBase, Buffer, LinearLayer, ModuleList, Parameter

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
        return ttml.ops.swiglu.swiglu(
            x,
            self.w1.weight.tensor,
            self.w2.weight.tensor,
            self.w3.weight.tensor,
        )


class _GatherTopK(ttml.autograd.Function):
    """Gather routing weights at top-k expert indices.

    Forward computes ``out[..., k] = routing_weights[..., topk_indices[..., k]]``.
    Backward scatters ``grad_output`` directly back into the full expert axis,
    avoiding per-expert slice backward concat/permute chains.
    """

    @staticmethod
    def forward(ctx, routing_weights, topk_indices_u32):
        routing_value = routing_weights.get_value()
        ctx.topk_indices_u32 = topk_indices_u32
        ctx.rw_shape = list(routing_value.shape)
        ctx.rw_layout = routing_value.layout

        # Gather routing weights at the top-k expert indices with a single
        # tile-layout gather along the expert axis:
        #   [B,1,S,E] gather index [B,1,S,K] -> [B,1,S,K]
        # Avoids materializing a [B,1,S*K,E] one-hot plus the
        # repeat_interleave / multiply / sum reduction (O(E) -> O(K) per token).
        index = ttnn.to_layout(topk_indices_u32, routing_value.layout)
        return ttnn.gather(routing_value, dim=-1, index=index)

    @staticmethod
    def backward(ctx, grad_output):
        # Top-k indices are unique for a token, so scatter is equivalent to
        # scatter-add here and avoids rebuilding every e:e+1 slice gradient.
        device = grad_output.device()
        topk_indices_rm = ttnn.to_layout(ctx.topk_indices_u32, ttnn.ROW_MAJOR_LAYOUT)
        grad_rm = ttnn.to_layout(grad_output, ttnn.ROW_MAJOR_LAYOUT)
        grad_routing = ttnn.zeros(ctx.rw_shape, grad_output.dtype, ttnn.ROW_MAJOR_LAYOUT, device)
        grad_routing = ttnn.scatter(grad_routing, -1, topk_indices_rm, grad_rm)
        return ttnn.to_layout(grad_routing, ctx.rw_layout)


class MoE(AbstractModuleBase):
    """Mixture of Experts with fully on-device routing.

    Routing: sigmoid/softmax scores -> optional group selection -> topk.
    Execution: dense masking loop over experts (full input to each).
    No CPU interaction during forward pass.
    """

    memory_marker_prefix = "DENSE_MOE"

    def __init__(
        self,
        config,
        *,
        moe_axis_name: str | None = None,
    ) -> None:
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

        # Routed experts: dense ``Expert`` modules, or EP-sharded gate/up/down
        # parameters (``SparseMoEEP``) created directly with a mesh mapper — no
        # dense-on-device weights and no host readback from a duplicate LinearLayer.
        H, I = config.dim, config.moe_inter_dim
        # moe_type decides how routed experts are sharded across moe_axis_name:
        #   sparse_ep — partition the expert list across the axis. Otherwise
        #   (dense / plain sparse / no axis) build replicated dense Expert modules.
        moe_type = str(getattr(config, "moe_type", "sparse_ep")).lower()
        ep_sharded = moe_axis_name is not None and moe_type == "sparse_ep"

        if ep_sharded:
            mesh = ttml.maybe_mesh()
            if mesh is None or not mesh.has_axis(moe_axis_name):
                raise ValueError(f"MoE: moe_axis_name={moe_axis_name!r} requires an open mesh with that axis")
            D = mesh.axis_size(moe_axis_name)
            E = config.n_routed_experts
            if E % D != 0:
                raise ValueError(
                    f"MoE: n_routed_experts ({E}) must be divisible by axis "
                    f"{moe_axis_name!r} size ({D}) for EP-sharded experts"
                )
            self.e_local = E // D
            self._ep_axis_name = moe_axis_name
            # Mapper shards dim 0 of the init tensor across the EP mesh axis.
            # Init shape (D, 1, I, H) → each EP shard gets (1, 1, I, H), i.e.
            # the i-th Parameter on shard r holds global expert r*E_local + i.
            mapper = ttml.mesh().axis_mapper(moe_axis_name, tdim=0)
            k_in = math.sqrt(1.0 / H)
            init_gate = ttml.init.uniform(-k_in, k_in)
            k_mid = math.sqrt(1.0 / I)
            init_down = ttml.init.uniform(-k_mid, k_mid)
            self.w_gate = []
            self.w_up = []
            self.w_down = []
            for i in range(self.e_local):
                gate = Parameter(init_gate((D, 1, I, H), mapper))
                up = Parameter(init_gate((D, 1, I, H), mapper))
                down = Parameter(init_down((D, 1, H, I), mapper))
                setattr(self, f"w_gate_{i}", gate)
                setattr(self, f"w_up_{i}", up)
                setattr(self, f"w_down_{i}", down)
                self.w_gate.append(gate)
                self.w_up.append(up)
                self.w_down.append(down)
            self.experts = ModuleList([])
        else:
            self.w_gate = []
            self.w_up = []
            self.w_down = []
            self.experts = ModuleList(
                [Expert(config.dim, config.moe_inter_dim) for _ in range(config.n_routed_experts)]
            )

        # Shared expert(s). If MoE is running EP-sharded, shard the shared MLP
        # across the same axis so it stops being a fully-replicated FFN
        # duplicating weights, optimizer state, and compute on every chip.
        shared_axis = moe_axis_name if ep_sharded else None
        if config.n_shared_experts > 0:
            self.shared_experts = DeepSeekMLP(
                config.dim,
                config.n_shared_experts * config.moe_inter_dim,
                tp_axis_name=shared_axis,
            )
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

        # Local expert-id vector [E] consumed by moe_group / moe_ungroup. It is
        # constant across forwards, so build it once here instead of per-step.
        # EP shards it across the expert axis (each chip sees its local
        # [r*E_local, (r+1)*E_local) slice); every other layout leaves the
        # mapper None so it is replicated. Only the sparse variants read it.
        leids_mapper = ttml.mesh().axis_mapper(moe_axis_name, tdim=0) if ep_sharded else None
        self._leids = ttnn.from_torch(
            torch.arange(config.n_routed_experts, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=leids_mapper,
        )

    def _memory_snapshot(self, x: ttml.autograd.Tensor, phase: str) -> ttml.autograd.Tensor:
        layer_id = getattr(self, "_debug_layer_id", "unknown")
        label = f"{self.memory_marker_prefix}_L{layer_id}_{phase}"
        return memory_snapshot(x, f"{label}_FWD", f"{label}_BWD")

    def gather_topk_weights(self, routing_weights, topk_indices_u32):
        return _GatherTopK.apply(routing_weights, topk_indices_u32)

    def prepare_routing_weights(self, scores, topk_indices, *, gather_topk: bool = False):
        """Build routing mask/weights and update token-count state.

        Dense MoE consumes the full ``[B,1,S,E]`` routing weights. Sparse MoE
        additionally needs gathered top-k weights ``[B,1,S,K]`` for group/ungroup.
        """
        B, _, S, _ = list(scores.get_value().shape)
        device = scores.get_value().device()
        topk_indices_u32 = ttnn.typecast(topk_indices, ttnn.DataType.UINT32)
        topk_indices_rm = ttnn.to_layout(topk_indices, ttnn.ROW_MAJOR_LAYOUT)

        expert_mask_all = ttnn.zeros([B, 1, S, self.num_experts], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_src = ttnn.ones([B, 1, S, self.n_activated], ttnn.DataType.BFLOAT16, ttnn.ROW_MAJOR_LAYOUT, device)
        expert_mask_all = ttnn.scatter(expert_mask_all, -1, topk_indices_rm, expert_src)
        expert_mask_all = ttnn.to_layout(expert_mask_all, ttnn.TILE_LAYOUT)

        if self.score_func == "sigmoid":
            routing_weights = moe_routing_normalize(scores, expert_mask_all, self.route_scale, 1e-20)
        elif self.route_scale != 1.0:
            routing_weights = ttml.ops.binary.mul(scores, self.route_scale)
        else:
            routing_weights = scores

        mask_bs_flat = ttnn.reshape(expert_mask_all, [1, 1, B * S, self.num_experts])
        batch_counts = ttnn.sum(mask_bs_flat, dim=-2, keepdim=True)
        new_counts = ttnn.add(self._token_counts.tensor.get_value(), batch_counts)
        self._token_counts.tensor.set_value(new_counts)

        scores_for_routing = self.gather_topk_weights(routing_weights, topk_indices_u32) if gather_topk else None
        return expert_mask_all, routing_weights, scores_for_routing, topk_indices_u32

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

            # Score each group by its top-2 experts. Sum in fp32 so near-equal
            # group sums don't collapse to the same bf16 value and route to a
            # topk tie-break.
            top2_vals, _top2_idx = ttnn.topk(ttnn.typecast(biased_grouped, ttnn.DataType.BFLOAT16), 2, dim=-1)
            top2_vals_f32 = ttnn.typecast(top2_vals, ttnn.DataType.FLOAT32)
            group_scores = ttnn.sum(top2_vals_f32, dim=-1, keepdim=True)
            group_scores = ttnn.reshape(group_scores, [B, 1, S, self.n_groups])

            # Center on the per-token mean before the bf16 cast topk needs:
            # bf16 ULP near 0 is far smaller than near ~0.86, so the fp32
            # ordering survives the cast.
            group_mean = ttnn.mean(group_scores, dim=-1, keepdim=True)
            group_scores_centered = ttnn.subtract(group_scores, group_mean)

            # Select top n_limited_groups groups
            _gv, top_group_indices = ttnn.topk(
                ttnn.typecast(group_scores_centered, ttnn.DataType.BFLOAT16), self.n_limited_groups, dim=-1
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
        x_in = self._memory_snapshot(x_in, "START")

        x_r = profiler_marker(x_in, "[START] [MoE] Routing")
        scores, _topk_values, topk_indices = self.compute_routing(x_r)

        expert_mask_all, routing_weights, _scores_for_routing, _topk_indices_u32 = self.prepare_routing_weights(
            scores, topk_indices
        )
        scores = profiler_marker(scores, "[END] [MoE] Routing")
        scores = self._memory_snapshot(scores, "AFTER_ROUTING")

        # ── 5. Per-expert weighted outputs ──
        output = None
        x_e = profiler_marker(x_in, "[START] [MoE] Experts")
        x_e = self._memory_snapshot(x_e, "BEFORE_EXPERTS")

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
                routing_weight = autograd_slice(routing_weights, [0, 0, 0, expert_idx], [B, 1, S, expert_idx + 1])
                weighted = ttml.ops.binary.mul(expert_out, routing_weight)
                weighted = ttml.ops.binary.mul(weighted, mask_tt)

            if output is None:
                output = weighted
            else:
                output = ttml.ops.binary.add(output, weighted)

        if output is None:
            output = ttml.autograd.create_tensor(ttml.core.zeros_like(x_in.get_value()))

        output = profiler_marker(output, "[END] [MoE] Experts")
        output = self._memory_snapshot(output, "AFTER_EXPERTS")

        # ── 5. Shared experts ──
        if self.shared_experts is not None:
            x_s = profiler_marker(x_in, "[START] [MoE] SharedExp")
            x_s = self._memory_snapshot(x_s, "BEFORE_SHARED")
            shared_out = self.shared_experts(x_s)
            shared_out = profiler_marker(shared_out, "[END] [MoE] SharedExp")
            shared_out = self._memory_snapshot(shared_out, "AFTER_SHARED")
            output = ttml.ops.binary.add(output, shared_out)

        output = profiler_marker(output, "[END] [MoE]")
        output = self._memory_snapshot(output, "END")
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
