# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V3 Mixture of Experts — TP-sharded variant of SparseMoE.

Mirrors `SparseMoE`'s route → group → moe_ffn_swiglu_fw → ungroup pipeline,
but every chip in the TP cluster holds all routed experts with each
expert's intermediate dim (I) sharded across the cluster axis. Per chip:

    1. all_gather tokens across cluster_axis     → full [B, 1, S, H]
    2. moe_group                                 → grouped [T_cap, H]
    3. moe_ffn_swiglu_fw with TP-sharded weights:
         w_gate, w_up: [H, I/D]  (column-parallel, output dim sharded)
         w_down:       [I/D, H]  (row-parallel,    input dim sharded)
       → partial output [T_cap, H] per chip
    4. moe_ungroup                               → partial [B, 1, S, H]
    5. all_reduce across cluster_axis            → full [B, 1, S, H]

See moe_forward_roofline.md ("Alternative approach: TP-sharded experts")
for the AI/ridge analysis vs the EP path.
"""

from __future__ import annotations

import torch

import ttnn
import ttml
from ttml.modules import Parameter

from .moe import MoE
from .moe_sparse import _to_layout
from .autograd_ops import (
    moe_routing_normalize,
)


def _first_replica_numpy(tensor: ttml.autograd.Tensor):
    """Return the first mesh replica as a host float32 array."""
    mesh = ttml.maybe_mesh()
    if mesh is None or mesh.num_devices() == 1:
        return tensor.to_numpy(ttnn.DataType.FLOAT32)

    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(
        ttml.autograd.AutoContext.get_instance().get_device(), 0
    )
    arr = tensor.to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    shape = tuple(tensor.get_value().shape)
    return arr.reshape((mesh.num_devices(),) + shape)[0]


def _make_sharded_param_from_dense_weight(dense_weight: ttml.autograd.Tensor, *, axis_name: str, tdim: int):
    """Create a leaf sharded parameter from a LinearLayer weight.

    LinearLayer stores [1, 1, out, in]. moe_ffn_swiglu_fw consumes
    [1, 1, in, out], so transpose on host once, then shard with the mapper.
    """
    weight_np = _first_replica_numpy(dense_weight).transpose(0, 1, 3, 2)
    mapper = ttml.mesh().axis_mapper(axis_name, tdim=tdim)
    return Parameter(ttml.autograd.Tensor.from_numpy(weight_np, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper))


class SparseMoETP(MoE):
    """TP-sharded sparse MoE.

    All experts live on every chip; each expert's intermediate dim is
    sharded across the configured `tp_axis_name` axis. Forward gathers
    the full token set, runs the local 1/D-sharded FFN on every expert,
    and all-reduces the partial outputs across the TP group.
    """

    def __init__(self, config) -> None:
        # Build the dense MoE (gate, dense Experts, shared experts, buffers).
        # We immediately copy the dense expert weights into persistent sharded
        # parameters so forward does not scatter weights every step.
        super().__init__(config)

        # Mesh axis to shard experts on. Falls back to "tp" so the class
        # is still usable when constructed directly without a config that
        # carries `moe_tp_axis_name`.
        self.axis_name = getattr(config, "moe_tp_axis_name", None) or "tp"
        self.cluster_axis = ttml.mesh().axis_index(self.axis_name)
        D = ttml.mesh().axis_size(self.axis_name)

        I = config.moe_inter_dim

        # Each expert's intermediate dim (I) is sharded I → I/D across the
        # TP axis. Divisibility is required (otherwise the shard is
        # uneven); sub-tile shards (I/D < 32 or not a multiple of 32)
        # still work but waste compute proportional to 32 / (I/D).
        if I % D != 0:
            raise ValueError(
                f"SparseMoETP: moe_inter_dim={I} must be divisible by " f"the '{self.axis_name}' axis size ({D})"
            )

        self.w_gate = []
        self.w_up = []
        self.w_down = []
        for e in range(config.n_routed_experts):
            # w1/w3 become [H, I] and shard intermediate/output dim I.
            gate = _make_sharded_param_from_dense_weight(
                self.experts[e].w1.weight.tensor, axis_name=self.axis_name, tdim=3
            )
            up = _make_sharded_param_from_dense_weight(
                self.experts[e].w3.weight.tensor, axis_name=self.axis_name, tdim=3
            )
            # w2 becomes [I, H] and shards intermediate/input dim I.
            down = _make_sharded_param_from_dense_weight(
                self.experts[e].w2.weight.tensor, axis_name=self.axis_name, tdim=2
            )

            # Register each Parameter by name; storing only in a Python list
            # would not be picked up by AbstractModuleBase.__setattr__.
            setattr(self, f"w_gate_{e}", gate)
            setattr(self, f"w_up_{e}", up)
            setattr(self, f"w_down_{e}", down)
            self.w_gate.append(gate)
            self.w_up.append(up)
            self.w_down.append(down)

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        device = ttml.autograd.AutoContext.get_instance().get_device()
        K = self.n_activated
        E = self.num_experts

        # Input is replicated across the TP axis (this is the convention
        # in this codebase — preceding TP modules end with all_reduce).
        # No gather needed.
        B, _, S, _dim = list(x.get_value().shape)

        # ── 2. routing (same as SparseMoE / dense MoE) ──
        scores, _topk_values, topk_indices = self.compute_routing(x)
        topk_indices_u32 = ttnn.typecast(topk_indices, ttnn.DataType.UINT32)
        mask_parts = []
        expert_count_scalars = []
        for expert_idx in range(E):
            match = ttnn.eq(topk_indices_u32, float(expert_idx))
            match_f = ttnn.typecast(match, ttnn.DataType.BFLOAT16)
            match_any = ttnn.sum(match_f, dim=-1, keepdim=True)
            mask_narrow_bf = ttnn.typecast(ttnn.gt(match_any, 0.0), ttnn.DataType.BFLOAT16)
            mask_parts.append(mask_narrow_bf)
            expert_count_scalars.append(ttnn.reshape(ttnn.sum(mask_narrow_bf), [1, 1, 1, 1]))

        if self.score_func == "sigmoid":
            full_mask = ttnn.concat(mask_parts, dim=-1)
            routing_weights = moe_routing_normalize(scores, full_mask, self.route_scale, 1e-20)
        else:
            if self.route_scale != 1.0:
                routing_weights = ttml.ops.binary.mul(scores, self.route_scale)
            else:
                routing_weights = scores

        scores_for_routing = _gather_topk_via_existing(routing_weights, topk_indices_u32, E)

        # Token-count accumulator (same as dense MoE).
        batch_counts = ttnn.concat(expert_count_scalars, dim=-1)
        new_counts = ttnn.add(self._token_counts.tensor.get_value(), batch_counts)
        self._token_counts.tensor.set_value(new_counts)

        # ── 3. moe_group ──
        metadata = ttnn.to_layout(ttnn.typecast(topk_indices, ttnn.DataType.UINT16), ttnn.ROW_MAJOR_LAYOUT)
        leids = ttnn.from_torch(
            torch.arange(E, dtype=torch.int32),
            dtype=ttnn.uint16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        x_rm = _to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        scores_for_routing_rm = _to_layout(scores_for_routing, ttnn.ROW_MAJOR_LAYOUT)
        group_out = ttml.ops.moe.moe_group_op(x_rm, metadata, scores_for_routing_rm, leids, int(E), int(K))
        grouped = group_out.grouped

        # ── 4. moe_ffn_swiglu_fw with TP-sharded weights ──
        # Each chip holds 1/D of every expert's intermediate axis. Local
        # matmul produces a partial [T_cap, H] that is summed across the
        # TP group by the all_reduce below.
        # broadcast(grouped) is identity in forward (grouped is already
        # replicated) but autograd-aware: its backward all_reduce-sums
        # the per-chip partial d(grouped) coming out of moe_ffn into the
        # full gradient. Without this the partials are silently dropped
        # before they reach moe_group's backward (i.e. the routing path).
        grouped = ttml.ops.distributed.broadcast(grouped, self.cluster_axis)
        w_gate = [self.w_gate[e].tensor for e in range(E)]
        w_up = [self.w_up[e].tensor for e in range(E)]
        w_down = [self.w_down[e].tensor for e in range(E)]
        y_grouped = ttml.ops.moe.moe_ffn_swiglu_fw(grouped, group_out.offsets, w_gate, w_up, w_down)

        # ── 5. moe_ungroup ──
        output_rm = ttml.ops.moe.moe_ungroup_op(
            y_grouped,
            group_out.grouped_scores,
            metadata,
            leids,
            group_out.plan,
            group_out.offsets,
            int(E),
            int(K),
            int(B),  # D in the wrapper's signature
            1,
            int(S),
        )
        output = _to_layout(output_rm, ttnn.TILE_LAYOUT)

        # ── 6. all_reduce across TP axis to combine partial sums ──
        # noop_backward=True: each chip's partial contributes equally to
        # the replicated output, so dL/d(partial_chip) = dL/d(output) flows
        # back identically to every chip — no second collective in backward
        # (matches RowParallelLinear's input_is_parallel=True case).
        output = ttml.ops.distributed.all_reduce(output, noop_backward=True, cluster_axis=self.cluster_axis)

        # Shared experts: their LinearLayer modules are NOT TP-sharded by
        # default; they replicate work and don't need the all_reduce. If a
        # downstream change makes them TP, swap to ColumnParallel/RowParallel.
        if self.shared_experts is not None:
            output = ttml.ops.binary.add(output, self.shared_experts(x))

        return output


def _gather_topk_via_existing(routing_weights, topk_indices_u32, num_experts):
    """Per-row gather without adding routing-module ops.

    Same construction as SparseMoE's gather: per-K, per-expert one_hot
    masked weighted sum. Defined here to keep moe_sparse_tp.py
    self-contained (the SparseMoE version is private to that module).
    """
    from .autograd_ops import autograd_concat, autograd_slice

    device = ttml.autograd.AutoContext.get_instance().get_device()
    K = list(topk_indices_u32.shape)[-1]
    rw_shape = list(routing_weights.get_value().shape)
    base_idx_shape = list(topk_indices_u32.shape)

    rw_per_expert = []
    for e in range(num_experts):
        end = list(rw_shape)
        end[-1] = e + 1
        rw_per_expert.append(autograd_slice(routing_weights, [0, 0, 0, e], end))

    arange_E_u32 = ttnn.arange(0, num_experts, 1, dtype=ttnn.uint32, device=device)
    arange_E_u32 = ttnn.reshape(arange_E_u32, [1, 1, 1, num_experts])

    out_per_k = []
    for k in range(K):
        start = [0] * len(base_idx_shape)
        end = list(base_idx_shape)
        start[-1] = k
        end[-1] = k + 1
        idx_k = ttnn.slice(topk_indices_u32, start, end)
        accumulator = None
        for e in range(num_experts):
            arange_e = ttnn.slice(arange_E_u32, [0, 0, 0, e], [1, 1, 1, e + 1])
            mask = ttnn.eq(idx_k, arange_e)
            mask = ttnn.typecast(mask, ttnn.float32)
            mask = ttnn.typecast(mask, ttnn.bfloat16)
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
            mask_at = ttml.autograd.Tensor(mask, False)
            contribution = ttml.ops.binary.mul(rw_per_expert[e], mask_at)
            accumulator = contribution if accumulator is None else ttml.ops.binary.add(accumulator, contribution)
        out_per_k.append(accumulator)
    return autograd_concat(out_per_k, dim=-1)
