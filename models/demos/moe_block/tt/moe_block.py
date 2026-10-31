# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MoE block (prefill) on ttnn.experimental.moe_block's C++ ops: fused head -> sparse 2D dispatch -> routed
experts + combine -> fused tail, with one L1 arena a call that the head and the combine run in.

Orchestration only: every program, plan and persistent buffer belongs to the ops and the block's MoEBlockResources
(created once a mesh, after the mesh is opened with its fabric config). A forward is four op calls plus the arena's
allocation and free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttnn


@dataclass
class MoEBlockWeights:
    """One layer's weights in the ops' layouts.

    head: ttnn.experimental.moe_block.prepare_fused_head_weights (shared expert + router).
    expert_ids, gate_up, down: the flat routed experts for routed_experts_combine_flat_config's plan
        (flat_routed_expert_weights.FlatRoutedExpert's gidx / w_gu / w_d).
    w_up: a latent model's prepare_fused_tail_weights (None without a latent)."""

    head: object
    expert_ids: ttnn.Tensor
    gate_up: ttnn.Tensor
    down: ttnn.Tensor
    w_up: Optional[ttnn.Tensor] = None


class MoEBlock:
    """One MoE block on ``mesh_device``. ``resources``: the block's MoEBlockResources on it; ``dtypes``: MoEDtypes;
    ``accumulators``: the combine's accumulator count (the head's routing planner is planned for it);
    ``max_tokens_per_expert`` / ``activation``: the flat routed experts' capacity and activation (se3_compute SE_ACT);
    ``layout`` / ``combine``: FlatLayoutOptions / CombineConfig (combine.n_acc is set to ``accumulators``)."""

    def __init__(
        self,
        mesh_device,
        resources,
        *,
        dtypes,
        accumulators: int,
        max_tokens_per_expert: int,
        activation: int,
        layout=None,
        combine=None,
        l1_free: Optional[int] = None,
    ):
        mb = ttnn.experimental.moe_block
        self.mb, self.mesh_device, self.resources, self.dtypes = mb, mesh_device, resources, dtypes
        self.accumulators, self.l1_free = accumulators, l1_free
        self.combine = combine if combine is not None else mb.CombineConfig()
        self.combine.n_acc = accumulators
        self.expert_kw = dict(
            max_tokens_per_expert=max_tokens_per_expert,
            activation=activation,
            layout=layout if layout is not None else mb.FlatLayoutOptions(),
            combine=self.combine,
        )
        # (plans the head and the combine once; every forward allocates this spec at the same L1 address)
        self.arena_spec = mb.moe_block_arena_spec(
            resources, dtypes=dtypes, accumulators=accumulators, l1_free=l1_free, **self.expert_kw
        )
        self.in_program_dispatch = resources.config.experts_during_dispatch

    def __call__(self, x: ttnn.Tensor, weights: MoEBlockWeights) -> ttnn.Tensor:
        """x: this chip's [1, 1, S/SP, H/TP] bfloat16 TILE, DRAM. Returns the residual's layout (fused_tail's)."""
        mb = self.mb
        arena = ttnn.allocate_tensor_on_device(self.arena_spec, self.mesh_device)
        head = mb.fused_head(
            x,
            weights.head,
            resources=self.resources,
            dtypes=self.dtypes,
            accumulators=self.accumulators,
            l1_free=self.l1_free,
            arena=arena,
        )
        if self.in_program_dispatch:  # (the dispatch is the combine's first phase: it fills xbuf from route)
            dispatched, route = head.xbuf, head.route
        else:
            dispatched, route = mb.sparse_dispatch_2d(head.xbuf, head.route, self.resources), None
        y_own = mb.routed_experts_combine(
            dispatched,
            head.counts,
            head.regions,
            head.token_index,
            head.table,
            head.plan,
            weights.expert_ids,
            weights.gate_up,
            weights.down,
            self.resources,
            route=route,
            arena=arena,
            **self.expert_kw,
        )
        ttnn.deallocate(arena)
        return mb.fused_tail(y_own, head.se, resources=self.resources, w_up=weights.w_up)
