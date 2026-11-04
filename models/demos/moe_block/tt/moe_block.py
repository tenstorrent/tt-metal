# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MoE block (prefill) on ttnn.experimental.moe_block's C++ ops: fused head -> sparse 2D dispatch -> routed
experts + combine -> fused tail, with one L1 arena a call that the head and the combine run in.

Orchestration only: every program, plan and persistent buffer belongs to the ops and the block's MoEBlockResources
(created once a mesh, after the mesh is opened with its fabric config). A forward is four op calls plus the arena's
allocation and free."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from loguru import logger

import ttnn


@dataclass
class MoEBlockWeights:
    """One layer's weights in the ops' layouts.

    head: ttnn.experimental.moe_block.prepare_fused_head_weights (shared expert + router).
    expert_ids, gate_up, down: the flat routed experts laid out for the combine's flat plan
        (flat_routed_expert_weights.FlatRoutedExpert's gidx / w_gu / w_d).
    w_up: a latent model's prepare_fused_tail_weights (None without a latent)."""

    head: object
    expert_ids: ttnn.Tensor
    gate_up: ttnn.Tensor
    down: ttnn.Tensor
    w_up: Optional[ttnn.Tensor] = None


class MoEBlock:
    """One MoE block on ``mesh_device``. ``resources``: the block's MoEBlockResources on it; ``dtypes``: MoEDtypes;
    ``accumulators``: the combine's accumulator count (the head's routing planner is planned for it); None: what the
    combine planner gives (moe_block_plan, which also picks the dispatch's row group);
    ``max_tokens_per_expert`` / ``activation``: the flat routed experts' capacity and activation (gate_up_compute
    SE_ACT);
    ``layout`` / ``combine``: FlatLayoutOptions / CombineConfig (combine.n_acc is set to ``accumulators``).
    Capacity and layout default to max(chunk, 256) and FlatLayoutOptions() - the combine fits the flat plan into its
    arena (the largest free L1 block less the static CBs below it; FlatLayoutOptions.arena_bytes)."""

    def __init__(
        self,
        mesh_device,
        resources,
        *,
        dtypes,
        activation: int,
        accumulators: Optional[int] = None,
        max_tokens_per_expert: Optional[int] = None,
        layout=None,
        combine=None,
        l1_free: Optional[int] = None,
    ):
        mb = ttnn.experimental.moe_block
        # (the ops' persistent words and semaphores live across calls: in L1_SMALL, where no other program's circular
        # buffers go - in plain L1 the model's programs between two MoE layers overwrote them. Open the mesh with an
        # l1_small_size that holds them: l1_small_bytes after the first call.)
        assert resources.config.persist_l1_small, "MoEBlock: create its resources with persist_l1_small=True"
        self.mb, self.mesh_device, self.resources, self.dtypes = mb, mesh_device, resources, dtypes
        self.l1_free = l1_free
        chunk = resources.config.chunk_tokens
        if max_tokens_per_expert is None:  # (an expert sees each token at most once; the flat op is built for >= 256)
            max_tokens_per_expert = max(chunk, 256)
        if layout is None:
            layout = mb.FlatLayoutOptions()
        self.combine = combine if combine is not None else mb.CombineConfig()
        self.expert_kw = dict(
            max_tokens_per_expert=max_tokens_per_expert,
            activation=activation,
            layout=layout,
            combine=self.combine,
        )
        # (the block's plan, once per resources: the combine's accumulators and the dispatch's row group, which the
        # ops take by default from then on)
        self.plan = mb.moe_block_plan(
            resources, dtypes=dtypes, accumulators=accumulators, l1_free=l1_free, **self.expert_kw
        )
        self.accumulators = self.plan.accumulators
        self.combine.n_acc = self.accumulators
        logger.info(
            f"MoEBlock: accumulators {self.accumulators}, dispatch group {self.plan.l1.dispatch_group} ({self.plan.note})"
        )
        # (plans the head and the combine once; every forward allocates this spec at the same L1 address)
        self.arena_spec = mb.moe_block_arena_spec(
            resources, dtypes=dtypes, accumulators=self.accumulators, l1_free=l1_free, **self.expert_kw
        )
        self.exchange_spec = (
            None  # (the dispatch + combine's arena when the head's does not fit: the first call sets it)
        )
        self.one_arena = None
        self.chunk_words = None  # (a padded chunk's scalars given as ints: two persistent device words, rewritten)
        # (the head planner's DRAM fallback - a shape whose resident layouts do not fit the head's L1: x gathered over
        # TP into DRAM ahead of the head, which reloads it a row chunk at a time; the plan the op runs, same l1_free)
        fit = mb.plan_fused_head(resources, dtypes, self.accumulators, l1_free)
        self.x_dram = bool(fit.ok and len(fit.planned) and fit.planned[0].lay.xdram)

    def _one_arena_fits(self) -> bool:
        """One arena of the head's size a forward when it sits above the block's own programs' static circular buffers
        (allocated top-down in the largest free L1 block, it reaches from the block's top down by its size), else two:
        the head's, then the exchange's (planned above them). The head's and the dispatch's CBs are in the arena; the
        flat expert's static CBs are what is left below it."""
        free = int(ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank)
        kw = self.expert_kw
        static = self.mb.routed_experts_combine_static_cb_bytes(
            self.resources,
            max_tokens_per_expert=kw["max_tokens_per_expert"],
            activation=kw["activation"],
            layout=kw["layout"],
        )
        head = int(self.arena_spec.memory_config.shard_spec.shape[0]) * 64
        bound = free - static - 1024  # (1 KB for the allocator's alignment)
        fits = head <= bound
        logger.info(
            f"MoEBlock: arena {head} B a core, room above the block's static CBs {bound} B (largest free L1 block "
            f"{free} B, static CBs {static} B): {'one arena' if fits else 'two arenas (the exchange planned above them)'}"
        )
        return fits

    def chunk_scalars(self, actual_start: int, actual_end: int) -> tuple:
        """The block's persistent pair of one-word device tensors (replicated, uint32, DRAM - main's metadata scalars'
        form) holding a padded chunk's actual_start / actual_end, rewritten with these values (host writes: not inside a
        trace capture - a traced caller copies its own metadata scalars, or passes them as they are)."""
        chunk = self.resources.config.chunk_tokens
        assert (
            actual_start <= actual_end <= actual_start + chunk
        ), f"MoEBlock: chunk [{actual_start}, {actual_end}) holds more than {chunk} tokens or is reversed"
        word = lambda v: torch.tensor([[[[v]]]], dtype=torch.int64)  # noqa: E731
        if self.chunk_words is None:
            self.chunk_words = tuple(
                ttnn.from_torch(
                    word(v),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.mesh_device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                for v in (actual_start, actual_end)
            )
        else:
            for v, t in zip((actual_start, actual_end), self.chunk_words):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(word(v), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT), t
                )
        return self.chunk_words

    def __call__(
        self,
        x: ttnn.Tensor,
        weights: MoEBlockWeights,
        actual_start: Union[int, ttnn.Tensor, None] = None,
        actual_end: Union[int, ttnn.Tensor, None] = None,
        metadata: Optional[tuple] = None,
    ) -> ttnn.Tensor:
        """x: this chip's [1, 1, S/SP, H/TP] bfloat16 TILE, DRAM. Returns the residual's layout (fused_tail's).

        A padded chunk: actual_start / actual_end, main's chunked-prefill metadata scalars (the chunk's first real
        token's absolute position, one past its last) - one-element uint32 device tensors in DRAM, replicated, read on
        device every call (a trace replays with their values of the moment), or ints (host-written words, eager only);
        or metadata = main's (slot_id, actual_start, actual_end) tensors as they are. The chunk is in main's KV-pad-aware
        rotated layout over the SP rows (moe_padding_config; chunk-aligned start: the natural order, the first
        end - start tokens real). Padding tokens are not routed, dispatched, computed by the routed experts or combined;
        their rows of the output are the shared expert's only (the dense parts still compute them; discard them).
        None: every row is real."""
        mb = self.mb
        if metadata is not None:
            assert actual_start is None and actual_end is None, "MoEBlock: metadata or actual_start / actual_end"
            actual_start, actual_end = metadata[1], metadata[2]
        assert (actual_start is None) == (actual_end is None), "MoEBlock: a padded chunk takes actual_start and _end"
        if isinstance(actual_start, int) or isinstance(actual_end, int):
            assert isinstance(actual_start, int) and isinstance(actual_end, int), "MoEBlock: two ints or two tensors"
            actual_start, actual_end = self.chunk_scalars(actual_start, actual_end)
        if self.exchange_spec is None:  # (the first call, before any arena: one arena or two)
            self.one_arena = self._one_arena_fits()
        # (the DRAM fallback's x_full [1, 1, rows, H]: this mesh row's x gathered over the TP axis)
        x_full = (
            ttnn.all_gather(x, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if self.x_dram else None
        )
        arena = ttnn.allocate_tensor_on_device(self.arena_spec, self.mesh_device)
        self.arena_address = int(arena.buffer_address())
        head = mb.fused_head(
            x,
            weights.head,
            resources=self.resources,
            dtypes=self.dtypes,
            accumulators=self.accumulators,
            x_full=x_full,
            l1_free=self.l1_free,
            arena=arena,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        if x_full is not None:
            ttnn.deallocate(x_full)
        self.last_head = head  # (its persistent outputs: a model's checks read the routing rows)
        if self.exchange_spec is None:  # (once: the exchange's own arena - the combine's, without the head's bytes)
            self.exchange_spec = (
                False
                if self.one_arena
                else mb.routed_experts_combine_arena_spec(
                    self.resources, rows_cap=int(head.table.shape[-2]) * 32, **self.expert_kw
                )
            )
        if self.exchange_spec:
            # (two arenas a forward for a head larger than the exchange's arena - K3: the head's
            # arena reaches below the flat expert's static circular buffers; the exchange's is planned above them)
            ttnn.deallocate(arena)
            arena = ttnn.allocate_tensor_on_device(self.exchange_spec, self.mesh_device)
        # (the dispatch's CBs in the call's arena - between the head's use of it and the combine's)
        dispatched = mb.sparse_dispatch_2d(head.xbuf, head.route, self.resources, arena=arena)
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
            arena=arena,
            **self.expert_kw,
        )
        ttnn.deallocate(arena)
        return mb.fused_tail(y_own, head.se, resources=self.resources, w_up=weights.w_up)


def l1_small_bytes(mesh_device) -> int:
    """L1_SMALL allocated a bank on mesh_device (after a block's first call: its persistent words and semaphores)."""
    return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank)
