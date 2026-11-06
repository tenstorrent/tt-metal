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
    arena (the largest free L1 block less the static CBs below it; FlatLayoutOptions.arena_bytes).

    Token loops (MoEBlockConfig.token_loops): a chunk whose head does not fit at once runs the head and the tail k times
    (each chip's rows i chunk / (SP k) .. a call) and the dispatch, the routed experts and the combine once on the whole
    chunk - each expert's weights read once. 0 (the default): the fewest loops whose plan fits, a resident head
    preferred to the head's DRAM fallback - 1 wherever the chunk fits as it is. The resources are set for it here
    (set_token_loops), so prepare the weights after the block is made (resources / accumulators / expert_kw)."""

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
        tail_dtype=None,
    ):
        mb = ttnn.experimental.moe_block
        # (the ops' persistent words and semaphores live across calls: in L1_SMALL, where no other program's circular
        # buffers go - in plain L1 the model's programs between two MoE layers overwrote them. Open the mesh with an
        # l1_small_size that holds them: l1_small_bytes after the first call.)
        assert resources.config.persist_l1_small, "MoEBlock: create its resources with persist_l1_small=True"
        self.mb, self.mesh_device, self.resources, self.dtypes = mb, mesh_device, resources, dtypes
        self.l1_free = l1_free
        self.tail_dtype = ttnn.bfloat8_b if tail_dtype is None else tail_dtype  # (a latent model's W_up tiles)
        if layout is None:
            layout = mb.FlatLayoutOptions()
        self.combine = combine if combine is not None else mb.CombineConfig()
        n_acc = self.combine.n_acc
        ask = (accumulators, max_tokens_per_expert, activation, layout)
        # (padding rows: a chunk whose rows a mesh row are not whole 32 x TP rows - the head's rows rule - runs with
        # each SP row padded up to them; the padding rows are padding tokens of a padded chunk, their output dropped)
        topo = resources.topology
        rows, unit = resources.config.chunk_tokens // topo["R"], 32 * topo["C"]
        self.pad_rows = -rows % unit
        self.pad = self.pad_rows * topo["R"]
        self.full_words = None  # (a chunk with padding rows and no span of the caller's: (0, chunk) device words)
        if self.pad or resources.config.pad_tokens:
            cfg = resources.config
            resources.set_token_loops(cfg.token_loops, cfg.chunk_splits, self.pad)
        cfg = resources.config
        if cfg.token_loops != 0 and cfg.chunk_splits != 0:  # (the caller's loops and splits)
            self._plan(*ask, n_acc)
        else:
            self._plan_loops(ask, n_acc, fixed_k=cfg.token_loops, fixed_b=cfg.chunk_splits)
        self.loops = max(1, int(resources.config.token_loops))
        self.splits = max(1, int(resources.config.chunk_splits))
        logger.info(
            f"MoEBlock: accumulators {self.accumulators}, dispatch group {self.plan.l1.dispatch_group} ({self.plan.note})"
            + (
                f", {self.splits} splits x {self.loops} token loops of {resources.config.loop_tokens} tokens"
                if self.loops * self.splits > 1
                else ""
            )
        )
        self.exchange_spec = (
            None  # (the dispatch + combine's arena when the head's does not fit: the first call sets it)
        )
        self.one_arena = None
        self.chunk_words = None  # (a padded chunk's scalars given as ints: two persistent device words, rewritten)

    def _plan(self, accumulators, max_tokens_per_expert, activation, layout, n_acc):
        """The block's plans for the resources' token loops (raises where one does not fit)."""
        mb, resources, dtypes, l1_free = self.mb, self.resources, self.dtypes, self.l1_free
        # (an expert sees each token of a split at most once - the experts run once a split; the flat op is built for
        # >= 256)
        cap = max(resources.config.split_tokens, 256)
        self.combine.n_acc = n_acc
        self.expert_kw = dict(
            max_tokens_per_expert=(
                cap
                if max_tokens_per_expert is None
                else min(max_tokens_per_expert, cap)
                if resources.config.chunk_splits > 1
                else max_tokens_per_expert
            ),
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
        # (plans the head and the combine once; every forward allocates this spec at the same L1 address)
        self.arena_spec = mb.moe_block_arena_spec(
            resources, dtypes=dtypes, accumulators=self.accumulators, l1_free=l1_free, **self.expert_kw
        )
        # (the head planner's DRAM fallback - a shape whose resident layouts do not fit the head's L1: x gathered over
        # TP into DRAM ahead of the head, which reloads it a row chunk at a time; the plan the op runs, same l1_free)
        fit = mb.plan_fused_head(resources, dtypes, self.accumulators, l1_free)
        self.x_dram = bool(fit.ok and len(fit.planned) and fit.planned[0].lay.xdram)
        # (the tail of a call: its L1 grows with the loop's rows - host only, uncached)
        mb.plan_fused_tail(resources, dtype=self.tail_dtype)

    def _plan_loops(self, ask, n_acc, fixed_k=0, fixed_b=0):
        """The (splits, loops) to run (resources' token_loops / chunk_splits 0): one call wherever the chunk plans as it
        is (the head's DRAM fallback included - nothing that fits changes); else the fewest splits (each re-reads every
        expert's weights), and of their loops whose plans fit the one with the least predicted head time (k x the head
        planner's predicted call) - a resident head before a DRAM fallback."""
        resources = self.resources
        chunk, sp = resources.config.plan_tokens, resources.topology["R"]
        try:  # (one call, or the caller's fixed loops / splits with the other at 1)
            resources.set_token_loops(fixed_k, fixed_b, self.pad)
            self._plan(*ask, n_acc)
            return
        except RuntimeError as ex:
            first_error = ex
            logger.info(f"MoEBlock: chunk {chunk} in one call does not fit: {_short(ex)}")
        # (only the dimensions the caller left at 0 are searched)
        bs = [fixed_b] if fixed_b else [b for b in range(1, 65) if chunk % (b * sp * 32) == 0]
        for b in bs:
            # (each chip's loop rows a whole number of tiles: x is sliced by rows)
            ks = [fixed_k] if fixed_k else [k for k in range(1, 65) if chunk % (b * k * sp * 32) == 0]
            fits = []  # (DRAM fallback, predicted head us of the split, k)
            for k in ks:
                if (k, b) == (fixed_k or 1, fixed_b or 1):
                    continue  # (the one tried above)
                resources.set_token_loops(k, b, self.pad)
                try:
                    self._plan(*ask, n_acc)
                except RuntimeError as ex:
                    logger.info(f"MoEBlock: chunk {chunk} as {b} splits x {k} token loops does not fit: {_short(ex)}")
                    continue
                fit = self.mb.plan_fused_head(resources, self.dtypes, self.accumulators, self.l1_free)
                t = k * float(fit.planned[0].t) if fit.ok and len(fit.planned) else float("inf")
                fits.append((self.x_dram, t, k))
                logger.info(f"MoEBlock: chunk {chunk} as {b} splits x {k} token loops: head {t:.0f} us predicted")
            if fits:
                _, _, k = min(fits)
                resources.set_token_loops(k, b, self.pad)
                self._plan(*ask, n_acc)
                return
        raise first_error

    def _one_arena_fits(self) -> bool:
        """One arena of the head's size a forward when it sits above the block's own programs' static circular buffers
        (allocated top-down in the largest free L1 block, it reaches from the block's top down by its size), else two:
        the head's, then the exchange's (planned above them). Every CB of the ops that run in the arena (head, dispatch,
        flat experts, combine) is in it, so nothing sits below it (static 0) and the head's arena, planned on the free
        L1, always fits: one arena. The check stays for a caller whose flat layout brings static CBs back."""
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

    def _words(self, a: int, b: int) -> tuple:
        """a pair of one-word device tensors (replicated, uint32, DRAM) holding a and b"""
        word = lambda v: torch.tensor([[[[v]]]], dtype=torch.int64)  # noqa: E731
        return tuple(
            ttnn.from_torch(
                word(v),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for v in (a, b)
        )

    def chunk_scalars(self, actual_start: int, actual_end: int) -> tuple:
        """The block's persistent pair of one-word device tensors (replicated, uint32, DRAM - main's metadata scalars'
        form) holding a padded chunk's actual_start / actual_end, rewritten with these values (host writes: not inside a
        trace capture - a traced caller copies its own metadata scalars, or passes them as they are)."""
        chunk = self.resources.config.chunk_tokens
        assert (
            actual_start <= actual_end <= actual_start + chunk
        ), f"MoEBlock: chunk [{actual_start}, {actual_end}) holds more than {chunk} tokens or is reversed"
        if self.chunk_words is None:
            self.chunk_words = self._words(actual_start, actual_end)
        else:
            word = lambda v: torch.tensor([[[[v]]]], dtype=torch.int64)  # noqa: E731
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
        if metadata is not None:
            assert actual_start is None and actual_end is None, "MoEBlock: metadata or actual_start / actual_end"
            actual_start, actual_end = metadata[1], metadata[2]
        assert (actual_start is None) == (actual_end is None), "MoEBlock: a padded chunk takes actual_start and _end"
        if isinstance(actual_start, int) or isinstance(actual_end, int):
            assert isinstance(actual_start, int) and isinstance(actual_end, int), "MoEBlock: two ints or two tensors"
            actual_start, actual_end = self.chunk_scalars(actual_start, actual_end)
        if self.exchange_spec is None:  # (the first call, before any arena: one arena or two)
            self.one_arena = self._one_arena_fits()
        if self.pad_rows:  # (padding rows: each chip's rows padded up to whole 32 x TP rows; their output dropped)
            if actual_start is None:
                if self.full_words is None:  # (made at the first call: eager - a trace captures them as they are)
                    self.full_words = self._words(0, self.resources.config.chunk_tokens)
                actual_start, actual_end = self.full_words
            rows = int(x.shape[-2])
            xp = ttnn.pad(x, [(0, 0), (0, 0), (0, self.pad_rows), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            yp = self._forward_splits(xp, weights, actual_start, actual_end)
            ttnn.deallocate(xp)
            y = ttnn.slice(yp, [0, 0, 0, 0], [1, 1, rows, int(yp.shape[-1])], [1, 1, 1, 1],
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            ttnn.deallocate(yp)
            return y
        return self._forward_splits(x, weights, actual_start, actual_end)

    def _forward_splits(self, x, weights, actual_start, actual_end) -> ttnn.Tensor:
        """The chunk as the block's splits (one without)."""
        if self.splits == 1:
            return self._forward(x, weights, actual_start, actual_end)
        # (splits: the whole block a piece of each chip's rows - every piece reads the experts' weights again)
        rows, width = int(x.shape[-2]), int(x.shape[-1])
        r = rows // self.splits
        outs = []
        for j in range(self.splits):
            xj = ttnn.slice(x, [0, 0, j * r, 0], [1, 1, (j + 1) * r, width], [1, 1, 1, 1],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            outs.append(self._forward(xj, weights, actual_start, actual_end, split=j))
            ttnn.deallocate(xj)
        y = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return y

    def _forward(self, x, weights, actual_start, actual_end, split: int = 0) -> ttnn.Tensor:
        """One split (the whole chunk without splits): the head and the tail k times (token loops: each chip's rows
        i r .. (i + 1) r a call), the dispatch, the routed experts and the combine once - the head's last call planned
        every loop's tokens."""
        mb, k = self.mb, self.loops
        rows, width = int(x.shape[-2]), int(x.shape[-1])
        r = rows // k
        xs = [
            x
            if k == 1
            else ttnn.slice(
                x, [0, 0, i * r, 0], [1, 1, (i + 1) * r, width], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )  # fmt: skip
            for i in range(k)
        ]
        # (the DRAM fallback's x_full [1, 1, rows, H]: each loop's x of this mesh row gathered over the TP axis - before
        # the arena: the gather's CBs need the L1)
        x_fulls = [
            ttnn.all_gather(xi, dim=3, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG) if self.x_dram else None
            for xi in xs
        ]
        arena = ttnn.allocate_tensor_on_device(self.arena_spec, self.mesh_device)
        self.arena_address = int(arena.buffer_address())
        ses = []
        for i in range(k):
            head = mb.fused_head(
                xs[i],
                weights.head,
                resources=self.resources,
                dtypes=self.dtypes,
                accumulators=self.accumulators,
                x_full=x_fulls[i],
                l1_free=self.l1_free,
                arena=arena,
                actual_start=actual_start,
                actual_end=actual_end,
                **({"loop": split * k + i} if k * self.splits > 1 else {}),
            )
            if x_fulls[i] is not None:
                ttnn.deallocate(x_fulls[i])
            if xs[i] is not x:
                ttnn.deallocate(xs[i])
            ses.append(head.se)  # (each loop's shared expert output: its own tensor)
        self.last_head = head  # (its persistent outputs: a model's checks read the routing rows - every loop's)
        if self.exchange_spec is None:  # (once: the exchange's own arena - the combine's, without the head's bytes)
            self.exchange_spec = (
                False
                if self.one_arena
                else mb.routed_experts_combine_arena_spec(
                    self.resources, rows_cap=int(head.table.shape[-2]) * 32, **self.expert_kw
                )
            )
        if self.exchange_spec:
            # (two arenas a forward only for a head arena larger than the room above static CBs - none since every
            # CB is in the arena; the exchange's arena is then planned above them)
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
        if k == 1:
            return mb.fused_tail(y_own, head.se, resources=self.resources, w_up=weights.w_up)
        # (the tail a loop: the owner rows i S .. (i + 1) S of the combine's output - loop i's tokens)
        S, hr = int(y_own.shape[-2]) // k, int(y_own.shape[-1])
        outs = []
        for i in range(k):
            yi = ttnn.slice(y_own, [0, 0, i * S, 0], [1, 1, (i + 1) * S, hr], [1, 1, 1, 1],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            outs.append(mb.fused_tail(yi, ses[i], resources=self.resources, w_up=weights.w_up))
            ttnn.deallocate(yi)
        y = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return y


def _short(ex) -> str:
    """a TT_FATAL's message without its backtrace"""
    msg = " ".join(str(ex).split())
    return msg.split(" backtrace:")[0][-300:]


def l1_small_bytes(mesh_device) -> int:
    """L1_SMALL allocated a bank on mesh_device (after a block's first call: its persistent words and semaphores)."""
    return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank)
