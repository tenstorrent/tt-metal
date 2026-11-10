# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MoE block (prefill) on ttnn.experimental.moe_block's C++ ops: MoE head -> sparse 2D dispatch -> routed
experts + combine -> MoE tail, with one L1 arena a call that the head and the combine run in.

Orchestration only: every program, plan and persistent buffer belongs to the ops and the block's MoEBlockResources
(created once a mesh, after the mesh is opened with its fabric config). A forward is four op calls plus the arena's
allocation and free."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Union

import torch
from loguru import logger

import ttnn


@contextmanager
def _zone(name: str):
    """A Tracy host zone named name (seen when the run is under python -m tracy; near free otherwise)."""
    ttnn.start_tracy_zone("moe_block", name, 0)
    try:
        yield
    finally:
        ttnn.stop_tracy_zone(name)


@dataclass
class MoEBlockWeights:
    """One layer's weights in the ops' layouts.

    head: ttnn.experimental.moe_block.prepare_moe_head_weights (shared expert + router).
    expert_ids, gate_up, down: the flat routed experts laid out for the combine's flat plan
        (flat_routed_expert_weights.FlatRoutedExpert's gidx / w_gu / w_d).
    w_up: a latent model's prepare_moe_tail_weights (None without a latent)."""

    head: object
    expert_ids: ttnn.Tensor
    gate_up: ttnn.Tensor
    down: ttnn.Tensor
    w_up: Optional[ttnn.Tensor] = None


class MoEBlock:
    """One MoE block on ``mesh_device``. ``resources``: the block's MoEBlockResources on it; ``dtypes``: MoEDtypes;
    ``accumulators``: the combine's accumulator count (the head's routing planner is planned for it); None: what the
    combine planner gives (moe_block_plan, which also picks the dispatch's row group);
    ``max_tokens_per_expert``: the flat routed experts' capacity (their activation is the resources' dims');
    ``layout`` / ``combine``: FlatLayoutOptions / CombineConfig (combine.n_acc is set to ``accumulators``).
    Capacity and layout default to max(chunk, 256) and FlatLayoutOptions() - the combine fits the flat plan into its
    arena (the largest free L1 block less the static CBs below it; FlatLayoutOptions.arena_bytes).

    Row blocks: a chunk whose rows do not fit the MoE head's L1 at once runs as the head's row blocks - the head's own
    choice (plan_moe_head's HeadFit.row_blocks), one launch whose kernels loop over the blocks; the tail follows the
    head's block layout (moe_tail row_blocks = MoEHeadOutputs.row_blocks), the dispatch, the routed experts and the
    combine run once on the whole chunk (each expert's weights read once).

    Splits (MoEBlockConfig.chunk_splits, chosen here): a chunk that does not fit in one call at all (usually the
    combine's landing pool / accumulator rings) runs the whole block as the fewest pieces that fit - each piece reads
    every routed expert's weights again, so it is loudly warned about. The resources are set for the splits here
    (set_chunk_splits), so prepare the weights after the block is made (resources / accumulators / expert_kw)."""

    def __init__(
        self,
        mesh_device,
        resources,
        *,
        dtypes,
        accumulators: Optional[int] = None,
        max_tokens_per_expert: Optional[int] = None,
        layout=None,
        combine=None,
    ):
        mb = ttnn.experimental.moe_block
        # (the ops' persistent words and semaphores live across calls: in L1_SMALL, where no other program's circular
        # buffers go - in plain L1 the model's programs between two MoE layers overwrote them. Open the mesh with an
        # l1_small_size that holds them: l1_small_bytes after the first call.)
        assert resources.config.persist_l1_small, "MoEBlock: create its resources with persist_l1_small=True"
        self.mb, self.mesh_device, self.resources, self.dtypes = mb, mesh_device, resources, dtypes
        # (the mesh axes: SP on resources.config.sp_axis - 1 for the 4x4 tray-half stage - and TP on the other)
        self.sp_axis = int(resources.config.sp_axis)
        self.tp_axis = 1 - self.sp_axis
        if layout is None:
            layout = mb.FlatLayoutOptions()
        self.combine = combine if combine is not None else mb.CombineConfig()
        n_acc = self.combine.n_acc
        ask = (accumulators, max_tokens_per_expert, layout)
        # (padding rows: a chunk whose rows a mesh row are not whole 32 x TP rows - the head's rows rule - runs with
        # each SP row padded up to them - MoEBlockResources sets MoEBlockConfig.pad_tokens; the padding rows are
        # padding tokens of a padded chunk, their output dropped)
        topo = resources.topology
        self.pad = int(resources.config.pad_tokens)
        self.pad_rows = self.pad // topo["R"]
        self.full_words = None  # (a chunk with padding rows and no span of the caller's: (0, chunk) device words)
        self._plan_splits(ask, n_acc, fixed_b=int(resources.config.chunk_splits))
        self.splits = max(1, int(resources.config.chunk_splits))
        self.row_blocks = int(self.head_fit.row_blocks)
        logger.info(
            f"MoEBlock: accumulators {self.accumulators}, dispatch group {self.plan.l1.dispatch_group} ({self.plan.note})"
            + (f", the head in {self.row_blocks} row blocks" if self.row_blocks > 1 else "")
            + (f", {self.splits} splits of {resources.config.split_tokens} tokens" if self.splits > 1 else "")
        )
        self.chunk_words = None  # (a padded chunk's scalars given as ints: two persistent device words, rewritten)
        self.meta_words = None  # (a forward given as host chunks: its metadata words, a persistent device tensor)

    def _plan(self, accumulators, max_tokens_per_expert, layout, n_acc):
        """The block's plans for the resources' splits (raises RuntimeError naming the op that does not fit)."""
        mb, resources, dtypes = self.mb, self.resources, self.dtypes
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
            layout=layout,
            combine=self.combine,
        )
        # (the block's plan, once per resources: the combine's accumulators and the dispatch's row group, which the
        # ops take by default from then on)
        chunk = resources.config.split_tokens
        self.plan = _fits(
            f"the dispatch / routed experts + combine plan (moe_block_plan: sparse_dispatch_2d, routed_experts_combine) "
            f"of {chunk} tokens",
            lambda: mb.moe_block_plan(resources, dtypes=dtypes, accumulators=accumulators, **self.expert_kw),
        )
        self.accumulators = self.plan.accumulators
        self.combine.n_acc = self.accumulators
        # (the head picks its row blocks from its L1 fit; the tail follows its block layout)
        fit = mb.plan_moe_head(resources, dtypes, self.accumulators)
        if not fit:
            raise RuntimeError(f"MoEBlock: moe_head does not fit {chunk} tokens in one call: {fit.reason}")
        self.head_fit = fit
        # (plans the head and the combine once; every forward allocates this spec at the same L1 address)
        self.arena_spec = _fits(
            f"the block's arena (moe_head + routed_experts_combine) for {chunk} tokens",
            lambda: mb.moe_block_arena_spec(resources, dtypes=dtypes, accumulators=self.accumulators, **self.expert_kw),
        )
        # (the tail of a call: its L1 grows with a row block's rows - host only, uncached)
        _fits(
            f"moe_tail ({chunk} tokens as the head's {fit.row_blocks} row blocks)",
            lambda: mb.plan_moe_tail(resources, dtype=ttnn.bfloat8_b, row_blocks=fit.row_blocks),
        )  # (a latent model's W_up tiles)

    def _plan_splits(self, ask, n_acc, fixed_b=0):
        """The splits to run (resources' chunk_splits; 0: the fewest that fit): one call wherever the chunk plans as it
        is (the head picks its own row blocks); else the fewest splits whose plans fit - each split re-reads every
        routed expert's weights, so more than one is warned about loudly."""
        resources = self.resources
        chunk, sp = int(resources.config.plan_tokens), int(resources.topology["R"])
        bs = [fixed_b] if fixed_b else [b for b in range(1, 65) if chunk % (b * sp * 32) == 0]
        first_error = None
        for b in bs:
            resources.set_chunk_splits(b)
            try:
                self._plan(*ask, n_acc)
            except RuntimeError as ex:
                first_error = first_error or ex
                logger.info(f"MoEBlock: chunk {chunk} as {b} splits does not fit: {_short(ex)}")
                continue
            if b > 1:
                _warn_splits(chunk, b, first_error)
            return
        raise first_error

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
        metadata: Union[tuple, ttnn.Tensor, None] = None,
        chunks: Optional[list] = None,
    ) -> ttnn.Tensor:
        """x: this chip's [1, 1, S/SP, H/TP] bfloat16 TILE, DRAM. Returns the residual's layout (moe_tail's).

        A padded chunk: actual_start / actual_end, main's chunked-prefill metadata scalars (the chunk's first real
        token's absolute position, one past its last) - one-element uint32 device tensors in DRAM, replicated, read on
        device every call (a trace replays with their values of the moment), or ints (host-written words, eager only);
        or metadata = main's (slot_id, actual_start, actual_end) tensors as they are. The chunk is in main's KV-pad-aware
        rotated layout over the SP rows (moe_padding_config; chunk-aligned start: the natural order, the first
        end - start tokens real). Padding tokens are not routed, dispatched, computed by the routed experts or combined,
        and the head's shared expert and the tail skip the row tiles holding none: the padding row tiles of the output
        are zero (without a latent; with one, the tail computes them).
        None: every row is real.

        A packed forward (MoEBlockConfig.forward_chunks F, common/prefill/prefill_chunk.py): x holds F chunks, each SP
        row every chunk's rows back to back, each chunk in main's rotated layout with its own span. Its padding comes
        as metadata = the forward's metadata words (one uint32 device tensor: at F = 1 (slot, start, end), above
        {num_chunks, 0} + F records - tt-d-gen's PrefillBatchMetadata as the device holds it), read on device every
        call; or chunks = the forward's PrefillChunks (anything with slot_id / actual_start / actual_end, or
        (slot, start, end) tuples; empty ones at the tail), written to the block's persistent words (eager only)."""
        with _zone("MoEBlock forward"):
            return self._call(x, weights, actual_start, actual_end, metadata, chunks)

    def _call(self, x, weights, actual_start, actual_end, metadata, chunks) -> ttnn.Tensor:
        span = self._span(actual_start, actual_end, metadata, chunks)
        if self.pad_rows:  # (padding rows: each chip's rows padded up to whole 32 x TP rows; their output dropped)
            if not span:
                if self.full_words is None:  # (made at the first call: eager - a trace captures them as they are)
                    F = max(1, int(self.resources.config.forward_chunks))
                    if F == 1:
                        words = self._words(0, self.resources.config.chunk_tokens)
                        self.full_words = dict(actual_start=words[0], actual_end=words[1])
                    else:
                        c = self.resources.config.chunk_tokens // F
                        self.full_words = dict(metadata=self._meta_tensor(forward_words([(0, 0, c)] * F, F)))
                span = self.full_words
            rows = int(x.shape[-2])
            xp = ttnn.pad(x, [(0, 0), (0, 0), (0, self.pad_rows), (0, 0)], 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            yp = self._forward_splits(xp, weights, span)
            ttnn.deallocate(xp)
            y = ttnn.slice(yp, [0, 0, 0, 0], [1, 1, rows, int(yp.shape[-1])], [1, 1, 1, 1],
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            ttnn.deallocate(yp)
            return y
        return self._forward_splits(x, weights, span)

    def _span(self, actual_start, actual_end, metadata, chunks) -> dict:
        """the head's padded-forward kwargs: {} (every row real), actual_start / actual_end, or metadata (words)"""
        F = max(1, int(self.resources.config.forward_chunks))
        given = [actual_start is not None or actual_end is not None, metadata is not None, chunks is not None]
        assert sum(given) <= 1, "MoEBlock: actual_start / actual_end, metadata or chunks - one of them"
        if chunks is not None:
            words = forward_words(chunks, F)
            if self.meta_words is None:
                self.meta_words = self._meta_tensor(words)
            else:
                host = ttnn.from_torch(torch.tensor([[[words]]], dtype=torch.int64), dtype=ttnn.uint32,
                                       layout=ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
                ttnn.copy_host_to_device_tensor(host, self.meta_words)
            return dict(metadata=self.meta_words)
        if isinstance(metadata, ttnn.Tensor):
            return dict(metadata=metadata)
        if metadata is not None:  # (main's (slot_id, actual_start, actual_end) tensors)
            actual_start, actual_end = metadata[1], metadata[2]
        assert (actual_start is None) == (actual_end is None), "MoEBlock: a padded chunk takes actual_start and _end"
        if actual_start is None:
            return {}
        assert F == 1, f"MoEBlock: a forward of {F} chunks takes metadata or chunks, not one chunk's span"
        if isinstance(actual_start, int) or isinstance(actual_end, int):
            assert isinstance(actual_start, int) and isinstance(actual_end, int), "MoEBlock: two ints or two tensors"
            actual_start, actual_end = self.chunk_scalars(actual_start, actual_end)
        return dict(actual_start=actual_start, actual_end=actual_end)

    def _meta_tensor(self, words) -> ttnn.Tensor:
        """the forward's metadata words as a one-page uint32 device tensor (replicated, DRAM)"""
        return ttnn.from_torch(
            torch.tensor([[[words]]], dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _forward_splits(self, x, weights, span) -> ttnn.Tensor:
        """The chunk as the block's splits (one without)."""
        if self.splits == 1:
            return self._forward(x, weights, span)
        # (splits: the whole block a piece of each chip's rows - every piece reads the experts' weights again)
        rows, width = int(x.shape[-2]), int(x.shape[-1])
        r = rows // self.splits
        outs = []
        for j in range(self.splits):
            xj = ttnn.slice(x, [0, 0, j * r, 0], [1, 1, (j + 1) * r, width], [1, 1, 1, 1],
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip
            outs.append(self._forward(xj, weights, span, split=j))
            ttnn.deallocate(xj)
        y = ttnn.concat(outs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in outs:
            ttnn.deallocate(t)
        return y

    def _forward(self, x, weights, span, split: int = 0) -> ttnn.Tensor:
        """One split (the whole chunk without splits): the head (its own row blocks), the dispatch, the routed experts,
        the combine and the tail (the head's row blocks) once."""
        mb = self.mb
        on_split = {"split": split} if self.splits > 1 else {}
        with _zone("MoEBlock arena"):
            arena = ttnn.allocate_tensor_on_device(self.arena_spec, self.mesh_device)
        self.arena_address = int(arena.buffer_address())
        with _zone("MoEBlock moe_head"):
            head = mb.moe_head(
                x,
                weights.head,
                resources=self.resources,
                dtypes=self.dtypes,
                accumulators=self.accumulators,
                arena=arena,
                **span,
                **on_split,
            )
        self.last_head = head  # (its persistent outputs: a model's checks read the routing rows)
        # (the dispatch's CBs in the call's arena - between the head's use of it and the combine's)
        with _zone("MoEBlock sparse_dispatch_2d"):
            dispatched = mb.sparse_dispatch_2d(head.dispatch_buffer, head.route_words, self.resources, arena=arena)
        with _zone("MoEBlock routed_experts_combine"):
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
        # (the tail in the head's row-block layout; a padded forward's span and split, as the head's)
        with _zone("MoEBlock moe_tail"):
            return mb.moe_tail(
                y_own,
                head.shared_expert_output,
                resources=self.resources,
                w_up=weights.w_up,
                row_blocks=head.row_blocks,
                **span,
                **(on_split if span else {}),
            )


INVALID_SLOT = 0xFFFFFFFF


def to_mesh_order(items: list, mesh_shape, sp_axis: int) -> list:
    """Per-chip items in the block's chip order (SP row r, TP column c: index r C + c) reordered to the mesh's row-major
    device order (ShardTensorToMesh's): the identity with SP on mesh axis 0; with SP on axis 1 the mesh device (mr, mc)
    is the block's chip (mc, mr)."""
    if sp_axis == 0:
        return list(items)
    rows, cols = (int(v) for v in tuple(mesh_shape))
    assert len(items) == rows * cols, f"{len(items)} items for a {rows}x{cols} mesh"
    return [items[mc * rows + mr] for mr in range(rows) for mc in range(cols)]


def sp_tp(mesh_shape, sp_axis: int) -> tuple:
    """(SP, TP): the block's chip rows and columns of a mesh with SP on sp_axis."""
    shape = tuple(int(v) for v in tuple(mesh_shape))
    return shape[sp_axis], shape[1 - sp_axis]


def forward_words(chunks, forward_chunks: int) -> list:
    """The forward's metadata words (common/prefill/prefill_chunk.py forward_metadata_words) for chunks - PrefillChunk
    likes (slot_id, actual_start, actual_end) or (slot, start, end) tuples, the empty ones (start == end) at the tail -
    as uint32: at one chunk a forward (slot, start, end), above {num_chunks, 0} + forward_chunks records."""
    cs = [(c.slot_id, c.actual_start, c.actual_end) if hasattr(c, "actual_start") else tuple(c) for c in chunks]
    assert 1 <= len(cs) <= forward_chunks, f"MoEBlock: {len(cs)} chunks for a forward of {forward_chunks}"
    cs += [(0, 0, 0)] * (forward_chunks - len(cs))
    if forward_chunks == 1:
        return [cs[0][0] & 0xFFFFFFFF, cs[0][1], cs[0][2]]
    real = [c for c in cs if c[2] != c[1]]
    assert cs[: len(real)] == real, f"MoEBlock: empty chunks go after every non-empty one: {cs}"
    words = [len(real), 0] + [w & 0xFFFFFFFF for c in real for w in c]
    return words + [INVALID_SLOT, 0, 0] * (forward_chunks - len(real))


def _fits(what: str, plan):
    """plan() or a RuntimeError naming what does not fit in one call"""
    try:
        return plan()
    except RuntimeError as ex:
        raise RuntimeError(f"MoEBlock: {what} does not fit in one call: {_short(ex)}") from ex


def _warn_splits(chunk: int, splits: int, why) -> None:
    """The loud warning of a block that runs as splits (once a block's construction)."""
    bar = "!" * 118
    logger.warning(
        f"\n{bar}\n"
        f"MoEBlock PERFORMANCE WARNING: chunk {chunk} does NOT fit in one call - the whole block runs as {splits} SPLITS "
        f"of {chunk // splits} tokens.\n"
        f"  Why: {_short(why) if why is not None else 'the one-call plan failed'}\n"
        f"  Cost: every routed expert's weights are read from DRAM {splits} times (once a split); prefill MoE is weight-"
        f"bandwidth bound, so the routed experts' time grows ~{splits}x, and the head, dispatch, combine and tail run "
        f"{splits} times on 1/{splits} of the tokens.\n"
        f"  Fix: run a smaller chunk per call (MoEBlockConfig.chunk_tokens, e.g. {chunk // splits}), a bigger stage "
        f"(more chips per SP row / TP), or give the combine more L1 (its landing pool / accumulator rings).\n"
        f"{bar}"
    )


def _short(ex) -> str:
    """a TT_FATAL's message without its backtrace"""
    msg = " ".join(str(ex).split())
    return msg.split(" backtrace:")[0][-300:]


def l1_small_bytes(mesh_device) -> int:
    """L1_SMALL allocated a bank on mesh_device (after a block's first call: its persistent words and semaphores)."""
    return int(ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_allocated_per_bank)
