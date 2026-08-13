# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side `block_residual` lifecycle for Kimi K3 attention residuals, plus
the stack driver that walks it.

The stream keeps `reference.attn_res.attn_res`'s `prefix_sum` / `num_sealed` /
`seal` / `accumulate` / `block_size` surface, and `attn_res_stack_split` keeps that
module's `attn_res_stack` signature, so the depth harness drives both backends off
one seal schedule and the read count cannot diverge between them. The in-layer order
is per-backend, because only this one owns tensors; the harness's per-layer PCC
curve is what gates that half.
"""

from dataclasses import dataclass

import ttnn

BLOCK_SIZE = 12


@dataclass
class TtAttnResState:
    """Owned live and sealed tensors transferred between pipeline segments."""

    prefix_sum: ttnn.Tensor | None
    block_residual: ttnn.Tensor | None

    def deallocate(self):
        if self.prefix_sum is not None:
            ttnn.deallocate(self.prefix_sum)
        if self.block_residual is not None:
            ttnn.deallocate(self.block_residual)
        self.prefix_sum = None
        self.block_residual = None

    def take_packed(self) -> ttnn.Tensor:
        """Consume this state into one D2D tensor: sealed snapshots, then live prefix."""
        if self.prefix_sum is None:
            raise ValueError("AttnRes state has no live prefix to pack")
        if self.block_residual is None:
            packed = self.prefix_sum
        else:
            packed = ttnn.concat([self.block_residual, self.prefix_sum], dim=1)
            ttnn.deallocate(self.block_residual)
            ttnn.deallocate(self.prefix_sum)
        self.prefix_sum = None
        self.block_residual = None
        return packed

    @classmethod
    def from_packed(cls, packed: ttnn.Tensor, *, num_sealed: int) -> "TtAttnResState":
        """Validate and consume one D2D tensor into sealed snapshots and live prefix."""
        if num_sealed < 0:
            raise ValueError(f"num_sealed must be non-negative, got {num_sealed}")
        if len(packed.shape) != 4 or packed.shape[0] != 1 or packed.shape[1] != num_sealed + 1:
            raise ValueError(f"packed AttnRes state must be [1, {num_sealed + 1}, N, H], got {tuple(packed.shape)}")
        if num_sealed == 0:
            return cls(prefix_sum=packed, block_residual=None)

        stop = tuple(packed.shape)
        block_residual = None
        try:
            block_residual = ttnn.slice(
                packed,
                (0, 0, 0, 0),
                (stop[0], num_sealed, stop[2], stop[3]),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            prefix_sum = ttnn.slice(
                packed,
                (0, num_sealed, 0, 0),
                stop,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        except Exception:
            if block_residual is not None:
                ttnn.deallocate(block_residual)
            ttnn.deallocate(packed)
            raise
        ttnn.deallocate(packed)
        return cls(prefix_sum=prefix_sum, block_residual=block_residual)


class TtAttnResStream(object):
    """One live stream and write-once snapshots, on device.

    Writes are plain `+=` with weight one; AttnRes rewrites only the read.
    `prefix_sum` is None between a seal and the next `accumulate` — the layer
    pipeline places no read site in that window, so `merge` asserts rather than
    guessing.

    **The stream owns its tensors.** Construction transfers ownership of
    `hidden_states`, and `accumulate` takes ownership of `module_out`. That makes
    the first `seal` a zero-copy ownership move into `block_residual` instead of a
    clone, and lets every later `seal` free what it superseded.

    Args:
        op: a `TtAttnRes`.
        hidden_states: `[1, 1, N, d]` token embeddings, the first live stream.
        block_size: layers per block; seals fire at `layer_idx % block_size == 0`.
        block_residual: optional sealed snapshots transferred from the preceding segment.
    """

    def __init__(self, op, hidden_states, block_size=BLOCK_SIZE, block_residual=None):
        self.op = op
        self._prefix_sum = hidden_states
        self._pending = None
        self.block_residual = block_residual
        self.block_size = block_size

    @property
    def prefix_sum(self):
        """The live stream, with any deferred write settled.

        Reading it costs the plain sum `accumulate` deferred, so a caller that
        goes on to take statistics over it should go through `merge` instead.
        """
        self._flush()
        return self._prefix_sum

    @property
    def num_sealed(self):
        return 0 if self.block_residual is None else self.block_residual.shape[1]

    def merge(self, partial, shift, mass, q, site):
        """A read, settling the deferred write against this query.

        The read holds both addends of the deferred sum for as long as it takes to
        score them, so the sum comes out of the same pass and the write costs no
        dispatch at all.
        """
        assert self._prefix_sum is not None, "no live stream between seal and accumulate"

        if self._pending is None:
            return self.op.merge(partial, shift, mass, self._prefix_sum, q, site)

        merged, total = self.op.merge(partial, shift, mass, self._prefix_sum, q, site, pending=self._pending)
        ttnn.deallocate(self._prefix_sum)
        ttnn.deallocate(self._pending)
        self._prefix_sum, self._pending = total, None
        return merged

    def seal(self):
        self._flush()
        assert self._prefix_sum is not None, "nothing to seal"
        if self.block_residual is None:
            self.block_residual = self._prefix_sum
        else:
            grown = ttnn.concat([self.block_residual, self._prefix_sum], dim=1)
            ttnn.deallocate(self.block_residual)
            ttnn.deallocate(self._prefix_sum)
            self.block_residual = grown
        self._prefix_sum = None

    def accumulate(self, module_out):
        """Take ownership of a module output, deferring the sum itself.

        The read that consumes the sum also reduces it, and one op does both —
        but it needs the query, which only the read has. So the addend is held
        until then; anything reaching the stream first settles it plainly.
        """
        if self._prefix_sum is None:
            self._prefix_sum = module_out
            return
        self._flush()
        self._pending = module_out

    def _flush(self):
        if self._pending is None:
            return
        total = ttnn.add(self._prefix_sum, self._pending)
        ttnn.deallocate(self._prefix_sum)
        ttnn.deallocate(self._pending)
        self._prefix_sum, self._pending = total, None

    def deallocate(self):
        for tensor in (self._prefix_sum, self._pending, self.block_residual):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self._prefix_sum, self._pending = None, None
        self.block_residual = None

    def take_state(self):
        """Settle deferred accumulation and transfer the stream tensors to the caller."""
        self._flush()
        assert self._prefix_sum is not None, "cannot hand off a stream between seal and accumulate"
        state = TtAttnResState(prefix_sum=self._prefix_sum, block_residual=self.block_residual)
        self._prefix_sum = None
        self.block_residual = None
        return state


def attn_res_segment(
    op,
    state,
    layer_indices,
    q_pre,
    q_post,
    attn_fns,
    mlp_fns,
    block_size=BLOCK_SIZE,
):
    """Walk a contiguous model-layer segment and return its transferable residual state.

    Unlike :func:`attn_res_stack_split`, this entry point accepts and returns an owned
    :class:`TtAttnResState`. That is what a pipeline rank must hand to its successor.
    This stage validates K3's natural 31/31/31 split at layers 31 and 62. Complete
    per-segment groups are batched normally; an empirical workaround pads only a
    successor segment's leading group to 24 query columns (see issue #53029).

    Ownership of ``state`` transfers into this function after argument validation;
    ownership of the returned state transfers back to the caller. Module callables borrow
    their input and transfer ownership of their output, matching ``attn_res_stack_split``.
    A device synchronization at the return boundary lands the writes before another process
    or rank transports the state.
    """
    layer_indices = tuple(layer_indices)
    if not layer_indices:
        raise ValueError("an AttnRes segment must contain at least one layer")
    if layer_indices != tuple(range(layer_indices[0], layer_indices[0] + len(layer_indices))):
        raise ValueError(f"AttnRes segment layers must be contiguous, got {layer_indices}")
    if not isinstance(state, TtAttnResState):
        raise TypeError(f"state must be TtAttnResState, got {type(state).__name__}")
    if state.prefix_sum is None:
        raise ValueError("AttnRes segment state has no live prefix")
    if layer_indices[0] == 0 and state.block_residual is not None:
        raise ValueError("AttnRes segment starting at layer 0 must not have sealed input state")
    if layer_indices[0] != 0 and state.block_residual is None:
        raise ValueError(f"AttnRes segment starting at layer {layer_indices[0]} requires sealed input state")
    lengths = {len(layer_indices), len(q_pre), len(q_post), len(attn_fns), len(mlp_fns)}
    if len(lengths) != 1:
        raise ValueError(
            "segment layers, queries, and module lists must have equal lengths: "
            f"layers={len(layer_indices)}, q_pre={len(q_pre)}, q_post={len(q_post)}, "
            f"attn={len(attn_fns)}, mlp={len(mlp_fns)}"
        )

    # Group the known read schedule by sealed-set epoch. The pre-attention read
    # on a seal layer belongs to the outgoing epoch; its post-attention read is
    # the first site over the newly grown sealed set.
    query_groups = []
    current_group = []
    has_sealed = state.block_residual is not None
    for local_idx, layer_idx in enumerate(layer_indices):
        if has_sealed:
            current_group.append(q_pre[local_idx])
        if layer_idx % block_size == 0:
            if current_group:
                query_groups.append(tuple(current_group))
            current_group = []
            has_sealed = True
        current_group.append(q_post[local_idx])
    if current_group:
        query_groups.append(tuple(current_group))
    # The natural 31/31/31 K3 split silently produced wrong aggregate output when
    # successor leading fragments used their natural batch widths. The root cause is
    # not isolated: short trailing batches in the same walk are correct. Pad only the
    # successor's leading fragment to the ordinary epoch width and consume its real
    # sites. This keeps one inter_block collective per handoff. Track and remove this
    # empirical workaround under https://github.com/tenstorrent/tt-metal/issues/53029.
    grouped_queries = [(queries, len(queries)) for queries in query_groups]
    if state.block_residual is not None:
        leading, active_sites = grouped_queries[0]
        epoch_sites = 2 * block_size
        if len(leading) < epoch_sites:
            grouped_queries[0] = (leading + (leading[-1],) * (epoch_sites - len(leading)), active_sites)

    groups = iter(grouped_queries)
    pending = iter(())
    partials = shifts = masses = None
    stream = TtAttnResStream(
        op,
        state.prefix_sum,
        block_size=block_size,
        block_residual=state.block_residual,
    )
    # Ownership moved into ``stream`` after validation. Empty the caller's handle so
    # defensive cleanup after either success or failure cannot double-free its tensors.
    state.prefix_sum = None
    state.block_residual = None

    def free_batch():
        nonlocal partials, shifts, masses
        if partials is None:
            return
        ttnn.deallocate(partials)
        ttnn.deallocate(shifts)
        ttnn.deallocate(masses)
        partials = shifts = masses = None

    def begin_group():
        nonlocal pending, partials, shifts, masses
        free_batch()
        try:
            queries, active_sites = next(groups)
        except StopIteration as error:
            raise AssertionError("AttnRes segment consumed more query groups than scheduled") from error
        partials, shifts, masses = op.inter_block(stream.block_residual, queries)
        pending = enumerate(queries[:active_sites])

    def read():
        try:
            site, query = next(pending)
        except StopIteration as error:
            raise AssertionError("AttnRes segment consumed its query group before the next seal") from error
        return stream.merge(partials, shifts, masses, query, site)

    next_state = None
    transient_h = None
    transient_out = None
    try:
        if stream.block_residual is not None:
            begin_group()

        for local_idx, layer_idx in enumerate(layer_indices):
            borrowed = stream.num_sealed == 0
            h = stream.prefix_sum if borrowed else read()
            transient_h = None if borrowed else h

            if layer_idx % block_size == 0:
                stream.seal()
                begin_group()

            transient_out = attn_fns[local_idx](h)
            if transient_h is not None:
                ttnn.deallocate(transient_h)
                transient_h = None
            stream.accumulate(transient_out)
            transient_out = None

            h = read()
            transient_h = h
            transient_out = mlp_fns[local_idx](h)
            ttnn.deallocate(transient_h)
            transient_h = None
            stream.accumulate(transient_out)
            transient_out = None

        # Settle the final deferred write before the state leaves this segment.
        next_state = stream.take_state()
        free_batch()
        try:
            next(pending)
        except StopIteration:
            pass
        else:
            raise AssertionError("unconsumed AttnRes segment query site")
        try:
            next(groups)
        except StopIteration:
            pass
        else:
            raise AssertionError("unconsumed AttnRes segment query group")
        ttnn.synchronize_device(op.mesh_device)
        return next_state
    except Exception:
        if transient_h is not None:
            ttnn.deallocate(transient_h)
        if transient_out is not None:
            ttnn.deallocate(transient_out)
        free_batch()
        if next_state is None:
            stream.deallocate()
        else:
            next_state.deallocate()
        raise


def finalize_attn_res(op, state, q_out):
    """Apply the model-level AttnRes read and consume ``state``."""
    if state.prefix_sum is None:
        raise ValueError("output AttnRes state has no live prefix")
    if state.block_residual is None:
        raise ValueError("output AttnRes requires at least one sealed snapshot")
    partials = shifts = masses = None
    try:
        partials, shifts, masses = op.inter_block(state.block_residual, [q_out])
        return op.merge(partials, shifts, masses, state.prefix_sum, q_out, 0)
    finally:
        if partials is not None:
            ttnn.deallocate(partials)
            ttnn.deallocate(shifts)
            ttnn.deallocate(masses)
        state.deallocate()


def _block_sites(layers, q_pre, q_post, q_out, block_size):
    """Queries grouped by the sealed set they read, in the order the walk consumes them.

    A `block_residual` value outlives the block that installed it by one read. The seal
    fires in the middle of layer `L`, between its two reads, so the sealed set spans layer
    `L`'s post-attention read through layer `L + block_size`'s pre-attention read. Groups
    are cut on the read sequence every `2 * block_size` sites for that reason, not on layer
    boundaries — and the trailing group is short whenever the stack does not divide evenly.
    """
    order = []
    for layer_idx in range(layers):
        if layer_idx > 0:
            order.append(q_pre[layer_idx])
        order.append(q_post[layer_idx])
    order.append(q_out)

    sites = 2 * block_size
    return [order[start : start + sites] for start in range(0, len(order), sites)]


def attn_res_stack_split(op, hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, hook=None):
    """Walk a whole stack's residual bookkeeping.

    Same schedule and same reads as `reference.attn_res.attn_res.attn_res_stack`, equal to it
    up to rounding, with the sealed half of every read hoisted into one `inter_block` per
    block. A sealed snapshot is write-once, so that half is loop-invariant across a block's
    read sites; only `merge`, which folds in the live stream, has to stay per site.

    Layer 0's pre-attention read is skipped for want of a snapshot, so every executed read
    runs at `S >= 1` and `inter_block`'s non-None precondition holds for all of them.

    `attn_fn` and `mlp_fn` stand in for everything between two reads. Each module's own input
    layernorm folds into its callable — that norm is distinct from the `*_res_norm` already
    folded into `q_pre` / `q_post`, and both exist in the checkpoint. A callable borrows the
    tensor it is handed and must not free it; ownership of what it returns passes to the stream.

    Args:
        op: a `TtAttnRes`.
        hidden_states: `[1, 1, N, d/tp_factor]` token embeddings, placed with
            `op.stream_mapper`. Ownership passes to the stream.
        q_pre, q_post: folded queries from `op.to_query`, one per layer.
            `q_pre[0]` is never read — layer 0 has no sealed snapshot — so a
            93-layer stack holds 187 queries and takes 186 reads.
        q_out: folded query for the single model-level read.
        attn_fns, mlp_fns: per-layer callables, see above.
        block_size: layers per block; seals fire at `layer_idx % block_size == 0`.
        hook: optional `(layer_idx, stream) -> None`, called after each layer.

    Returns:
        `[1, 1, N, d/tp_factor]` — what `model.norm` sees. The stream is freed
        here; the returned tensor is the caller's.
    """
    stream = TtAttnResStream(op, hidden_states, block_size=block_size)
    blocks = iter(_block_sites(len(attn_fns), q_pre, q_post, q_out, block_size))
    pending = iter(())
    partials = shifts = masses = None

    def read():
        site, q = next(pending)
        return stream.merge(partials, shifts, masses, q, site)

    for layer_idx, (attn_fn, mlp_fn) in enumerate(zip(attn_fns, mlp_fns)):
        # Only reach for the stream itself when no read follows — the property settles the
        # write that `merge` would otherwise fold into its own pass.
        borrowed = stream.num_sealed == 0
        h = stream.prefix_sum if borrowed else read()

        if layer_idx % block_size == 0:
            stream.seal()
            # The read just above is the outgoing block's last site, so replacing `pending`
            # and freeing its batches here strands nothing. `inter_block` has to follow the
            # seal, not precede it — the snapshot it batches over is the one just installed.
            queries = next(blocks)
            if shifts is not None:
                ttnn.deallocate(partials)
                ttnn.deallocate(shifts)
                ttnn.deallocate(masses)
            partials, shifts, masses = op.inter_block(stream.block_residual, queries)
            pending = enumerate(queries)

        stream.accumulate(attn_fn(h))
        if not borrowed:
            ttnn.deallocate(h)

        h = read()
        stream.accumulate(mlp_fn(h))
        ttnn.deallocate(h)

        if hook is not None:
            hook(layer_idx, stream)

    out = read()
    ttnn.deallocate(partials)
    ttnn.deallocate(shifts)
    ttnn.deallocate(masses)
    stream.deallocate()
    return out
