# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-side `block_residual` lifecycle for Kimi K3 attention residuals, and the
walk over it.

`TtAttnResWalk` exposes the schedule as four calls a transformer makes from inside the
layer loop it already owns, so adopting AttnRes moves no loop.

The stream keeps `reference.kimi_k3.attn_res.attn_res`'s `running_sum` / `num_sealed` / `seal` /
`accumulate` / `block_size` surface, so the reference walk and this one run off one seal
schedule and the read count cannot diverge between them. The in-layer order is
per-backend, because only this one owns tensors.
"""

import ttnn
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import walk_sites

BLOCK_SIZE = 12


class TtAttnResStream(object):
    """One live stream and write-once snapshots, on device.

    Writes are plain `+=` with weight one; AttnRes rewrites only the read.
    `running_sum` is None between a seal and the next `accumulate` — the layer
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
    """

    def __init__(self, op, hidden_states, block_size=BLOCK_SIZE):
        self.op = op
        self._running_sum = hidden_states
        self._pending = None
        self.block_residual = None
        self.block_size = block_size

    @property
    def running_sum(self):
        """The live stream, with any deferred write settled.

        Reading it costs the plain sum `accumulate` deferred, so a caller that
        goes on to take statistics over it should go through `merge` instead.
        """
        self._flush()
        return self._running_sum

    @property
    def num_sealed(self):
        return 0 if self.block_residual is None else self.block_residual.shape[1]

    def merge(self, partial, shift, mass, q, site):
        """A read, settling the deferred write against this query.

        The read holds both addends of the deferred sum for as long as it takes to
        score them, so the sum comes out of the same pass and the write costs no
        dispatch at all.
        """
        assert self._running_sum is not None, "no live stream between seal and accumulate"

        if self._pending is None:
            return self.op.merge(partial, shift, mass, self._running_sum, q, site)

        merged, total = self.op.merge(partial, shift, mass, self._running_sum, q, site, pending=self._pending)
        ttnn.deallocate(self._running_sum)
        ttnn.deallocate(self._pending)
        self._running_sum, self._pending = total, None
        return merged

    def seal(self):
        self._flush()
        assert self._running_sum is not None, "nothing to seal"
        if self.block_residual is None:
            self.block_residual = self._running_sum
        else:
            grown = ttnn.concat([self.block_residual, self._running_sum], dim=1)
            ttnn.deallocate(self.block_residual)
            ttnn.deallocate(self._running_sum)
            self.block_residual = grown
        self._running_sum = None

    def accumulate(self, module_out):
        """Take ownership of a module output, deferring the sum itself.

        The read that consumes the sum also reduces it, and one op does both —
        but it needs the query, which only the read has. So the addend is held
        until then; anything reaching the stream first settles it plainly.
        """
        if self._running_sum is None:
            self._running_sum = module_out
            return
        self._flush()
        self._pending = module_out

    def _flush(self):
        if self._pending is None:
            return
        total = ttnn.add(self._running_sum, self._pending)
        ttnn.deallocate(self._running_sum)
        ttnn.deallocate(self._pending)
        self._running_sum, self._pending = total, None

    def deallocate(self):
        for tensor in (self._running_sum, self._pending, self.block_residual):
            if tensor is not None:
                ttnn.deallocate(tensor)
        self._running_sum, self._pending = None, None
        self.block_residual = None


def _block_sites(layers, q_pre, q_post, q_out, block_size):
    """Queries grouped by the sealed set they read, in the order the walk consumes them.

    A `block_residual` value outlives the block that installed it by one read. The seal
    fires in the middle of layer `L`, between its two reads, so the sealed set spans layer
    `L`'s post-attention read through layer `L + block_size`'s pre-attention read. Groups
    are cut on the read sequence every `2 * block_size` sites for that reason, not on layer
    boundaries — and the trailing group is short whenever the stack does not divide evenly.
    """
    order = walk_sites(q_pre[:layers], q_post[:layers], q_out)
    sites = 2 * block_size
    return [order[start : start + sites] for start in range(0, len(order), sites)]


class TtAttnResWalk(object):
    """One forward pass's residual bookkeeping, driven by the caller's layer loop.

    A transformer owns its own layer loop, so the schedule is exposed as four calls it
    makes from where it already is:

        walk = TtAttnResWalk(op, embeddings, q_pre, q_post, q_out, len(layers))
        for layer_idx, layer in enumerate(layers):
            h, borrowed = walk.open_layer(layer_idx)   # pre-attention read
            walk.write(attention(h))
            if not borrowed:
                ttnn.deallocate(h)
            h = walk.read()                            # post-attention read
            walk.write(mlp(h))
            ttnn.deallocate(h)
        hidden = walk.finish()                         # model-level read

    `q_pre` and `q_post` hold one folded query per layer, `q_out` one for the model-level
    read. `q_pre[0]` is never issued — layer 0 has nothing sealed to read against — so a
    93-layer stack holds 187 queries and takes 186 reads.

    `open_layer` is what a caller cannot reconstruct from the parts: it owes a read, and
    on a block boundary also a seal and the batch that follows it, in that order and no
    other. The read belongs to the outgoing block, so it precedes the seal; the batch is
    over the snapshot the seal just installed, so it follows it.

    A read costs one dispatch and no read is optional — `write` defers its sum into the
    next one. A caller that skips reads and sums by hand is running the reference
    architecture's residual, not this one.

    Everything here is one pass's state and none of it survives `finish`. The op is not:
    it holds the weights and outlives every walk built on it.
    """

    def __init__(self, op, hidden_states, q_pre, q_post, q_out, num_layers, block_size=BLOCK_SIZE):
        self.op = op
        self.block_size = block_size
        self.stream = TtAttnResStream(op, hidden_states, block_size=block_size)
        self._blocks = iter(_block_sites(num_layers, q_pre, q_post, q_out, block_size))
        self._pending = iter(())
        self._partials = self._shifts = self._masses = None

    def read(self):
        """The next read site, in the order the schedule issues them."""
        site, query = next(self._pending)
        return self.stream.merge(self._partials, self._shifts, self._masses, query, site)

    def write(self, module_out):
        """Add a module's output to the live stream, taking ownership of it."""
        self.stream.accumulate(module_out)

    def open_layer(self, layer_idx):
        """This layer's pre-attention read, and the seal and batch it may owe first.

        Returns `(h, borrowed)`. Layer 0 has nothing sealed to read against, so it borrows
        the live stream itself and the caller must not free what it got — every later layer
        owns its `h`.
        """
        borrowed = self.stream.num_sealed == 0
        h = self.stream.running_sum if borrowed else self.read()

        if layer_idx % self.block_size == 0:
            self.stream.seal()
            # The read above was the outgoing block's last site, so replacing the batches
            # here strands nothing.
            queries = next(self._blocks)
            self._free_batches()
            self._partials, self._shifts, self._masses = self.op.inter_block(self.stream.block_residual, queries)
            self._pending = enumerate(queries)
        return h, borrowed

    def finish(self):
        """The single model-level read, and the end of the walk.

        Returns what `model.norm` sees. The walk's own tensors are freed here; the returned
        one is the caller's.
        """
        out = self.read()
        self._free_batches()
        self.stream.deallocate()
        return out

    def _free_batches(self):
        if self._shifts is None:
            return
        ttnn.deallocate(self._partials)
        ttnn.deallocate(self._shifts)
        ttnn.deallocate(self._masses)
        self._partials = self._shifts = self._masses = None
