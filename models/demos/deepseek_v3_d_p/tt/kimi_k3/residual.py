# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The residual stream, as something a block talks to rather than a tensor it carries.

Every other model in this package threads one activation through its layer loop and folds each
module's output into it with `ttnn.add`. Kimi-K3 cannot: under AttnRes the value a module reads is
a softmax mixture of the live partial sum with one sealed snapshot per earlier 12-layer block, the
seal happens *inside* a layer between its two reads, and no read is optional — `TtAttnResWalk`
defers each write into the next read so the read op settles it for free. A block that skipped a
read and summed by hand would be running the plain-residual architecture with extra steps.

So the block calls six methods and holds no residual of its own. Two implementations satisfy them:

* `PlainResidualStream` — one activation and `ttnn.add`, bit-identical to what `TtPrefillBlock`
  does today. It is not here to retrofit other models; it is here so a K3 run can be bisected,
  swapping AttnRes out to tell an AttnRes bug apart from a KDA or MoE one.
* `TtAttnResResidual` — the real one, a thin adapter over `TtAttnResWalk`.

**Ownership is the part that does not survive being left implicit.** `open`/`read` hand back a
tensor the caller may *use* but not free; `write` *takes* one. The two implementations differ in
who owns the returned value — AttnRes's read allocates, the plain stream's does not — so `release`
exists to end the borrow without the block having to know which. Making them uniform by cloning in
the plain path instead would cost a 2.3 MB device copy at every read site of every layer, which at
61 layers is about a millisecond per chunk spent to avoid one method.
"""

from __future__ import annotations

from typing import Protocol

import ttnn


class ResidualStream(Protocol):
    """What a Kimi-K3 block needs from the residual, and nothing more."""

    def open(self, local_layer_idx: int) -> ttnn.Tensor:
        """This layer's pre-attention value, BORROWED until the matching `release`.

        Also performs whatever the layer boundary owes before that read — under AttnRes, the block
        seal and the sealed-set batch that follows it, in that order.
        """
        ...

    def read(self) -> ttnn.Tensor:
        """The pre-FFN value. Same borrow contract as `open`."""
        ...

    def write(self, module_out: ttnn.Tensor) -> None:
        """Fold a module's output in, TAKING OWNERSHIP of it. Once per `open`/`read`."""
        ...

    def release(self, borrowed: ttnn.Tensor) -> None:
        """End a borrow. After this the block must not touch the tensor either way."""
        ...

    def finish(self) -> ttnn.Tensor:
        """What `model.norm` sees. Frees the stream's own tensors; the return is the caller's."""
        ...

    def discard(self) -> None:
        """Abandon the stream with no final read — what a `kv_only` last layer leaves behind."""
        ...

    def handoff(self) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """End at a pipeline boundary: `(live_sum, sealed_set)`, ownership to the caller.

        The exit for a rank that is not the last. `finish` spends the model-level query, of
        which the stack has exactly one, so an earlier rank must not call it.
        """
        ...

    @property
    def num_sealed(self) -> int:
        """How many snapshots the stream would score a read against right now."""
        ...


class PlainResidualStream:
    """One running sum and `ttnn.add` — today's semantics, with nothing added.

    `write` reassigns rather than deallocating the previous sum, which is exactly what
    `TtPrefillBlock.forward` does at its two `x = ttnn.add(x, ...)` sites: the superseded activation
    is released when its last Python reference drops. Deallocating it here would be a behaviour
    change dressed up as tidiness.
    """

    def __init__(self, hidden_states: ttnn.Tensor):
        self._hidden = hidden_states

    def open(self, local_layer_idx: int) -> ttnn.Tensor:  # noqa: ARG002 - no per-layer bookkeeping
        return self._hidden

    def read(self) -> ttnn.Tensor:
        return self._hidden

    def write(self, module_out: ttnn.Tensor) -> None:
        self._hidden = ttnn.add(self._hidden, module_out)
        ttnn.deallocate(module_out)

    def release(self, borrowed: ttnn.Tensor) -> None:
        """A no-op: the running sum is the stream's, and `write` already superseded it."""

    def current(self) -> ttnn.Tensor:
        """The live sum, for the taps that want it without consuming a read site."""
        return self._hidden

    def finish(self) -> ttnn.Tensor:
        return self._hidden

    def discard(self) -> None:
        ttnn.deallocate(self._hidden)

    def handoff(self) -> tuple[ttnn.Tensor, None]:
        """One live sum and nothing sealed — the plain arm has no sealed set to carry."""
        hidden, self._hidden = self._hidden, None
        return hidden, None

    @property
    def num_sealed(self) -> int:
        return 0


class TtAttnResResidual:
    """`TtAttnResWalk` behind the same six methods.

    The walk already exposes `open_layer` / `read` / `write` / `finish`; what it does not expose is
    who owns what, and that is the whole of the adapter. `open_layer` returns `(h, borrowed)`, where
    `borrowed` is true exactly at layer 0 — nothing is sealed yet, so the layer reads the live
    stream itself instead of a mixture, and freeing it would free the stream. Every later `open` and
    every `read` allocates, and the block must give it back.

    One `TtAttnRes` op serves a whole stack (it holds all 187 folded queries and the shared exchange
    scratch); one walk serves one forward pass. AttnRes state is per token, so a chunk carries its
    own walk and nothing crosses a chunk boundary — the opposite of the KDA carry.
    """

    def __init__(self, walk):
        self._walk = walk
        self._borrowed = None

    def open(self, local_layer_idx: int) -> ttnn.Tensor:
        hidden, borrowed = self._walk.open_layer(local_layer_idx)
        self._borrowed = hidden if borrowed else None
        return hidden

    def read(self) -> ttnn.Tensor:
        return self._walk.read()

    def write(self, module_out: ttnn.Tensor) -> None:
        self._walk.write(module_out)

    def release(self, borrowed: ttnn.Tensor) -> None:
        if borrowed is self._borrowed:
            # Layer 0's borrow of the live stream. The walk still owns it.
            self._borrowed = None
            return
        ttnn.deallocate(borrowed)

    def current(self) -> ttnn.Tensor:
        """The live partial sum, with any deferred write settled first."""
        return self._walk.stream.running_sum

    def finish(self) -> ttnn.Tensor:
        return self._walk.finish()

    def handoff(self) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        return self._walk.handoff()

    @property
    def num_sealed(self) -> int:
        return self._walk.stream.num_sealed

    def discard(self) -> None:
        """Free the walk without its model-level read.

        `finish` is the only exit the walk offers, and it costs a read. A `kv_only` last layer takes
        no post-attention read and reaches no model-level one, so the schedule simply ends with two
        of its `2N` sites unconsumed — the `_pending` iterator is left un-exhausted, which is inert.
        What is not inert is the sealed set and the block's `inter_block` batches, so free those.
        """
        self._walk._free_batches()
        self._walk.stream.deallocate()
