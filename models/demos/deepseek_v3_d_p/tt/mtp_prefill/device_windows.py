# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device MTP shift-windows for the prefill runner (GLM-5.2, issue #53533).

``token_windows.py`` solves the same problem on the HOST: it slices python lists and hands each
window to an ``embed_fn`` that uploads it. That is right for a test driving the transformer
directly, and wrong for the serving runner, where the tokens arrive over the H2D socket and never
touch host memory. This module is the device twin -- same ``(k, H^k) -> embedding`` contract that
``TtMTPPredictor.forward`` consumes, no host round trip anywhere in it.

Two things make that possible.

**1. The overhang rides the trunk's own H2D row.** One socket delivers chip ``c`` the contiguous
``[c*L, c*L + L + overhang)``, and the runner cuts it at ``L`` into the ``[1, 1, L]`` trunk the model
has always been handed and the ``overhang`` ids that follow it. Because a chip's lookahead is the ids
immediately past its OWN shard, MTP level ``k`` -- which wants position ``p + k`` on the row whose
hidden sits at ``p`` -- reads the SAME local slice ``[k, k+L)`` on every chip. No SP ring-shift, no
cross-chip rotation. See ``runner_utils.MTP_OVERHANG_ALIGN`` for why the overhang is a whole tile.

**2. What crosses the D2D socket is the union EMBEDDING, not the ids.** The first rank gathers the
two id tensors separately -- out of the table it already loads -- and ships both stacked under the
hidden. The trunk gather is not extra work: its result IS this chunk's model input, so the whole
union costs ``L + overhang`` gathered rows, not ``L`` for the model and another ``L + overhang`` for
MTP. Downstream ranks slice their windows out of what arrived, so the
LAST rank runs its levels with no embedding table at all (453.75 MiB/chip at GLM-5.2's 154880-entry
vocab). ``slice(embed(ids)) == embed(slice(ids))`` row-for-row, so the two paths agree bit-exactly by
construction rather than by measurement -- which is why the ids no longer need a codec to survive a
bf16 wire.

**3. The last chunk GENERATES the ids the prompt does not have, on device.** Past a request's real
length the stream has no more tokens, so level ``k``'s window at the last real row has nothing to
read. ``MTPEmbedSource`` solves that on the host -- LM head, argmax, feed the id back in -- which is
a host round trip per level, exactly what this path exists to avoid. Here the same chain runs on
device: ``argmax(lm_head(H^k))`` -> ``embed`` -> SP ``all_gather`` -> a one-hot matmul that writes
the result into the union at global position ``actual_end + k`` (:class:`MTPDeviceGeneration`).
Writing it at that ONE position is the whole simplification: every level's window slice then picks
it up by itself. ``generated_tokens`` still comes back empty -- the ids never leave the device.

Only the FINAL chunk of a request does any of this; an interior chunk's windows are pure prompt
slices, and the producer says which is which in the 4th PrefillMetadata word.
"""

from __future__ import annotations

from typing import Optional

import ttnn

__all__ = ["MTPUnionEmbedding", "MTPDeviceEmbedSource", "MTPDeviceGeneration"]


class MTPUnionEmbedding:
    """One chunk's MTP source rows: the ``L + overhang`` embeddings covering this chip's trunk
    positions and the ``overhang`` positions that follow them.

    Held as an ordered list of row BLOCKS rather than one tensor, because the two ranks that build it
    hold it differently and neither should pay a copy to look like the other:

    * :meth:`from_ids` -- the FIRST rank, holding the trunk and overhang id tensors the H2D row was
      cut into. It gathers each separately, so the blocks are ``[trunk, overhang]`` and
      :attr:`trunk`, which is this chunk's model input, is the leading block at no cost.
    * :meth:`from_embedding` -- any downstream rank, holding the contiguous union that arrived packed
      under the hidden on the D2D socket. One block.

    :meth:`window` is the only thing that needs the rows contiguous, and it joins them once
    (:meth:`_row_major`). A rank that runs no level -- the first rank of a pipeline, every
    intermediate rank -- therefore never joins them at all: it stacks :attr:`parts` straight into the
    D2D activation, which writes the same bytes a joined union would have.
    """

    def __init__(self, parts: list, *, num_levels: int, window_len: int):
        self.num_levels = int(num_levels)
        self.window_len = int(window_len)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self._parts: list = list(parts)
        assert self._parts, "a union needs at least one row block"
        self._rows: Optional[ttnn.Tensor] = None
        # Set by clear_rows/add_patch: the union AFTER generation patches, as one tensor. Kept beside
        # _parts rather than replacing them because a middle rank re-packs the blocks it received --
        # and because .trunk, which is this chunk's model input, must keep pointing at the embedding
        # the trunk actually ran on.
        self._patched: Optional[ttnn.Tensor] = None
        rows = sum(int(p.shape[-2]) for p in self._parts)
        assert rows >= self.window_len + self.num_levels, (
            f"union embedding is {rows} rows, needs at least window_len + K = {self.window_len} + "
            f"{self.num_levels}. The producer and the runner must agree on PREFILL_MTP_LEVELS."
        )

    @classmethod
    def from_ids(
        cls, chunk_ids: ttnn.Tensor, overhang_ids: ttnn.Tensor, embed_fn, *, num_levels: int
    ) -> "MTPUnionEmbedding":
        """Gather the union from the two id tensors the H2D row was cut into. Neither is consumed.

        TWO gathers, deliberately, rather than one over a rejoined id row. The trunk gather is not
        extra work -- its result IS the model's input for this chunk (:attr:`trunk`), the tensor the
        first rank would gather inside ``forward`` anyway. Rejoining the ids first would make the
        union one gather and then force a SECOND, identical gather of the same ``L`` rows for the
        model: 640 redundant rows per chip per chunk at the production shape.

        ``embed_fn`` is the trunk's own embedding gather: ``[sp, 1, N]`` uint32 ids ->
        ``[1, 1, N, H/tp]`` bf16 TILE, UNMASKED. Position-0 masking belongs to the caller
        (:class:`MTPDeviceEmbedSource`), which applies it per window.
        """
        window_len = int(chunk_ids.shape[-1])
        return cls(
            [embed_fn(chunk_ids), embed_fn(overhang_ids)],
            num_levels=num_levels,
            window_len=window_len,
        )

    @classmethod
    def from_embedding(cls, embedding: ttnn.Tensor, num_levels: int, window_len: int) -> "MTPUnionEmbedding":
        """Take ownership of a received union: ``[1, 1, window_len + overhang, H/tp]`` bf16 TILE."""
        return cls([embedding], num_levels=num_levels, window_len=window_len)

    @property
    def parts(self) -> list:
        """The union as its row blocks, in row order. What the D2D pack stacks under the hidden."""
        assert self._parts, "union embedding already deallocated"
        return list(self._parts)

    @property
    def overhang(self) -> int:
        """Rows the union holds past this chip's trunk shard: ``U - window_len``."""
        assert self._parts, "union embedding already deallocated"
        return sum(int(p.shape[-2]) for p in self._parts) - self.window_len

    @property
    def trunk(self) -> ttnn.Tensor:
        """This chunk's trunk embedding -- the first ``window_len`` rows, as its own tensor.

        The model input on the first rank. Owned HERE, because the D2D pack re-reads it after
        ``forward`` returns; the caller must not free it, :meth:`deallocate` does.

        Defined only on a :meth:`from_ids` union, where the trunk is a block in its own right. A
        received union is one contiguous tensor whose leading rows are not separable without a copy,
        so this asserts rather than quietly hand back all ``L + overhang`` of them.
        """
        assert self._parts, "union embedding already deallocated"
        rows = int(self._parts[0].shape[-2])
        assert rows == self.window_len, (
            f"leading block is {rows} rows, not the {self.window_len}-row trunk -- .trunk exists "
            "only on a union built by from_ids (the first rank)"
        )
        return self._parts[0]

    def window(self, shift: int) -> ttnn.Tensor:
        """MTP window ``shift`` (1..K): rows ``[shift, shift + window_len)`` as
        ``[1, 1, window_len, H/tp]`` bf16 TILE. Caller frees it."""
        assert 1 <= shift <= self.num_levels, f"shift {shift} out of range [1, {self.num_levels}]"
        src = self._row_major()
        s = list(src.shape)
        rows = ttnn.slice(src, [0, 0, shift, 0], [s[0], s[1], shift + self.window_len, s[3]])
        window = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
        ttnn.deallocate(rows)
        return window

    def clear_rows(self, keep_mask: ttnn.Tensor) -> None:
        """Multiply the union by ``[sp, 1, U, H/tp]`` ``keep_mask`` (zeroing the generation rows)."""
        self._apply(lambda src: ttnn.multiply(src, keep_mask))

    def add_patch(self, select: ttnn.Tensor, embeddings: ttnn.Tensor) -> None:
        """Add ``select @ embeddings`` into the union: ``[sp, 1, U, 32*sp] @ [1, 1, 32*sp, H/tp]``.

        ``select`` is one-hot, so this writes one embedding row into the union rows that hold the
        generated position and leaves every other row exactly as it was.
        """

        def _patch(src):
            delta = ttnn.matmul(select, embeddings)
            out = ttnn.add(src, delta)
            ttnn.deallocate(delta)
            return out

        self._apply(_patch)

    def deallocate(self) -> None:
        for t in self._parts:
            ttnn.deallocate(t)
        self._parts = []
        for name in ("_patched", "_rows"):
            t = getattr(self, name)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)

    def _current(self) -> tuple:
        """``(the union as ONE tile tensor, whether the caller must free it)``."""
        if self._patched is not None:
            return self._patched, False
        assert self._parts, "union embedding already deallocated"
        if len(self._parts) == 1:
            return self._parts[0], False
        return ttnn.concat(self._parts, dim=-2), True

    def _apply(self, fn) -> None:
        """Replace the union with ``fn(union)``, freeing what it replaces and the stale ROW_MAJOR copy.

        Never touches ``_parts``: those are the received/gathered blocks, which the D2D pack and
        ``.trunk`` still read.
        """
        src, temp = self._current()
        out = fn(src)
        if temp or src is self._patched:
            ttnn.deallocate(src)
        self._patched = out
        if self._rows is not None:
            ttnn.deallocate(self._rows)
            self._rows = None

    def _row_major(self) -> ttnn.Tensor:
        """ROW_MAJOR copy of the JOINED union, materialized once and reused until it is invalidated.

        A window starts at row ``k`` for ``k`` in 1..K, which is never a multiple of 32, and
        ``ttnn.slice`` on TILE_LAYOUT only cuts on tile boundaries -- so the rows must be untilized,
        and joined first when they arrived as separate blocks. Kept alive beside the blocks rather
        than replacing them, because a middle rank re-packs the tiled ones -- 2 MiB/chip at the
        production shape, on a rank that just gave back 453.75. Generation invalidates it once per
        level (:meth:`_apply`), so the last chunk untilizes ``K`` times instead of once.
        """
        if self._rows is None:
            joined, temp = self._current()
            self._rows = ttnn.to_layout(joined, ttnn.ROW_MAJOR_LAYOUT)
            if temp:
                ttnn.deallocate(joined)
        return self._rows


class MTPDeviceGeneration:
    """Everything the LAST chunk needs to fill the positions the prompt does not reach.

    Built once per chunk by the caller (it owns the mesh geometry and the LM head) and handed to
    :class:`MTPDeviceEmbedSource`, which drives it level by level.

    Args:
        keep_mask: ``[sp, 1, U, H/tp]`` ones, zero on every row generation will write. Applied to
            the union once, before level 0.
        selects: ``K`` one-hot ``[sp, 1, U, 32*sp]`` selectors, ``selects[k]`` placing level ``k``'s
            token at global position ``actual_end + k``. All ``K`` are host-known up front: the
            source row is the LM head's ``(device_id, token_offset)`` for ``actual_isl - 1``, which
            is the same row at every level.
        embed_fn: ``H^k -> [1, 1, 32*sp, H/tp]`` -- lm_head, argmax, embed, SP all-gather. The
            gathered block is what ``selects[k]`` indexes into.
    """

    def __init__(self, keep_mask: ttnn.Tensor, selects: list, embed_fn):
        self.keep_mask = keep_mask
        self.selects = list(selects)
        self.embed_fn = embed_fn
        assert self.selects, "generation needs one selector per level"

    def deallocate(self) -> None:
        for t in [self.keep_mask, *self.selects]:
            ttnn.deallocate(t)
        self.selects = []


class MTPDeviceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable, sourced entirely on device.

    Drop-in for :class:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows.MTPEmbedSource`:
    same ``(k, H^k) -> ttnn.Tensor`` signature, same ``generated_tokens`` property.

    Without ``generation`` (every interior chunk) it ignores ``H^k`` entirely -- each window is a
    row slice of the union the socket delivered. With it (the last chunk of a request) it runs the
    device generation chain on ``H^k`` and patches the union before slicing, so level ``k``'s last
    real row reads the token level ``k`` just generated and the earlier rows read the tokens the
    earlier levels generated.

    ``generated_tokens`` is empty either way: the ids are argmaxed, embedded and consumed on device
    and never come back to host. Nothing on the runner path reads them.
    """

    def __init__(self, union: MTPUnionEmbedding, mask_fn=None, generation: "Optional[MTPDeviceGeneration]" = None):
        self.union = union
        self.mask_fn = mask_fn
        self.generation = generation
        self.num_levels = union.num_levels
        self._next_level = 0

    @property
    def generated_tokens(self) -> list:
        """Always empty: the generated ids never leave the device. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed):
        assert 0 <= k < self.num_levels, f"level {k} out of range [0, {self.num_levels})"
        if self.generation is not None:
            # Strict level order, once each -- same contract as MTPEmbedSource. The patches are
            # INCREMENTAL: level k's window reads positions actual_end+k-j for j in 0..k, so every
            # earlier level's token must already be in the union.
            assert k == self._next_level, f"generation must run levels in order; expected {self._next_level}, got {k}"
            self._next_level += 1
            if k == 0:
                self.union.clear_rows(self.generation.keep_mask)
            gathered = self.generation.embed_fn(prev_normed)
            self.union.add_patch(self.generation.selects[k], gathered)
            ttnn.deallocate(gathered)
        window = self.union.window(k + 1)
        return self.mask_fn(window) if self.mask_fn is not None else window
