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

Deliberately NOT here: generation. ``MTPEmbedSource`` resolves the last chunk's ``K`` trailing
positions by running the LM head once per level and feeding the sampled token back in. That is a
host round trip per level by definition, and the runner path exists to avoid host round trips, so
the device source carries whatever the producer put in those slots (its own pad id) and reports no
generated tokens. It costs the last ``k`` rows of level ``k`` on the FINAL chunk of a request --
``K*(K+1)/2`` rows out of ``chunk_size*K`` -- and nothing at all on an interior chunk, where every
window is a pure prompt slice.
"""

from __future__ import annotations

from typing import Optional

import ttnn

__all__ = ["MTPUnionEmbedding", "MTPDeviceEmbedSource"]


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

    def deallocate(self) -> None:
        for t in self._parts:
            ttnn.deallocate(t)
        self._parts = []
        if self._rows is not None:
            ttnn.deallocate(self._rows)
            self._rows = None

    def _row_major(self) -> ttnn.Tensor:
        """ROW_MAJOR copy of the JOINED union, materialized once and reused by all K windows.

        A window starts at row ``k`` for ``k`` in 1..K, which is never a multiple of 32, and
        ``ttnn.slice`` on TILE_LAYOUT only cuts on tile boundaries -- so the rows must be untilized,
        and joined first when they arrived as separate blocks. Kept alive beside the blocks rather
        than replacing them, because a middle rank re-packs the tiled ones -- 2 MiB/chip at the
        production shape, on a rank that just gave back 453.75.
        """
        if self._rows is None:
            assert self._parts, "union embedding already deallocated"
            joined = self._parts[0] if len(self._parts) == 1 else ttnn.concat(self._parts, dim=-2)
            self._rows = ttnn.to_layout(joined, ttnn.ROW_MAJOR_LAYOUT)
            if joined is not self._parts[0]:
                ttnn.deallocate(joined)
        return self._rows


class MTPDeviceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable, sourced entirely on device.

    Drop-in for :class:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows.MTPEmbedSource`:
    same ``(k, H^k) -> ttnn.Tensor`` signature, same ``generated_tokens`` property. It ignores
    ``H^k`` -- with no host LM-head round trip there is nothing to derive from it -- so
    ``generated_tokens`` is always empty and the last chunk's trailing rows carry the producer's pad
    ids. See this module's header for exactly which rows that is.
    """

    def __init__(self, union: MTPUnionEmbedding, mask_fn=None):
        self.union = union
        self.mask_fn = mask_fn
        self.num_levels = union.num_levels

    @property
    def generated_tokens(self) -> list:
        """Always empty: this source never runs the LM head. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed):
        assert 0 <= k < self.num_levels, f"level {k} out of range [0, {self.num_levels})"
        window = self.union.window(k + 1)
        return self.mask_fn(window) if self.mask_fn is not None else window
