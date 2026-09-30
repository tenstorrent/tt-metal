# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device MTP shift windows for the prefill runner.

Prefill's tokens never reach host memory, so every level's ``(k, H^k) -> embedding`` window is built
on device: a row slice of the embedded union, plus generated ids on a request's final chunk.
"""

from __future__ import annotations

from typing import Optional

import ttnn

__all__ = ["MTPSeam", "MTPUnionEmbedding", "MTPDeviceEmbedSource", "MTPDeviceGeneration"]


class MTPSeam:
    """The seam of a chunk that starts off a ``window_len`` boundary, and the rows that mend it.

    Such a chunk is rotated so that ONE chip, the seam chip, holds two position runs: rows ``[0, row)``
    end where the next chip's rows begin, and rows ``[row, window_len)`` continue into its own
    lookahead. A window shifted by ``d`` therefore needs, in rows ``[row - d, row)``, the next chip's
    leading rows. :meth:`splice_rows` brings them over; every other chip keeps its own rows there, so
    one SPMD program serves the whole mesh.
    """

    def __init__(self, *, row: int, neighbour: int, on_seam: ttnn.Tensor, off_seam: ttnn.Tensor, gather):
        self.row = int(row)
        self.neighbour = int(neighbour)
        self.on_seam = on_seam
        self.off_seam = off_seam
        self.gather = gather
        assert self.row % ttnn.TILE_SIZE == 0, f"seam row {self.row} is not tile-aligned"

    @classmethod
    def for_chunk(
        cls, chunk_start: int, window_len: int, sp_factor: int, chip_index: ttnn.Tensor, gather
    ) -> "Optional[MTPSeam]":
        """The seam of the chunk at ``chunk_start``, or None when every chip holds one run.

        ``chip_index`` is ``[1, 1, 1, 1]`` holding each chip's SP index; ``gather`` all-gathers a
        ``[1, 1, 32, H/tp]`` tile over SP into ``[1, 1, 32*sp, H/tp]``, in SP order.
        """
        offset = chunk_start % window_len
        if sp_factor == 1 or offset == 0:
            return None
        assert chunk_start % ttnn.TILE_SIZE == 0, (
            f"chunk_start={chunk_start} is not tile-aligned; the KV writer only resumes on a multiple of "
            f"{ttnn.TILE_SIZE}"
        )
        seam_chip = (chunk_start // window_len) % sp_factor
        return cls(
            row=window_len - offset,
            neighbour=(seam_chip + 1) % sp_factor,
            on_seam=ttnn.eq(chip_index, float(seam_chip)),
            off_seam=ttnn.ne(chip_index, float(seam_chip)),
            gather=gather,
        )

    def splice_rows(self, union: ttnn.Tensor) -> ttnn.Tensor:
        """``[1, 1, 32, H/tp]`` ROW_MAJOR: the neighbour's leading rows on the seam chip and this
        chip's own rows ``[row, row + 32)`` everywhere else. ``union`` is a tile tensor, not consumed.
        """
        width = int(union.shape[-1])
        tile = ttnn.TILE_SIZE
        head = ttnn.slice(union, [0, 0, 0, 0], [1, 1, tile, width])
        heads = self.gather(head)
        ttnn.deallocate(head)
        neighbour = ttnn.slice(heads, [0, 0, tile * self.neighbour, 0], [1, 1, tile * (self.neighbour + 1), width])
        ttnn.deallocate(heads)
        own = ttnn.slice(union, [0, 0, self.row, 0], [1, 1, self.row + tile, width])
        taken = ttnn.multiply(neighbour, self.on_seam)
        kept = ttnn.multiply(own, self.off_seam)
        ttnn.deallocate(neighbour)
        ttnn.deallocate(own)
        rows = ttnn.add(taken, kept)
        ttnn.deallocate(taken)
        ttnn.deallocate(kept)
        rows_rm = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(rows)
        return rows_rm

    def deallocate(self) -> None:
        for t in (self.on_seam, self.off_seam):
            if t is not None:
                ttnn.deallocate(t)
        self.on_seam = self.off_seam = None


class MTPUnionEmbedding:
    """One chunk's MTP source rows: this chip's trunk embeddings and the lookahead rows after them.

    Held as an ordered list of row blocks; only :meth:`window` joins them.
    """

    def __init__(self, parts: list, *, num_levels: int, window_len: int):
        self.num_levels = int(num_levels)
        self.window_len = int(window_len)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self._parts: list = list(parts)
        assert self._parts, "a union needs at least one row block"
        self._rows: Optional[ttnn.Tensor] = None
        self._patched: Optional[ttnn.Tensor] = None
        self._seam: Optional[MTPSeam] = None
        self._seam_rows: Optional[ttnn.Tensor] = None
        rows = sum(int(p.shape[-2]) for p in self._parts)
        assert rows >= self.window_len + self.num_levels, (
            f"union embedding is {rows} rows, needs at least window_len + K = {self.window_len} + "
            f"{self.num_levels}. The producer and the runner must agree on PREFILL_MTP_LEVELS."
        )

    @classmethod
    def from_ids(
        cls, chunk_ids: ttnn.Tensor, mtp_ids: ttnn.Tensor, embed_fn, *, num_levels: int
    ) -> "MTPUnionEmbedding":
        """Gather the union from the two id tensors the H2D row was cut into. Neither is consumed."""
        window_len = int(chunk_ids.shape[-1])
        return cls(
            [embed_fn(chunk_ids), embed_fn(mtp_ids)],
            num_levels=num_levels,
            window_len=window_len,
        )

    @classmethod
    def from_embedding(cls, embedding: ttnn.Tensor, num_levels: int, window_len: int) -> "MTPUnionEmbedding":
        """Take ownership of a received union: ``[1, 1, window_len + num_mtp_tokens, H/tp]`` bf16 TILE."""
        return cls([embedding], num_levels=num_levels, window_len=window_len)

    @property
    def parts(self) -> list:
        """The union as its row blocks, in row order. What the D2D pack stacks under the hidden."""
        assert self._parts, "union embedding already deallocated"
        return list(self._parts)

    @property
    def num_mtp_tokens(self) -> int:
        """Rows the union holds past this chip's trunk shard: ``U - window_len``."""
        assert self._parts, "union embedding already deallocated"
        return sum(int(p.shape[-2]) for p in self._parts) - self.window_len

    @property
    def trunk(self) -> ttnn.Tensor:
        """This chunk's trunk embedding -- the leading ``window_len`` rows, as its own tensor."""
        assert self._parts, "union embedding already deallocated"
        rows = int(self._parts[0].shape[-2])
        assert rows == self.window_len, (
            f"leading block is {rows} rows, not the {self.window_len}-row trunk -- .trunk exists "
            "only on a union built by from_ids (the first rank)"
        )
        return self._parts[0]

    def window(self, shift: int) -> ttnn.Tensor:
        """MTP window ``shift`` (1..K): rows ``[shift, shift + window_len)`` as
        ``[1, 1, window_len, H/tp]`` bf16 TILE, with the seam's rows spliced in when one is set.
        Caller frees it."""
        assert 1 <= shift <= self.num_levels, f"shift {shift} out of range [1, {self.num_levels}]"
        src = self._row_major()
        s = list(src.shape)
        if self._seam is None:
            rows = ttnn.slice(src, [0, 0, shift, 0], [s[0], s[1], shift + self.window_len, s[3]])
        else:
            seam_row = self._seam.row
            pieces = [
                ttnn.slice(src, [0, 0, shift, 0], [s[0], s[1], seam_row, s[3]]),
                ttnn.slice(self._spliced(), [0, 0, 0, 0], [s[0], s[1], shift, s[3]]),
                ttnn.slice(src, [0, 0, seam_row + shift, 0], [s[0], s[1], self.window_len + shift, s[3]]),
            ]
            rows = ttnn.concat(pieces, dim=-2)
            for piece in pieces:
                ttnn.deallocate(piece)
        window = ttnn.to_layout(rows, ttnn.TILE_LAYOUT)
        ttnn.deallocate(rows)
        return window

    def set_seam(self, seam: "Optional[MTPSeam]") -> None:
        """Make :meth:`window` mend ``seam`` (None: plain shifts). The union does not own it."""
        assert seam is None or (self.num_levels < seam.row < self.window_len and self.num_levels <= ttnn.TILE_SIZE), (
            f"a seam at row {seam.row} of {self.window_len} cannot be mended for K={self.num_levels}: the "
            f"splice needs K < row < window_len and K <= {ttnn.TILE_SIZE} (one tile of the neighbour's rows)"
        )
        self._drop_spliced()
        self._seam = seam

    def clear_seam(self) -> None:
        self.set_seam(None)

    def clear_rows(self, keep_mask: ttnn.Tensor) -> None:
        """Multiply the union by ``[sp, 1, U, H/tp]`` ``keep_mask`` (zeroing the generation rows)."""
        self._apply(lambda src: ttnn.multiply(src, keep_mask))

    def add_patch(self, select: ttnn.Tensor, embeddings: ttnn.Tensor) -> None:
        """Add ``select @ embeddings`` into the union: ``[sp, 1, U, 32*sp] @ [1, 1, 32*sp, H/tp]``.

        ``select`` is one-hot, so this writes one embedding row and leaves every other row as it was.
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
        self._seam = None
        for name in ("_patched", "_rows", "_seam_rows"):
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
        """Replace the union with ``fn(union)``, freeing what it replaces and the stale ROW_MAJOR copy."""
        src, temp = self._current()
        out = fn(src)
        if temp or src is self._patched:
            ttnn.deallocate(src)
        self._patched = out
        if self._rows is not None:
            ttnn.deallocate(self._rows)
            self._rows = None
        self._drop_spliced()

    def _row_major(self) -> ttnn.Tensor:
        """ROW_MAJOR copy of the joined union, materialized once and reused until invalidated.

        A window starts at row ``k``, never a tile boundary, and ``ttnn.slice`` only cuts tiles.
        """
        if self._rows is None:
            joined, temp = self._current()
            self._rows = ttnn.to_layout(joined, ttnn.ROW_MAJOR_LAYOUT)
            if temp:
                ttnn.deallocate(joined)
        return self._rows

    def _spliced(self) -> ttnn.Tensor:
        """The seam's :meth:`MTPSeam.splice_rows` of the CURRENT union, rebuilt after every patch --
        generation may write the neighbour's leading rows."""
        if self._seam_rows is None:
            assert self._parts, "union embedding already deallocated"
            src = self._patched if self._patched is not None else self._parts[0]
            self._seam_rows = self._seam.splice_rows(src)
        return self._seam_rows

    def _drop_spliced(self) -> None:
        if self._seam_rows is not None:
            ttnn.deallocate(self._seam_rows)
            self._seam_rows = None


class MTPDeviceGeneration:
    """Everything the last chunk needs to fill the positions the prompt does not reach.

    ``keep_mask`` zeroes the rows generation will write, ``selects[k]`` places level ``k``'s token at
    global position ``actual_end + k``, and ``embed_fn`` is the lm_head/argmax/embed/gather chain.
    """

    def __init__(self, keep_mask: ttnn.Tensor, selects: list, embed_fn):
        self.keep_mask = keep_mask
        self.selects = list(selects)
        self.embed_fn = embed_fn
        assert self.selects, "generation needs one selector per level"

    def deallocate(self) -> None:
        for t in [self.keep_mask, *self.selects]:
            if t is not None:
                ttnn.deallocate(t)
        self.selects = []


class MTPDeviceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable, sourced entirely on device.

    Levels below ``provided_levels`` slice the union the socket delivered; at and above it the
    generation chain runs on ``H^k`` and patches the union first. The ids never come back to host.
    """

    def __init__(
        self,
        union: MTPUnionEmbedding,
        generation: "Optional[MTPDeviceGeneration]" = None,
        provided_levels: int = 0,
    ):
        self.union = union
        self.generation = generation
        self.num_levels = union.num_levels
        self.provided_levels = int(provided_levels)
        assert (
            0 <= self.provided_levels <= self.num_levels
        ), f"provided_levels {self.provided_levels} outside [0, {self.num_levels}]"
        assert (generation is None) == (self.provided_levels == self.num_levels), (
            "generation must be present exactly when some level has to produce its own token: "
            f"provided_levels={self.provided_levels} of {self.num_levels}, generation="
            f"{'set' if generation is not None else 'None'}"
        )
        self._next_level = self.provided_levels

    @property
    def generated_tokens(self) -> list:
        """Always empty: the generated ids never leave the device. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed):
        assert 0 <= k < self.num_levels, f"level {k} out of range [0, {self.num_levels})"
        if self.generation is not None and k >= self.provided_levels:
            assert k == self._next_level, f"generation must run levels in order; expected {self._next_level}, got {k}"
            self._next_level += 1
            if k == self.provided_levels:
                self.union.clear_rows(self.generation.keep_mask)
            gathered = self.generation.embed_fn(prev_normed)
            self.union.add_patch(self.generation.selects[k], gathered)
            ttnn.deallocate(gathered)
        return self.union.window(k + 1)
