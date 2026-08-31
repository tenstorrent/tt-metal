# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device MTP shift-windows for the prefill runner (GLM-5.2, issue #53533).

``token_windows.py`` solves the same problem on the HOST: it slices python lists and hands each
window to an ``embed_fn`` that uploads it. That is right for a test driving the transformer
directly, and wrong for the serving runner, where the tokens arrive over the H2D socket and never
touch host memory. This module is the device twin -- same ``(k, H^k) -> embedding`` contract that
``TtMTPPredictor.forward`` consumes, no host round trip anywhere in it.

Three things make that possible.

**1. The H2D shard OVERLAPS.** The trunk's input is ``[sp, 1, L]`` with chip ``c`` holding positions
``[c*L, (c+1)*L)``. With ``K`` MTP levels the producer instead sends chip ``c`` the slice
``[c*L, c*L + L + K)`` -- ``K`` tokens of overlap with chip ``c+1``. MTP level ``k`` wants position
``p + k + 1`` on the row whose hidden sits at ``p``, so on chip ``c`` it wants
``[c*L + k+1, c*L + k+1 + L)``, which is the SAME local slice ``[k+1, k+1+L)`` on every chip. One
uniform ``ttnn.slice`` on a ROW_MAJOR uint32 tensor, no SP ring-shift, no cross-chip rotation. The
trunk's own input is the local slice ``[0, L)`` of that same tensor.

**2. Token ids cross the D2D socket as base-256 digits.** The pipeline socket carries bf16, and bf16
holds integers exactly only to 256 -- a 154880-entry vocab would be silently rounded. Each id is
carried as three PRE-SCALED digits ``[t%256, 256*((t//256)%256), 65536*(t//65536)]``: each of those
values is exactly representable in bf16 (255*2^8 and 2*2^16 are 8-significant-bit numbers), so
decoding is ``typecast(f32) + sum(dim=-1)`` -- exact, and with no multiply to get wrong. Encoding is
``ttnn.embedding`` against a ``[vocab, 32]`` digit table, i.e. a gather, which is bit-exact by
construction. A matmul-based decode was MEASURED inexact (580/640 rows wrong even at HiFi4 with
fp32 accumulate) and is deliberately not used.

**3. One 32-column block per level.** ``TOKEN_COLS`` is one tile width, so every group boundary in
the packed activation is tile-aligned and the unpack is a plain ``ttnn.slice`` -- the same shape of
operation DFlash's ``_unpack_activation`` already does. ``K`` levels widen the per-chip D2D
activation from ``H/tp`` to ``H/tp + 32K`` (1536 -> 1664 for GLM-5.2 at ``tp=4``, ``K=4``).

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

import torch

import ttnn
from models.demos.common.prefill.runners.runner_utils import MTP_TOKEN_COLS as TOKEN_COLS
from models.demos.common.prefill.runners.runner_utils import mtp_token_block_cols as token_block_cols

DIGIT_BASE = 256
NUM_DIGITS = 3
"""Base-256 digits per id. Three covers vocab < 16.7M; GLM-5.2's is 154880."""

MAX_VOCAB = DIGIT_BASE**NUM_DIGITS

# TOKEN_COLS / token_block_cols are re-exported from the common runner rather than defined here: the
# runner sizes the D2D socket and this module fills it, so a divergence between the two is a silent
# wrong-width transport. One tile width, so only the first NUM_DIGITS columns are ever nonzero.
__all__ = [
    "TOKEN_COLS",
    "token_block_cols",
    "DIGIT_BASE",
    "NUM_DIGITS",
    "MAX_VOCAB",
    "build_digit_table",
    "TokenDigitCodec",
    "MTPTokenWindows",
    "MTPDeviceEmbedSource",
]


def build_digit_table(mesh_device, vocab_size: int) -> ttnn.Tensor:
    """The replicated ``[vocab_size, TOKEN_COLS]`` bf16 pre-scaled digit table.

    Row ``t`` is ``[t % 256, 256 * (t//256 % 256), 65536 * (t//65536), 0, ...]``. Every entry has at
    most 8 significant bits, so it survives bf16 exactly, and the three sum to ``t``.
    """
    assert 0 < vocab_size <= MAX_VOCAB, f"vocab_size {vocab_size} outside [1, {MAX_VOCAB}] for {NUM_DIGITS} digits"
    t = torch.arange(vocab_size, dtype=torch.int64)
    tbl = torch.zeros(vocab_size, TOKEN_COLS, dtype=torch.float32)
    for d in range(NUM_DIGITS):
        tbl[:, d] = ((t // DIGIT_BASE**d) % DIGIT_BASE).float() * float(DIGIT_BASE**d)
    return ttnn.from_torch(
        tbl,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class TokenDigitCodec:
    """Encodes/decodes token ids to and from the bf16 digit block the D2D socket carries.

    Built lazily and only where the transport actually needs it: a single-rank runner reads its
    windows straight off the H2D uint32 tensor and never allocates the table (~10 MB per chip).
    """

    def __init__(self, mesh_device, vocab_size: int):
        self.mesh_device = mesh_device
        self.vocab_size = int(vocab_size)
        self._table: Optional[ttnn.Tensor] = None

    @property
    def table(self) -> ttnn.Tensor:
        if self._table is None:
            self._table = build_digit_table(self.mesh_device, self.vocab_size)
        return self._table

    def encode(self, ids: ttnn.Tensor) -> ttnn.Tensor:
        """``[.., 1, L]`` uint32 ROW_MAJOR ids -> ``[.., 1, L, TOKEN_COLS]`` bf16 TILE digits."""
        return ttnn.unsqueeze_to_4D(ttnn.embedding(ids, self.table, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16))

    def decode(self, group: ttnn.Tensor) -> ttnn.Tensor:
        """``[.., 1, L, TOKEN_COLS]`` bf16 digits -> ``[.., 1, L]`` uint32 ROW_MAJOR ids.

        The digits are pre-scaled, so this is a reduction and two layout casts -- no multiply. The
        29 padding columns are zero and contribute nothing; the partial sums stay under 2^18, well
        inside f32's exact-integer range, so the reduction is exact rather than merely accurate.
        """
        f32 = ttnn.typecast(group, ttnn.float32)
        summed = ttnn.sum(f32, dim=-1, keepdim=True)
        ttnn.deallocate(f32)
        u32 = ttnn.typecast(summed, ttnn.uint32)
        ttnn.deallocate(summed)
        rm = ttnn.to_layout(u32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(u32)
        # Drop the trailing 1 by rebuilding the shape from the tensor's own dims: correct whether
        # `.shape` reports the mesh-global or the per-chip extent, which differ for a socket tensor.
        ids = ttnn.reshape(rm, ttnn.Shape(list(rm.shape)[:-1]))
        return ids

    def deallocate(self) -> None:
        if self._table is not None:
            ttnn.deallocate(self._table)
            self._table = None


class MTPTokenWindows:
    """One chunk's MTP token windows, however they reached this rank.

    Two sources, one interface:

    * :meth:`from_tokens` -- the FIRST rank, holding the H2D socket's ``[sp, 1, L+K]`` uint32 tensor.
      A window is a ``ttnn.slice`` of it, and so is the trunk's own input.
    * :meth:`from_block` -- any downstream rank, holding the ``[.., 1, L, 32K]`` bf16 digit block
      that came in packed alongside the hidden. A window is that block's group ``k``, decoded.

    An INTERMEDIATE rank never runs MTP, so it never decodes: :meth:`block` hands its received block
    straight back for re-packing, and the ids stay in digit form for the whole traverse.
    """

    def __init__(self, *, num_levels: int, window_len: int, tokens=None, block=None):
        assert (tokens is None) != (block is None), "MTPTokenWindows takes exactly one of tokens/block"
        self.num_levels = int(num_levels)
        self.window_len = int(window_len)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        self._tokens = tokens
        self._block = block

    @classmethod
    def from_tokens(cls, tokens: ttnn.Tensor, num_levels: int, window_len: int) -> "MTPTokenWindows":
        """Take ownership of the H2D chunk tensor: ``[sp, 1, window_len + num_levels]`` uint32 RM."""
        return cls(num_levels=num_levels, window_len=window_len, tokens=tokens)

    @classmethod
    def from_block(cls, block: ttnn.Tensor, num_levels: int, window_len: int) -> "MTPTokenWindows":
        """Take ownership of a received digit block: ``[.., 1, window_len, 32*num_levels]`` bf16."""
        return cls(num_levels=num_levels, window_len=window_len, block=block)

    @property
    def has_tokens(self) -> bool:
        """True when the ids are here as integers (first rank) rather than as digits."""
        return self._tokens is not None

    def trunk_tokens(self) -> ttnn.Tensor:
        """The trunk's own ``[sp, 1, window_len]`` input: local slice ``[0, window_len)``.

        Same rows the non-MTP runner would have received, so the trunk's forward is unchanged.
        """
        assert self._tokens is not None, "trunk_tokens() needs the uint32 chunk (first rank only)"
        return self._slice_tokens(0)

    def window(self, shift: int) -> ttnn.Tensor:
        """MTP window ``shift`` (1..K) as ``[.., 1, window_len]`` uint32 RM ids. Caller frees it."""
        assert 1 <= shift <= self.num_levels, f"shift {shift} out of range [1, {self.num_levels}]"
        if self._tokens is not None:
            return self._slice_tokens(shift)
        raise AssertionError("window() on a digit-block source needs a codec; use window_from_block()")

    def window_from_block(self, shift: int, codec: TokenDigitCodec) -> ttnn.Tensor:
        """Same as :meth:`window`, decoding group ``shift - 1`` when the ids arrived as digits."""
        if self._tokens is not None:
            return self.window(shift)
        assert 1 <= shift <= self.num_levels, f"shift {shift} out of range [1, {self.num_levels}]"
        s = list(self._block.shape)
        lo = (shift - 1) * TOKEN_COLS
        group = ttnn.slice(self._block, [0, 0, 0, lo], [s[0], s[1], s[2], lo + TOKEN_COLS])
        ids = codec.decode(group)
        ttnn.deallocate(group)
        return ids

    def block(self, codec: TokenDigitCodec) -> ttnn.Tensor:
        """The ``[.., 1, window_len, 32*K]`` bf16 digit block to pack next to the hidden.

        Encodes the K windows on the first rank; passes the received block through (and gives up
        ownership of it) on an intermediate rank, so a middle stage costs no encode and no decode.
        """
        if self._block is not None:
            out, self._block = self._block, None
            return out
        groups = []
        for k in range(self.num_levels):
            ids = self.window(k + 1)
            groups.append(codec.encode(ids))
            ttnn.deallocate(ids)  # encode() gathers out of it and does not consume it
        if len(groups) == 1:
            return groups[0]
        packed = ttnn.concat(groups, dim=3)
        for g in groups:
            ttnn.deallocate(g)
        return packed

    def deallocate(self) -> None:
        for name in ("_tokens", "_block"):
            t = getattr(self, name)
            if t is not None:
                ttnn.deallocate(t)
                setattr(self, name, None)

    def _slice_tokens(self, shift: int) -> ttnn.Tensor:
        s = list(self._tokens.shape)
        assert len(s) == 3, f"expected a [sp, 1, L+K] token tensor, got shape {s}"
        assert s[-1] >= self.window_len + self.num_levels, (
            f"H2D row is {s[-1]} ids, needs at least window_len + K = {self.window_len} + "
            f"{self.num_levels}. The producer and the runner must agree on PREFILL_MTP_LEVELS."
        )
        return ttnn.slice(self._tokens, [0, 0, shift], [s[0], s[1], shift + self.window_len])


class MTPDeviceEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable, sourced entirely on device.

    Drop-in for :class:`~models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows.MTPEmbedSource`:
    same ``(k, H^k) -> ttnn.Tensor`` signature, same ``generated_tokens`` property. It ignores
    ``H^k`` -- with no host LM-head round trip there is nothing to derive from it -- so
    ``generated_tokens`` is always empty and the last chunk's trailing rows carry the producer's pad
    ids. See this module's header for exactly which rows that is.
    """

    def __init__(self, windows: MTPTokenWindows, embed_fn, codec: Optional[TokenDigitCodec] = None):
        self.windows = windows
        self.embed_fn = embed_fn
        self.codec = codec
        self.num_levels = windows.num_levels
        assert windows.has_tokens or codec is not None, "a digit-block source needs a codec to decode its windows"

    @property
    def generated_tokens(self) -> list:
        """Always empty: this source never runs the LM head. Kept for interface parity."""
        return []

    def __call__(self, k: int, prev_normed):
        assert 0 <= k < self.num_levels, f"level {k} out of range [0, {self.num_levels})"
        ids = self.windows.window_from_block(k + 1, self.codec)
        return self.embed_fn(ids)
