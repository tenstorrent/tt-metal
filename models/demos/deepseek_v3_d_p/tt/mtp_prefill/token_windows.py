# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Token windows for GLM-5.2 MTP prefill (issue #53533).

MTP level ``k`` at position ``p`` consumes ``embed(t_{p+k})``, so every chunk needs ``K`` tokens
from past its own right edge, and the LAST chunk of a request needs ``K`` tokens that do not exist
yet. This module owns those two problems and nothing else.

    GEN_SLOT             sentinel for a token that has not been generated yet
    mtp_extended_stream  one chunk's C ids  -> its C+K stream    <- what a serving path calls
    mtp_chunk_stream     whole prompt + idx -> the same          <- offline/test convenience
    MTPEmbedSource       C+K stream         -> level k's embedded window, one call per level

The stream is indexed by chunk-local POSITION, so level ``k``'s window is the plain slice
``stream[k+1 : k+1+C]``. That is the whole point: no per-level index arithmetic left to get wrong.

**Shifting happens on the host, on ids, before sharding.** Sharding is a fixed row -> position
permutation applied to the window's *contents*, so applying it to an already-shifted window lands
``t_{p+k}`` on the row whose hidden sits at ``p``. Shifting after sharding would instead need an SP
ring-shift, because chip ``c``'s last ``k`` rows want chip ``c+1``'s first ``k`` tokens. That claim
is measured on the mesh by ``tests/mtp_prefill/test_mtp_device_windows.py``.

**The sequence axis does not grow.** ``+K`` is a token lookahead, not extra rows of compute: every
level writes KV at position ``p`` with the trunk's rope and actual_start/end, every window is ``C``
long, and every activation stays ``[1, 1, L, H/tp]``. Only the embedding fed in is shifted.

Pure stdlib -- no ``ttnn``, no ``torch``. Device work reaches it only through ``embed_fn``.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

GEN_SLOT = -1
"""Sentinel for a last-chunk position whose token has not been generated yet.

It never reaches the device: :meth:`MTPEmbedSource.window` resolves every slot below the last real
token from the generated list and rewrites any still-unresolved slot (always a pad row) to
``pad_token``. A sentinel reaching ``TtParallelEmbedding`` would index the vocab table at -1.
"""


def mtp_extended_stream(
    chunk_tokens: Sequence[int],
    num_levels: int,
    *,
    real_len: Optional[int] = None,
    lookahead: Optional[Sequence[int]] = None,
    pad_token: int = 0,
) -> list[int]:
    """One chunk's ``C`` ids -> its ``C + K`` stream, indexed by chunk-local position.

    Interior chunk: pass the next chunk's first ``K`` ids as ``lookahead``.
    Last chunk:     pass ``lookahead=None``; the ``K`` positions after the last real token become
    :data:`GEN_SLOT`, filled in by :class:`MTPEmbedSource` from each level's own lm_head.

    Args:
        chunk_tokens: this chunk's ``C`` ids for ``[s, s+C)``. Entries at or after ``real_len`` are
            the chunk's own padding and are replaced by ``pad_token``.
        num_levels: ``K``.
        real_len: this chunk's real-token count (``actual_end - actual_start``). Defaults to ``C``.
        lookahead: positions ``[s+C, s+C+K)``, or ``None`` for the last chunk of a request.
        pad_token: id written into pad positions. Any in-vocab id works -- pad rows sit past
            ``actual_end``, where the trunk's own hidden and KV are already garbage.

    Returns:
        ``C + K`` ids, indexed by chunk-local position.
    """
    c = len(chunk_tokens)
    k = int(num_levels)
    assert k >= 1, f"num_levels must be >= 1, got {k}"
    real = c if real_len is None else int(real_len)
    assert 0 <= real <= c, f"real_len {real} out of range for a chunk of {c}"

    if lookahead is None:
        # Last chunk: [ prompt | K generation slots | pad ]. Length is C + K for any real_len,
        # including real_len == C (a prompt that is an exact multiple of the chunk size).
        stream = list(chunk_tokens[:real]) + [GEN_SLOT] * k + [pad_token] * (c - real)
    else:
        la = list(lookahead)
        assert real == c, (
            f"an interior chunk is fully real, got real_len={real} for a chunk of {c}; "
            "pass lookahead=None for the last chunk of a request"
        )
        assert len(la) >= k, f"lookahead must supply at least {k} ids, got {len(la)}"
        stream = list(chunk_tokens) + la[:k]

    assert len(stream) == c + k, f"extended stream is {len(stream)}, expected {c + k}"
    return stream


def mtp_chunk_stream(
    all_tokens: Sequence[int],
    chunk_idx: int,
    chunk_size: int,
    num_levels: int,
    *,
    pad_token: int = 0,
) -> tuple[list[int], int]:
    """A whole prompt + a chunk index -> that chunk's ``(stream, real_len)``.

    Convenience over :func:`mtp_extended_stream`: it picks interior-vs-last for you. A serving path
    that receives one chunk at a time calls that function directly instead.

    Interior/last is the one choice a caller must not get wrong, and it is invisible in every
    per-chunk check -- the stream is still ``C+K`` long and every window still ``C`` wide. It shows
    up only as level ``k`` reading the wrong token on a chunk's last ``k`` rows.

    Args:
        all_tokens: the request's REAL prompt ids, unpadded. ``len(all_tokens)`` is ``P``.
        chunk_idx: which ``chunk_size``-sized chunk of them to build.
        chunk_size: ``C``, the padded chunk length.
        num_levels: ``K``.
        pad_token: id written into pad positions.

    Returns:
        ``(stream, real_len)`` -- ``C + K`` ids indexed by chunk-local position, and this chunk's
        real-token count, which is its ``actual_isl``.
    """
    total = len(all_tokens)
    k = int(num_levels)
    c = int(chunk_size)
    s = int(chunk_idx) * c
    assert 0 <= s < total, f"chunk {chunk_idx} starts at {s}, past the {total} real tokens"

    real = min(c, total - s)
    chunk = list(all_tokens[s : s + real]) + [pad_token] * (c - real)

    if s + c >= total:
        return mtp_extended_stream(chunk, k, real_len=real, lookahead=None, pad_token=pad_token), real

    tail = total - (s + c)
    assert tail >= k, (
        f"chunk {chunk_idx} is interior but only {tail} real token(s) follow it, fewer than K={k}: "
        f"its level-{k} window reaches position {s + c + k - 1}, past the prompt end {total}, so "
        f"those rows want tokens generated from a hidden that chunk {chunk_idx + 1} has not produced "
        "yet -- chunks run in order, so this is a causality problem, not bookkeeping. Rebalance the "
        f"last two chunks so the final one carries at least K={k} real tokens."
    )
    return mtp_extended_stream(chunk, k, lookahead=list(all_tokens[s + c : s + c + k]), pad_token=pad_token), real


class MTPEmbedSource:
    """``TtMTPPredictor.forward``'s ``embeds`` callable: level ``k`` -> its embedded window.

    Call it once per level, in order. Each call does three things:

      1. SLICE  window ``k+1`` out of the C+K stream: ``ext[k+1 : k+1+C]``.
      2. FILL   on the LAST chunk only, resolve the :data:`GEN_SLOT`s -- generating ``t_{P+k}`` by
                running ``next_token_fn`` on the hidden it was just handed.
      3. EMBED  hand the resulting ``C`` ids to ``embed_fn``.

    On an interior chunk step 2 is skipped entirely: every window is a pure prompt slice,
    ``next_token_fn`` is never called, and no host round trip happens.

    **Why a callable and not a list of K tensors.** On the last chunk level ``k``'s INPUT depends on
    level ``k-1``'s OUTPUT: ``t_{P+k} = argmax lm_head(H^k)`` at the last real row. Nothing can be
    materialized before the loop runs. Cost is one lm_head round trip per level -- ``K`` of them, or
    ``K-1`` when ``seed_token`` is given -- once per request, on the final chunk only.

    **Which rows are generated.** Window ``shift`` row ``j`` holds global position ``s + shift + j``.
    Real prompt covers ``j < real_len - shift``; rows ``[real_len - shift, real_len)`` -- exactly
    ``shift`` of them -- are the generated tokens ``t_P .. t_{P+shift-1}``; rows ``>= real_len`` are
    pad, as they are for the trunk.

    Args:
        extended: this chunk's stream from :func:`mtp_extended_stream`.
        chunk_size: ``C``.
        num_levels: ``K``.
        embed_fn: ``list[int] -> ttnn.Tensor``. Shards, uploads and embeds one window, and on the
            chunk starting at absolute position 0 zeroes row 0. Injected so this module stays
            device-free; the transformer passes its own ``_mtp_embed_window``.
        next_token_fn: ``H^k -> int``, the greedy token at the last real row. Required whenever the
            stream carries generation slots. The transformer passes ``_mtp_next_token_fn``.
        real_len: this chunk's real-token count; where the generation slots start. Defaults to ``C``.
        seed_token: ``t_P``, if the caller already sampled it off the trunk (the transformer does,
            at ``tt_prefill_transformer.py:636``). Saves one redundant 32-row lm_head. Pass ``None``
            to have level 1 derive it from ``H^0`` instead -- same tensor, same row, so the two
            agree by construction.
        pad_token: id substituted for any generation slot still unresolved when a window is built.
            Such a slot is always a pad row.
    """

    def __init__(
        self,
        extended: Sequence[int],
        chunk_size: int,
        num_levels: int,
        *,
        embed_fn: Callable[[list[int]], object],
        next_token_fn: Optional[Callable[[object], int]] = None,
        real_len: Optional[int] = None,
        seed_token: Optional[int] = None,
        pad_token: int = 0,
    ):
        self.chunk_size = int(chunk_size)
        self.num_levels = int(num_levels)
        assert self.num_levels >= 1, f"num_levels must be >= 1, got {self.num_levels}"
        assert len(extended) == self.chunk_size + self.num_levels, (
            f"extended stream is {len(extended)}, expected {self.chunk_size} + {self.num_levels}; "
            "build it with mtp_extended_stream()"
        )
        self._ext = list(extended)
        self.real_len = self.chunk_size if real_len is None else int(real_len)
        assert 0 <= self.real_len <= self.chunk_size
        self.embed_fn = embed_fn
        self.next_token_fn = next_token_fn
        self.seed_token = None if seed_token is None else int(seed_token)
        self.pad_token = int(pad_token)
        self._gen: list[int] = []

        self.generating = GEN_SLOT in self._ext
        if self.generating:
            slots = self._ext[self.real_len : self.real_len + self.num_levels]
            assert slots == [GEN_SLOT] * self.num_levels, (
                f"generation slots must be the {self.num_levels} positions right after the last real "
                f"token ({self.real_len}); got {slots}"
            )
            # Level 1 can be seeded, but every later level needs its predecessor's lm_head.
            if self.num_levels > 1 or self.seed_token is None:
                assert next_token_fn is not None, (
                    "the last chunk needs next_token_fn to produce t_P..t_{P+K-1}: level k's window "
                    "requires t_{P+k-1} = argmax lm_head(H^{k-1}) at the last real row"
                )

    @property
    def generated_tokens(self) -> list[int]:
        """``[t_P, t_{P+1}, ...]`` produced so far. Always empty on an interior chunk."""
        return list(self._gen)

    def window(self, shift: int) -> list[int]:
        """Window ``shift`` as ``C`` in-vocab ids, generation slots resolved."""
        assert 0 <= shift <= self.num_levels, f"shift {shift} out of range [0, {self.num_levels}]"
        ext = self._ext
        if self.generating:
            assert len(self._gen) >= shift, (
                f"window {shift} needs {shift} generated token(s) for rows "
                f"[{self.real_len - shift}, {self.real_len}), have {len(self._gen)}"
            )
            ext = list(ext)
            for i, tok in enumerate(self._gen):
                ext[self.real_len + i] = int(tok)
            ext = [self.pad_token if t == GEN_SLOT else t for t in ext]
        return ext[shift : shift + self.chunk_size]

    def __call__(self, k: int, prev_normed):
        """``TtMTPPredictor``'s hook: level ``k`` (0-based) wants the shift-``k+1`` embedding.

        ``prev_normed`` is ``H^k`` -- the trunk output after ``model.norm`` at ``k=0``, and level
        ``k``'s ``shared_head.norm`` output after that. On the last chunk it is exactly the tensor
        whose lm_head yields ``t_{P+k}``, the one token this level's window is still missing.
        """
        assert 0 <= k < self.num_levels, f"level {k} out of range [0, {self.num_levels})"
        if self.generating:
            assert len(self._gen) == k, (
                f"level {k} called with {len(self._gen)} generated token(s); MTPEmbedSource must be "
                "driven in level order, once each -- it is stateful across the recurrence"
            )
            if k == 0 and self.seed_token is not None:
                self._gen.append(self.seed_token)
            else:
                self._gen.append(int(self.next_token_fn(prev_normed)))
        return self.embed_fn(self.window(k + 1))
