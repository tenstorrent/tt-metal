# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DS41F-0073 marked decoder chunks: the split rule of tt-d-gen's ``pd_decoder_cone`` (both workers of a P/D pair:
``PrefixIndexer::decoder_prefix_cap`` / ``decoder_chunk_start``), mirrored in plain Python (no ttnn) for the runtime's check
and the host tests. ``LASTCHUNK_DECODER.md`` (this directory) has the design.

The decode ring's state after a prompt is exact once every decoder layer's window is: in the decoder (20..39) the only
cross-position paths are each layer's window (127 positions back) and layer 20's entries / index keys, which the encoder output
fixes. So a decoder started at ``c`` from zero windows is exact from ``c + 20 x 127 + 1`` on (``CONE``), at every layer -- the
same argument that sizes today's ring replay (pd_migrate_tail 2541, disagg_prefill_plan.md). The prefill runs the decoder on
the chunks covering the migrated prefix's last ``CONE`` positions (``decoder_start``); the decode side replays only the rest of
the prompt (at least ``tail`` = 1 token: the last prompt token yields the first output token).
"""

from __future__ import annotations

CONE = 20 * 127 + 1  # 2541: layers 20..39, each reaching 127 positions back, +1 for the position itself


def migrated_prefix_cap(prompt_len: int, tail: int, align: int, max_tokens: int = 0) -> int:
    """Today's rule (``PrefixIndexer::migrated_prefix_cap`` with a non-zero tail): floor(S - tail, align), at most
    floor(max_tokens, align); 0 = the prompt is served by the decode side alone."""
    if prompt_len <= tail:
        return 0
    a = align or 1
    n = (prompt_len - tail) // a * a
    if max_tokens and n > max_tokens:
        n = max_tokens // a * a
    return n if n >= a else 0


def decoder_prefix_cap(
    prompt_len: int,
    *,
    chunk: int,
    cone: int = CONE,
    tail: int = 1,
    align: int = 128,
    max_tokens: int = 0,
    single_chunk: bool = False,
) -> int:
    """The migrated prefix with the decoder in the prefill: floor(S - tail, align) (tail >= 1), at most floor(max_tokens,
    align). ``single_chunk`` (policy L): when the prefix's last chunk is shorter than the cone, cut the prefix back to that
    chunk's start, so exactly one (full) chunk runs the decoder and the decode side replays the cut part (< cone + align).
    Only with chunk >= cone (else no single chunk holds a cone: the rule is M's). Policy M (False): keep the longest prefix;
    the decoder then runs on up to two chunks (chunk >= cone)."""
    t = max(1, int(tail))
    if prompt_len <= t:
        return 0
    a = align or 1
    n = (prompt_len - t) // a * a
    if max_tokens and n > max_tokens:
        n = max_tokens // a * a
    if single_chunk and chunk >= cone and n > chunk:
        last = n - (n - 1) // chunk * chunk  # (0, chunk]
        if last < cone:
            n -= last
    return n if n >= a else 0


def decoder_start(prefix: int, chunk: int, cone: int = CONE) -> int:
    """The start of the first decoder chunk for a ``prefix``-token burst (chunks at multiples of ``chunk`` from 0): the chunk
    holding position prefix - cone (0 when the prefix is no longer than the cone)."""
    return 0 if prefix <= cone else (prefix - cone) // chunk * chunk


def is_decoder_chunk(end: int, prefix: int, cone: int = CONE) -> bool:
    """tt-d-gen's mark (slot word bit 28): the chunk ending at ``end`` overlaps the prefix's last ``cone`` positions."""
    return end > prefix - cone


def cone_exact(prefix: int, start, cone: int = CONE) -> bool:
    """Is the export at ``prefix`` exact when the decoder started at ``start`` (None = never)?"""
    return start is not None and (start == 0 or prefix - start >= cone)
