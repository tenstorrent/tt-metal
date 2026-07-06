# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Frozen prompt-prefix KV reuse for DiffusionGemma serving (APC prototype, #47466).

DiffusionGemma serves with a **model-owned contiguous** KV cache
(``tt_model.tt_kv_cache``, one active sequence). Prefill writes the prompt K/V into
positions ``[0:cache_len]`` (causal), and the denoise phase cross-attends to that
frozen ``[0:cache_len]`` prefix (read-only). Because prefill attention is causal,
the K/V written at position ``i`` is a pure function of ``tokens[0:i]`` and the
absolute position ``i`` (RoPE is absolute) — independent of the total prefill
length. So when a new request's *aligned* prompt token span is a byte-identical
leading span of the K/V already resident in the contiguous cache, that span is
**bit-identical** to what a fresh prefill would write, and the prefill forward can
be skipped entirely.

This module is the host-side registry that decides that. It holds **no device
state** — the K/V lives in the model cache; this only tracks *which aligned prompt
tokens the contiguous cache currently holds* so a subsequent request can decide
whether ``[0:new_cache_len]`` is already correct.

Scope (deliberately bit-exact only):

- **reuse** iff ``new_aligned == resident_aligned[:new_cache_len]`` and
  ``new_cache_len <= resident_cache_len``. Covers the exact-full-match and
  aligned-proper-prefix cases (see ``doc/vllm_integration/prefix_cache/README.md``).
- a **partial-prefix miss** (shared prefix but the new prompt extends/differs in the
  suffix) is *not* reusable here: the suffix must cross-attend to the cached prefix
  during prefill = chunked/prefix prefill, which the Gemma4 backbone does not
  support and which the vLLM paged-cache ownership change (#47488) is the real home
  for. Such requests fall back to a full prefill and are counted so the saving a
  paged path *could* have captured is visible.

Enable via ``DG_PREFIX_CACHE=1`` (default OFF). With the flag off — or when no
``PrefixKVCache`` is attached to the session — the prefill path is byte-identical to
the pre-existing one.
"""

from __future__ import annotations

import os
from numbers import Integral
from typing import NamedTuple


def prefix_cache_enabled() -> bool:
    """Return True when ``DG_PREFIX_CACHE`` opts into frozen prompt-prefix reuse."""
    return os.environ.get("DG_PREFIX_CACHE", "0").lower() in ("1", "true", "yes", "on")


class ReusePlan(NamedTuple):
    """Decision for one prefill: whether the resident cache prefix can be reused."""

    reuse: bool  # skip the prefill forward — [0:cache_len] is already correct
    matched_len: int  # longest common aligned-prefix length vs the resident record
    prompt_len: int  # logical prompt length of the incoming request
    cache_len: int  # 32-aligned span the incoming request occupies
    # A shared prefix that could NOT be reused bit-exactly here (suffix differs /
    # extends); a paged / chunked-prefill path (#47488) could reuse ``matched_len``.
    partial_prefix: bool


def _common_prefix_len(a: tuple, b: tuple) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixKVCache:
    """Host-side registry of the aligned prompt tokens the contiguous cache holds.

    Not thread-safe and single-sequence by construction (the contiguous model cache
    backs one active sequence — concurrent batched serving is #47488 + #47557). One
    instance is shared across serving sessions that share a ``tt_model`` so reuse
    works request-to-request.
    """

    def __init__(self):
        self._resident_aligned: tuple[int, ...] | None = None
        self._resident_prompt_len: int | None = None
        self._resident_cache_len: int | None = None
        # Stats (host-only, for evidence / logging).
        self.hits = 0
        self.misses = 0
        self.partial_prefix_misses = 0
        self.tokens_reused = 0
        self.prefill_time_saved_s = 0.0
        # Rolling average of observed real prefill wall-times, used to attribute a
        # per-reuse saving (a reuse does no prefill, so its saving is estimated from
        # what a comparable real prefill costs).
        self._prefill_time_sum_s = 0.0
        self._prefill_time_n = 0

    @property
    def resident_cache_len(self) -> int | None:
        return self._resident_cache_len

    @property
    def avg_prefill_time_s(self) -> float:
        if self._prefill_time_n == 0:
            return 0.0
        return self._prefill_time_sum_s / self._prefill_time_n

    def observe_prefill_time(self, seconds: float) -> None:
        """Record a real prefill wall-time to estimate future reuse savings."""
        if seconds is None or seconds < 0:
            return
        self._prefill_time_sum_s += float(seconds)
        self._prefill_time_n += 1

    @staticmethod
    def _validate_span(prompt_len: int, cache_len: int) -> None:
        for name, value in (("prompt_len", prompt_len), ("cache_len", cache_len)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if cache_len < prompt_len:
            raise ValueError(f"cache_len ({cache_len}) must cover prompt_len ({prompt_len})")

    def plan(self, aligned_tokens, prompt_len: int, cache_len: int) -> ReusePlan:
        """Decide whether ``[0:cache_len]`` is already resident and reusable.

        ``aligned_tokens`` is the padded prompt id sequence (length ``cache_len``)
        exactly as ``prefill_prompt_tokens`` would write it. Reuse requires a
        byte-identical leading span, so the incoming aligned prompt (including any
        32-tile pad) must equal the resident aligned prompt over its whole length.
        """
        aligned = tuple(int(t) for t in aligned_tokens)
        if len(aligned) != cache_len:
            raise ValueError(f"aligned_tokens length {len(aligned)} must equal cache_len {cache_len}")
        self._validate_span(prompt_len, cache_len)

        if self._resident_aligned is None:
            return ReusePlan(False, 0, prompt_len, cache_len, partial_prefix=False)

        matched = _common_prefix_len(aligned, self._resident_aligned)
        reuse = (
            cache_len <= self._resident_cache_len
            and matched == cache_len
            and self._resident_aligned[:cache_len] == aligned
        )
        # A "partial prefix" is a genuine shared prefix (>= one full 32-tile) that we
        # cannot reuse bit-exactly: the incoming prompt extends or differs past the
        # shared span. That is the #47488 / chunked-prefill case.
        partial = (not reuse) and matched >= 32
        return ReusePlan(reuse, matched, prompt_len, cache_len, partial_prefix=partial)

    def record(self, aligned_tokens, prompt_len: int, cache_len: int) -> None:
        """Set the resident state to the current request's own prompt prefix.

        Always the *current* request's ``[0:cache_len]`` span: that is exactly what
        the contiguous cache holds after this prefill, and it is guaranteed intact
        once this request starts committing (commit appends at ``[cache_len:]``).
        """
        aligned = tuple(int(t) for t in aligned_tokens)
        if len(aligned) != cache_len:
            raise ValueError(f"aligned_tokens length {len(aligned)} must equal cache_len {cache_len}")
        self._validate_span(prompt_len, cache_len)
        self._resident_aligned = aligned
        self._resident_prompt_len = prompt_len
        self._resident_cache_len = cache_len

    def note_reuse(self, plan: ReusePlan, *, prefill_time_saved_s: float = 0.0) -> None:
        """Update stats after a reuse and re-anchor the resident record to it."""
        self.hits += 1
        self.tokens_reused += plan.cache_len
        self.prefill_time_saved_s += max(0.0, float(prefill_time_saved_s))

    def note_miss(self, plan: ReusePlan) -> None:
        self.misses += 1
        if plan.partial_prefix:
            self.partial_prefix_misses += 1

    def invalidate(self) -> None:
        """Forget the resident state (e.g. the contiguous cache was reallocated)."""
        self._resident_aligned = None
        self._resident_prompt_len = None
        self._resident_cache_len = None

    def stats(self) -> dict:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "partial_prefix_misses": self.partial_prefix_misses,
            "tokens_reused": self.tokens_reused,
            "prefill_time_saved_s": self.prefill_time_saved_s,
            "resident_cache_len": self._resident_cache_len,
        }
