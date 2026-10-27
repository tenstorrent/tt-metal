# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Draft-token sources for the pup's greedy speculative decode (``PUP_SPEC_DRAFTER=mtp|ngram|none``).

Pure host code (no torch ops, no ttnn). All drafters satisfy ``pup_spec_accept.Drafter``: ``start(prompt_ids)`` once
per request, then ``propose(context_ids, committed, first_pos)`` once after every seed / verify that does not end the
request, returning up to ``k`` draft ids. Drafts only change WHICH tokens get verified, never the output: acceptance
is exact (``pup_spec_accept.accept_block``).

- :class:`NGramDrafter`: prompt lookup over ``prompt + generated`` (an offline study's best general variant: n-grams
  tried longest first from ``n_max`` down to ``n_min`` = 2, no unigram fallback, most recent earlier match). Image /
  video placeholder ids never take part in a match and cut a proposal short.
- :class:`MtpDrafter`: the device MTP head through ``PupSpecEngine.propose`` (which also keeps the MTP KV cache in
  step with the committed tokens, hence the exactly-once rule).
- :class:`NoDrafter`: zero drafts every step (the verify-formulated greedy reference; measures verify cost per token).
"""

from __future__ import annotations

from typing import Sequence

DRAFTERS = ("mtp", "ngram", "none")


class NGramDrafter:
    """Incremental prompt-lookup index. An n-gram ending at context index ``i`` is indexed once its continuation
    ``ctx[i + 1]`` exists; a later occurrence replaces an earlier one (most recent match). ``source="prompt"``
    indexes only n-grams whose continuation lies inside the prompt (the speed300 best variant)."""

    def __init__(self, k: int, n_max: int = 4, n_min: int = 2, exclude_ids: Sequence[int] = (), source: str = "all"):
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not 1 <= n_min <= n_max:
            raise ValueError(f"need 1 <= n_min <= n_max, got {n_min}, {n_max}")
        if source not in ("all", "prompt"):
            raise ValueError(f"source must be 'all' or 'prompt', got {source!r}")
        self.k = int(k)
        self.ns = tuple(range(int(n_max), int(n_min) - 1, -1))
        self.exclude = frozenset(int(t) for t in exclude_ids)
        self.source = source
        self._ctx: list[int] = []
        self._idx: dict[int, dict[tuple, int]] = {n: {} for n in self.ns}
        self._prompt_len = 0

    def start(self, prompt_ids: Sequence[int]) -> None:
        self._ctx = []
        self._idx = {n: {} for n in self.ns}
        self._prompt_len = len(prompt_ids)
        self._extend(prompt_ids)

    def _extend(self, toks: Sequence[int]) -> None:
        ctx = self._ctx
        for t in toks:
            ctx.append(int(t))
            i = len(ctx) - 2  # n-grams ending at i now have their continuation ctx[i + 1]
            if i < 0:
                continue
            if self.source == "prompt" and i > self._prompt_len - 2:
                continue
            if ctx[i + 1] in self.exclude:
                continue
            for n in self.ns:
                s = i - n + 1
                if s < 0:
                    continue
                key = tuple(ctx[s : i + 1])
                if self.exclude and not self.exclude.isdisjoint(key):
                    continue
                self._idx[n][key] = i

    def propose(self, context_ids: Sequence[int], committed: Sequence[int] = (), first_pos: int = 0) -> list[int]:
        """Up to ``k`` ids following the most recent earlier occurrence of the longest matching suffix of
        ``context_ids`` (= prompt + generated so far; the drafter extends its index with the new tail)."""
        n_had = len(self._ctx)
        if len(context_ids) < n_had or (n_had and int(context_ids[n_had - 1]) != self._ctx[-1]):
            raise ValueError("context is not an extension of the indexed one: call start() for a new request")
        self._extend(context_ids[len(self._ctx) :])
        ctx = self._ctx
        L = len(ctx)
        for n in self.ns:
            if L < n:
                continue
            key = tuple(ctx[L - n :])
            p = self._idx[n].get(key)
            if p is None:
                continue
            out = []
            for t in ctx[p + 1 : p + 1 + self.k]:
                if t in self.exclude:
                    break
                out.append(t)
            return out
        return []


class MtpDrafter:
    """The device MTP head: ``engine.propose(committed, first_pos)`` (K ids, or [] near the KV capacity)."""

    def __init__(self, engine):
        self.engine = engine
        self.k = int(engine.K)

    def start(self, prompt_ids: Sequence[int]) -> None:
        pass  # the engine's prefill already warmed the MTP prompt KV and the anchor row

    def propose(self, context_ids: Sequence[int], committed: Sequence[int], first_pos: int) -> list[int]:
        return [int(t) for t in self.engine.propose([int(t) for t in committed], int(first_pos))][: self.k]


class NoDrafter:
    def __init__(self, k: int):
        self.k = int(k)

    def start(self, prompt_ids: Sequence[int]) -> None:
        pass

    def propose(self, context_ids: Sequence[int], committed: Sequence[int] = (), first_pos: int = 0) -> list[int]:
        return []


def make_drafter(
    kind: str,
    *,
    k: int,
    engine=None,
    exclude_ids: Sequence[int] = (),
    n_max: int = 4,
    n_min: int = 2,
    source: str = "all",
):
    if kind == "mtp":
        if engine is None:
            raise ValueError("the mtp drafter needs the spec engine")
        return MtpDrafter(engine)
    if kind == "ngram":
        return NGramDrafter(k, n_max=n_max, n_min=n_min, exclude_ids=exclude_ids, source=source)
    if kind == "none":
        return NoDrafter(k)
    raise ValueError(f"unknown drafter {kind!r}; expected one of {DRAFTERS}")


__all__ = ["DRAFTERS", "MtpDrafter", "NGramDrafter", "NoDrafter", "make_drafter"]
