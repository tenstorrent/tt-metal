"""Per-iteration evidence for the text-correctness gate.

The legacy gate compared the FIRST 32 tokens, returned a single
boolean, and forgot everything between iterations. That hid the
medgemma-4b-it failure mode (first ~36 tokens coherent, then
repeated Devanagari/Chinese garbage for the rest of the 200-token
output) because the gate's window ended just before the collapse.
This module exists to capture the *full* output of every iteration
in enough structured detail that:

1. The :class:`Comparator` can decide ``ok`` based on the *whole*
   output, not just a 32-token slice.
2. :mod:`.diagnose` can spot regime shifts ("token 36 starts
   repeating", "first non-ASCII run begins at token 41") and
   convert them into actionable hypotheses ("KV-cache index off-
   by-one at sliding-window boundary", "RoPE position wraps at
   max_position_embeddings").
3. :mod:`.hypothesis` can compare evidence across iterations to
   tell whether a suspect was actually tested.

What goes into an :class:`TextEvidence`
---------------------------------------
* ``tt_text`` — the FULL decoded text out of the demo (not
  truncated). The medgemma collapse happens at ~char 200; the
  legacy gate truncated to ~120 chars and missed it.
* ``tt_tokens`` — re-tokenized version of ``tt_text`` (so the
  comparator can do token-level alignment).
* ``hf_text``, ``hf_tokens`` — HF CPU reference for the same
  prompt, generated lazily (the dispatcher caches this so we
  don't re-run HF on every iteration).
* ``collapse_position`` — the token index where the output starts
  to look obviously wrong (see :func:`find_first_collapse`).
  ``None`` if no collapse is detected.
* ``regime_shifts`` — list of ``(start_token_idx, kind, detail)``
  tuples documenting WHERE the output changes character. Used by
  :mod:`.diagnose` to map collapse positions to architectural
  features (e.g. sliding-window boundary, RoPE wrap, max_seq_len).
* ``mismatch_segments`` — token-level diff with HF: a list of
  ``(tt_start, tt_end, hf_start, hf_end, kind)`` describing
  insertions, deletions, and substitutions. Used in the repair
  prompt to show the LLM exactly where divergence happens.

The dataclass is intentionally serializable (no numpy / torch /
loguru objects in the payload) so it can be JSON-dumped to the
audit log and replayed in unit tests without a HF mirror.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple


DEFAULT_SCAN_LIMIT = 256


COLLAPSE_REPEAT_WINDOW = 8


COLLAPSE_REPEAT_THRESHOLD = 0.625


NON_ASCII_RUN_THRESHOLD = 0.5


NON_ASCII_RUN_MIN_LEN = 12


_TRIVIAL_TOKEN_CHARS = set(string.whitespace)


CHAR_REPEAT_MIN_LEN = 20


@dataclass
class RegimeShift:
    """A point in the output where the text's character changes.

    Attributes:
      start_token_idx  — first token where the regime applies.
      kind             — short tag identifying the regime
                         (``"repetition"``, ``"non_ascii_run"``,
                         ``"language_switch"``, …). Stable; consumed
                         by :mod:`.diagnose`.
      detail           — free-form human-readable explanation
                         (e.g. ``"7-of-8 token-id 12345"``).
    """

    start_token_idx: int
    kind: str
    detail: str = ""

    def __repr__(self) -> str:
        return f"RegimeShift(start={self.start_token_idx}, " f"kind={self.kind!r}, detail={self.detail!r})"


@dataclass
class TextEvidence:
    """Full, structured record of one demo run's textual output.

    Built by :func:`build_text_evidence`. Consumed by:

    * :class:`text.TextComparator.compare` (the engine="evidence"
      compare step) — to decide pass/fail based on the WHOLE output.
    * :class:`hypothesis.HypothesisState.update_from_evidence` —
      to track regime shifts across iterations.
    * :func:`planner.render_repair_prompt` — to render the
      side-by-side mismatch view and the collapse-position banner.

    Fields:
      tt_text            — full TT-decoded text (un-truncated).
      tt_tokens          — re-tokenized version of ``tt_text``.
      hf_text            — HF reference text (same prompt, same
                            tokenizer; truncated to len of TT).
      hf_tokens          — HF reference token ids.
      collapse_position  — first token index where the output
                            starts to look degenerate, or ``None``
                            if no collapse was detected.
      regime_shifts      — see :class:`RegimeShift`.
      prefix_match_count — number of tokens at the START of the
                            output that match HF exactly. The
                            medgemma case had prefix_match_count
                            ≈ 36 (first 36 tokens fine).
      mismatch_count     — total non-matching tokens within the
                            compared window.
      compared_tokens    — number of tokens actually compared
                            (min(len(tt_tokens), len(hf_tokens),
                            scan_limit)).
      scan_limit         — the configured scan window.
      input_hint         — the prompt the demo saw, propagated
                            from :class:`base.Evidence.input_hint`.
    """

    tt_text: str
    tt_tokens: List[int]
    hf_text: str = ""
    hf_tokens: List[int] = field(default_factory=list)
    collapse_position: Optional[int] = None
    regime_shifts: List[RegimeShift] = field(default_factory=list)
    prefix_match_count: int = 0
    mismatch_count: int = 0
    compared_tokens: int = 0
    scan_limit: int = DEFAULT_SCAN_LIMIT
    input_hint: str = ""

    @property
    def mismatch_ratio(self) -> float:
        if self.compared_tokens <= 0:
            return 0.0
        return self.mismatch_count / self.compared_tokens

    @property
    def collapsed(self) -> bool:
        """Did the output exhibit a detectable collapse anywhere
        within the scan window?"""
        return self.collapse_position is not None

    @property
    def first_regime_shift(self) -> Optional[RegimeShift]:
        if not self.regime_shifts:
            return None
        return min(self.regime_shifts, key=lambda r: r.start_token_idx)

    def summary(self) -> str:
        """One-line summary for log lines."""
        bits = [
            f"compared={self.compared_tokens}",
            f"prefix_match={self.prefix_match_count}",
            f"mismatch={self.mismatch_count}",
        ]
        if self.collapse_position is not None:
            bits.append(f"collapse@tok{self.collapse_position}")
        if self.regime_shifts:
            kinds = ",".join(sorted({r.kind for r in self.regime_shifts}))
            bits.append(f"regimes={kinds}")
        return " ".join(bits)


def find_first_collapse(
    tokens: Sequence[int],
    text: str,
    *,
    repeat_window: int = COLLAPSE_REPEAT_WINDOW,
    repeat_threshold: float = COLLAPSE_REPEAT_THRESHOLD,
) -> Optional[int]:
    """Find the first token index where the output starts to look
    degenerate, or return ``None`` if no collapse is detected.

    Two heuristics, the earlier of which wins:

    * Repetition collapse: there exists an index ``k`` such that
      within ``tokens[k : k + repeat_window]`` at least
      ``repeat_threshold * repeat_window`` of the tokens are the
      same id. Captures the medgemma "stuck on one token"
      pattern.
    * Non-ASCII run: there exists a contiguous stretch of
      ``text`` with at least ``NON_ASCII_RUN_MIN_LEN`` characters
      where ≥ ``NON_ASCII_RUN_THRESHOLD`` are non-printable-ASCII.
      Captures the medgemma "switches to Kannada" pattern. We map
      the char index back to a token index by walking the
      tokenizer's offset_mapping; if we don't have one, we
      approximate as ``int(char_idx / len(text) * len(tokens))``.

    The earliest match wins because if the output starts repeating
    AT the same point it starts speaking Devanagari, we report
    that joint collapse position once (not twice). The detail
    string in the returned :class:`RegimeShift` makes the actual
    cause clear.
    """
    if not tokens:
        return None

    rep_idx: Optional[int] = None
    if len(tokens) >= repeat_window:
        for k in range(len(tokens) - repeat_window + 1):
            window = tokens[k : k + repeat_window]
            cnt = Counter(window)
            most_common_count = cnt.most_common(1)[0][1]
            if most_common_count / repeat_window >= repeat_threshold:
                rep_idx = k
                break

    nonascii_token_idx: Optional[int] = None
    if text:
        run_start: Optional[int] = None
        run_nonascii = 0
        run_total = 0
        for i, ch in enumerate(text):
            is_print_ascii = ch in string.printable
            if not is_print_ascii:
                if run_start is None:
                    run_start = i
                run_nonascii += 1
                run_total += 1
            else:
                if (
                    run_start is not None
                    and run_total >= NON_ASCII_RUN_MIN_LEN
                    and run_nonascii / max(run_total, 1) >= NON_ASCII_RUN_THRESHOLD
                ):
                    break
                run_start = None
                run_nonascii = 0
                run_total = 0

        if (
            run_start is not None
            and run_total >= NON_ASCII_RUN_MIN_LEN
            and run_nonascii / max(run_total, 1) >= NON_ASCII_RUN_THRESHOLD
        ):
            char_idx = run_start
            nonascii_token_idx = int(round(char_idx / max(len(text), 1) * len(tokens)))
            nonascii_token_idx = min(max(nonascii_token_idx, 0), len(tokens) - 1)

    candidates = [x for x in (rep_idx, nonascii_token_idx) if x is not None]
    if not candidates:
        return None
    return min(candidates)


def detect_regime_shifts(
    tokens: Sequence[int],
    text: str,
    *,
    repeat_window: int = COLLAPSE_REPEAT_WINDOW,
    repeat_threshold: float = COLLAPSE_REPEAT_THRESHOLD,
) -> List[RegimeShift]:
    """Return ALL detected regime shifts, not just the earliest.

    Each shift is a :class:`RegimeShift`. Used by :mod:`.diagnose`
    to map specific shifts to architectural features.
    """
    shifts: List[RegimeShift] = []

    if len(tokens) >= repeat_window:
        last_dominant: Optional[int] = None
        for k in range(len(tokens) - repeat_window + 1):
            window = tokens[k : k + repeat_window]
            cnt = Counter(window)
            dom_tok, dom_count = cnt.most_common(1)[0]
            if dom_count / repeat_window >= repeat_threshold:
                if dom_tok != last_dominant:
                    shifts.append(
                        RegimeShift(
                            start_token_idx=k,
                            kind="repetition",
                            detail=(f"{dom_count}/{repeat_window} tokens are " f"token-id {dom_tok}"),
                        )
                    )
                    last_dominant = dom_tok
            else:
                last_dominant = None

    if text:
        in_run = False
        run_start_char: int = 0
        run_nonascii = 0
        run_total = 0
        for i, ch in enumerate(text):
            is_print_ascii = ch in string.printable
            if not is_print_ascii:
                if not in_run:
                    in_run = True
                    run_start_char = i
                    run_nonascii = 0
                    run_total = 0
                run_nonascii += 1
                run_total += 1
            else:
                if in_run:
                    if (
                        run_total >= NON_ASCII_RUN_MIN_LEN
                        and run_nonascii / max(run_total, 1) >= NON_ASCII_RUN_THRESHOLD
                    ):
                        char_idx = run_start_char
                        tok_idx = int(round(char_idx / max(len(text), 1) * len(tokens)))
                        shifts.append(
                            RegimeShift(
                                start_token_idx=tok_idx,
                                kind="non_ascii_run",
                                detail=(
                                    f"~{run_total} chars at char "
                                    f"{char_idx}, "
                                    f"non-ASCII ratio "
                                    f"{run_nonascii / max(run_total, 1):.0%}"
                                ),
                            )
                        )
                in_run = False
                run_total = 0
                run_nonascii = 0

        if (
            in_run
            and run_total >= NON_ASCII_RUN_MIN_LEN
            and run_nonascii / max(run_total, 1) >= NON_ASCII_RUN_THRESHOLD
        ):
            char_idx = run_start_char
            tok_idx = int(round(char_idx / max(len(text), 1) * len(tokens)))
            shifts.append(
                RegimeShift(
                    start_token_idx=tok_idx,
                    kind="non_ascii_run",
                    detail=(
                        f"~{run_total} chars at char {char_idx}, "
                        f"non-ASCII ratio "
                        f"{run_nonascii / max(run_total, 1):.0%}"
                    ),
                )
            )

    if text:
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]

            if ch.isalnum() or ch.isspace():
                i += 1
                continue

            j = i + 1
            while j < n and text[j] == ch:
                j += 1
            run_len = j - i
            if run_len >= CHAR_REPEAT_MIN_LEN:
                char_idx = i
                tok_idx = int(round(char_idx / max(len(text), 1) * len(tokens)))
                tok_idx = min(max(tok_idx, 0), max(len(tokens) - 1, 0))
                shifts.append(
                    RegimeShift(
                        start_token_idx=tok_idx,
                        kind="char_repetition",
                        detail=(
                            f"run of {run_len} consecutive '{ch}' "
                            f"chars at char {char_idx} (BPE-blind "
                            f"detector; token-level missed this)"
                        ),
                    )
                )
            i = j

    shifts.sort(key=lambda s: s.start_token_idx)
    return shifts


def count_prefix_match(
    tt_tokens: Sequence[int],
    hf_tokens: Sequence[int],
) -> int:
    """Return the number of tokens at the START of both sequences
    that match exactly. The medgemma case had ~36 here, which is
    the strongest single signal that "the model works at first
    and then collapses" (different from "the model never works")."""
    n = 0
    for a, b in zip(tt_tokens, hf_tokens):
        if a != b:
            break
        n += 1
    return n


def build_text_evidence(
    *,
    tt_text: str,
    tt_tokens: Sequence[int],
    hf_text: str = "",
    hf_tokens: Sequence[int] = (),
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    input_hint: str = "",
) -> TextEvidence:
    """Construct a :class:`TextEvidence` from raw inputs.

    Pure (no I/O, no HF, no torch). Callers should run HF, run
    the tokenizer, then call this with the resulting sequences.
    """
    tt_list = list(tt_tokens)[:scan_limit]
    hf_list = list(hf_tokens)[:scan_limit]

    prefix_match = count_prefix_match(tt_list, hf_list)
    compared = min(len(tt_list), len(hf_list))
    mismatch = compared - sum(1 for a, b in zip(tt_list, hf_list) if a == b)

    shifts = detect_regime_shifts(tt_list, tt_text)
    collapse = shifts[0].start_token_idx if shifts else None

    return TextEvidence(
        tt_text=tt_text,
        tt_tokens=tt_list,
        hf_text=hf_text,
        hf_tokens=hf_list,
        collapse_position=collapse,
        regime_shifts=shifts,
        prefix_match_count=prefix_match,
        mismatch_count=mismatch,
        compared_tokens=compared,
        scan_limit=scan_limit,
        input_hint=input_hint,
    )


@dataclass
class MismatchSegment:
    """A contiguous region where TT and HF diverged."""

    tt_start: int
    tt_end: int
    hf_start: int
    hf_end: int
    tt_text: str
    hf_text: str
    kind: str


_TOKEN_SPLIT_RE = re.compile(r"\s+")


def first_mismatch_segment(
    evidence: TextEvidence,
    *,
    context_tokens: int = 8,
) -> Optional[MismatchSegment]:
    """Return the first contiguous mismatch region in ``evidence``,
    or ``None`` if TT and HF agree token-for-token.

    Used by the planner to surface the EXACT spot where divergence
    starts in the repair prompt; the LLM is far more effective
    when it sees the boundary than when it sees the full 200-
    token diff."""
    tt = evidence.tt_tokens
    hf = evidence.hf_tokens
    n = min(len(tt), len(hf))
    start = -1
    for i in range(n):
        if tt[i] != hf[i]:
            start = i
            break
    if start < 0:
        if len(tt) == len(hf):
            return None

        start = n
    end_tt = min(start + context_tokens, len(tt))
    end_hf = min(start + context_tokens, len(hf))

    tt_snip = (
        evidence.tt_text[: 80 + end_tt * 4][start * 3 : start * 3 + 80]
        if evidence.tt_text
        else " ".join(map(str, tt[start:end_tt]))
    )
    hf_snip = (
        evidence.hf_text[: 80 + end_hf * 4][start * 3 : start * 3 + 80]
        if evidence.hf_text
        else " ".join(map(str, hf[start:end_hf]))
    )
    kind = "prefix-divergence" if start == 0 else "interior"
    return MismatchSegment(
        tt_start=start,
        tt_end=end_tt,
        hf_start=start,
        hf_end=end_hf,
        tt_text=tt_snip,
        hf_text=hf_snip,
        kind=kind,
    )


__all__ = [
    "DEFAULT_SCAN_LIMIT",
    "MismatchSegment",
    "RegimeShift",
    "TextEvidence",
    "build_text_evidence",
    "count_prefix_match",
    "detect_regime_shifts",
    "find_first_collapse",
    "first_mismatch_segment",
]
