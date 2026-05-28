# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Validation helpers for the SeamlessM4T-v2 T2TT demo.

Provides a thin wrapper around ``sacrebleu`` so callers can compute a
BLEU score over a list of (hypothesis, reference) pairs without having
to deal with the corpus_bleu API directly.

Usage::

    from models.demos.facebook_seamless_m4t_v2_large.demo.validate import bleu

    score = bleu(hyps=["Bonjour le monde."], refs=["Bonjour le monde."])
    print(score)        # -> 100.0

Notes:
    - The score returned is a percentage in [0.0, 100.0] (sacrebleu's
      convention).
    - We use a single-reference setup: each hypothesis is scored against
      exactly one reference string.
    - ``sacrebleu.corpus_bleu`` takes references as a list-of-lists
      (one inner list per reference position) — we wrap accordingly.
"""

from __future__ import annotations

from typing import List


def bleu(hyps: List[str], refs: List[str]) -> float:
    """Corpus BLEU (0..100) of ``hyps`` against single-reference ``refs``.

    Args:
        hyps: list of translation hypotheses.
        refs: list of reference translations (same length as ``hyps``).

    Returns:
        float BLEU score in ``[0, 100]``.

    Raises:
        ValueError: if list lengths mismatch or either is empty.
        ImportError: if ``sacrebleu`` is not installed.
    """
    if len(hyps) != len(refs):
        raise ValueError(f"hyps ({len(hyps)}) and refs ({len(refs)}) must be the same length")
    if not hyps:
        raise ValueError("bleu() requires at least one (hyp, ref) pair")

    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("sacrebleu is required for BLEU evaluation; install with `pip install sacrebleu`") from exc

    # sacrebleu expects references as List[List[str]] -- one inner list per ref index.
    score = sacrebleu.corpus_bleu(hyps, [refs])
    return float(score.score)
