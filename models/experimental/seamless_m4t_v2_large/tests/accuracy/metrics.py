# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Self-contained, tokenizer-free task-quality metrics for SeamlessM4T v2.

The per-op PCC suite verifies numerical correlation against the torch reference, but cannot catch a
*task-level* regression where every op stays in-PCC yet the decoded text is wrong (e.g. long-audio
ASR silently transcribing in the wrong language). These metrics close that gap: they score decoded
text against a reference (HF output or ground truth) with **tokenizer-free** measures so they work
across scripts (Devanagari, Latin, …) without a per-language tokenizer.

- ``corpus_chrf`` — micro-averaged character n-gram F-beta (the primary metric; language-agnostic).
- ``corpus_char_bleu`` — character-level BLEU (a script-agnostic BLEU proxy).
- ``corpus_wer`` / ``corpus_cer`` — word / character error rate (for ASR same-language transcription).
- ``script_fraction`` — fraction of letters in a target Unicode script; the cheap language check that
  catches a wrong-language flip (e.g. ASR producing English when ``tgt_lang='hin'``) with no reference.

``score_corpus`` prefers ``sacrebleu`` when importable and falls back to the built-ins. chrF/char-BLEU
are ported from ``models/demos/seamless_m4t_v2/evaluation/metrics.py`` (the ``yito`` branch).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


# --- chrF / char-BLEU (tokenizer-free corpus translation metrics) ----------------------------


def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i : i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def corpus_chrf(hyps: Sequence[str], refs: Sequence[str], max_n: int = 6, beta: float = 2.0) -> float:
    """Micro-averaged character n-gram F-beta (chrF), as a 0-100 score."""
    eps = 1e-9
    f_scores = []
    for n in range(1, max_n + 1):
        match = tp = tt = 0
        for h, r in zip(hyps, refs):
            hn, rn = _char_ngrams(h, n), _char_ngrams(r, n)
            match += sum((hn & rn).values())
            tp += sum(hn.values())
            tt += sum(rn.values())
        prec = match / (tp + eps)
        rec = match / (tt + eps)
        denom = beta * beta * prec + rec
        f_scores.append((1 + beta * beta) * prec * rec / (denom + eps))
    return 100.0 * sum(f_scores) / max_n


def corpus_char_bleu(hyps: Sequence[str], refs: Sequence[str], max_n: int = 4) -> float:
    """Character-level BLEU (0-100); a script-agnostic proxy for word-tokenized BLEU."""
    eps = 1e-9
    precisions = []
    for n in range(1, max_n + 1):
        match = total = 0
        for h, r in zip(hyps, refs):
            hn, rn = _char_ngrams(h, n), _char_ngrams(r, n)
            match += sum((hn & rn).values())
            total += sum(hn.values())
        precisions.append((match + eps) / (total + eps))
    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    hyp_len = sum(len(h.replace(" ", "")) for h in hyps)
    ref_len = sum(len(r.replace(" ", "")) for r in refs)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / (hyp_len + eps))
    return 100.0 * bp * geo


def score_corpus(hyps: Sequence[str], refs: Sequence[str]) -> dict:
    """Return {'bleu', 'chrf', 'backend'}; prefer sacrebleu, else the built-ins."""
    try:
        import sacrebleu

        bleu = sacrebleu.corpus_bleu(list(hyps), [list(refs)]).score
        chrf = sacrebleu.corpus_chrf(list(hyps), [list(refs)]).score
        return {"bleu": bleu, "chrf": chrf, "backend": "sacrebleu"}
    except Exception:  # noqa: BLE001 - sacrebleu unavailable
        return {
            "bleu": corpus_char_bleu(hyps, refs),
            "chrf": corpus_chrf(hyps, refs),
            "backend": "builtin(char-bleu+chrf)",
        }


# --- WER / CER (ASR same-language transcription) ---------------------------------------------


def _edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance between two token sequences (lists of words or chars)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def corpus_wer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Aggregate word error rate (edit distance over words / total reference words)."""
    dist = words = 0
    for h, r in zip(hyps, refs):
        rw = r.split()
        dist += _edit_distance(h.split(), rw)
        words += len(rw)
    return dist / max(1, words)


def corpus_cer(hyps: Sequence[str], refs: Sequence[str]) -> float:
    """Aggregate character error rate (edit distance over chars / total reference chars).

    Character-level is the right granularity for scripts without whitespace word boundaries
    (Devanagari, CJK), where WER is unreliable.
    """
    dist = chars = 0
    for h, r in zip(hyps, refs):
        rc = list(r.replace(" ", ""))
        dist += _edit_distance(list(h.replace(" ", "")), rc)
        chars += len(rc)
    return dist / max(1, chars)


# --- Language / script check (cheap, reference-free wrong-language detector) ------------------

_SCRIPT_RANGES = {
    "deva": [("ऀ", "ॿ")],  # Devanagari (Hindi/Marathi/…)
    "latin": [("a", "z"), ("A", "Z"), ("À", "ɏ")],  # Latin incl. accented
}


def script_fraction(text: str, script: str) -> float:
    """Fraction of *letters* in ``text`` that belong to ``script`` (e.g. 'deva', 'latin').

    Returns 0.0 for empty/letterless text. A near-1.0 value means the text is in that script; this is
    the cheap, reference-free check that flags a wrong-language decode (the assertion that was missing
    when long-audio ASR silently produced English for ``tgt_lang='hin'``).
    """
    ranges = _SCRIPT_RANGES[script]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    in_script = sum(any(lo <= c <= hi for lo, hi in ranges) for c in letters)
    return in_script / len(letters)
