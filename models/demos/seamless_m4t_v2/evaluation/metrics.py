# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Self-contained MT metrics for the S2TT benchmark, used when `sacrebleu` is not
importable in the runtime. chrF is tokenizer-free and well suited to Japanese;
a character-level BLEU is provided as a Japanese-friendly proxy for sacrebleu's
`ja-mecab` BLEU.

`score_corpus` prefers sacrebleu (BLEU with ja-mecab + chrF) and transparently
falls back to these implementations.
"""

from __future__ import annotations

import math
from collections import Counter


def _char_ngrams(s: str, n: int) -> Counter:
    s = s.replace(" ", "")
    return Counter(s[i : i + n] for i in range(len(s) - n + 1)) if len(s) >= n else Counter()


def corpus_chrf(hyps, refs, max_n: int = 6, beta: float = 2.0) -> float:
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


def corpus_char_bleu(hyps, refs, max_n: int = 4) -> float:
    """Character-level BLEU (0-100); Japanese-friendly proxy when mecab is absent."""
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


def score_corpus(hyps, refs) -> dict:
    """Return {'bleu', 'chrf', 'backend'}; prefer sacrebleu, else built-ins."""
    try:
        import sacrebleu

        try:
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="ja-mecab").score
            backend = "sacrebleu(ja-mecab)"
        except Exception:  # noqa: BLE001
            bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
            backend = "sacrebleu(default)"
        chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
        return {"bleu": bleu, "chrf": chrf, "backend": backend}
    except Exception:  # noqa: BLE001 - sacrebleu unavailable
        return {
            "bleu": corpus_char_bleu(hyps, refs),
            "chrf": corpus_chrf(hyps, refs),
            "backend": "builtin(char-bleu+chrf)",
        }
