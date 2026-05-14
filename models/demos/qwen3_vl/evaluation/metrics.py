# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Metric implementations for VQA benchmarks."""

import re
import string
import unicodedata
from typing import Union


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _levenshtein(s1: str, s2: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[-1]


def anls(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity (ANLS) used in DocVQA / InfoVQA.

    For each ground truth, compute NLS = 1 - edit_dist / max_len.
    If NLS < threshold, score is 0.  Returns max over all ground truths.
    """
    pred = prediction.strip()
    best = 0.0
    for gt in ground_truths:
        gt = gt.strip()
        max_len = max(len(pred), len(gt))
        if max_len == 0:
            nls = 1.0
        else:
            nls = 1.0 - _levenshtein(pred.lower(), gt.lower()) / max_len
        score = nls if nls >= threshold else 0.0
        best = max(best, score)
    return best


def vqa_accuracy(prediction: str, ground_truths: list[str]) -> float:
    """VQA accuracy: fraction of annotators whose answer matches.

    Standard VQA metric: min(count_matching / 3, 1.0).
    ground_truths may contain multiple annotator answers.
    """
    pred_norm = _normalize_text(prediction)
    matches = sum(1 for gt in ground_truths if _normalize_text(gt) == pred_norm)
    return min(matches / 3, 1.0)


def relaxed_accuracy(prediction: str, ground_truths: list[str], tolerance: float = 0.05) -> float:
    """Relaxed accuracy used by ChartQA.

    For numeric answers: correct if |pred - gt| / |gt| <= tolerance.
    For non-numeric answers: exact string match after normalization.
    """
    pred_norm = _normalize_text(prediction)
    for gt in ground_truths:
        gt_norm = _normalize_text(gt)
        if pred_norm == gt_norm:
            return 1.0
        try:
            pred_val = float(pred_norm.replace(",", ""))
            gt_val = float(gt_norm.replace(",", ""))
            if gt_val == 0:
                if pred_val == 0:
                    return 1.0
            elif abs(pred_val - gt_val) / abs(gt_val) <= tolerance:
                return 1.0
        except ValueError:
            pass
    return 0.0


def exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Normalized exact match (used for multiple-choice benchmarks)."""
    pred_norm = _normalize_text(prediction)
    return 1.0 if any(_normalize_text(gt) == pred_norm for gt in ground_truths) else 0.0


def extract_mcq_answer(prediction: str, choices=None, num_choices: int = 5) -> str:
    """Extract single-letter or labelled answer from model output.

    Handles patterns like: 'A', '(A)', 'A.', 'Answer: A', 'The answer is A'.
    Falls back to checking if the prediction contains one of the choices.

    Args:
        prediction: Raw model output string.
        choices: List of choice texts for direct-match fallback.
        num_choices: Number of choices (determines valid letter range A-?).
    """
    pred = prediction.strip()
    max_letter = chr(ord("A") + max(num_choices - 1, 4))  # at least A-E

    # Build pattern to match any valid letter for this question
    valid = f"A-{max_letter}" if max_letter > "E" else "A-E"
    valid_lower = valid.lower()

    # Try explicit letter extraction patterns (strongest signal first)
    patterns = [
        rf"^[Tt]he answer is[:\s]+\(?([{valid}{valid_lower}])\)?",
        rf"^[Aa]nswer[:\s]+\(?([{valid}{valid_lower}])\)?",
        rf"^\(?([{valid}{valid_lower}])\)?[.\s]",
        rf"\b([{valid}{valid_lower}])\s*[:\)]\s",
        rf"^([{valid}{valid_lower}])$",
    ]
    for pat in patterns:
        m = re.search(pat, pred)
        if m:
            return m.group(1).upper()

    # Check if any choice TEXT appears verbatim in the prediction
    if choices:
        pred_lower = pred.lower()
        for i, choice in enumerate(choices):
            c = str(choice).lower().strip()
            if c and (c in pred_lower or pred_lower in c):
                return chr(ord("A") + i)

    # Last resort: first valid letter anywhere in the text
    m = re.search(rf"\b([{valid}])\b", pred.upper())
    if m:
        return m.group(1)

    return pred[:1].upper() if pred else ""
