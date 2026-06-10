# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Validation metric helpers for the dots.ocr e2e gates.

One thin function per metric (generation skill §7). The ocr use case
gates on corpus WER (use_case.validation_metric = "wer").
"""


def wer(hyps: list[str], refs: list[str]) -> float:
    """Corpus word error rate of hypotheses against references (0 is perfect)."""
    import jiwer

    return float(jiwer.wer(refs, hyps))
