# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""mmmu_pro_aa: Artificial Analysis's multiple-choice prompt and answer extraction on MMMU-Pro (standard, 10 options).

Prompt and regexes are quoted from artificialanalysis.ai/methodology/intelligence-benchmarking ("Multiple Choice
Questions" and "Multiple Choice Extraction Regex", read 2026-11-07). The question keeps its ``<image n>`` tags; the
images go in front of the text in the order of their first mention, as lmms-eval's mmmu_pro task sends them (AA does
not say how it places images)."""

import ast
import re

from lmms_eval.tasks.mmmu_pro.utils import mmmu_pro_doc_to_visual  # noqa: F401

from mimo_lmms.failures import request_failed

# MMMU-Pro's ``options`` column is a Python list literal of at most ten strings ("['a', 'b', ...]"); the longest in
# the standard split is a few thousand characters. A value far outside that is not parsed.
MAX_OPTIONS_CHARS = 100_000

LETTERS = "ABCDEFGHIJ"
PRIMARY = re.compile(r"(?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])")
FALLBACKS = [
    re.compile(r"\\boxed\{[^}]*([A-Z])[^}]*\}"),
    re.compile(r"answer is ([a-zA-Z])"),
    re.compile(r"answer is \(([a-zA-Z])"),
    re.compile(r"([A-Z])\)\s*[^A-Z]*"),
    re.compile(r"([A-Z])\s+is\s+the\s+correct\s+answer"),
    re.compile(r"([A-Z])\s*$"),
    re.compile(r"([A-Z])\s*\."),
    re.compile(r"([A-Z])\s*[^\w]"),
]


def aa_prompt(question, options):
    letters = LETTERS[: len(options)]
    head = (
        "Answer the following multiple choice question. The last line of your response should be in the following "
        f"format: 'Answer: {'/'.join(letters)}' (e.g. 'Answer: A')."
    )
    lines = [head, question] + [f"{letter}) {option}" for letter, option in zip(letters, options)]
    return "\n".join(lines)


def aa_extract(response):
    """AA's multi-stage extraction: a single-letter response as is, else the primary 'Answer: X' pattern, else the
    fallbacks in order; the last match of the first pattern that matches."""
    text = (response or "").strip()
    if len(text) == 1 and text.isalpha():
        return text.upper()
    for pattern in [PRIMARY] + FALLBACKS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].upper()
    return None


def parse_options(text):
    """The dataset's ``options`` literal as a list of strings: bounded in size before it is parsed (``literal_eval``
    builds whatever nesting the text holds) and checked to be the flat list the prompt needs."""
    if not isinstance(text, str) or len(text) > MAX_OPTIONS_CHARS:
        raise ValueError(f"MMMU-Pro options: expected a list literal of at most {MAX_OPTIONS_CHARS} characters")
    options = ast.literal_eval(text)
    if not isinstance(options, (list, tuple)) or not 1 <= len(options) <= len(LETTERS):
        raise ValueError(f"MMMU-Pro options: expected 1 to {len(LETTERS)} options, got {str(options)[:80]!r}")
    return [str(option) for option in options]


def aa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return aa_prompt(doc["question"], parse_options(doc["options"]))


def aa_process_results(doc, results):
    if request_failed(results[0]):  # never through the extraction: "[MIMO_REQUEST_FAILED ..." would read as D
        return {"aa_accuracy": 0.0}
    pred = aa_extract(results[0])
    return {"aa_accuracy": float(pred is not None and pred == doc["answer"].strip().upper())}
