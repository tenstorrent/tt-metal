# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vgi_bench: Seldon's VGI-Bench public split with its pinned prompt and answer parsing.

`build_prompt` (multiple-choice branch) and `parse_answer` are copied verbatim from vgibench/prompts.py of
github.com/Seldon-Foundation/VGIBench (MIT, commit c7b4dc3, HARNESS_VERSION 1.0.0): the official score is
`vgi_accuracy`. OpenRouter's benchmark page describes its grading as "the letter on its final Answer: line", while
Seldon's parser takes the FIRST "Answer:" match (then the first standalone letter), so `vgi_accuracy_last_answer_line`
scores the same replies with the last "Answer:" line instead (same fallback), to show whether the choice matters.
"""

import os
import re

from mimo_lmms.failures import request_failed

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# 'Answer: c', 'Answer - (C)', 'Answer: **d**' ...
ANSWER_LINE = re.compile(r"answer\s*[:\-]\s*\**\s*\(?\s*([a-zA-Z])\b", re.IGNORECASE)


def build_prompt(question):
    options = "\n".join(f"{LETTERS[i]}) {a}" for i, a in enumerate(question["answers"]))
    return (
        "Watch the video and answer the multiple-choice question about it.\n\n"
        f"Question: {question['question']}\n\n"
        f"{options}\n\n"
        "Reply with the letter of the correct answer only."
    )


def parse_answer(text, n_options):
    """Seldon's parser: the explicit 'Answer:' line first, then the first standalone letter."""
    match = ANSWER_LINE.search(text)
    if match:
        index = LETTERS.index(match.group(1).lower())
        if index < n_options:
            return index
    for match in re.finditer(r"\b([a-zA-Z])\b", text):
        index = LETTERS.index(match.group(1).lower())
        if index < n_options:
            return index
    return None


def parse_answer_last_line(text, n_options):
    """As parse_answer, but the LAST 'Answer:' match wins (OpenRouter's description of its grading)."""
    for match in reversed(list(ANSWER_LINE.finditer(text))):
        index = LETTERS.index(match.group(1).lower())
        if index < n_options:
            return index
    return parse_answer(text, n_options)


def vgi_doc_to_visual(doc):
    """The video by URL, as Seldon's harness does. With VGI_LOCAL_VIDEO_DIR set (a local, sha256-checked copy of the
    videos named by their URL basename) a server that reads local media (vLLM --allowed-local-media-path) gets a
    file:// URL to the same bytes instead, so it does not download hundreds of MB from the CDN for every question."""
    url = doc["video_url"]
    local_dir = os.environ.get("VGI_LOCAL_VIDEO_DIR")
    if local_dir:
        path = os.path.join(local_dir, url.rsplit("/", 1)[-1])
        if os.path.isfile(path):
            return ["file://" + os.path.abspath(path)]
    return [url]


def vgi_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return build_prompt(doc)


def vgi_process_results(doc, results):
    text = results[0] or ""
    n = len(doc["answers"])
    if request_failed(text):  # never through the parser: the first standalone letter of an error body is an answer
        return {"vgi_accuracy": 0.0, "vgi_accuracy_last_answer_line": 0.0}
    return {
        "vgi_accuracy": float(parse_answer(text, n) == doc["correct_answer"]),
        "vgi_accuracy_last_answer_line": float(parse_answer_last_line(text, n) == doc["correct_answer"]),
    }
