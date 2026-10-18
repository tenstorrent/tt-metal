# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Prompt layout and answer parsing shared by the GSM8K training and eval scripts."""

from __future__ import annotations

import re
from typing import Optional

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"

SYSTEM_PROMPT = (
    "Respond in the following format:\n" f"{THINK_OPEN}\n...\n{THINK_CLOSE}\n" f"{ANSWER_OPEN}\n...\n{ANSWER_CLOSE}\n"
)

FORMAT_RE = re.compile(
    re.escape(THINK_OPEN)
    + r".*?"
    + re.escape(THINK_CLOSE)
    + r"\s*"
    + re.escape(ANSWER_OPEN)
    + r".*?"
    + re.escape(ANSWER_CLOSE),
    re.DOTALL,
)

STRICT_FORMAT_RE = re.compile(
    r"^\s*"
    + re.escape(THINK_OPEN)
    + r"\n.*?\n"
    + re.escape(THINK_CLOSE)
    + r"\n"
    + re.escape(ANSWER_OPEN)
    + r"\n.*?\n"
    + re.escape(ANSWER_CLOSE)
    + r"\s*$",
    re.DOTALL,
)

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def format_prompt(question: str) -> str:
    """Plain "Question:/Answer:" completion layout the SFT checkpoint was trained on; no chat template."""
    return f"{SYSTEM_PROMPT}\nQuestion: {question}\nAnswer:"


def normalize_number(s: str) -> str:
    s = s.strip().rstrip(".").replace(",", "").replace("$", "").replace("%", "")
    if not s:
        return s
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        return s


def extract_hash_answer(gold: str) -> str:
    """GSM8K gold answers end with '#### <number>'."""
    return normalize_number(gold.split("####")[-1])


def extract_tag_answer(text: str) -> Optional[str]:
    """Last number inside the last <answer>...</answer> block."""
    if ANSWER_OPEN not in text:
        return None
    body = text.split(ANSWER_OPEN)[-1].split(ANSWER_CLOSE)[0]
    nums = _NUM_RE.findall(body)
    return normalize_number(nums[-1]) if nums else None
