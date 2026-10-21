# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-a-haystack prompt for the GLM-5.3-Flash long-context runs (host only).

``filler_sentences`` / ``build_needle_ids`` / ``eos_ids`` and the constants ``NEEDLE`` / ``QUESTION`` /
``NEEDLE_KEY`` are copied VERBATIM from ``tests/test_mesh_long_context.py`` at commit 8c5a3cb5902 (branch
arg/glm53_mesh): the mesh run and the HF CPU references build their prompt through that exact function (same
seeds, same text, the think block closed after the generation prompt), so the single-chip smoke's ids match
them token for token (N=4096 -> 4079 ids, N=16384 -> 16359 ids). ``build_needle_prompt`` is a thin wrapper
with the ``(tokenizer, n_tokens, depth, needle, question)`` signature: with the canonical needle / question it
returns exactly ``build_needle_ids``'s ids; other strings go through the same algorithm.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

NEEDLE = "The secret passphrase for the vault is 'amber-falcon-7391'."
QUESTION = "What is the secret passphrase for the vault? Answer with the passphrase only."
NEEDLE_KEY = "amber-falcon-7391"

_TOPICS = [
    "the harbour",
    "a mountain pass",
    "the old library",
    "a night market",
    "the river delta",
    "an orchard",
    "the observatory",
    "a tram depot",
    "the glass factory",
    "a lighthouse",
    "the salt flats",
    "a bakery",
]
_VERBS = ["described", "measured", "repainted", "catalogued", "photographed", "mapped", "restored", "visited"]


def filler_sentences(n: int, seed: int = 0) -> str:
    """``n`` varied, non-repeating filler sentences (counters keep every sentence distinct)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        t = _TOPICS[int(torch.randint(len(_TOPICS), (1,), generator=g))]
        v = _VERBS[int(torch.randint(len(_VERBS), (1,), generator=g))]
        k = int(torch.randint(10, 999, (1,), generator=g))
        out.append(f"On day {i + 1} the survey team {v} {t} and logged {k} entries before noon.")
    return " ".join(out)


def build_needle_ids(tok, n_tokens: int, depth: float) -> tuple[list[int], str]:
    """Chat-templated prompt of about ``n_tokens`` tokens: filler with the needle at ``depth``, then the question."""

    # the generation prompt ends with ``<|assistant|><think>`` (this template has no no-think switch); closing the
    # think block right away makes the greedy continuation the answer itself, so a short generation can contain the
    # needle (the HF reference builds its prompt through this same function)
    close_think = tok.encode("</think>", add_special_tokens=False)

    def ids_of(text: str) -> list[int]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):  # transformers >= 5 returns a BatchEncoding
            ids = ids["input_ids"]
        return [int(t) for t in ids] + close_think

    def prompt(n_sent: int) -> str:
        a = filler_sentences(max(1, int(n_sent * depth)), seed=1)
        b = filler_sentences(max(1, n_sent - int(n_sent * depth)), seed=2)
        return f"{a} {NEEDLE} {b}\n\n{QUESTION}"

    n_sent = max(2, n_tokens // 20)
    for _ in range(4):  # converge the filler length to the requested token count
        text = prompt(n_sent)
        ids = ids_of(text)
        if abs(len(ids) - n_tokens) <= max(64, n_tokens // 200) or n_sent <= 2:
            return ids, text
        n_sent = max(2, int(n_sent * n_tokens / len(ids)))
    return ids, text


def eos_ids(snapshot: Path, tok) -> set[int]:
    ids = set()
    gc = snapshot / "generation_config.json"
    if gc.exists():
        v = json.loads(gc.read_text()).get("eos_token_id", [])
        ids.update(v if isinstance(v, list) else [v])
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return ids


# ------------------------------------------------------------------------------------ wrappers (this branch)
def _needle_ids(tok, n_tokens: int, depth: float, needle: str, question: str) -> tuple[list[int], str]:
    """``build_needle_ids`` with the needle / question as parameters (the same algorithm, line for line)."""
    close_think = tok.encode("</think>", add_special_tokens=False)

    def ids_of(text: str) -> list[int]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        return [int(t) for t in ids] + close_think

    def prompt(n_sent: int) -> str:
        a = filler_sentences(max(1, int(n_sent * depth)), seed=1)
        b = filler_sentences(max(1, n_sent - int(n_sent * depth)), seed=2)
        return f"{a} {needle} {b}\n\n{question}"

    n_sent = max(2, n_tokens // 20)
    for _ in range(4):
        text = prompt(n_sent)
        ids = ids_of(text)
        if abs(len(ids) - n_tokens) <= max(64, n_tokens // 200) or n_sent <= 2:
            return ids, text
        n_sent = max(2, int(n_sent * n_tokens / len(ids)))
    return ids, text


def build_needle_prompt(
    tokenizer, n_tokens: int, depth: float, needle: str = NEEDLE, question: str = QUESTION
) -> torch.Tensor:
    """Token ids ``[S]`` int64 of the needle prompt; the canonical needle / question give exactly
    ``build_needle_ids(tokenizer, n_tokens, depth)`` (the mesh run's and the HF references' ids)."""
    if (needle, question) == (NEEDLE, QUESTION):
        ids, _ = build_needle_ids(tokenizer, n_tokens, depth)
    else:
        ids, _ = _needle_ids(tokenizer, n_tokens, depth, needle, question)
    return torch.tensor(ids, dtype=torch.int64)


def needle_token_start(tokenizer, n_tokens: int, depth: float) -> int:
    """The index of the needle's first token in ``build_needle_ids``'s ids (by tokenizing the text before it)."""
    ids, text = build_needle_ids(tokenizer, n_tokens, depth)
    before = text[: text.index(NEEDLE)]
    head = tokenizer.apply_chat_template(
        [{"role": "user", "content": before}], add_generation_prompt=False, tokenize=True
    )
    if hasattr(head, "input_ids"):
        head = head["input_ids"]
    return len(head)


def retrieved(text: str, key: str = NEEDLE_KEY) -> bool:
    """Whether the decoded continuation contains the needle key (compared without whitespace, case-insensitive)."""
    norm = lambda s: "".join(ch for ch in s if not ch.isspace()).lower()
    return norm(key) in norm(text)
