# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Prompt and answer logic adapted from strands-labs/strands-decider (Apache-2.0).

import json
from dataclasses import dataclass

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_QUESTION_FRACTION = 0.75
MAX_BATCH = 32  # strands fits the window per chunk of this many questions
NOUL_SLOT_LABELS = ("false", "true")
NOUL_DEFAULT_CRITERIA = {
    "false": "the statement does not hold for this state",
    "true": "the statement holds for this state",
}
HEADERS = {
    "noul": "Decide whether the statement is true of the state.",
    "choice": "Select exactly one option.",
    "score": "Rate the state against the ordered levels below (lowest first).",
}


@dataclass(frozen=True)
class EncodedQuestion:
    name: str
    kind: str  # "noul" | "choice" | "score"
    labels: list[str]  # slot labels in slot order
    input_ids: list[int]  # full row: state tokens + question tokens
    option_positions: list[int]  # absolute index of the last token of each option line
    query_position: int  # absolute index of the pooled query token (last token)
    descriptions: tuple[str, ...] = ()  # rubric text per slot (score legend)


def render_content(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


def resolve_kind(spec: dict):
    """Question kind; when `type` is missing, resolve like upstream's pydantic union
    NoulQuestion | ChoiceQuestion | ScoreQuestion (leftmost valid member wins ties)."""
    kind = spec.get("type")
    if kind is not None or "type" in spec:
        return kind
    criteria = spec.get("criteria")
    if isinstance(criteria, list):
        return "score"
    if isinstance(criteria, dict) and not set(criteria).issubset({"true", "false"}):
        return "choice"
    return "noul"


def validate_questions(questions: dict) -> None:
    if not questions:
        raise ValueError("at least one question is required")
    for name, spec in questions.items():
        kind = resolve_kind(spec)
        if kind not in HEADERS:
            raise ValueError(f"question {name!r}: unknown type {kind!r}")
        if "instructions" not in spec:
            raise ValueError(f"question {name!r}: instructions required")
        criteria = spec.get("criteria")
        if kind == "noul":
            if criteria is not None and not set(criteria).issubset({"true", "false"}):
                raise ValueError("noul criteria keys must be a subset of {'true', 'false'}")
        elif kind == "choice":
            if not isinstance(criteria, dict) or len(criteria) < 2:
                raise ValueError("choice requires at least 2 options")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise ValueError(f"choice allows at most {MAX_CHOICE_OPTIONS} options")
        else:
            if not isinstance(criteria, list) or not (MIN_SCORE_LEVELS <= len(criteria) <= MAX_SCORE_LEVELS):
                raise ValueError(f"score requires between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels")


def render_state(state) -> str:
    return f"<state>\n{render_content(state)}\n</state>\n"


def _option_block(pairs) -> tuple[str, list[tuple[int, int]]]:
    lines, spans, cursor = [], [], 0
    for i, (name, desc) in enumerate(pairs):
        desc = " ".join((desc or "").split())
        line = f"{i + 1}. {name}" + (f" — {desc}" if desc else "")
        lines.append(line)
        spans.append((cursor, cursor + len(line)))
        cursor += len(line) + 1
    return "\n".join(lines), spans


def _render(spec: dict):
    kind = resolve_kind(spec)
    instructions = render_content(spec["instructions"])
    criteria = spec.get("criteria")
    if kind == "noul":
        crit = {**NOUL_DEFAULT_CRITERIA, **(criteria or {})}
        pairs = [(lbl, render_content(crit[lbl])) for lbl in NOUL_SLOT_LABELS]
    elif kind == "choice":
        pairs = [(name, render_content(desc)) for name, desc in criteria.items()]
    else:
        pairs = [(str(i), desc) for i, desc in enumerate(criteria)]
    block, block_spans = _option_block(pairs)
    prefix = f'<question type="{kind}">\n{HEADERS[kind]}\n{instructions}\n<options>\n'
    text = prefix + block + "\n</options>\n</question>\n<answer>"
    base = len(prefix)
    return (
        text,
        [n for n, _ in pairs],
        [(base + s, base + e) for s, e in block_spans],
        tuple((d or "").strip() for _, d in pairs),
    )


def render_question(name: str, spec: dict) -> tuple[str, list[str], list[tuple[int, int]]]:
    text, labels, spans, _ = _render(spec)
    return text, labels, spans


def option_token_index(offsets, spans, base: int = 0) -> list[int]:
    out = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was truncated through its option list"
            )
        out.append(last)
    return out


def _fit(tokenizer, state_text: str, question_texts: list[str], max_len: int):
    enc = tokenizer(question_texts, add_special_tokens=False, return_offsets_mapping=True)
    q, offs = enc["input_ids"], enc["offset_mapping"]
    longest = max(len(x) for x in q)
    reserve = min(longest, max(1, int(max_len * MAX_QUESTION_FRACTION)))
    cut = [max(0, len(x) - reserve) for x in q]
    offs = [o[c:] for o, c in zip(offs, cut)]
    q = [x[c:] for x, c in zip(q, cut)]
    state_budget = max(1, max_len - reserve)
    s = tokenizer(state_text, add_special_tokens=True, truncation=True, max_length=state_budget)["input_ids"]
    return s, q, offs


def encode_request(tokenizer, state, questions: dict, max_length: int = 4096) -> list[EncodedQuestion]:
    validate_questions(questions)
    state_text = render_state(state)
    names = list(questions)
    rendered = [_render(questions[n]) for n in names]
    out = []
    for g in range(0, len(names), MAX_BATCH):
        chunk = rendered[g : g + MAX_BATCH]
        s, q, offs = _fit(tokenizer, state_text, [r[0] for r in chunk], max_length)
        for i, (text, labels, spans, descs) in enumerate(chunk):
            ids = s + q[i]
            positions = [len(s) + p for p in option_token_index(offs[i], spans, 0)]
            out.append(
                EncodedQuestion(
                    name=names[g + i],
                    kind=resolve_kind(questions[names[g + i]]),
                    labels=labels,
                    input_ids=ids,
                    option_positions=positions,
                    query_position=len(ids) - 1,
                    descriptions=descs,
                )
            )
    return out
