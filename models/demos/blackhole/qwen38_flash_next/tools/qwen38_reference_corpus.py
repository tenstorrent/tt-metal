#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The teacher-forced reference corpus ``Q38-REF-v1`` and the columns compared over it.

The corpus (``tools/reference/corpus-q38-ref-v1.json`` + ``.jsonl``, frozen with sha256s) holds 36 items in five
parts: the twelve shipped acceptance prompts as their records render them (``acceptance``), the same twelve rendered
as the server renders a request today (thinking on, medium) plus four requests with tools (``served``), the first
1024 tokens of two books (``book``), four evaluation items rendered as the evaluation harness sends them (``eval``)
and two long natural-text prompts scored in 32-position windows (``long``).  Every column runs the same token
stream: for a text item the text itself is the teacher; for a prompt item the prompt is followed by
CONTINUATION_TOKENS teacher tokens, the HF argmax chain, which the HF column produces and the other columns read back
from the HF reference file (``--teacher``).  Position p is the output after consuming tokens 0..p; it predicts
token p+1.

Modes (``python -m ...tools.qwen38_reference_corpus MODE``):

    freeze   build the corpus files from the checkpoint tokenizer, the acceptance records, the two book texts and
             the frozen evaluation items (the committed corpus was made this way; ``verify`` re-checks it)
    hf       the Transformers ``qwen4_exp`` model on the CPU: bf16 weights, the final hidden state through an fp32
             LM head, fp32 log-softmax; per scored position the top-32 ids and log-probs and the teacher's log-prob
    oracle   the same through the CPU oracle ``tt/`` (never edited), with bf16 experts or BF4-emulated experts
             (``--experts bf4``: the host packer's shared-exponent rounding applied to every routed expert) and with
             the QSA K/V cache in bf16 or in one of the device's fp8 formats (``--kv``, ``qwen38_kv_cache_emulation``)
    device   a served chain's agreement records (the chat server's ``--agreement-reference`` run: argmax + the
             32-candidate row per position, one file per item) as a column; its log-probs are normalised over the
             row, so only the shared-support metrics apply (the server writes this column and its score itself)
    score    two columns: top-1 / top-5 / top-32 agreement, clear-margin top-1 (the reference's top-2 margin above
             ``--clear-margin`` logits), truncated KL over the shared top-32 support, first divergence per item,
             per-100-token segments of the text items
    verify   the corpus files against their sha256s (and a reference file against the corpus)

A reference file is one JSON document (``REFERENCE_SCHEMA``); its arrays are base64 little-endian int32 / float32 so
the ``score`` and ``device`` modes need only the standard library.  ``--full-logits DIR`` also keeps the scored
positions' fp16 logits (``torch.save``, one file per item) for exact KL later.
"""

from __future__ import annotations

import argparse
import base64
import bz2
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

MODEL_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = MODEL_DIR.parents[3]
REFERENCE_DIR = Path(__file__).with_name("reference")
ACCEPTANCE_DIR = Path(__file__).with_name("acceptance") / "greedy-prompts"
TALE_OF_TWO_CITIES = REPO_ROOT / "models" / "tt_transformers" / "tests" / "tale-of-two-cities.txt.bz2"

CORPUS_ID = "Q38-REF-v1"
MANIFEST_NAME = "corpus-q38-ref-v1.json"
ITEMS_NAME = "corpus-q38-ref-v1.jsonl"
MANIFEST_SCHEMA = "qwen38-reference-corpus/v1"
REFERENCE_SCHEMA = "qwen38-reference-logits/v1"
AGREEMENT_RECORDS_SCHEMA = "qwen38-agreement-records/v1"
SCORE_SCHEMA = "qwen38-reference-score/v1"
PARTS = ("acceptance", "served", "book", "eval", "long")
KV_CHOICES = (
    "bf16",
    "scaled_fp8",
    "scaled_fp8_rne",
    "fp8_e4m3",
    "fp8_e4m3_storage",
)  # = qwen38_kv_cache_emulation.KV_FORMATS
TOP_K = 32
CONTINUATION_TOKENS = 256
BOOK_TOKENS = 1024
LONG_TOKENS = (8_192, 32_704)  # the second is the 32k build's context limit less the 64-token headroom
LONG_WINDOW = 32
LONG_DEPTHS = (2_048, 8_192)  # plus the item's end; a window is the 32 positions before depth - 32
CLEAR_MARGIN = 0.125  # one bf16 ulp at logit magnitude 16..32: both observed near-tie flips were inside it
SEGMENT_TOKENS = 100
# The acceptance records' rendering (the session's constants) and the server's request defaults.
RECORD_FLAGS = {"enable_thinking": False, "reasoning_effort": "low"}
SERVED_FLAGS = {"enable_thinking": True, "reasoning_effort": "medium"}
EVAL_FLAGS = {"enable_thinking": False, "reasoning_effort": "low"}
GSM8K_TEMPLATE = "Q: {question}\nA: Let's think step by step."
HUMANEVAL_TEMPLATE = (
    "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```\n"
)
MOBY_DICK_START = ("CHAPTER 1. Loomings.", 2)  # the second occurrence: the body, not the table of contents
MOBY_DICK_END = "*** END OF THE PROJECT GUTENBERG EBOOK"
CHAT_TURN = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>\n", re.DOTALL)
EMPTY_THINK = "<think>\n\n</think>\n\n"

TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}
TOOL_PYTHON = {
    "type": "function",
    "function": {
        "name": "run_python",
        "description": "Run a Python snippet and return its stdout.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python source"}},
            "required": ["code"],
        },
    },
}
TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web and return the top results.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "count": {"type": "integer", "minimum": 1, "maximum": 10}},
            "required": ["query"],
        },
    },
}
TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}
TOOL_BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command in the workspace and return stdout and stderr.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
            "required": ["command"],
        },
    },
}
HERMES_SYSTEM_PROMPT = (
    "You are Hermes, an autonomous coding agent working in a terminal.  You have tools; call one whenever a step "
    "needs it, read the result, and continue until the task is done.  Keep replies short and factual.  Never invent "
    "file contents: read them.  When the task is finished, summarise what changed in three lines or fewer."
)


class ReferenceCorpusError(RuntimeError):
    pass


def _log(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, "t": round(time.time(), 3), **values}, sort_keys=True), flush=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """sha256 of the ids as little-endian uint32 (the long-context prompt's convention)."""

    return _sha256_bytes(struct.pack(f"<{len(token_ids)}I", *token_ids))


# -- packed arrays (standard library only) -----------------------------------------------------------------------


def pack_array(kind: str, values: Sequence[float | int]) -> str:
    """base64 of little-endian ``i`` (int32) or ``f`` (float32) values."""

    if kind not in ("i", "f"):
        raise ReferenceCorpusError(f"pack kind must be 'i' or 'f', got {kind!r}")
    return base64.b64encode(struct.pack(f"<{len(values)}{kind}", *values)).decode("ascii")


def unpack_array(kind: str, text: str) -> list[int] | list[float]:
    raw = base64.b64decode(text.encode("ascii"), validate=True)
    if kind not in ("i", "f") or len(raw) % 4:
        raise ReferenceCorpusError(f"packed array of kind {kind!r} has {len(raw)} bytes")
    return list(struct.unpack(f"<{len(raw) // 4}{kind}", raw))


# -- the corpus ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusItem:
    """One item: its ids (the whole text, or the prompt) and how many teacher tokens follow the prompt."""

    item_id: str
    part: str
    token_ids: tuple[int, ...]
    continuation_tokens: int  # 0 = a text item: the text is the teacher
    windows: tuple[tuple[int, int], ...]  # scored positions [start, end); () = every position with a teacher
    source: dict[str, Any] = field(default_factory=dict)

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def positions(self) -> int:
        """Positions with a teacher token: the text's, or the prompt's plus the continuation."""

        return self.prompt_tokens - 1 + self.continuation_tokens

    def scored_positions(self) -> list[int]:
        if not self.windows:
            return list(range(self.positions))
        positions: list[int] = []
        for start, end in self.windows:
            if not 0 <= start < end <= self.positions:
                raise ReferenceCorpusError(f"{self.item_id}: window {(start, end)} outside 0..{self.positions}")
            positions.extend(range(start, end))
        return positions

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "part": self.part,
            "prompt_tokens": self.prompt_tokens,
            "continuation_tokens": self.continuation_tokens,
            "teacher": "text" if self.continuation_tokens == 0 else "hf-argmax",
            "positions": self.positions,
            "scored_positions": len(self.scored_positions()),
            "windows": [list(window) for window in self.windows],
            "token_ids_sha256": token_ids_sha256(self.token_ids),
            "source": self.source,
        }


def _validate_item(document: dict[str, Any]) -> CorpusItem:
    ids = document.get("token_ids")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ReferenceCorpusError(f"{document.get('item_id')}: token_ids must be a nonempty list of ids")
    item = CorpusItem(
        item_id=str(document["item_id"]),
        part=str(document["part"]),
        token_ids=tuple(ids),
        continuation_tokens=int(document["continuation_tokens"]),
        windows=tuple((int(a), int(b)) for a, b in document.get("windows", [])),
        source=dict(document.get("source", {})),
    )
    if item.part not in PARTS or item.continuation_tokens < 0:
        raise ReferenceCorpusError(f"{item.item_id}: part {item.part!r}, continuation {item.continuation_tokens}")
    item.scored_positions()
    return item


def write_corpus(items: Sequence[CorpusItem], out_dir: Path, *, provenance: dict[str, Any]) -> dict[str, Any]:
    """The items file then the manifest (with the items file's sha256); returns the manifest."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if len({item.item_id for item in items}) != len(items):
        raise ReferenceCorpusError("item ids repeat")
    lines = [
        json.dumps(
            {
                "item_id": item.item_id,
                "part": item.part,
                "continuation_tokens": item.continuation_tokens,
                "windows": [list(window) for window in item.windows],
                "token_ids": list(item.token_ids),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in items
    ]
    items_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / ITEMS_NAME).write_bytes(items_bytes)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "corpus_id": CORPUS_ID,
        "items_file": ITEMS_NAME,
        "items_sha256": _sha256_bytes(items_bytes),
        "top_k": TOP_K,
        "continuation_tokens": CONTINUATION_TOKENS,
        "clear_margin_logits": CLEAR_MARGIN,
        "parts": {part: sum(1 for item in items if item.part == part) for part in PARTS},
        "positions": sum(item.positions for item in items),
        "scored_positions": sum(len(item.scored_positions()) for item in items),
        "provenance": provenance,
        "items": [item.manifest_entry() for item in items],
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_corpus(directory: Path = REFERENCE_DIR) -> tuple[dict[str, Any], list[CorpusItem]]:
    """The manifest and its items, every sha256 re-checked (actual vs recorded in the error)."""

    manifest_path = Path(directory) / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("corpus_id") != CORPUS_ID:
        raise ReferenceCorpusError(
            f"{manifest_path}: schema {manifest.get('schema')!r} corpus {manifest.get('corpus_id')!r}, expected "
            f"{MANIFEST_SCHEMA!r} {CORPUS_ID!r}"
        )
    items_bytes = (Path(directory) / manifest["items_file"]).read_bytes()
    actual = _sha256_bytes(items_bytes)
    if actual != manifest["items_sha256"]:
        raise ReferenceCorpusError(f"{manifest['items_file']}: sha256 {actual} vs manifest {manifest['items_sha256']}")
    items = [_validate_item(json.loads(line)) for line in items_bytes.decode("utf-8").splitlines() if line]
    entries = {entry["item_id"]: entry for entry in manifest["items"]}
    if [item.item_id for item in items] != list(entries):
        raise ReferenceCorpusError("the items file and the manifest list different items")
    for item in items:
        expected = entries[item.item_id]["token_ids_sha256"]
        if token_ids_sha256(item.token_ids) != expected:
            raise ReferenceCorpusError(f"{item.item_id}: ids sha256 {token_ids_sha256(item.token_ids)} vs {expected}")
        if item.positions != entries[item.item_id]["positions"]:
            raise ReferenceCorpusError(
                f"{item.item_id}: {item.positions} positions vs {entries[item.item_id]['positions']}"
            )
    return manifest, items


def select_items(items: Sequence[CorpusItem], parts: Sequence[str], ids: Sequence[str]) -> list[CorpusItem]:
    selected = [item for item in items if (not parts or item.part in parts) and (not ids or item.item_id in ids)]
    missing = set(ids) - {item.item_id for item in selected}
    if missing:
        raise ReferenceCorpusError(f"unknown item ids {sorted(missing)}")
    return selected


# -- freeze --------------------------------------------------------------------------------------------------------


def messages_from_transcript(text: str) -> list[dict[str, str]]:
    """The messages of a rendered chat transcript (the acceptance records' prompt text): every closed turn; an
    assistant turn's empty thinking block is dropped (the template re-emits it)."""

    turns = CHAT_TURN.findall(text)
    if not turns:
        raise ReferenceCorpusError("no closed <|im_start|>...<|im_end|> turns in the transcript")
    messages = []
    for role, content in turns:
        if role == "assistant" and content.startswith(EMPTY_THINK):
            content = content[len(EMPTY_THINK) :]
        messages.append({"role": role, "content": content})
    return messages


def _book_ids(tokenizer, text: str, count: int, *, label: str) -> list[int]:
    ids = [int(i) for i in tokenizer.encode(text, add_special_tokens=False)]
    if len(ids) < count:
        raise ReferenceCorpusError(f"{label}: {len(ids)} tokens, fewer than {count}")
    return ids[:count]


def moby_dick_body(raw: bytes) -> str:
    """The novel's body between the second chapter-1 heading and the Project Gutenberg end marker."""

    text = raw.decode("utf-8").replace("\r\n", "\n")
    marker, occurrence = MOBY_DICK_START
    start = -1
    for _ in range(occurrence):
        start = text.index(marker, start + 1)
    return text[start : text.index(MOBY_DICK_END, start)].rstrip() + "\n"


def _long_windows(tokens: int) -> tuple[tuple[int, int], ...]:
    """The 32 positions before the bridge at every standard depth inside the item, and at its end."""

    depths = [depth for depth in LONG_DEPTHS if depth <= tokens] + [tokens]
    return tuple((depth - 2 * LONG_WINDOW, depth - LONG_WINDOW) for depth in sorted(set(depths)))


def _tool_requests() -> list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]]:
    """The four requests with tools: one tool, three tools, a tool round with its result, a Hermes-shaped request."""

    weather_call = {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Kyoto"}}}
    return [
        ("tools-1", [], [TOOL_PYTHON]),  # the code prompt's messages are filled in by the caller
        ("tools-3", [], [TOOL_SEARCH, TOOL_WEATHER, TOOL_PYTHON]),
        (
            "tool-result",
            [
                {"role": "user", "content": "What's the weather in Kyoto right now?  One sentence."},
                {"role": "assistant", "content": "", "tool_calls": [weather_call]},
                {"role": "tool", "content": '{"city": "Kyoto", "temperature_c": 17, "condition": "light rain"}'},
            ],
            [TOOL_WEATHER],
        ),
        (
            "hermes",
            [
                {"role": "system", "content": HERMES_SYSTEM_PROMPT},
                {"role": "user", "content": "Summarise the README in the current directory."},
            ],
            [TOOL_READ_FILE, TOOL_BASH],
        ),
    ]


def freeze_corpus(
    *,
    checkpoint: Path,
    acceptance_dir: Path,
    tale_path: Path,
    moby_dick_path: Path,
    eval_items_dir: Path,
    out_dir: Path,
) -> dict[str, Any]:
    from models.demos.blackhole.qwen38_flash_next.chat import PINNED_TOKENIZER_ARTIFACTS, Qwen38OfficialChatTemplate
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_protocol as protocol
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_server import load_acceptance_records
    from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import SYSTEM_PROMPT

    tokenizer = Qwen38OfficialChatTemplate(checkpoint).tokenizer
    items: list[CorpusItem] = []

    def render(messages, tools, flags) -> tuple[int, ...]:
        if messages[0]["role"] != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}, *messages]
        return tuple(protocol.render_prompt(tokenizer, list(messages), list(tools), **flags))

    # acceptance: the records' prompt ids, whose transcript must re-render to the same ids (the extraction is exact)
    records = load_acceptance_records(acceptance_dir)
    record_messages: dict[str, list[dict[str, str]]] = {}
    for record in records:
        messages = messages_from_transcript(tokenizer.decode(record["prompt_token_ids"]))
        rendered = render(messages, [], RECORD_FLAGS)
        if rendered != tuple(record["prompt_token_ids"]):
            raise ReferenceCorpusError(
                f"{record['prompt']}: the transcript re-renders to {len(rendered)} ids, the record holds "
                f"{len(record['prompt_token_ids'])}; first difference at "
                f"{next((i for i, (a, b) in enumerate(zip(rendered, record['prompt_token_ids'])) if a != b), None)}"
            )
        record_messages[record["prompt"]] = messages
        items.append(
            CorpusItem(
                f"acceptance-{record['prompt']}",
                "acceptance",
                rendered,
                CONTINUATION_TOKENS,
                (),
                {"record": f"prompt-{record['prompt']}-greedy.json", "record_sha256": record["sha256"], **RECORD_FLAGS},
            )
        )
    # served: the same conversations under the server's request defaults, then the four requests with tools
    for record in records:
        items.append(
            CorpusItem(
                f"served-{record['prompt']}",
                "served",
                render(record_messages[record["prompt"]], [], SERVED_FLAGS),
                CONTINUATION_TOKENS,
                (),
                {"record": f"prompt-{record['prompt']}-greedy.json", **SERVED_FLAGS},
            )
        )
    for name, messages, tools in _tool_requests():
        if not messages:
            messages = record_messages["code" if name == "tools-1" else "fact"]
        items.append(
            CorpusItem(
                f"served-{name}",
                "served",
                render(messages, tools, SERVED_FLAGS),
                CONTINUATION_TOKENS,
                (),
                {"tools": [tool["function"]["name"] for tool in tools], "messages": messages, **SERVED_FLAGS},
            )
        )
    # book: tt-metal's Tier-1 text and the long-context prompt's opening
    tale_raw = tale_path.read_bytes()
    with bz2.open(tale_path, "rt", encoding="utf-8") as handle:
        tale_text = handle.read()
    items.append(
        CorpusItem(
            "book-tale-of-two-cities",
            "book",
            tuple(_book_ids(tokenizer, tale_text, BOOK_TOKENS, label="tale of two cities")),
            0,
            (),
            {"file": tale_path.name, "file_sha256": _sha256_bytes(tale_raw), "add_special_tokens": False},
        )
    )
    moby_raw = moby_dick_path.read_bytes()
    body = moby_dick_body(moby_raw)
    moby_ids = _book_ids(tokenizer, body, max(LONG_TOKENS), label="moby dick")
    moby_source = {
        "file": moby_dick_path.name,
        "file_sha256": _sha256_bytes(moby_raw),
        "body_sha256": _sha256_bytes(body.encode("utf-8")),
        "start_marker": MOBY_DICK_START[0],
        "start_occurrence": MOBY_DICK_START[1],
        "end_marker": MOBY_DICK_END,
        "add_special_tokens": False,
    }
    items.append(CorpusItem("book-moby-dick", "book", tuple(moby_ids[:BOOK_TOKENS]), 0, (), moby_source))
    # eval: the first two frozen items of GSM8K and HumanEval, as the evaluation harness sends them
    for name, template, key, item_key in (
        ("gsm8k", GSM8K_TEMPLATE, "question", "item_id"),
        ("humaneval", HUMANEVAL_TEMPLATE, "prompt", "task_id"),
    ):
        lines = (eval_items_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()[:2]
        for line in lines:
            row = json.loads(line)
            messages = [{"role": "user", "content": template.format(**{key: row[key]})}]
            items.append(
                CorpusItem(
                    f"eval-{name}-{row[item_key].replace('/', '-')}",
                    "eval",
                    render(messages, [], EVAL_FLAGS),
                    CONTINUATION_TOKENS,
                    (),
                    {
                        "set": name,
                        "item_id": row[item_key],
                        "line_sha256": _sha256_bytes(line.encode("utf-8")),
                        **EVAL_FLAGS,
                    },
                )
            )
    # long: the novel to 8k and to the long-context prompt's length, scored in windows
    for tokens in LONG_TOKENS:
        items.append(
            CorpusItem(
                f"long-moby-dick-{tokens}", "long", tuple(moby_ids[:tokens]), 0, _long_windows(tokens), moby_source
            )
        )

    provenance = {
        "tokenizer": {name: PINNED_TOKENIZER_ARTIFACTS[name] for name in ("tokenizer.json", "chat_template.jinja")},
        "acceptance_records": {record["prompt"]: record["sha256"] for record in records},
        "system_prompt": SYSTEM_PROMPT,
        "eval_templates": {"gsm8k": GSM8K_TEMPLATE, "humaneval": HUMANEVAL_TEMPLATE},
        "long_windows": "the 32 positions before the bridge at depths 2048 and 8192 and at the item's end",
    }
    return write_corpus(items, out_dir, provenance=provenance)


# -- the reference file ----------------------------------------------------------------------------------------------


@dataclass
class ReferenceItem:
    """One item's scored positions: per position the top-k ids (descending log-prob) and the teacher's log-prob."""

    item_id: str
    positions: list[int]
    teacher_ids: list[int]
    top_ids: list[list[int]]
    top_logprobs: list[list[float]]
    teacher_logprobs: list[float | None]

    def to_document(self) -> dict[str, Any]:
        k = len(self.top_ids[0]) if self.top_ids else TOP_K
        teacher = [float("nan") if value is None else value for value in self.teacher_logprobs]
        return {
            "item_id": self.item_id,
            "count": len(self.positions),
            "top_k": k,
            "positions": pack_array("i", self.positions),
            "teacher_ids": pack_array("i", self.teacher_ids),
            "top_ids": pack_array("i", [i for row in self.top_ids for i in row]),
            "top_logprobs": pack_array("f", [v for row in self.top_logprobs for v in row]),
            "teacher_logprobs": pack_array("f", teacher),
        }

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> "ReferenceItem":
        count, k = int(document["count"]), int(document["top_k"])
        flat_ids = unpack_array("i", document["top_ids"])
        flat_lp = unpack_array("f", document["top_logprobs"])
        if len(flat_ids) != count * k or len(flat_lp) != count * k:
            raise ReferenceCorpusError(f"{document['item_id']}: {len(flat_ids)} ids for {count} x {k}")
        teacher = unpack_array("f", document["teacher_logprobs"])
        return cls(
            str(document["item_id"]),
            unpack_array("i", document["positions"]),
            unpack_array("i", document["teacher_ids"]),
            [flat_ids[i * k : (i + 1) * k] for i in range(count)],
            [flat_lp[i * k : (i + 1) * k] for i in range(count)],
            [None if math.isnan(value) else value for value in teacher],
        )


def write_reference(
    path: Path, *, manifest: dict[str, Any], producer: dict[str, Any], items: Sequence[ReferenceItem]
) -> None:
    document = {
        "schema": REFERENCE_SCHEMA,
        "corpus_id": manifest["corpus_id"],
        "corpus_items_sha256": manifest["items_sha256"],
        "producer": producer,
        "items": [item.to_document() for item in items],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_reference(
    path: Path, *, manifest: dict[str, Any] | None = None
) -> tuple[dict[str, Any], dict[str, ReferenceItem]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != REFERENCE_SCHEMA:
        raise ReferenceCorpusError(f"{path}: schema {document.get('schema')!r}, expected {REFERENCE_SCHEMA!r}")
    if manifest is not None and document.get("corpus_items_sha256") != manifest["items_sha256"]:
        raise ReferenceCorpusError(
            f"{path}: corpus items sha256 {document.get('corpus_items_sha256')} vs manifest {manifest['items_sha256']}"
        )
    items = [ReferenceItem.from_document(entry) for entry in document["items"]]
    return document, {item.item_id: item for item in items}


# -- producers (torch) ---------------------------------------------------------------------------------------------


def topk_rows(logits, teacher_ids: Sequence[int | None], k: int = TOP_K):
    """fp32 log-softmax of ``logits`` [n, V]: the top-k ids and log-probs per row and the teacher's log-prob."""

    import torch

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = logprobs.topk(k, dim=-1)
    teacher = [None if token is None else float(logprobs[row, token]) for row, token in enumerate(teacher_ids)]
    return indices.tolist(), values.tolist(), teacher


def bfp4_round(weight):
    """The BF4 (Bfp4_b) value of every element of a linear weight [out, in]: blocks of 16 consecutive output rows share
    the block's largest exponent, each element keeps a sign and 3 mantissa bits (round to nearest, ties to even,
    clamped to 7), the way ``bf4_host_packer.cpp`` packs a tile's 16-element face rows along the matmul output."""

    import torch

    out_features = weight.shape[0]
    if out_features % 16:
        raise ReferenceCorpusError(f"BF4 blocks need 16 | out_features, got {out_features}")
    x = weight.detach().float().reshape(out_features // 16, 16, -1)
    magnitude = x.abs()
    _, exponent = torch.frexp(magnitude)  # |x| = m * 2^e, m in [0.5, 1): the biased exponent field is e - 1 + 127
    exponent = torch.where(magnitude > 0, exponent - 1, torch.full_like(exponent, -200))
    shared = exponent.amax(dim=1, keepdim=True)
    scale = torch.exp2((shared - 2).float())  # mantissa 3 bits: value = q * 2^(shared - 2), q in 0..7
    q = torch.round(magnitude / scale).clamp_(max=7.0)
    rounded = torch.sign(x) * q * scale
    return rounded.reshape(weight.shape).to(torch.bfloat16)


class _Runner:
    """A model column: ``forward`` consumes ids with a carried state and returns the final hidden states; the fp32 LM
    head is applied here so every column's logits are the same arithmetic."""

    head_weight = None  # fp32 [V, H]

    def forward(self, ids, state):  # -> (hidden [1, m, H], state)
        raise NotImplementedError

    def logits(self, hidden, rows: int = 512):
        import torch

        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        return torch.cat([flat[i : i + rows] @ self.head_weight.T for i in range(0, flat.shape[0], rows)])


HF_EXPERTS_INDEXED = "eager-indexed"


def indexed_eager_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """transformers' eager experts loop with its per-expert ``index_add_`` written as an indexed add: a token is
    routed to an expert once, so the rows are unique and the two are bitwise equal, and the bf16 ``index_add_`` CPU
    kernel (milliseconds per call at one row) is avoided."""

    import torch
    import torch.nn.functional as F

    final = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert in expert_hit:
        expert_index = expert[0]
        if expert_index == self.num_experts:
            continue
        top_k_position, token_index = torch.where(expert_mask[expert_index])
        current = hidden_states[token_index]
        gate, up = F.linear(current, self.gate_up_proj[expert_index]).chunk(2, dim=-1)
        current = F.linear(self.act_fn(gate) * up, self.down_proj[expert_index])
        current = current * top_k_weights[token_index, top_k_position, None]
        final[token_index] += current.to(final.dtype)
    return final


class HFRunner(_Runner):
    def __init__(self, checkpoint: Path, *, attn_implementation: str | None, experts_implementation: str | None):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

        if HF_EXPERTS_INDEXED not in ALL_EXPERTS_FUNCTIONS:
            ALL_EXPERTS_FUNCTIONS.register(HF_EXPERTS_INDEXED, indexed_eager_experts_forward)
        kwargs: dict[str, Any] = {"dtype": torch.bfloat16, "low_cpu_mem_usage": True, "local_files_only": True}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if experts_implementation:
            # the eager forms visit the routed experts only; the batched forms read all 512 experts' weights per token
            kwargs["experts_implementation"] = experts_implementation
        self.model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs).eval()
        inner = self.model.model
        self.text_model = getattr(inner, "language_model", inner)
        self.head_weight = self.model.lm_head.weight.detach().float()
        self.description = {
            "kind": "hf",
            "model_class": type(self.model).__name__,
            "text_model_class": type(self.text_model).__name__,
            "attn_implementation": getattr(self.model.config, "_attn_implementation", None),
            "experts_implementation": getattr(self.text_model.config, "_experts_implementation", None),
            "weights": "bf16",
            "experts": "bf16",
            "lm_head": "fp32 (bf16 final hidden state, fp32 weight copy), fp32 log-softmax",
        }

    def forward(self, ids, state):
        output = self.text_model(input_ids=ids, past_key_values=state, use_cache=True)
        return output.last_hidden_state, output.past_key_values


class OracleRunner(_Runner):
    def __init__(self, checkpoint: Path, *, experts: str, rss_limit_gib: float, kv: str = "bf16"):
        import torch

        from models.demos.blackhole.qwen38_flash_next.checkpoint import Qwen38Checkpoint
        from models.demos.blackhole.qwen38_flash_next.config import Qwen38Placement
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_kv_cache_emulation import (
            KVCacheRounding,
            install_kv_rounding,
        )
        from models.demos.blackhole.qwen38_flash_next.tt.model import Qwen38TextModelOracle
        from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38ExpertWeights

        if experts not in ("bf16", "bf4"):
            raise ReferenceCorpusError(f"experts must be bf16 or bf4, got {experts!r}")
        if kv not in KV_CHOICES:
            raise ReferenceCorpusError(f"kv must be one of {KV_CHOICES}, got {kv!r}")
        loaded = Qwen38Checkpoint(checkpoint)
        placement = Qwen38Placement(loaded.config, mesh_shape=(1, 4), physical_ids=(0, 1, 2, 3))
        self.oracle = Qwen38TextModelOracle(loaded, placement)
        io = self.oracle.model_io
        self.head_weight = torch.cat([io.lm_head_weight_shard(device) for device in range(4)]).float()
        self.rss_limit_gib = rss_limit_gib
        self.experts = experts
        self.quantised: dict[tuple[int, int], Any] = {}
        # The checkpoint's expert slices alias their shard's mmap: a cache of them holds one 3.5 GB mapping per
        # slice, and the address space runs out near 16k experts (ENOMEM from mmap, 45k mappings).  The cache
        # holds copies: BF4-rounded, or the bf16 values cloned off the mapping.
        original_layer = self.oracle.layer
        rounding = bfp4_round if experts == "bf4" else (lambda weight: weight.clone())

        def layer(layer_index: int):
            created = original_layer(layer_index)
            weights = created.mlp.weights
            if not getattr(weights, "_copy_wrapped", False):
                original_expert = weights.expert

                def expert(expert_index: int, _index=layer_index, _original=original_expert):
                    cached = self.quantised.get((_index, expert_index))
                    if cached is None:
                        exact = _original(expert_index)
                        cached = Qwen38ExpertWeights(
                            rounding(exact.gate_up), rounding(exact.down), exact.intermediate_size
                        )
                        self.quantised[(_index, expert_index)] = cached
                        weights._expert_cache.pop(expert_index, None)  # only the copy is kept
                    return cached

                weights.expert = expert  # type: ignore[method-assign]
                weights._copy_wrapped = True  # type: ignore[attr-defined]
            return created

        self.oracle.layer = layer  # type: ignore[method-assign]
        # The QSA K/V rounding wraps the layer factory after the expert wrapper (the two compose).
        self.kv_rounding = KVCacheRounding(kv)
        install_kv_rounding(self.oracle, self.kv_rounding)
        self.description = {
            "kind": "oracle",
            "model_class": type(self.oracle).__name__,
            "weights": "bf16",
            "experts": (
                "bf4-emulated (blocks of 16 along the matmul output, shared exponent, 3 mantissa bits)"
                if experts == "bf4"
                else "bf16"
            ),
            "kv_cache": self.kv_rounding.describe(),
            "lm_head": "fp32 (bf16 final hidden state, fp32 weight copy), fp32 log-softmax",
        }

    def forward(self, ids, state):
        output = self.oracle.forward(ids, state=state, return_logits=False)
        if _current_rss_gib() > self.rss_limit_gib:
            for layer in self.oracle._layers.values():
                layer.mlp.weights._expert_cache.clear()
            self.quantised.clear()
            _log("expert_cache_cleared", rss_gib=round(_current_rss_gib(), 1))
        return output.hidden_states, output.state


def _current_rss_gib() -> float:
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024**3)
    except OSError:
        return 0.0


def reference_item(
    runner: _Runner,
    item: CorpusItem,
    *,
    chunk_tokens: int,
    teacher_ids: Sequence[int] | None = None,
    top_k: int = TOP_K,
    full_logits_dir: Path | None = None,
) -> ReferenceItem:
    """Teacher-force one item through ``runner``: the prompt (or text) in chunks, then the continuation one token at a
    time, feeding ``teacher_ids`` when given (another column's chain) or the runner's own argmax (the HF column)."""

    import torch

    scored = item.scored_positions()
    wanted = set(scored)
    ids = list(item.token_ids)
    tokens = item.prompt_tokens
    rows: dict[int, tuple[list[int], list[float], float | None, int]] = {}
    kept_logits: dict[int, Any] = {}
    state = None
    started = time.monotonic()
    prefill_positions = 0

    def keep(first_position: int, logits, teachers: Sequence[int | None]) -> None:
        positions = [first_position + offset for offset in range(logits.shape[0])]
        chosen = [index for index, position in enumerate(positions) if position in wanted]
        if not chosen:
            return
        selected = logits[chosen]
        top_ids, top_lp, teacher_lp = topk_rows(selected, [teachers[index] for index in chosen], top_k)
        for row, index in enumerate(chosen):
            teacher_id = teachers[index]
            rows[positions[index]] = (
                top_ids[row],
                top_lp[row],
                teacher_lp[row],
                -1 if teacher_id is None else teacher_id,
            )
            if full_logits_dir is not None:
                kept_logits[positions[index]] = selected[row].to(torch.float16)

    with torch.inference_mode():
        for start in range(0, tokens, chunk_tokens):
            chunk = ids[start : start + chunk_tokens]
            hidden, state = runner.forward(torch.tensor([chunk], dtype=torch.long), state)
            logits = runner.logits(hidden)
            # position start + i predicts ids[start + i + 1]; the prompt's last position starts the continuation
            teachers: list[int | None] = [
                ids[p + 1] if p + 1 < tokens else None for p in range(start, start + len(chunk))
            ]
            if item.continuation_tokens and start + len(chunk) == tokens:
                teachers[-1] = int(logits[-1].argmax()) if teacher_ids is None else int(teacher_ids[0])
            keep(start, logits, teachers)
            prefill_positions += len(chunk)
            _log(
                "prefill_progress",
                item=item.item_id,
                tokens=prefill_positions,
                seconds=round(time.monotonic() - started, 1),
                rss_gib=round(_current_rss_gib(), 1),
            )
            if item.continuation_tokens and start + len(chunk) == tokens:
                next_token = teachers[-1]
        prefill_seconds = time.monotonic() - started
        step_seconds: list[float] = []
        chain: list[int] = []
        if item.continuation_tokens:
            chain.append(int(next_token))
            for step in range(1, item.continuation_tokens):
                step_started = time.monotonic()
                hidden, state = runner.forward(torch.tensor([[chain[-1]]], dtype=torch.long), state)
                logits = runner.logits(hidden)
                following = int(logits[-1].argmax()) if teacher_ids is None else int(teacher_ids[step])
                keep(tokens - 1 + step, logits, [following])
                chain.append(following)
                step_seconds.append(time.monotonic() - step_started)
                if step % 32 == 0:
                    _log(
                        "continuation_progress",
                        item=item.item_id,
                        step=step,
                        seconds_per_step=round(sum(step_seconds[-32:]) / len(step_seconds[-32:]), 3),
                        rss_gib=round(_current_rss_gib(), 1),
                    )
    if teacher_ids is not None and list(teacher_ids[: item.continuation_tokens]) != chain:
        raise ReferenceCorpusError(f"{item.item_id}: the fed chain drifted from the teacher chain")
    missing = [position for position in scored if position not in rows]
    if missing:
        raise ReferenceCorpusError(
            f"{item.item_id}: {len(missing)} scored positions without logits, first {missing[0]}"
        )
    _log(
        "item_done",
        item=item.item_id,
        prefill_tokens=prefill_positions,
        prefill_seconds=round(prefill_seconds, 2),
        prefill_tokens_per_second=round(prefill_positions / prefill_seconds, 2) if prefill_seconds else None,
        continuation_tokens=item.continuation_tokens,
        seconds_per_step=round(sum(step_seconds) / len(step_seconds), 3) if step_seconds else None,
        rss_gib=round(_current_rss_gib(), 1),
    )
    if full_logits_dir is not None:
        full_logits_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "item_id": item.item_id,
                "positions": scored,
                "logits_fp16": torch.stack([kept_logits[p] for p in scored]),
            },
            full_logits_dir / f"{item.item_id}.pt",
        )
    return ReferenceItem(
        item.item_id,
        scored,
        [rows[p][3] for p in scored],
        [rows[p][0] for p in scored],
        [rows[p][1] for p in scored],
        [rows[p][2] for p in scored],
    )


def produce_reference(
    runner: _Runner,
    manifest: dict[str, Any],
    items: Sequence[CorpusItem],
    *,
    out: Path,
    chunk_tokens: int,
    teacher: dict[str, ReferenceItem] | None,
    full_logits_dir: Path | None,
    producer: dict[str, Any],
) -> None:
    """Every selected item, the file rewritten after each so a restart (``--resume``) continues from the last one."""

    done: dict[str, ReferenceItem] = {}
    if out.exists():
        document, done = load_reference(out, manifest=manifest)
        if document["producer"] != producer:
            raise ReferenceCorpusError(f"{out}: producer {document['producer']} differs from {producer}; remove it")
        _log("resume", items_done=sorted(done))
    for item in items:
        if item.item_id in done:
            continue
        teacher_ids = None
        if item.continuation_tokens and teacher is not None:
            if item.item_id not in teacher:
                raise ReferenceCorpusError(f"{item.item_id}: not in the teacher reference")
            column = teacher[item.item_id]
            by_position = dict(zip(column.positions, column.teacher_ids))
            teacher_ids = [by_position[p] for p in range(item.prompt_tokens - 1, item.positions)]
        _log(
            "item_start",
            item=item.item_id,
            part=item.part,
            prompt_tokens=item.prompt_tokens,
            positions=len(item.scored_positions()),
        )
        done[item.item_id] = reference_item(
            runner, item, chunk_tokens=chunk_tokens, teacher_ids=teacher_ids, full_logits_dir=full_logits_dir
        )
        write_reference(
            out, manifest=manifest, producer=producer, items=[done[i.item_id] for i in items if i.item_id in done]
        )
    _log("reference_written", path=str(out), items=len(done), sha256=_sha256_file(out))


# -- device column -----------------------------------------------------------------------------------------------


def device_reference_items(records_dir: Path, corpus: Sequence[CorpusItem]) -> list[ReferenceItem]:
    """A served chain's agreement records (one ``<item_id>.json`` per item: argmax and the 32-candidate row per
    position) as a column; log-probs are normalised over the row (log-softmax of the candidates' logits)."""

    by_id = {item.item_id: item for item in corpus}
    items = []
    for path in sorted(records_dir.glob("*.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != AGREEMENT_RECORDS_SCHEMA:
            raise ReferenceCorpusError(
                f"{path}: schema {document.get('schema')!r}, expected {AGREEMENT_RECORDS_SCHEMA!r}"
            )
        item = by_id.get(document.get("item_id"))
        if item is None:
            raise ReferenceCorpusError(f"{path}: item {document.get('item_id')!r} is not in the corpus")
        positions, teacher_ids, top_ids, top_lp = [], [], [], []
        for entry in document["positions"]:
            ids = [int(i) for i in entry["candidate_ids"]]
            logits = [float(v) for v in entry["candidate_logits"]]
            if len(ids) != len(logits) or not ids:
                raise ReferenceCorpusError(f"{path}: position {entry.get('position')} candidate row is malformed")
            argmax = int(entry["argmax"])
            # descending logits; the device's own argmax first among a tie at the top (its resolve broke that tie)
            order = sorted(range(len(ids)), key=lambda i: (-logits[i], ids[i] != argmax, ids[i]))
            if ids[order[0]] != argmax:
                raise ReferenceCorpusError(
                    f"{path}: position {entry['position']} argmax {argmax} is not the row's top {ids[order[0]]}"
                )
            peak = logits[order[0]]
            normaliser = peak + math.log(sum(math.exp(v - peak) for v in logits))
            positions.append(int(entry["position"]))
            teacher_ids.append(int(entry["teacher_id"]))
            top_ids.append([ids[i] for i in order])
            top_lp.append([logits[i] - normaliser for i in order])
        items.append(ReferenceItem(item.item_id, positions, teacher_ids, top_ids, top_lp, [None] * len(positions)))
    if not items:
        raise ReferenceCorpusError(f"no agreement records under {records_dir}")
    return items


# -- score -------------------------------------------------------------------------------------------------------


def _truncated_kl(
    a_ids: Sequence[int], a_lp: Sequence[float], b_ids: Sequence[int], b_lp: Sequence[float]
) -> tuple[float, float] | None:
    """KL(a || b) over the ids both rows hold, each renormalised over that support; and a's mass on it."""

    b_by_id = dict(zip(b_ids, b_lp))
    shared = [(lp, b_by_id[i]) for i, lp in zip(a_ids, a_lp) if i in b_by_id]
    if not shared:
        return None
    a_mass = sum(math.exp(lp) for lp, _ in shared)
    b_mass = sum(math.exp(lp) for _, lp in shared)
    kl = sum(math.exp(lp) / a_mass * ((lp - math.log(a_mass)) - (blp - math.log(b_mass))) for lp, blp in shared)
    return max(kl, 0.0), a_mass


def score_item(a: ReferenceItem, b: ReferenceItem, *, clear_margin: float) -> dict[str, Any]:
    b_index = {position: row for row, position in enumerate(b.positions)}
    shared = [(row, b_index[position]) for row, position in enumerate(a.positions) if position in b_index]
    counts = {"top1": 0, "top5": 0, "a_in_b_top32": 0, "clear": 0, "clear_top1": 0, "kl_positions": 0}
    kl_sum = 0.0
    mass_sum = 0.0
    gap_sum = 0.0
    gaps = 0
    first_divergence = None
    segments: dict[int, list[int]] = {}
    for ra, rb in shared:
        a1, b1 = a.top_ids[ra][0], b.top_ids[rb][0]
        agree = a1 == b1
        counts["top1"] += agree
        counts["top5"] += b1 in a.top_ids[ra][:5]
        counts["a_in_b_top32"] += a1 in b.top_ids[rb]
        margin = a.top_logprobs[ra][0] - a.top_logprobs[ra][1]
        if margin > clear_margin:
            counts["clear"] += 1
            counts["clear_top1"] += agree
        if not agree and first_divergence is None:
            first_divergence = a.positions[ra]
        kl = _truncated_kl(a.top_ids[ra], a.top_logprobs[ra], b.top_ids[rb], b.top_logprobs[rb])
        if kl is not None:
            counts["kl_positions"] += 1
            kl_sum += kl[0]
            mass_sum += kl[1]
        if a.teacher_logprobs[ra] is not None and b.teacher_logprobs[rb] is not None:
            gap_sum += a.teacher_logprobs[ra] - b.teacher_logprobs[rb]
            gaps += 1
        segment = segments.setdefault(a.positions[ra] // SEGMENT_TOKENS, [0, 0])
        segment[0] += agree
        segment[1] += 1
    n = len(shared)
    rate = lambda value, total: None if not total else round(value / total, 4)  # noqa: E731
    return {
        "item_id": a.item_id,
        "positions": n,
        "top1": rate(counts["top1"], n),
        "top5": rate(counts["top5"], n),
        "a_in_b_top32": rate(counts["a_in_b_top32"], n),
        "clear_positions": counts["clear"],
        "clear_top1": rate(counts["clear_top1"], counts["clear"]),
        "kl_mean": rate(kl_sum, counts["kl_positions"]),
        "a_mass_on_shared_support": rate(mass_sum, counts["kl_positions"]),
        "teacher_logprob_gap_mean": rate(gap_sum, gaps),
        "first_divergence": first_divergence,
        "segments": {str(k * SEGMENT_TOKENS): rate(v[0], v[1]) for k, v in sorted(segments.items())},
        "_counts": counts,
        "_kl_sum": kl_sum,
        "_gap_sum": gap_sum,
        "_gaps": gaps,
    }


def score_references(
    a: dict[str, ReferenceItem], b: dict[str, ReferenceItem], corpus: Sequence[CorpusItem], *, clear_margin: float
) -> dict[str, Any]:
    """Every item both columns hold; per part and over the corpus the position-weighted rates."""

    parts = {item.item_id: item.part for item in corpus}
    items = [score_item(a[i], b[i], clear_margin=clear_margin) for i in a if i in b]
    if not items:
        raise ReferenceCorpusError("the two references share no items")

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        n = sum(row["positions"] for row in rows)
        clear = sum(row["_counts"]["clear"] for row in rows)
        kl_n = sum(row["_counts"]["kl_positions"] for row in rows)
        gaps = sum(row["_gaps"] for row in rows)
        return {
            "items": len(rows),
            "positions": n,
            "top1": round(sum(r["_counts"]["top1"] for r in rows) / n, 4) if n else None,
            "top5": round(sum(r["_counts"]["top5"] for r in rows) / n, 4) if n else None,
            "a_in_b_top32": round(sum(r["_counts"]["a_in_b_top32"] for r in rows) / n, 4) if n else None,
            "clear_positions": clear,
            "clear_top1": round(sum(r["_counts"]["clear_top1"] for r in rows) / clear, 4) if clear else None,
            "kl_mean": round(sum(r["_kl_sum"] for r in rows) / kl_n, 4) if kl_n else None,
            "teacher_logprob_gap_mean": round(sum(r["_gap_sum"] for r in rows) / gaps, 4) if gaps else None,
        }

    by_part = {}
    for part in PARTS:
        rows = [row for row in items if parts.get(row["item_id"]) == part]
        if rows:
            by_part[part] = aggregate(rows)
    return {
        "schema": SCORE_SCHEMA,
        "clear_margin_logits": clear_margin,
        "corpus": aggregate(items),
        "parts": by_part,
        "items": [{k: v for k, v in row.items() if not k.startswith("_")} for row in items],
    }


def format_score(score: dict[str, Any], *, a_name: str, b_name: str) -> str:
    lines = [f"a = {a_name}   b = {b_name}   clear margin {score['clear_margin_logits']} logits"]
    header = f"{'item':40} {'pos':>6} {'top1':>7} {'top5':>7} {'a in b32':>9} {'clear':>6} {'clr top1':>9} {'KL':>8} {'gap':>8} {'first div':>9}"
    lines.append(header)
    fmt = lambda value, width=7: f"{'-':>{width}}" if value is None else f"{value:>{width}.4f}"  # noqa: E731
    for row in score["items"]:
        lines.append(
            f"{row['item_id']:40} {row['positions']:>6} {fmt(row['top1'])} {fmt(row['top5'])} {fmt(row['a_in_b_top32'], 9)} "
            f"{row['clear_positions']:>6} {fmt(row['clear_top1'], 9)} {fmt(row['kl_mean'], 8)} "
            f"{fmt(row['teacher_logprob_gap_mean'], 8)} {'-' if row['first_divergence'] is None else row['first_divergence']:>9}"
        )
    for name, row in list(score["parts"].items()) + [("corpus", score["corpus"])]:
        lines.append(
            f"{name:40} {row['positions']:>6} {fmt(row['top1'])} {fmt(row['top5'])} {fmt(row['a_in_b_top32'], 9)} "
            f"{row['clear_positions']:>6} {fmt(row['clear_top1'], 9)} {fmt(row['kl_mean'], 8)} "
            f"{fmt(row['teacher_logprob_gap_mean'], 8)} {'':>9}"
        )
    return "\n".join(lines)


# -- main ----------------------------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)

    freeze = modes.add_parser("freeze", help="build the corpus files")
    freeze.add_argument("--checkpoint", type=Path, required=True)
    freeze.add_argument("--acceptance-dir", type=Path, default=ACCEPTANCE_DIR)
    freeze.add_argument("--tale", type=Path, default=TALE_OF_TWO_CITIES, help="tale-of-two-cities.txt.bz2")
    freeze.add_argument("--moby-dick", type=Path, required=True, help="Project Gutenberg ebook 2701 as plain text")
    freeze.add_argument("--eval-items", type=Path, required=True, help="directory with gsm8k.jsonl and humaneval.jsonl")
    freeze.add_argument("--out", type=Path, default=REFERENCE_DIR)

    for name in ("hf", "oracle"):
        run = modes.add_parser(name, help=f"produce the {name} column")
        run.add_argument("--checkpoint", type=Path, required=True)
        run.add_argument("--corpus", type=Path, default=REFERENCE_DIR)
        run.add_argument("--out", type=Path, required=True, help="the reference file (rewritten after every item)")
        run.add_argument("--parts", nargs="*", default=[], choices=PARTS)
        run.add_argument("--items", nargs="*", default=[])
        run.add_argument("--chunk-tokens", type=int, default=1024, help="prefill chunk (bounds the logits' memory)")
        run.add_argument("--threads", type=int, default=None)
        run.add_argument("--full-logits", type=Path, default=None, help="also keep fp16 logits per item here")
        run.add_argument(
            "--teacher", type=Path, default=None, help="a reference file whose argmax chain is fed (the HF column)"
        )
        if name == "hf":
            run.add_argument(
                "--attn-implementation", default=None, help="e.g. eager or sdpa; the model's default otherwise"
            )
            run.add_argument(
                "--experts-implementation",
                default=HF_EXPERTS_INDEXED,
                help="transformers' MoE experts kernel: eager-indexed (the eager loop, bitwise, without the slow bf16 "
                "index_add_), eager, grouped_mm, batched_mm",
            )
        else:
            run.add_argument("--experts", choices=("bf16", "bf4"), default="bf16")
            run.add_argument(
                "--kv",
                choices=KV_CHOICES,
                default="bf16",
                help="the QSA K/V cache format the oracle emulates at cache-write time (the device's sparse_sdpa formats)",
            )
            run.add_argument("--rss-limit-gib", type=float, default=300.0)

    device = modes.add_parser("device", help="convert a served chain's agreement records into a column")
    device.add_argument("--records", type=Path, required=True)
    device.add_argument("--corpus", type=Path, default=REFERENCE_DIR)
    device.add_argument("--out", type=Path, required=True)
    device.add_argument("--label", default="device", help="the producer label recorded in the file")

    score = modes.add_parser("score", help="compare two columns")
    score.add_argument("--a", type=Path, required=True, help="the reference column (HF)")
    score.add_argument("--b", type=Path, required=True)
    score.add_argument("--corpus", type=Path, default=REFERENCE_DIR)
    score.add_argument("--clear-margin", type=float, default=CLEAR_MARGIN)
    score.add_argument("--out", type=Path, default=None, help="write the score JSON here")

    verify = modes.add_parser("verify", help="check the corpus files (and a reference file against them)")
    verify.add_argument("--corpus", type=Path, default=REFERENCE_DIR)
    verify.add_argument("--reference", type=Path, default=None)
    verify.add_argument(
        "--stamp",
        type=Path,
        default=None,
        help="write the reference file's identity (name, sha256, bytes, items, producer) as JSON here: the repository "
        "keeps the stamp, the hosts keep the file",
    )
    verify.add_argument("--expect-stamp", type=Path, default=None, help="a stamp the reference file must match")
    return parser


def reference_stamp(path: Path, document: dict[str, Any], columns: dict[str, ReferenceItem]) -> dict[str, Any]:
    return {
        "schema": "qwen38-reference-stamp/v1",
        "file": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "corpus_id": document["corpus_id"],
        "corpus_items_sha256": document["corpus_items_sha256"],
        "items": sorted(columns),
        "positions": sum(len(column.positions) for column in columns.values()),
        "producer": document["producer"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "freeze":
        manifest = freeze_corpus(
            checkpoint=args.checkpoint,
            acceptance_dir=args.acceptance_dir,
            tale_path=args.tale,
            moby_dick_path=args.moby_dick,
            eval_items_dir=args.eval_items,
            out_dir=args.out,
        )
        print(json.dumps({k: v for k, v in manifest.items() if k != "items"}, indent=1, sort_keys=True))
        return 0
    if args.mode == "verify":
        manifest, items = load_corpus(args.corpus)
        print(
            f"{manifest['corpus_id']}: {len(items)} items, {manifest['positions']} positions, {manifest['scored_positions']} scored"
        )
        if args.reference is not None:
            document, columns = load_reference(args.reference, manifest=manifest)
            print(
                f"{args.reference}: {len(columns)} items, producer {json.dumps(document['producer'], sort_keys=True)}"
            )
            stamp = reference_stamp(args.reference, document, columns)
            if args.expect_stamp is not None:
                expected = json.loads(args.expect_stamp.read_text(encoding="utf-8"))
                differing = {k: (stamp.get(k), expected.get(k)) for k in expected if stamp.get(k) != expected.get(k)}
                if differing:
                    raise ReferenceCorpusError(
                        f"{args.reference} differs from the stamp {args.expect_stamp}: {differing}"
                    )
                print(f"matches the stamp {args.expect_stamp}")
            if args.stamp is not None:
                args.stamp.write_text(json.dumps(stamp, indent=1, sort_keys=True) + "\n", encoding="utf-8")
                print(f"stamp written to {args.stamp}: sha256 {stamp['sha256']}, {stamp['bytes']} bytes")
        return 0
    manifest, corpus = load_corpus(args.corpus)
    if args.mode == "device":
        items = device_reference_items(args.records, corpus)
        producer = {"kind": "device", "label": args.label, "normalisation": "log-softmax over the candidate row"}
        write_reference(args.out, manifest=manifest, producer=producer, items=items)
        print(f"{args.out}: {len(items)} items, sha256 {_sha256_file(args.out)}")
        return 0
    if args.mode == "score":
        _, a = load_reference(args.a, manifest=manifest)
        _, b = load_reference(args.b, manifest=manifest)
        score = score_references(a, b, corpus, clear_margin=args.clear_margin)
        print(format_score(score, a_name=str(args.a), b_name=str(args.b)))
        if args.out is not None:
            args.out.write_text(json.dumps(score, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    items = select_items(corpus, args.parts, args.items)
    teacher = None
    if args.teacher is not None:
        _, teacher = load_reference(args.teacher, manifest=manifest)
    elif any(item.continuation_tokens for item in items) and args.mode == "oracle":
        raise ReferenceCorpusError("the oracle column needs --teacher (the HF reference) for the prompt items")
    _log(
        "start",
        mode=args.mode,
        items=[item.item_id for item in items],
        threads=torch.get_num_threads(),
        chunk_tokens=args.chunk_tokens,
    )
    loaded = time.monotonic()
    if args.mode == "hf":
        runner: _Runner = HFRunner(
            args.checkpoint,
            attn_implementation=args.attn_implementation,
            experts_implementation=args.experts_implementation,
        )
    else:
        runner = OracleRunner(args.checkpoint, experts=args.experts, rss_limit_gib=args.rss_limit_gib, kv=args.kv)
    _log(
        "model_loaded",
        seconds=round(time.monotonic() - loaded, 1),
        rss_gib=round(_current_rss_gib(), 1),
        **runner.description,
    )
    producer = {
        **runner.description,
        "corpus_id": manifest["corpus_id"],
        "top_k": TOP_K,
        "chunk_tokens": args.chunk_tokens,
        "teacher": "own argmax" if teacher is None else "the HF reference's chain",
        "torch": torch.__version__,
    }
    if args.mode == "hf":
        import transformers

        producer["transformers"] = transformers.__version__
    produce_reference(
        runner,
        manifest,
        items,
        out=args.out,
        chunk_tokens=args.chunk_tokens,
        teacher=teacher,
        full_logits_dir=args.full_logits,
        producer=producer,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReferenceCorpusError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
