# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Independent candidate-classification gate for LLK persistent state.

This gate is deliberately independent of the effect-extraction model. It scans
every ``_llk_*`` definition and every reachable helper body for source tokens
that *look like* a persistent-state primitive -- an instruction/macro call
(``TTI_*`` / ``TT_OP_*`` / ``TT_*``), a persistent MMIO/config register-file
assignment (``cfg[...] =`` / ``regfile[...] =`` / ``mop_cfg[...] =``), or a
persistent software-shadow assignment -- and requires each such candidate to be
deliberately classified by the reviewed :mod:`candidate_model` as one of five
categories:

* ``persistent_immediate`` -- ``TTI_``/direct-MMIO/software write that mutates
  persistent architectural state immediately;
* ``persistent_deferred`` -- ``TT_OP_`` built instruction whose persistent
  effect activates when the enclosing MOP/replay program executes;
* ``programming`` -- MOP/replay program configuration writes;
* ``transient_datapath`` -- MOV/ELW/pool/SFPU data-plane mutation, summarized
  rather than enumerated instruction by instruction;
* ``reviewed_non_state`` -- reviewed ordering-only / no-state primitives.

An unclassified instruction or MMIO candidate is an audit error reported with
architecture, source path, line, and token so the reviewed model -- never the
gate -- is what gets updated when the instruction set grows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .inventory import (
    AuditModelError,
    _matching_delimiter,
    _skip_space,
)

CATEGORIES = {
    "persistent_immediate",
    "persistent_deferred",
    "programming",
    "transient_datapath",
    "reviewed_non_state",
}
_CLASSES = {"persistent", "deferred", "programming", "transient", "order_only"}
_ARCHITECTURES = ("wormhole_b0", "blackhole", "quasar")

_INSTRUCTION = re.compile(r"\b(TTI_|TT_OP_|TT_)([A-Za-z][A-Za-z0-9_]*)\b")
_MMIO = re.compile(r"\b(mop_cfg|regfile|cfg)\s*\[")
_HELPER_CALL = re.compile(r"\b(mailbox_write|mailbox_read)\b")
_ASSIGN_NAME = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")
_ASSIGNMENT_OPERATOR = re.compile(r"(?:<<|>>|[+\-*/%&|^])?=(?!=)")
_LOCAL_DECLARATION = re.compile(
    r"\b(?P<qualifiers>(?:(?:const|volatile|constexpr|register|static)\s+)*)"
    r"[A-Za-z_]\w*(?:::\w+)*(?:\s*<[^;{}=]+>)?\s*[*&]*\s+"
    r"(?P<name>[A-Za-z_]\w*)\s*(?==|;|,)"
)
_REVIEWABLE_SOFTWARE_NAME = re.compile(
    r"(?:configured_.*|.*_state|.*_context|shadow.*|.*replay_init|"
    r".*(?:format|fmt).*cache.*|.*cache.*(?:format|fmt).*)"
)


def load_candidate_model(path: Path | str | None = None) -> dict[str, Any]:
    """Load and validate the reviewed candidate-classification model."""
    if path is None:
        path = Path(__file__).with_name("candidate_model.json")
    try:
        model = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditModelError(f"cannot load candidate model: {error}") from error
    if not isinstance(model, dict) or model.get("schema_version") != 1:
        raise AuditModelError("candidate model requires schema_version 1")
    instructions = model.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise AuditModelError("candidate model requires a non-empty instructions list")
    for rule in instructions:
        if not isinstance(rule, dict) or "pattern" not in rule or "class" not in rule:
            raise AuditModelError("each instruction rule requires pattern and class")
        if rule["class"] not in _CLASSES:
            raise AuditModelError(
                f"instruction rule has an unknown class: {rule['class']}"
            )
        if not isinstance(rule.get("rationale"), str) or not rule["rationale"].strip():
            raise AuditModelError("each instruction rule requires a reviewed rationale")
        try:
            re.compile(rule["pattern"])
        except re.error as error:
            raise AuditModelError(
                f"invalid instruction pattern {rule['pattern']!r}: {error}"
            ) from error
        architectures = rule.get("architectures")
        if architectures is not None and (
            not isinstance(architectures, list)
            or not architectures
            or not set(architectures).issubset(_ARCHITECTURES)
        ):
            raise AuditModelError(
                "instruction rule architectures must name supported architectures"
            )
    mmio = model.get("mmio")
    if not isinstance(mmio, list) or not mmio:
        raise AuditModelError("candidate model requires a non-empty mmio list")
    for rule in mmio:
        if (
            not isinstance(rule, dict)
            or rule.get("class") not in _CLASSES
            or not rule.get("target")
        ):
            raise AuditModelError("each mmio rule requires target and a known class")
        if not isinstance(rule.get("rationale"), str) or not rule["rationale"].strip():
            raise AuditModelError("each mmio rule requires a reviewed rationale")
    helper_calls = model.get("helper_calls", [])
    if not isinstance(helper_calls, list):
        raise AuditModelError("candidate model helper_calls must be a list")
    for rule in helper_calls:
        if (
            not isinstance(rule, dict)
            or rule.get("class") not in _CLASSES
            or not rule.get("name")
            or not isinstance(rule.get("rationale"), str)
            or not rule["rationale"].strip()
        ):
            raise AuditModelError(
                "each helper_calls entry requires name, class, and a reviewed rationale"
            )
    software = model.get("software_state", [])
    if not isinstance(software, list):
        raise AuditModelError("candidate model software_state must be a list")
    for rule in software:
        if (
            not isinstance(rule, dict)
            or not rule.get("name")
            or not isinstance(rule.get("rationale"), str)
            or not rule["rationale"].strip()
        ):
            raise AuditModelError(
                "each software_state entry requires name and a reviewed rationale"
            )
    return model


def _category(cls: str, prefix: str | None) -> str:
    if cls == "deferred":
        return "persistent_deferred"
    if cls == "persistent":
        return "persistent_deferred" if prefix == "TT_OP_" else "persistent_immediate"
    return {
        "programming": "programming",
        "transient": "transient_datapath",
        "order_only": "reviewed_non_state",
    }[cls]


def _instruction_class(
    base: str, architecture: str, instructions: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for rule in instructions:
        architectures = rule.get("architectures")
        if architectures is not None and architecture not in architectures:
            continue
        if re.fullmatch(rule["pattern"], base):
            return rule
    return None


def classify_candidates(
    records: list[dict[str, Any]],
    helpers: list[dict[str, Any]],
    model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return one classified candidate per recognized persistent-state token.

    Candidates carry ``category`` = ``None`` when no reviewed rule matches; such
    candidates are the audit errors surfaced by :func:`enforce_candidates`.
    """
    model = model or load_candidate_model()
    instructions = model["instructions"]
    mmio_class = {rule["target"]: rule["class"] for rule in model["mmio"]}
    helper_class = {
        rule["name"]: rule["class"] for rule in model.get("helper_calls", [])
    }
    software_names = {rule["name"] for rule in model.get("software_state", [])}

    candidates: list[dict[str, Any]] = []
    for record in list(records) + list(helpers):
        architecture = record["architecture"]
        masked = record["_masked_body"]
        body = record["_body"]
        body_start_line = record["_body_start_line"]
        path = record["source"]["path"]
        parameter_names = {parameter["name"] for parameter in record["_parameters"]}
        local_names = {
            match.group("name")
            for match in _LOCAL_DECLARATION.finditer(masked)
            if "static" not in match.group("qualifiers").split()
        }

        def emit(
            start: int,
            end: int,
            kind: str,
            category: str | None,
            cls: str | None,
            base: str | None,
        ) -> None:
            candidates.append(
                {
                    "architecture": architecture,
                    "function": record["name"],
                    "source_path": path,
                    "line": body_start_line + body.count("\n", 0, start),
                    "token": body[start:end],
                    "kind": kind,
                    "base": base,
                    "class": cls,
                    "category": category,
                }
            )

        for match in _INSTRUCTION.finditer(masked):
            position = _skip_space(masked, match.end())
            following = masked[position : position + 1]
            if following not in ("(", ";"):
                continue  # a bare reference/constant, not an instruction call/statement
            prefix, base = match.group(1), match.group(2)
            rule = _instruction_class(base, architecture, instructions)
            if rule is None:
                emit(match.start(), match.end(), "instruction", None, None, base)
            else:
                emit(
                    match.start(),
                    match.end(),
                    "instruction",
                    _category(rule["class"], prefix),
                    rule["class"],
                    base,
                )

        for match in _MMIO.finditer(masked):
            bracket = match.end() - 1
            close = _matching_delimiter(masked, bracket, "[", "]")
            if close is None:
                continue
            after = _skip_space(masked, close + 1)
            if _ASSIGNMENT_OPERATOR.match(masked, after) is None:
                continue  # a read or comparison, not a persistent write
            target = match.group(1)
            cls = mmio_class[target]
            emit(
                match.start(),
                match.end() - 1,
                "mmio",
                _category(cls, None),
                cls,
                target,
            )

        for match in _HELPER_CALL.finditer(masked):
            position = _skip_space(masked, match.end())
            if masked[position : position + 1] != "(":
                continue
            name = match.group(1)
            cls = helper_class.get(name)
            emit(
                match.start(),
                match.end(),
                "helper_call",
                _category(cls, None) if cls is not None else None,
                cls,
                name,
            )

        for match in _ASSIGN_NAME.finditer(masked):
            name = match.group(1)
            if name in parameter_names or name in local_names:
                continue
            qualified = masked[max(0, match.start(1) - 2) : match.start(1)] == "::"
            recognizable = (
                name in software_names
                or qualified
                or _REVIEWABLE_SOFTWARE_NAME.fullmatch(name) is not None
            )
            if not recognizable:
                continue
            reviewed = name in software_names
            emit(
                match.start(1),
                match.end(1),
                "software",
                "persistent_immediate" if reviewed else None,
                "persistent" if reviewed else None,
                name,
            )

    return candidates


def enforce_candidates(
    records: list[dict[str, Any]],
    helpers: list[dict[str, Any]],
    model: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Classify candidates and fail closed for any unreviewed instruction/MMIO token."""
    model = model or load_candidate_model()
    candidates = classify_candidates(records, helpers, model)
    unknown = [candidate for candidate in candidates if candidate["category"] is None]
    if unknown:
        unknown.sort(
            key=lambda item: (
                item["architecture"],
                item["source_path"],
                item["line"],
                item["token"],
            )
        )
        first = unknown[0]
        raise AuditModelError(
            f"unreviewed persistent-state candidate: {first['architecture']} "
            f"{first['source_path']}:{first['line']} token {first['token']!r} "
            f"in {first['function']} ({len(unknown)} unreviewed candidate(s) total)"
        )
    return candidates
