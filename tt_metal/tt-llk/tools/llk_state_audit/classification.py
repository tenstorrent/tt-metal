"""Complete, deterministic definition classification and audit validation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .candidates import enforce_candidates, load_candidate_model
from .effects import build_effects
from .inventory import scan_functions, scan_helpers


def classify(
    root: Path | str,
    effect_model: Path | str | None = None,
    *,
    reviewed: Path | str | None = None,
    _records: list[dict[str, Any]] | None = None,
    _effects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify every discovered definition exactly once.

    A definition is included when it owns a reviewed state sink (or its name
    states a lifecycle operation), is a wrapper only when its unique target is
    present in the same architecture, and otherwise records a conservative
    exclusion reason.
    """
    records = scan_functions(root, effect_model)[0] if _records is None else _records
    effects = (
        build_effects(root, effect_model)["effects"] if _effects is None else _effects
    )
    direct_effect_functions = {
        (
            effect["architecture"],
            effect["function"],
            effect["evidence"]["source_path"],
            effect["evidence"]["body_fingerprint"],
        )
        for effect in effects
        if effect["evidence"]["kind"] == "direct"
    }
    transitive_effect_functions = {
        (
            effect["architecture"],
            effect["function"],
            effect["evidence"]["source_path"],
            effect["evidence"]["body_fingerprint"],
        )
        for effect in effects
        if effect["evidence"]["kind"] == "transitive"
    }
    keys = {(record["architecture"], record["name"]) for record in records}
    definitions: list[dict[str, Any]] = []
    for record in records:
        key = (record["architecture"], record["name"])
        source_key = (*key, record["source"]["path"], record["body_fingerprint"])
        target_key = (
            (record["architecture"], record["canonical_target"])
            if record["canonical_target"]
            else None
        )
        if record["lifecycle"] == "wrapper" and target_key in keys:
            classification, reason = (
                "wrapper",
                "single canonical same-architecture target",
            )
        elif (
            source_key in direct_effect_functions
            or source_key in transitive_effect_functions
            or record["lifecycle"]
            in {
                "init",
                "reinit",
                "uninit",
                "configure",
                "clear",
                "reset",
                "done",
                "restore",
            }
        ):
            if source_key in direct_effect_functions:
                reason = "recognized direct state primitive"
            elif source_key in transitive_effect_functions:
                reason = "recognized state effect through analyzed LLK call"
            else:
                reason = "lifecycle/state-name ownership"
            classification = "included"
        else:
            classification, reason = "excluded", _exclusion_reason(record)
        definitions.append(
            {
                "architecture": record["architecture"],
                "name": record["name"],
                "classification": classification,
                "reason": reason,
                "canonical_target": (
                    record["canonical_target"] if classification == "wrapper" else None
                ),
                "source": record["source"],
                "body_fingerprint": record["body_fingerprint"],
                "stability": record["stability_tier"],
            }
        )
    definitions.sort(
        key=lambda item: (
            item["architecture"],
            item["source"]["path"],
            item["source"]["start_line"],
            item["name"],
        )
    )
    result = {
        "schema_version": 2,
        "definitions": definitions,
        "drift": _drift(definitions, reviewed),
    }
    return result


def audit(root: Path | str, effect_model: Path | str | None = None) -> dict[str, Any]:
    """Return inventory, total classification, normalized effects and checks."""
    records, model = scan_functions(root, effect_model)
    helpers = scan_helpers(root, effect_model, _model=model)
    candidates = enforce_candidates(records, helpers, load_candidate_model())
    inventory = {
        "schema_version": 1,
        "functions": [_public(record) for record in records],
    }
    effect_data = build_effects(
        root,
        effect_model,
        _records=records,
        _model=model,
        _helpers=helpers,
    )
    classification = classify(
        root,
        effect_model,
        _records=records,
        _effects=effect_data["effects"],
    )
    _validate(records, effect_data["effects"], classification["definitions"])
    summary = _summary(records, effect_data["effects"], classification["definitions"])
    summary["candidate_summary"] = _candidate_summary(candidates)
    return {
        "schema_version": 2,
        "inventory": inventory,
        "classification": classification["definitions"],
        "effects": effect_data["effects"],
        "summary": summary,
    }


def _candidate_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: Counter[str] = Counter(
        candidate["category"] for candidate in candidates
    )
    by_architecture_category: dict[str, dict[str, int]] = defaultdict(dict)
    for candidate in candidates:
        bucket = by_architecture_category[candidate["architecture"]]
        bucket[candidate["category"]] = bucket.get(candidate["category"], 0) + 1
    return {
        "total": len(candidates),
        "by_category": dict(sorted(by_category.items())),
        "by_architecture": {
            architecture: dict(sorted(counts.items()))
            for architecture, counts in sorted(by_architecture_category.items())
        },
    }


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _exclusion_reason(record: dict[str, Any]) -> str:
    name = record["name"].lower()
    body = record["_masked_body"]
    if "wait" in name or "get" in name or "is_" in name:
        return "read/wait-only"
    if not record["direct_calls"] and not record["state_sinks"]:
        return "pure helper or no reviewed persistent sink"
    if not record["state_sinks"]:
        return "data-path-only or no reviewed persistent sink"
    return "no reviewed persistent sink"


def _drift(
    definitions: list[dict[str, Any]], reviewed: Path | str | None
) -> list[dict[str, Any]]:
    if reviewed is None:
        return []
    try:
        baseline = json.loads(Path(reviewed).read_text(encoding="utf-8"))
        previous = baseline.get("definitions", [])
    except (OSError, json.JSONDecodeError):
        return [{"reason": "reviewed manifest unreadable"}]
    before = {
        (
            item["architecture"],
            item["name"],
            item["source"]["path"],
            item["source"]["start_line"],
        ): item
        for item in previous
    }
    drift = []
    for item in definitions:
        key = (
            item["architecture"],
            item["name"],
            item["source"]["path"],
            item["source"]["start_line"],
        )
        old = before.get(key)
        if old is None or old.get("body_fingerprint") != item["body_fingerprint"]:
            drift.append(
                {
                    "architecture": item["architecture"],
                    "name": item["name"],
                    "reason": "definition added or body fingerprint changed",
                }
            )
    return drift


def _validate(
    records: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    definitions: list[dict[str, Any]],
) -> None:
    record_keys = {_identity(item) for item in records}
    definition_keys = {_identity(item) for item in definitions}
    if len(definitions) != len(definition_keys) or record_keys != definition_keys:
        raise ValueError("every discovered definition must be classified exactly once")
    by_key = {(item["architecture"], item["name"]): item for item in records}
    by_definition = {(item["architecture"], item["name"]): item for item in definitions}
    for effect in effects:
        candidates = [
            record
            for record in records
            if record["architecture"] == effect["architecture"]
            and record["name"] == effect["function"]
            and record["source"]["path"] == effect["evidence"]["source_path"]
            and record["source"]["start_line"]
            <= effect["evidence"]["line"]
            <= record["source"]["end_line"]
        ]
        if len(candidates) != 1:
            raise ValueError("effect refers to an unclassified function")
        record = candidates[0]
        evidence = effect["evidence"]
        if evidence["body_fingerprint"] != record["body_fingerprint"]:
            raise ValueError("effect source fingerprint is not retained")
        if (
            not record["source"]["start_line"]
            <= evidence["line"]
            <= record["source"]["end_line"]
        ):
            raise ValueError("effect evidence line outside function range")
    for definition in definitions:
        if definition["classification"] == "wrapper":
            target = (definition["architecture"], definition["canonical_target"])
            if target not in by_definition:
                raise ValueError("wrapper lacks a same-architecture canonical function")


def _identity(item: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        item["architecture"],
        item["name"],
        item["source"]["path"],
        item["source"]["start_line"],
    )


def _summary(
    records: list[dict[str, Any]],
    effects: list[dict[str, Any]],
    definitions: list[dict[str, Any]],
) -> dict[str, Any]:
    def count(values: list[str]) -> dict[str, int]:
        return dict(sorted(Counter(values).items()))

    by_architecture: dict[str, dict[str, int]] = defaultdict(
        lambda: {"definitions": 0, "effects": 0}
    )
    for record in records:
        by_architecture[record["architecture"]]["definitions"] += 1
    for effect in effects:
        by_architecture[effect["architecture"]]["effects"] += 1
    return {
        "definition_count": len(records),
        "effect_count": len(effects),
        "by_architecture": dict(sorted(by_architecture.items())),
        "by_stability": count([item["stability"] for item in definitions]),
        "by_lifecycle": count([item["lifecycle"] for item in records]),
        "by_classification": count([item["classification"] for item in definitions]),
    }
