# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared resolver for centralized model perf/accuracy targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_MODEL_TARGETS_PATH = str(Path(__file__).resolve().parents[2] / "model_targets.yaml")

_SKU_ALIASES = {
    "wormhole": {"n300", "n150", "wh_n150", "wh_n300", "tg", "wh_llmbox_perf", "wh_galaxy_perf"},
    "t3k": {"t3k" },
    "glx": {"glx", "bhglx" },
    "blackhole": {"blackhole", "bh", "p150", "p300", "p150x8"},
}


def _normalize_token(value: Any) -> str:
    return str(value).strip().lower()


def normalize_sku(sku: Any) -> str:
    token = _normalize_token(sku)
    for canonical, aliases in _SKU_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def load_model_targets(targets_yaml_path: str | None = None) -> dict[str, Any]:
    path = Path(targets_yaml_path or DEFAULT_MODEL_TARGETS_PATH)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid centralized targets file format: {path}")
    return data


def _model_matches(model_key: str, model_name: str, model_block: dict[str, Any]) -> bool:
    model_norm = _normalize_token(model_name)
    if _normalize_token(model_key) == model_norm:
        return True
    aliases = model_block.get("aliases", [])
    return any(_normalize_token(alias) == model_norm for alias in aliases)


def _entry_matches(entry: dict[str, Any], batch_size: int | None, seq_len: int | None) -> bool:
    entry_batch = entry.get("batch_size")
    entry_seq = entry.get("seq_len")
    if batch_size is not None and entry_batch is not None and entry_batch != batch_size:
        return False
    if seq_len is not None and entry_seq is not None and entry_seq != seq_len:
        return False
    return True


def _entry_specificity(entry: dict[str, Any]) -> int:
    score = 0
    if entry.get("batch_size") is not None:
        score += 1
    if entry.get("seq_len") is not None:
        score += 1
    return score


def resolve_target_entry(
    model_name: str,
    sku: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
    targets_yaml_path: str | None = None,
    include_todo: bool = False,
) -> dict[str, Any] | None:
    targets_doc = load_model_targets(targets_yaml_path)
    targets = targets_doc.get("targets", {})
    sku_norm = normalize_sku(sku)

    for model_key, model_block in targets.items():
        if not isinstance(model_block, dict) or not _model_matches(model_key, model_name, model_block):
            continue

        skus = model_block.get("skus", {})
        for sku_key, sku_block in skus.items():
            if normalize_sku(sku_key) != sku_norm:
                continue
            entries = sku_block.get("entries", [])
            matches = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                if not include_todo and _normalize_token(entry.get("status", "active")) == "todo":
                    continue
                if _entry_matches(entry, batch_size=batch_size, seq_len=seq_len):
                    matches.append(entry)
            if not matches:
                return None
            return sorted(matches, key=_entry_specificity, reverse=True)[0]
    return None


def resolve_perf_targets(
    model_name: str,
    sku: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
    targets_yaml_path: str | None = None,
) -> dict[str, float] | None:
    entry = resolve_target_entry(
        model_name=model_name,
        sku=sku,
        batch_size=batch_size,
        seq_len=seq_len,
        targets_yaml_path=targets_yaml_path,
        include_todo=False,
    )
    if not entry:
        return None
    perf = entry.get("perf") or {}
    return perf if isinstance(perf, dict) else None


def resolve_accuracy_targets(
    model_name: str,
    sku: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
    targets_yaml_path: str | None = None,
) -> dict[str, float] | None:
    entry = resolve_target_entry(
        model_name=model_name,
        sku=sku,
        batch_size=batch_size,
        seq_len=seq_len,
        targets_yaml_path=targets_yaml_path,
        include_todo=False,
    )
    if not entry:
        return None
    accuracy = entry.get("accuracy") or {}
    return accuracy if isinstance(accuracy, dict) else None
