# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared resolver for centralized model perf/accuracy targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TARGETS_YAML_PATH_DEFAULT = str(Path(__file__).resolve().parents[2] / "model_targets.yaml")

# Keep SKU aliases explicit and unambiguous so models with multiple
# SKU blocks (e.g. wh_n150 + wh_llmbox_perf) resolve correctly.
# Note on p300x2: bh_quietbox_2 is 2x P300 boards = 4 dies, which today's
# determine_device_name labels "P150x4". That label is kept here as an alias
# so existing call sites still resolve to the right entries; the canonical
# SKU name is p300x2 because that matches the actual hardware.
_SKU_ALIASES = {
    "wh_n150": {"wh_n150", "n150"},
    "wh_n300": {"wh_n300", "n300"},
    "wh_llmbox_perf": {"wh_llmbox_perf", "wh_llmbox", "t3k"},
    "wh_galaxy_perf": {"wh_galaxy_perf", "wh_galaxy", "tg", "glx"},
    "bh_p100": {"bh_p100", "p100"},
    "bh_p150": {"bh_p150", "p150"},
    "bh_p300": {"bh_p300", "p300"},
    "bh_loudbox": {"bh_loudbox", "p150x8"},
    "p300x2": {"p300x2", "p150x4", "bh_quietbox_2"},
    "bh_galaxy_perf": {"bh_galaxy_perf", "bh_galaxy", "bhglx"},
    "blackhole": {"blackhole", "bh"},
}

DEFAULT_PERF_TOLERANCE = 0.15


def _normalize_token(value: Any) -> str:
    """Normalize external string-like values for case-insensitive matching."""
    return str(value).strip().lower()


def normalize_sku(sku: Any) -> str:
    """Map a SKU alias to a canonical SKU token."""
    token = _normalize_token(sku)
    for canonical, aliases in _SKU_ALIASES.items():
        if token in aliases:
            return canonical
    return token


def is_tolerance_key(metric_name: Any) -> bool:
    """Return True when a key denotes tolerance config, not a target metric."""
    return isinstance(metric_name, str) and (metric_name == "tolerance" or metric_name.endswith("_tolerance"))


def metric_tolerance_key_candidates(metric_name: str) -> list[str]:
    """Return supported per-metric tolerance key variants for a metric."""
    slash_normalized = metric_name.replace("/", "_")
    keys = [f"{metric_name}_tolerance", f"{slash_normalized}_tolerance"]
    # Preserve deterministic order while removing duplicates.
    return list(dict.fromkeys(keys))


def resolve_metric_tolerance(metric_name: str, thresholds: dict[str, Any], default_tolerance: float) -> float:
    """Resolve tolerance for a metric using per-metric keys with fallback default."""
    for key in metric_tolerance_key_candidates(metric_name):
        value = thresholds.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    generic = thresholds.get("tolerance")
    if isinstance(generic, (int, float)) and not isinstance(generic, bool):
        return float(generic)
    return default_tolerance


def load_model_targets() -> dict[str, Any]:
    """Load centralized targets YAML from the configured default path."""
    path = Path(TARGETS_YAML_PATH_DEFAULT)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid centralized targets file format: {path}")
    return data


def _model_matches(model_key: str, model_name: str, model_block: dict[str, Any]) -> bool:
    """Check if a model key (or alias) matches the requested model."""
    model_norm = _normalize_token(model_name)
    if _normalize_token(model_key) == model_norm:
        return True
    aliases = model_block.get("aliases", [])
    return any(_normalize_token(alias) == model_norm for alias in aliases)


def _entry_matches(entry: dict[str, Any], batch_size: int | None, seq_len: int | None) -> bool:
    """Match entry dimensions using strict fallback semantics for None values."""
    entry_batch = entry.get("batch_size")
    entry_seq = entry.get("seq_len")

    if batch_size is None:
        if entry_batch is not None:
            return False
    elif entry_batch is not None and entry_batch != batch_size:
        return False

    if seq_len is None:
        if entry_seq is not None:
            return False
    elif entry_seq is not None and entry_seq != seq_len:
        return False
    return True


def _entry_specificity(entry: dict[str, Any]) -> int:
    """Rank entries so the most specific matching target is selected."""
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
    include_todo: bool = False,
) -> dict[str, Any] | None:
    """Resolve the best matching centralized target entry for model and SKU."""
    targets_doc = load_model_targets()
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
                # This SKU key matched but has no entry for the requested
                # batch/seq; keep scanning other SKU keys / models instead of
                # giving up the whole search.
                continue
            return sorted(matches, key=_entry_specificity, reverse=True)[0]
    return None


def resolve_perf_targets(
    model_name: str,
    sku: str,
    batch_size: int | None = None,
    seq_len: int | None = None,
) -> dict[str, float] | None:
    """Resolve only the perf metrics for a given model/SKU combo."""
    entry = resolve_target_entry(
        model_name=model_name,
        sku=sku,
        batch_size=batch_size,
        seq_len=seq_len,
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
) -> dict[str, float] | None:
    """Resolve only the accuracy metrics for a given model/SKU combo."""
    entry = resolve_target_entry(
        model_name=model_name,
        sku=sku,
        batch_size=batch_size,
        seq_len=seq_len,
        include_todo=False,
    )
    if not entry:
        return None
    accuracy = entry.get("accuracy") or {}
    return accuracy if isinstance(accuracy, dict) else None
