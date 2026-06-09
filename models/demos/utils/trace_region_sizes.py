# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared resolver for centralized trace region sizes."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

from models.demos.utils.model_targets import _model_matches, normalize_sku

TRACE_REGION_SIZES_YAML_PATH = Path(__file__).resolve().parents[2] / "model_trace_region_sizes.yaml"

# Fallback when no YAML entry matches (local dev without HF_MODEL override).
DEFAULT_TRACE_REGION_SIZE = 50_000_000


@functools.cache
def load_trace_region_sizes() -> dict[str, Any]:
    """Load centralized trace region sizes YAML from the configured default path."""
    with TRACE_REGION_SIZES_YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trace region sizes file format: {TRACE_REGION_SIZES_YAML_PATH}")
    return data


def is_trace_region_size_placeholder(trace_region_size: int | None) -> bool:
    """Return True when a test did not set a custom trace region size."""
    return trace_region_size is None or trace_region_size == DEFAULT_TRACE_REGION_SIZE


def should_apply_trace_region_override(device_params: dict, override_trace_region_size: int | None) -> bool:
    """Return True when centralized YAML should replace the test's trace region size."""
    if not override_trace_region_size:
        return False
    return is_trace_region_size_placeholder(device_params.get("trace_region_size"))


def apply_trace_region_override(device_params: dict, override_trace_region_size: int | None) -> int | None:
    """Apply centralized trace region size unless the test set a custom value."""
    if should_apply_trace_region_override(device_params, override_trace_region_size):
        device_params["trace_region_size"] = override_trace_region_size
        return override_trace_region_size
    return device_params.get("trace_region_size")


def resolve_trace_region_size(model_name: str | None, sku: str | None) -> int | None:
    """Resolve trace region size in bytes for a model/SKU pair, or None if not configured."""
    if not model_name or not sku:
        return None

    sizes_doc = load_trace_region_sizes()
    sizes = sizes_doc.get("sizes", {})
    sku_norm = normalize_sku(sku)

    for model_key, model_block in sizes.items():
        if not isinstance(model_block, dict) or not _model_matches(model_key, model_name, model_block):
            continue

        skus = model_block.get("skus", {})
        for sku_key, sku_block in skus.items():
            if normalize_sku(sku_key) != sku_norm:
                continue
            if not isinstance(sku_block, dict):
                continue
            value = sku_block.get("trace_region_size")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
    return None
