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

# trace_region_size=0 lets the runtime allocate trace buffers dynamically (see deepseek-v3 demo).
TRACE_REGION_SIZE_DYNAMIC = 0

# Populated from device_params parametrize dict; resolved at fixture time via apply_trace_model_key().
TRACE_MODEL_KEY_PARAM = "_trace_model_key"


class TraceRegionSizeNotConfiguredError(ValueError):
    """Raised when trace_region_size is missing from model_trace_region_sizes.yaml."""


def _missing_trace_region_size_message(model_name: str, sku: str) -> str:
    return (
        f"trace_region_size is not configured for model={model_name!r} and SKU={sku!r}. "
        f"Add a (model, SKU) entry with trace_region_size to {TRACE_REGION_SIZES_YAML_PATH}."
    )


@functools.cache
def load_trace_region_sizes() -> dict[str, Any]:
    """Load centralized trace region sizes YAML from the configured default path."""
    with TRACE_REGION_SIZES_YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trace region sizes file format: {TRACE_REGION_SIZES_YAML_PATH}")
    return data


def resolve_trace_region_size(model_name: str | None, sku: str | None) -> int:
    """Resolve trace region size in bytes for a model/SKU pair from the centralized YAML."""
    if not model_name:
        raise ValueError("model_name is required to resolve trace_region_size")
    if not sku:
        raise ValueError("sku is required to resolve trace_region_size")

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
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value

    raise TraceRegionSizeNotConfiguredError(_missing_trace_region_size_message(model_name, sku_norm))


def resolve_demo_trace_region_size(model_key: str) -> int:
    """Resolve trace region size for a demo using the current cluster SKU."""
    from models.demos.utils.device_sku import get_current_device_sku_name

    return resolve_trace_region_size(model_key, get_current_device_sku_name())


def apply_trace_model_key(device_params: dict[str, Any]) -> dict[str, Any]:
    """Resolve trace_region_size from YAML when TRACE_MODEL_KEY_PARAM is set in device_params."""
    params = device_params.copy()
    model_key = params.pop(TRACE_MODEL_KEY_PARAM, None)
    if model_key is not None:
        params["trace_region_size"] = resolve_demo_trace_region_size(model_key)
    return params


def build_trace_device_params(model_key: str, **extra: Any) -> dict[str, Any]:
    """Build device_params dict with trace_region_size resolved from YAML."""
    return {"trace_region_size": resolve_demo_trace_region_size(model_key), **extra}
