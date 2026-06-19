# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared resolver for centralized trace region sizes.

This module is the single entry point for answering "how many bytes of
``trace_region_size`` should I reserve when opening a device for model X on
this machine?". The values live in one source of truth,
``models/model_trace_region_sizes.yaml``, keyed by model and SKU (the
cluster/board type, e.g. ``wh_n150``, ``wh_llmbox_perf``, ``bh_p150``).
This replaces the per-model/per-SKU literals that demos and tests used to
hardcode.

Resolution:
    ``resolve_trace_region_size(model_name, sku)`` matches ``model_name``
    against each YAML entry's key or ``aliases`` (case-insensitive) and the
    ``sku`` against its ``skus`` block (via ``normalize_sku`` so aliases like
    ``t3k`` -> ``wh_llmbox_perf`` work), returning the configured size in bytes.

Two ways callers consume it:
    * Eager -- call ``resolve_trace_region_size`` /
      ``resolve_demo_trace_region_size`` / ``build_trace_device_params``
      directly when opening a device or building a ``device_params`` dict
      (the SKU is read from the current cluster via ``get_current_device_sku_name``).
    * Deferred (pytest) -- put ``TRACE_MODEL_KEY_PARAM: "<model-key>"`` in a
      parametrized ``device_params`` dict; the ``device_params`` fixture calls
      ``apply_trace_model_key`` to resolve it to ``trace_region_size`` just
      before the device opens.

Fail-loud by design: if a (model, SKU) pair is not configured, resolution
raises ``TraceRegionSizeNotConfiguredError`` rather than falling back to a
default. There is no implicit default size -- a value is only ever returned
when it is explicitly present in the YAML. The one special value is ``0``
(``TRACE_REGION_SIZE_DYNAMIC``): when a YAML entry sets it explicitly (e.g.
``deepseek-v3``), it tells the runtime to allocate trace buffers dynamically
instead of reserving a fixed region up front.
"""

from __future__ import annotations

import functools
import re
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


def _append_unique(candidates: list[str], value: str) -> None:
    if value and value not in candidates:
        candidates.append(value)


def hf_model_name_candidates(hf_model: str) -> list[str]:
    """Build model name candidates from HF_MODEL env values, including hub cache paths."""
    candidates: list[str] = []
    _append_unique(candidates, hf_model)

    basename = hf_model.strip("/").split("/")[-1]
    _append_unique(candidates, basename)

    hub_match = re.search(r"models--([^/]+)--([^/]+)", hf_model)
    if hub_match:
        _append_unique(candidates, f"{hub_match.group(1)}/{hub_match.group(2)}")

    base_match = re.search(r"(.*?\d+[bB])-", basename)
    if base_match:
        _append_unique(candidates, base_match.group(1))

    return candidates


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
