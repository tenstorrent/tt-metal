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
    * Direct (full physical device) -- call ``resolve_trace_region_size`` /
      ``resolve_demo_trace_region_size`` / ``build_trace_device_params`` when a
      demo opens the whole physical device itself (the SKU is read from the
      current cluster via ``get_current_device_sku_name``, where physical ==
      logical).
    * Deferred (pytest mesh_device fixture) -- put
      ``TRACE_MODEL_KEY_PARAM: "<model-key>"`` in a parametrized
      ``device_params`` dict. The ``mesh_device`` fixture resolves it to
      ``trace_region_size`` at device-open time using the SKU of the logical
      submesh actually opened (derived from the mesh shape / ``data_parallel`` /
      ``MESH_DEVICE``), not the physical cluster.

Unconfigured pairs default to dynamic allocation: if a (model, SKU) pair is
not present in the YAML, resolution logs an info message and returns
``TRACE_REGION_SIZE_DYNAMIC`` (``0``), which tells the runtime to allocate
trace buffers dynamically instead of reserving a fixed region up front. A
non-zero size is only ever returned when it is explicitly present in the YAML;
``deepseek-v3`` also sets ``0`` explicitly to opt into the same dynamic
behavior.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from models.demos.utils.model_targets import _model_matches, normalize_sku

TRACE_REGION_SIZES_YAML_PATH = Path(__file__).resolve().parents[2] / "model_trace_region_sizes.yaml"

# trace_region_size=0 lets the runtime allocate trace buffers dynamically (see deepseek-v3 demo).
TRACE_REGION_SIZE_DYNAMIC = 0

# Set in a device_params parametrize dict; the mesh_device fixture pops it and resolves
# trace_region_size from the YAML using the logical submesh SKU at device-open time.
TRACE_MODEL_KEY_PARAM = "_trace_model_key"


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


@functools.cache
def load_trace_region_sizes() -> dict[str, Any]:
    """Load centralized trace region sizes YAML from the configured default path."""
    with TRACE_REGION_SIZES_YAML_PATH.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid trace region sizes file format: {TRACE_REGION_SIZES_YAML_PATH}")
    return data


def _find_configured_trace_region_size(model_name: str, sku_norm: str) -> int | None:
    """Return the configured trace_region_size for a model/normalized-SKU pair, or None."""
    sizes = load_trace_region_sizes().get("sizes", {})
    for model_key, model_block in sizes.items():
        if not isinstance(model_block, dict) or not _model_matches(model_key, model_name, model_block):
            continue
        for sku_key, sku_block in model_block.get("skus", {}).items():
            if normalize_sku(sku_key) != sku_norm or not isinstance(sku_block, dict):
                continue
            value = sku_block.get("trace_region_size")
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                return value
    return None


def resolve_trace_region_size(model_name: str | None, sku: str | None) -> int:
    """Resolve trace region size in bytes for a model/SKU pair from the centralized YAML.

    Returns the configured value, or ``TRACE_REGION_SIZE_DYNAMIC`` (0, dynamic
    allocation) -- with an info log -- when the pair is not configured.
    """
    if not model_name:
        raise ValueError("model_name is required to resolve trace_region_size")
    if not sku:
        raise ValueError("sku is required to resolve trace_region_size")

    sku_norm = normalize_sku(sku)
    value = _find_configured_trace_region_size(model_name, sku_norm)
    if value is not None:
        return value

    logger.info(
        f"No trace_region_size configured for model={model_name!r} and SKU={sku_norm!r}; "
        f"defaulting to dynamic allocation (trace_region_size={TRACE_REGION_SIZE_DYNAMIC})."
    )
    return TRACE_REGION_SIZE_DYNAMIC


def resolve_trace_region_size_for_candidates(model_name_candidates: list[str], sku: str | None) -> int:
    """Resolve the first configured candidate's size, else dynamic allocation (0).

    Used by the HF_MODEL env path, which derives several model-name candidates
    from a single HF model string. A configured candidate always wins over the
    dynamic-allocation default.
    """
    if not sku:
        raise ValueError("sku is required to resolve trace_region_size")

    sku_norm = normalize_sku(sku)
    for model_name in model_name_candidates:
        if not model_name:
            continue
        value = _find_configured_trace_region_size(model_name, sku_norm)
        if value is not None:
            return value

    logger.info(
        f"No trace_region_size configured for model candidates {list(model_name_candidates)!r} "
        f"and SKU={sku_norm!r}; defaulting to dynamic allocation (trace_region_size={TRACE_REGION_SIZE_DYNAMIC})."
    )
    return TRACE_REGION_SIZE_DYNAMIC


def resolve_demo_trace_region_size(model_key: str) -> int:
    """Resolve trace region size using the physical cluster SKU.

    Intended for demos that open the full physical device directly (so physical ==
    logical). Tests/demos that open a logical submesh via the ``mesh_device`` fixture
    should instead pass ``TRACE_MODEL_KEY_PARAM`` in ``device_params`` so the fixture
    resolves against the logical submesh SKU.
    """
    from models.demos.utils.device_sku import get_current_device_sku_name

    return resolve_trace_region_size(model_key, get_current_device_sku_name())


def build_trace_device_params(model_key: str, **extra: Any) -> dict[str, Any]:
    """Build device_params dict with trace_region_size resolved from YAML."""
    return {"trace_region_size": resolve_demo_trace_region_size(model_key), **extra}
