# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from models.demos.utils.trace_region_sizes import (
    TRACE_REGION_SIZES_YAML_PATH,
    TraceRegionSizeNotConfiguredError,
    load_trace_region_sizes,
    resolve_trace_region_size,
)


def _iter_yaml_trace_region_entries():
    doc = load_trace_region_sizes()
    sizes = doc.get("sizes", {})
    for model_key, model_block in sizes.items():
        if not isinstance(model_block, dict):
            continue

        model_names = [model_key]
        aliases = model_block.get("aliases", [])
        if isinstance(aliases, list):
            model_names.extend(aliases)

        skus = model_block.get("skus", {})
        if not isinstance(skus, dict):
            continue

        for sku_key, sku_block in skus.items():
            if not isinstance(sku_block, dict):
                continue
            expected = sku_block.get("trace_region_size")
            if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
                continue
            for model_name in model_names:
                yield model_name, sku_key, expected


def test_trace_region_sizes_yaml_schema():
    doc = yaml.safe_load(TRACE_REGION_SIZES_YAML_PATH.read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    assert doc.get("version") == 1

    sizes = doc.get("sizes")
    assert isinstance(sizes, dict) and sizes

    for model_name, model_block in sizes.items():
        assert isinstance(model_block, dict), f"{model_name}: expected dict block"
        skus = model_block.get("skus")
        assert isinstance(skus, dict) and skus, f"{model_name}: missing skus"
        for sku_name, sku_block in skus.items():
            value = sku_block.get("trace_region_size")
            assert isinstance(value, int) and value > 0, f"{model_name}/{sku_name}: invalid trace_region_size"


@pytest.mark.parametrize("model_name,sku,expected_size", list(_iter_yaml_trace_region_entries()))
def test_resolve_trace_region_size_matches_yaml(model_name, sku, expected_size):
    assert resolve_trace_region_size(model_name, sku) == expected_size


@pytest.mark.parametrize(
    "model_name,legacy_sku,expected_size",
    [
        ("Llama-3.1-8B", "N150", 25000000),
        ("Llama-3.1-8B", "T3K", 50000000),
        ("Llama-3.3-70B", "P150x4", 96000000),
        ("meta-llama/Llama-3.1-8B-Instruct", "bh_quietbox_2", 52000000),
    ],
)
def test_resolve_trace_region_size_legacy_sku_aliases(model_name, legacy_sku, expected_size):
    assert resolve_trace_region_size(model_name, legacy_sku) == expected_size


def test_resolve_trace_region_size_raises_when_not_configured():
    with pytest.raises(TraceRegionSizeNotConfiguredError, match="trace_region_size is not configured"):
        resolve_trace_region_size("unknown-model", "wh_n150")


def test_load_trace_region_sizes_is_cached():
    load_trace_region_sizes.cache_clear()
    first = load_trace_region_sizes()
    second = load_trace_region_sizes()
    assert first is second
