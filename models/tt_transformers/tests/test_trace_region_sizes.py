# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from models.demos.utils.model_targets import normalize_sku
from models.demos.utils.trace_region_sizes import (
    TRACE_REGION_SIZES_YAML_PATH,
    TraceRegionSizeNotConfiguredError,
    apply_trace_region_override,
    is_trace_region_size_placeholder,
    load_trace_region_sizes,
    resolve_trace_region_size,
    should_apply_trace_region_override,
)

# Regression table from the former trace_region_size_dict in trace_region_config.py.
_LEGACY_TRACE_REGION_SIZE_DICT = {
    "Llama-3.1-8B": {
        "N150": 25000000,
        "N300": 38000000,
        "T3K": 50000000,
        "TG": 50000000,
        "P150": 52000000,
        "P300": 52000000,
    },
    "Llama-3.3-70B": {
        "T3K": 90000000,
        "TG": 80000000,
        "P150": 80000000,
        "P300": 80000000,
        "P150x4": 96000000,
        "P150x8": 84000000,
    },
    "Llama-3.1-70B": {
        "T3K": 90000000,
        "TG": 90000000,
        "P150": 90000000,
        "P300": 90000000,
        "P150x4": 90000000,
        "P150x8": 90000000,
    },
    "Llama-3.2-90B": {
        "T3K": 20000000,
    },
    "Qwen3-32B": {
        "T3K": 90000000,
        "TG": 200000000,
        "P150": 90000000,
        "P300": 90000000,
        "P150x4": 90000000,
        "P150x8": 90000000,
    },
    "GPT-OSS-20B": {
        "T3K": 50000000,
        "TG": 50000000,
    },
    "GPT-OSS-120B": {
        "T3K": 50000000,
        "TG": 50000000,
    },
    "Qwen2.5-72B": {
        "T3K": 70000000,
        "TG": 70000000,
    },
    "gemma-3-27b": {
        "T3K": 70000000,
        "TG": 70000000,
        "P150x4": 70000000,
    },
    "DeepSeek-R1-Distill-Llama-70B": {
        "P150x4": 90000000,
    },
    "Llama-3.2-3B": {
        "N150": 10000000,
    },
    "Qwen2.5-VL-7B": {
        "N300": 10000000,
    },
}


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


def test_legacy_dict_values_preserved_in_yaml():
    for model_name, device_map in _LEGACY_TRACE_REGION_SIZE_DICT.items():
        for legacy_device, expected_size in device_map.items():
            resolved = resolve_trace_region_size(model_name, normalize_sku(legacy_device))
            assert resolved == expected_size, f"{model_name}/{legacy_device}"


def test_hf_alias_resolution():
    assert resolve_trace_region_size("meta-llama/Llama-3.1-8B-Instruct", "wh_n150") == 25000000


def test_tt_transformers_specific_entries():
    assert resolve_trace_region_size("meta-llama/Llama-3.1-8B-Instruct", "p300x2") == 52000000
    assert resolve_trace_region_size("meta-llama/Llama-3.1-8B-Instruct", "bh_quietbox_2") == 52000000
    assert resolve_trace_region_size("meta-llama/Llama-3.2-11B-Vision-Instruct", "wh_llmbox_perf") == 17400000
    assert resolve_trace_region_size("mistralai/Mixtral-8x7B-v0.1", "wh_llmbox_perf") == 250000000
    assert resolve_trace_region_size("mistralai/Mistral-Small-3.1-24B-Instruct-2503", "wh_n150") == 30000000
    assert resolve_trace_region_size("Qwen/Qwen2.5-72B-Instruct", "bh_p150") == 100000000
    assert resolve_trace_region_size("Qwen/Qwen2.5-32B-Instruct", "bh_p300") == 100000000
    assert resolve_trace_region_size("gpt-oss-20b", "wh_llmbox_perf") == 50000000


def test_is_trace_region_size_placeholder():
    assert is_trace_region_size_placeholder(None)
    assert not is_trace_region_size_placeholder(50_000_000)
    assert not is_trace_region_size_placeholder(216580672)
    assert not is_trace_region_size_placeholder(102000000)


def test_should_apply_trace_region_override():
    override_size = 80000000
    assert should_apply_trace_region_override({}, override_size)
    assert should_apply_trace_region_override({"trace_region_size": None}, override_size)
    assert not should_apply_trace_region_override({"trace_region_size": 216580672}, override_size)
    assert not should_apply_trace_region_override({"trace_region_size": 102000000}, override_size)
    assert not should_apply_trace_region_override({}, None)


def test_apply_trace_region_override():
    device_params = {"trace_region_size": 216580672}
    assert apply_trace_region_override(device_params, 80000000) == 216580672
    assert device_params["trace_region_size"] == 216580672

    device_params = {}
    assert apply_trace_region_override(device_params, 52000000) == 52000000
    assert device_params["trace_region_size"] == 52000000


def test_resolve_trace_region_size_raises_when_not_configured():
    with pytest.raises(TraceRegionSizeNotConfiguredError, match="trace_region_size is not configured"):
        resolve_trace_region_size("unknown-model", "wh_n150")


def test_load_trace_region_sizes_is_cached():
    load_trace_region_sizes.cache_clear()
    first = load_trace_region_sizes()
    second = load_trace_region_sizes()
    assert first is second
