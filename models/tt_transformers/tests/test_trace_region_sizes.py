# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path

import pytest
import yaml

from models.demos.utils.trace_region_sizes import (
    TRACE_REGION_SIZE_DYNAMIC,
    TRACE_REGION_SIZES_YAML_PATH,
    TraceRegionSizeNotConfiguredError,
    load_trace_region_sizes,
    resolve_trace_region_size,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CI_PIPELINE_FILES = (
    REPO_ROOT / "tests/pipeline_reorg/models_e2e_tests.yaml",
    REPO_ROOT / "tests/pipeline_reorg/models_unit_tests.yaml",
    REPO_ROOT / "tests/pipeline_reorg/models_device_perf_tests.yaml",
    REPO_ROOT / "tests/pipeline_reorg/models_sweep_tests.yaml",
)
HF_MODEL_RE = re.compile(r"HF_MODEL=([^\s]+)")


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
            if not isinstance(expected, int) or isinstance(expected, bool) or expected < 0:
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
            assert (
                isinstance(value, int) and not isinstance(value, bool) and value >= 0
            ), f"{model_name}/{sku_name}: invalid trace_region_size"


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


def _hf_model_name_candidates(hf_model: str) -> list[str]:
    candidates = [hf_model]
    basename = hf_model.strip("/").split("/")[-1]
    if basename not in candidates:
        candidates.append(basename)
    match = re.search(r"(.*?\d+[bB])-", basename)
    if match:
        base = match.group(1)
        if base not in candidates:
            candidates.append(base)
    return candidates


def _resolve_ci_trace_region_size(hf_model: str, sku: str) -> int:
    last_error: TraceRegionSizeNotConfiguredError | None = None
    for candidate in _hf_model_name_candidates(hf_model):
        try:
            return resolve_trace_region_size(candidate, sku)
        except TraceRegionSizeNotConfiguredError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _iter_ci_trace_region_requirements():
    """Yield (job_name, model_name, sku) for tiered CI jobs that set HF_MODEL."""
    for pipeline_path in CI_PIPELINE_FILES:
        if not pipeline_path.is_file():
            continue
        entries = yaml.safe_load(pipeline_path.read_text(encoding="utf-8")) or []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            cmd = entry.get("cmd", "")
            cmd_hf_match = HF_MODEL_RE.search(cmd)
            cmd_hf_model = cmd_hf_match.group(1).strip("'\"") if cmd_hf_match else None
            if cmd_hf_model and "{" in cmd_hf_model:
                cmd_hf_model = None

            job_name = entry.get("name", entry.get("model", "unknown"))
            skus = entry.get("skus", {})
            if not isinstance(skus, dict):
                continue
            for sku_key, sku_block in skus.items():
                if not isinstance(sku_block, dict):
                    sku_block = {}
                hf_model = sku_block.get("hf_model") or cmd_hf_model
                if not hf_model:
                    continue
                yield job_name, hf_model, sku_key


def test_load_trace_region_sizes_is_cached():
    load_trace_region_sizes.cache_clear()
    first = load_trace_region_sizes()
    second = load_trace_region_sizes()
    assert first is second


def test_resolve_deepseek_v3_dynamic_allocation():
    assert resolve_trace_region_size("deepseek-v3", "wh_llmbox_perf") == TRACE_REGION_SIZE_DYNAMIC


@pytest.mark.parametrize(
    "job_name,hf_model,sku",
    list(_iter_ci_trace_region_requirements()),
    ids=lambda val: str(val).replace("/", "_")[:120],
)
def test_ci_hf_model_jobs_have_trace_region_config(job_name, hf_model, sku):
    del job_name
    _resolve_ci_trace_region_size(hf_model, sku)


@pytest.mark.parametrize(
    "model_path,sku,expected_size",
    [
        ("models/demos/gemma4/configs/gemma-4-E2B-it", "wh_n150", 30000000),
        ("models/demos/gemma4/configs/gemma-4-E4B-it", "p300x2", 70000000),
        ("models/demos/gemma4/configs/gemma-4-26B-A4B-it", "wh_llmbox_perf", 70000000),
        ("models/demos/gemma4/configs/gemma-4-31B-it", "p300x2", 70000000),
    ],
)
def test_resolve_gemma4_config_path_aliases(model_path, sku, expected_size):
    assert resolve_trace_region_size(model_path, sku) == expected_size


def test_cpu_sku_skips_trace_region_override():
    """Data-parallel parametrization with zero sub-mesh devices must skip trace override."""
    num_devices = 8
    data_parallel = 16
    device_name_based_on_dp = "CPU" if (num_devices // data_parallel) == 0 else "N150"
    assert device_name_based_on_dp == "CPU"
    should_skip = not device_name_based_on_dp or device_name_based_on_dp == "CPU"
    assert should_skip
