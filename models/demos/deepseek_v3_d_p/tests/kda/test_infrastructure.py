# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-cache and constructor-policy contracts for the KDA layer."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import kda_state_dict_sha256
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = run_for_blackhole()


def _forward(layer: ttKDA, hidden: torch.Tensor, state: KdaState) -> torch.Tensor:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, _ = layer.forward(hidden_tt, state)
    return ttnn.to_torch(output)


def _cache_artifact_names(path: Path) -> set[str]:
    return {artifact.name for artifact in path.glob("*.tensorbin")}


def test_cache_identity_changes_with_weight_content() -> None:
    baseline = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    changed = {"weight": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)}

    assert kda_state_dict_sha256(baseline) == kda_state_dict_sha256(baseline)
    assert kda_state_dict_sha256(baseline) != kda_state_dict_sha256(changed)


def test_cache_key_covers_semantic_config_and_placement(tmp_path: Path) -> None:
    baseline = make_config()
    bounded = replace(baseline, gate_lower_bound=-5.0)
    cases = {
        "baseline": (baseline, 1),
        "bounded-gate": (bounded, 1),
        "tp-axis-0": (baseline, 0),
    }
    names = {}
    for case, (config, tensor_parallel_axis) in cases.items():
        case_path = tmp_path / case
        KDAWeights.build_ttnn_cache(
            random_weights(config),
            case_path,
            "layer_0.kda",
            config,
            object(),
            tensor_parallel_axis=tensor_parallel_axis,
        )
        names[case] = _cache_artifact_names(case_path)

    assert names["baseline"]
    assert names["baseline"] != names["bounded-gate"]
    assert names["baseline"] != names["tp-axis-0"]


def test_missing_cache_is_incomplete(tmp_path: Path) -> None:
    assert not KDAWeights.check_cache_complete(tmp_path / "missing", "layer_0.kda", make_config(), object())


def test_weight_loader_rejects_empty_source_weights(expect_error) -> None:
    with expect_error(ValueError, "state_dict must be non-empty"):
        load_kda_weights(object(), make_config(), {})


def test_cache_build_requires_source_weights(tmp_path: Path, expect_error) -> None:
    with expect_error(ValueError, "requires a state_dict"):
        KDAWeights.build_ttnn_cache(None, tmp_path, "layer_0.kda", make_config(), object())


@pytest.mark.use_module_device
def test_cache_only_load_rejects_corrupt_tensor(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(random_weights(config), tmp_path, cache_prefix, config, device)
    next(tmp_path.glob("*.tensorbin")).write_bytes(b"corrupt")

    with expect_error(RuntimeError, "too small"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)


@pytest.mark.use_module_device
def test_cached_and_in_memory_layers_match(device: ttnn.Device, tmp_path: Path) -> None:
    config = make_config()
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(151), dtype=torch.bfloat16)
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(state_dict, tmp_path, cache_prefix, config, device)

    in_memory_layer = ttKDA(device, config, state_dict)
    cached_layer = ttKDA(
        device,
        config,
        weights=KDAWeights.from_cache(tmp_path, cache_prefix, config, device),
    )
    cache_only_layer = ttKDA(device, config, None, weight_cache_path=tmp_path, layer_idx=0)
    outputs = {
        "in-memory": _forward(in_memory_layer, hidden, in_memory_layer.allocate_state()),
        "preloaded-cache": _forward(cached_layer, hidden, cached_layer.allocate_state()),
        "cache-only": _forward(cache_only_layer, hidden, cache_only_layer.allocate_state()),
    }

    assert_accurate(outputs["in-memory"], outputs["preloaded-cache"], name="preloaded cache", pcc_threshold=0.9999)
    assert_accurate(outputs["in-memory"], outputs["cache-only"], name="cache-only", pcc_threshold=0.9999)


@pytest.mark.use_module_device
def test_program_config_resolution(device: ttnn.Device) -> None:
    config = make_config()
    program_config = replace(KDAProgramConfig(), qkv_channel_chunk_size=128, tp_ccl_topology=ttnn.Topology.Ring)
    layer = ttKDA(device, config, random_weights(config), program_config=program_config)

    assert layer.qkv_convolution_program_config.channel_chunk_size == 96
    assert layer.tp_ccl_topology == ttnn.Topology.Ring


@pytest.mark.parametrize(
    "config_type,kwargs,message",
    [
        pytest.param(KDAProgramConfig, {"qkv_channel_chunk_size": 0}, "positive multiple", id="zero-qkv-chunk"),
        pytest.param(KDAProgramConfig, {"qkv_channel_chunk_size": -32}, "positive multiple", id="negative-qkv-chunk"),
        pytest.param(KDAProgramConfig, {"qkv_channel_chunk_size": 31}, "positive multiple", id="unaligned-qkv-chunk"),
        pytest.param(
            KDARecurrenceProgramConfig,
            {"local_scan_strategy": "invalid"},
            "local_scan_strategy",
            id="unknown-scan-strategy",
        ),
        pytest.param(
            KDARecurrenceProgramConfig,
            {"summary_group_chunks": 0},
            "summary_group_chunks must be positive",
            id="nonpositive-summary-group",
        ),
    ],
)
def test_program_config_rejects_invalid_values(
    config_type: type[KDAProgramConfig] | type[KDARecurrenceProgramConfig],
    kwargs: dict[str, object],
    message: str,
    expect_error,
) -> None:
    with expect_error(ValueError, message):
        config_type(**kwargs)
