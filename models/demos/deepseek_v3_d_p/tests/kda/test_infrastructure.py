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
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_small_kda_test_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig, KDARecurrenceProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, make_actual_start

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
        output, _ = layer.forward(hidden_tt, state, make_actual_start(layer.device))
    return ttnn.to_torch(output)


def _cache_artifact_names(path: Path) -> set[str]:
    return {artifact.name for artifact in path.glob("*.tensorbin")}


def test_cache_identity_changes_with_weight_content() -> None:
    baseline = {"weight": torch.tensor([1.0, 2.0], dtype=torch.bfloat16)}
    changed = {"weight": torch.tensor([1.0, 3.0], dtype=torch.bfloat16)}

    assert kda_state_dict_sha256(baseline) == kda_state_dict_sha256(baseline)
    assert kda_state_dict_sha256(baseline) != kda_state_dict_sha256(changed)


def test_cache_key_covers_semantic_config_and_placement(tmp_path: Path) -> None:
    baseline = make_small_kda_test_config()
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
    assert not KDAWeights.check_cache_complete(
        tmp_path / "missing", "layer_0.kda", make_small_kda_test_config(), object()
    )


def test_weight_loader_rejects_empty_source_weights(expect_error) -> None:
    with expect_error(ValueError, "state_dict must be non-empty"):
        load_kda_weights(object(), make_small_kda_test_config(), {})


def test_cache_build_requires_source_weights(tmp_path: Path, expect_error) -> None:
    with expect_error(ValueError, "requires a state_dict"):
        KDAWeights.build_ttnn_cache(None, tmp_path, "layer_0.kda", make_small_kda_test_config(), object())


@pytest.mark.use_module_device
def test_cache_only_load_rejects_corrupt_tensor(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_small_kda_test_config()
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(random_weights(config), tmp_path, cache_prefix, config, device)
    next(tmp_path.glob("*.tensorbin")).write_bytes(b"corrupt")

    with expect_error(RuntimeError, "too small"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)


@pytest.mark.use_module_device
def test_cached_and_in_memory_layers_match(device: ttnn.Device, tmp_path: Path) -> None:
    config = make_small_kda_test_config()
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(151), dtype=torch.bfloat16)
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(state_dict, tmp_path, cache_prefix, config, device)

    in_memory_layer = ttKDA(device, config, state_dict, active_seq_len=32)
    cached_layer = ttKDA(
        device, config, weights=KDAWeights.from_cache(tmp_path, cache_prefix, config, device), active_seq_len=32
    )
    cache_only_layer = ttKDA(device, config, None, weight_cache_path=tmp_path, layer_idx=0, active_seq_len=32)
    outputs = {
        "in-memory": _forward(in_memory_layer, hidden, in_memory_layer.allocate_state()),
        "preloaded-cache": _forward(cached_layer, hidden, cached_layer.allocate_state()),
        "cache-only": _forward(cache_only_layer, hidden, cache_only_layer.allocate_state()),
    }

    assert_accurate(
        outputs["in-memory"],
        outputs["preloaded-cache"],
        name="preloaded cache",
        pcc_threshold=0.9999,
        nrmse_threshold=1e-06,
    )
    assert_accurate(
        outputs["in-memory"], outputs["cache-only"], name="cache-only", pcc_threshold=0.9999, nrmse_threshold=1e-06
    )


@pytest.mark.use_module_device
def test_program_config_resolution(device: ttnn.Device) -> None:
    config = make_small_kda_test_config()
    program_config = replace(KDAProgramConfig(), qkv_channel_chunk_size=128, tp_ccl_topology=ttnn.Topology.Ring)
    layer = ttKDA(device, config, random_weights(config), program_config=program_config, active_seq_len=32)

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


def test_kimi_k3_configuration_does_not_depend_on_sequence_length():
    from models.demos.deepseek_v3_d_p.tt.kda.config import kimi_k3_program_config

    config = kimi_k3_program_config(tp_ccl_topology=ttnn.Topology.Linear)
    assert config.recurrence.summary_group_chunks is None
    assert config.recurrence.local_scan_strategy == "grouped"
    assert config.derived_projection_matmuls


_BLACKHOLE_GRID = ttnn.CoreCoord(12, 10)


@pytest.mark.parametrize(
    "local_rows,heads,expected_group_chunks",
    [
        # Galaxy SP8xTP4 (24 heads per device): one group per head fills the grid at every length up to T=20480.
        *[(rows, 24, rows // 32) for rows in (128, 640, 768, 896, 1280, 1408, 2560)],
        # LoudBox SP1xTP8 at T=5120 (12 heads): five groups split each head's value tiles over idle workers.
        (5120, 12, 32),
        # LoudBox SP4xTP2 at T=5120 (48 heads): more groups would not add workers.
        (1280, 48, 40),
        # A prime chunk count can only run as one group.
        (37 * 32, 24, 37),
    ],
)
def test_auto_summary_group_chunks(local_rows, heads, expected_group_chunks):
    from models.demos.deepseek_v3_d_p.tt.kda.config import auto_summary_group_chunks

    num_chunks = local_rows // 32
    group_chunks = auto_summary_group_chunks(_BLACKHOLE_GRID, heads, num_chunks, value_tiles=4)
    assert group_chunks == expected_group_chunks
    assert num_chunks % group_chunks == 0


def test_auto_summary_group_chunks_rejects_more_heads_than_workers(expect_error):
    from models.demos.deepseek_v3_d_p.tt.kda.config import auto_summary_group_chunks

    with expect_error(ValueError, "summary owners"):
        auto_summary_group_chunks(_BLACKHOLE_GRID, 121, 4, value_tiles=4)


@pytest.mark.parametrize("rows", [32 * tiles for tiles in range(10, 129, 3)])
@pytest.mark.parametrize(
    "k,n", [(7168, 12440), (3072, 7168), (7168, 6220), (1536, 7168)], ids=["in", "out", "in8", "out8"]
)
def test_projection_matmul_config_is_valid_for_any_length(rows, k, n):
    from models.demos.deepseek_v3_d_p.tt.kda.config import (
        KDA_PROJECTION_CB_L1_BYTES_PER_CORE,
        KIMI_K3_PROJECTION_DTYPE,
        _projection_cb_bytes,
        projection_matmul_config,
    )

    config = projection_matmul_config(_BLACKHOLE_GRID, rows, k, n, KIMI_K3_PROJECTION_DTYPE)
    assert config is not None
    assert config.per_core_M * _BLACKHOLE_GRID.y >= rows // 32
    assert config.per_core_N * _BLACKHOLE_GRID.x >= -(-n // 32)
    assert config.per_core_M % config.out_block_h == 0 and config.out_block_h >= min(config.per_core_M, 2)
    assert config.out_block_h % config.out_subblock_h == 0 and config.per_core_N % config.out_subblock_w == 0
    assert config.out_subblock_h * config.out_subblock_w <= 4
    assert (k // 32) % config.in0_block_w == 0
    assert (
        _projection_cb_bytes(config.out_block_h, config.in0_block_w, config.per_core_N, KIMI_K3_PROJECTION_DTYPE)
        <= KDA_PROJECTION_CB_L1_BYTES_PER_CORE
    )


def test_projection_matmul_config_keeps_tuned_schedules():
    from models.demos.deepseek_v3_d_p.tt.kda.config import projection_matmul_config

    # Galaxy SP8xTP4 at T=5120: the schedules tuned at 640 rows per device with bf16 weights.
    projection = projection_matmul_config(_BLACKHOLE_GRID, 640, 7168, 12440, ttnn.bfloat16)
    assert (projection.per_core_M, projection.out_block_h, projection.in0_block_w) == (2, 2, 4)
    assert (projection.out_subblock_h, projection.out_subblock_w, projection.per_core_N) == (1, 3, 33)
    projection = projection_matmul_config(_BLACKHOLE_GRID, 640, 3072, 7168, ttnn.bfloat16)
    assert (projection.per_core_M, projection.out_block_h, projection.in0_block_w) == (2, 2, 8)
    assert (projection.out_subblock_h, projection.out_subblock_w, projection.per_core_N) == (2, 1, 19)
    # Below one row tile per grid row, or in one-row-tile passes, the auto-selected schedule is faster.
    assert projection_matmul_config(_BLACKHOLE_GRID, 256, 7168, 12440, ttnn.bfloat16) is None
    assert projection_matmul_config(_BLACKHOLE_GRID, 1280, 7168, 24880, ttnn.bfloat16) is None
