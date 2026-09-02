# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-cache and construction-policy tests awaiting infrastructure consolidation."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate

pytestmark = [
    run_for_blackhole(),
    pytest.mark.use_module_device,
]


def _forward(
    layer: ttKDA,
    hidden: torch.Tensor,
    state: KdaState,
) -> tuple[torch.Tensor, KdaState]:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, next_state = layer.forward(hidden_tt, state)
    return ttnn.to_torch(output), next_state


def test_offline_cache_and_cache_only_layer_pcc(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    state_dict = random_weights(config)
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(151), dtype=torch.bfloat16)
    golden_output, _ = kda_forward_reference(hidden, state_dict, config)
    cache_prefix = "layer_0.kda"

    assert not KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    with expect_error(FileNotFoundError, "incomplete KDA TTNN cache"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)

    KDAWeights.build_ttnn_cache(state_dict, tmp_path, cache_prefix, config, device)
    assert KDAWeights.check_cache_complete(tmp_path, cache_prefix, config, device)
    cached_weights = KDAWeights.from_cache(tmp_path, cache_prefix, config, device)
    cached_layer = ttKDA(device, config, weights=cached_weights)
    cached_output, _ = _forward(cached_layer, hidden, cached_layer.allocate_state())
    assert_accurate(golden_output, cached_output, name="loaded-cache output", pcc_threshold=0.999)

    cache_only_layer = ttKDA(device, config, None, weight_cache_path=tmp_path, layer_idx=0)
    cache_only_output, _ = _forward(cache_only_layer, hidden, cache_only_layer.allocate_state())
    assert_accurate(golden_output, cache_only_output, name="cache-only output", pcc_threshold=0.999)


def test_cache_only_load_rejects_corrupt_tensorbin(device: ttnn.Device, tmp_path: Path, expect_error) -> None:
    config = make_config()
    cache_prefix = "layer_0.kda"
    KDAWeights.build_ttnn_cache(random_weights(config), tmp_path, cache_prefix, config, device)
    next(tmp_path.glob("*.tensorbin")).write_bytes(b"corrupt")

    with expect_error(RuntimeError, "too small"):
        KDAWeights.from_cache(tmp_path, cache_prefix, config, device)


def test_program_config_is_resolved_at_construction(device: ttnn.Device) -> None:
    config = make_config()
    program_config = replace(KDAProgramConfig(), qkv_channel_chunk_size=128, tp_ccl_topology=ttnn.Topology.Ring)
    layer = ttKDA(device, config, random_weights(config), program_config=program_config)
    assert layer.qkv_convolution_program_config.channel_chunk_size == 96
    assert layer.tp_ccl_topology == ttnn.Topology.Ring
