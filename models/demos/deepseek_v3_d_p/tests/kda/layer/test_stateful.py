# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""State continuity and trace-replay tests for the TTNN KDA layer."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = run_for_blackhole()


def _forward(layer: ttKDA, hidden: torch.Tensor, state: KdaState) -> tuple[torch.Tensor, KdaState]:
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


def test_segmented_prefill_matches_reference_and_reuses_program(
    device: ttnn.Device,
    isolated_program_cache: None,
) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(1, 64, config.hidden_size, generator=torch.Generator().manual_seed(73)).to(torch.bfloat16)
    golden_first, golden_state = kda_forward_reference(hidden[:, :32], weights, config)
    golden_second, golden_state = kda_forward_reference(hidden[:, 32:], weights, config, golden_state)

    layer = ttKDA(device, config, weights)
    state = layer.allocate_state()
    actual_first, state = _forward(layer, hidden[:, :32], state)
    cache_entries_after_first = device.num_program_cache_entries()
    assert cache_entries_after_first > 0, "first KDA forward must populate the program cache"
    actual_second, state = _forward(layer, hidden[:, 32:], state)
    assert device.num_program_cache_entries() == cache_entries_after_first

    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    assert_accurate(golden_first, actual_first, name="first prefill segment output", pcc_threshold=0.999)
    assert_accurate(golden_second, actual_second, name="second prefill segment output", pcc_threshold=0.999)
    assert_accurate(
        golden_state.recurrent,
        ttnn.to_torch(state.recurrent),
        name="segmented recurrent state",
        pcc_threshold=0.999,
    )
    assert_accurate(
        golden_convolution,
        ttnn.to_torch(state.convolution),
        name="segmented convolution state",
        pcc_threshold=0.999,
    )


@pytest.mark.use_module_device
def test_trace_replay_matches_eager_without_mutating_input_state(device: ttnn.Device) -> None:
    config = make_config()
    hidden = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(1917)).to(torch.bfloat16)
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    layer = ttKDA(device, config, random_weights(config))

    with ttnn.manage_config("throw_exception_on_fallback", True):
        eager_output, _ = layer.forward(hidden_tt, layer.allocate_state())
    ttnn.synchronize_device(device)
    eager_result = ttnn.to_torch(eager_output)

    input_state = layer.allocate_state()
    input_state_before = (ttnn.to_torch(input_state.recurrent), ttnn.to_torch(input_state.convolution))
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, next_state = layer.forward(hidden_tt, input_state)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    assert next_state.recurrent is not input_state.recurrent
    assert next_state.convolution is not input_state.convolution

    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    replayed_output = ttnn.to_torch(output)
    input_state_after = (ttnn.to_torch(input_state.recurrent), ttnn.to_torch(input_state.convolution))
    ttnn.release_trace(device, trace_id)

    assert_bit_identical(eager_result, replayed_output, name="traced layer output")
    for name, expected, actual in zip(("recurrent", "convolution"), input_state_before, input_state_after):
        assert_bit_identical(expected, actual, name=f"traced input {name} state")
