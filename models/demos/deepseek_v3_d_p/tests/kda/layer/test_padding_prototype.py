# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Throwaway correctness coverage for the current KDA tail-padding prototype."""

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import make_config, random_weights
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]


def _run(layer: ttKDA, hidden: torch.Tensor, length: int) -> tuple[torch.Tensor, KdaState]:
    hidden_tt = ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=layer.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with ttnn.manage_config("throw_exception_on_fallback", True):
        output, state = layer.forward(hidden_tt, layer.allocate_state(), length)
    return ttnn.to_torch(output), state


def test_tail_padding_matches_trimmed_reference_and_ignores_padding(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    length = 32
    real = torch.randn(1, length, config.hidden_size, generator=torch.Generator().manual_seed(1701)).to(torch.bfloat16)
    padding_a = torch.zeros(1, 64 - length, config.hidden_size, dtype=torch.bfloat16)
    padding_b = torch.randn(
        1, 64 - length, config.hidden_size, generator=torch.Generator().manual_seed(1702), dtype=torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(real, weights, config)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    layer = ttKDA(device, config, weights)

    output_a, state_a = _run(layer, torch.cat((real, padding_a), dim=1), length)
    output_b, state_b = _run(layer, torch.cat((real, padding_b), dim=1), length)

    assert tuple(output_a.shape) == (1, length, config.hidden_size)
    assert_accurate(golden_output, output_a, name="valid output", pcc_threshold=0.999)
    assert_accurate(
        golden_state.recurrent,
        ttnn.to_torch(state_a.recurrent),
        name="recurrent state",
        pcc_threshold=0.999,
    )
    assert_accurate(
        golden_convolution,
        ttnn.to_torch(state_a.convolution),
        name="convolution state",
        pcc_threshold=0.999,
    )
    assert_bit_identical(output_a, output_b, name="padding-invariant output")
    assert_bit_identical(
        ttnn.to_torch(state_a.recurrent),
        ttnn.to_torch(state_b.recurrent),
        name="padding-invariant recurrent state",
    )
    assert_bit_identical(
        ttnn.to_torch(state_a.convolution),
        ttnn.to_torch(state_b.convolution),
        name="padding-invariant convolution state",
    )
