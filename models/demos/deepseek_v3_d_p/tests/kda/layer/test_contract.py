# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Public contract tests for the composed TTNN KDA layer."""

from dataclasses import replace

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference
from models.demos.deepseek_v3_d_p.tests.kda.utils import (
    collect_mesh_accuracy_and_determinism_results,
    make_config,
    random_weights,
)
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.kda import KdaState, ttKDA
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import assert_accurate, assert_bit_identical

pytestmark = [run_for_blackhole(), pytest.mark.use_module_device]


def _hidden_to_device(hidden: torch.Tensor, device: ttnn.Device) -> ttnn.Tensor:
    return ttnn.from_torch(
        hidden,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _forward(layer: ttKDA, hidden: torch.Tensor, state: KdaState) -> tuple[ttnn.Tensor, KdaState]:
    with ttnn.manage_config("throw_exception_on_fallback", True):
        return layer.forward(_hidden_to_device(hidden, layer.device), state)


def _assert_state_metadata(state: KdaState, config) -> None:
    assert tuple(state.recurrent.shape) == (1, config.num_heads, config.head_k_dim, config.head_v_dim)
    assert state.recurrent.dtype == ttnn.float32
    assert state.recurrent.layout == ttnn.TILE_LAYOUT
    assert state.recurrent.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(state.convolution.shape) == (
        1,
        config.conv_kernel_size - 1,
        config.q_dim + config.k_dim + config.v_dim,
    )
    assert state.convolution.dtype == ttnn.bfloat16
    assert state.convolution.layout == ttnn.ROW_MAJOR_LAYOUT
    assert state.convolution.memory_config() == ttnn.DRAM_MEMORY_CONFIG


def test_layer_matches_reference_and_is_deterministic(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    sequence = 32
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(41 + sequence)).to(
        torch.bfloat16
    )
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)
    layer = ttKDA(device, config, weights)
    hidden_tt = _hidden_to_device(hidden, device)

    def run() -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        with ttnn.manage_config("throw_exception_on_fallback", True):
            output, state = layer.forward(hidden_tt, layer.allocate_state())
        return output, state.recurrent, state.convolution

    (output_tt, recurrent_tt, convolution_tt), mismatch_markers = collect_mesh_accuracy_and_determinism_results(run)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)
    assert_accurate(golden_output, ttnn.to_torch(output_tt), name="layer output", pcc_threshold=0.999)
    assert_accurate(
        golden_state.recurrent, ttnn.to_torch(recurrent_tt), name="layer recurrent state", pcc_threshold=0.999
    )
    assert_accurate(
        golden_convolution,
        ttnn.to_torch(convolution_tt),
        name="layer convolution state",
        pcc_threshold=0.999,
    )
    assert all(marker.item() == 0 for marker in mismatch_markers), "KDA layer is not bit-identical across runs"


def test_allocate_state_contract(device: ttnn.Device) -> None:
    config = make_config()
    layer = ttKDA(device, config, random_weights(config))
    first = layer.allocate_state()
    second = layer.allocate_state()

    _assert_state_metadata(first, config)
    _assert_state_metadata(second, config)
    assert first.recurrent.buffer_address() != second.recurrent.buffer_address()
    assert first.convolution.buffer_address() != second.convolution.buffer_address()


def test_forward_contract(device: ttnn.Device) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden = torch.randn(1, 64, config.hidden_size, generator=torch.Generator().manual_seed(109)).to(torch.bfloat16)
    golden_output, golden_state = kda_forward_reference(hidden, weights, config)
    layer = ttKDA(device, config, weights)
    input_state = layer.allocate_state()
    input_recurrent_before = ttnn.to_torch(input_state.recurrent)
    input_convolution_before = ttnn.to_torch(input_state.convolution)
    recurrent_address = input_state.recurrent.buffer_address()
    convolution_address = input_state.convolution.buffer_address()

    first_output, next_state = _forward(layer, hidden[:, :32], input_state)
    second_output, final_state = _forward(layer, hidden[:, 32:], next_state)
    actual_output = torch.cat((ttnn.to_torch(first_output), ttnn.to_torch(second_output)), dim=1)
    golden_convolution = torch.cat(
        (golden_state.q_convolution, golden_state.k_convolution, golden_state.v_convolution), dim=-1
    ).to(torch.bfloat16)

    assert tuple(first_output.shape) == (1, 32, config.hidden_size)
    assert first_output.dtype == ttnn.float32
    assert first_output.layout == ttnn.TILE_LAYOUT
    assert first_output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    _assert_state_metadata(final_state, config)
    assert next_state.recurrent is not input_state.recurrent
    assert next_state.convolution is not input_state.convolution
    assert input_state.recurrent.buffer_address() == recurrent_address
    assert input_state.convolution.buffer_address() == convolution_address
    assert_bit_identical(input_recurrent_before, ttnn.to_torch(input_state.recurrent), name="input recurrent state")
    assert_bit_identical(
        input_convolution_before, ttnn.to_torch(input_state.convolution), name="input convolution state"
    )
    assert_accurate(golden_output, actual_output, name="segmented layer output", pcc_threshold=0.999)
    assert_accurate(
        golden_state.recurrent,
        ttnn.to_torch(final_state.recurrent),
        name="segmented recurrent state",
        pcc_threshold=0.999,
    )
    assert_accurate(
        golden_convolution,
        ttnn.to_torch(final_state.convolution),
        name="segmented convolution state",
        pcc_threshold=0.999,
    )


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("axes", id="sp-and-tp-axes-must-be-distinct"),
        pytest.param("weight_sources", id="weight-sources-are-mutually-exclusive"),
        pytest.param("grouped_nonsquare", id="grouped-scan-requires-square-state"),
        pytest.param("batch_state", id="state-allocation-requires-batch-one"),
    ],
)
def test_layer_rejects_invalid_construction(case: str, device: ttnn.Device, expect_error) -> None:
    config = make_config()
    state_dict = random_weights(config)

    if case == "axes":
        with expect_error(ValueError, "requires distinct 2D SP/TP axes"):
            ttKDA(device, config, state_dict, sp_axis=0, tp_axis=0)
    elif case == "weight_sources":
        weights = ttKDA(device, config, state_dict).weights
        with expect_error(ValueError, "either constructed KDAWeights or host state_dict"):
            ttKDA(device, config, state_dict, weights=weights)
    elif case == "grouped_nonsquare":
        nonsquare_config = replace(config, head_v_dim=64)
        base_program_config = KDAProgramConfig()
        program_config = replace(
            base_program_config,
            recurrence=replace(base_program_config.recurrence, local_scan_strategy="grouped"),
        )
        with expect_error(ValueError, "grouped KDA affine prefix currently requires K == V"):
            ttKDA(device, nonsquare_config, random_weights(nonsquare_config), program_config=program_config)
    else:
        layer = ttKDA(device, config, state_dict)
        with expect_error(ValueError, "batch_size=1"):
            layer.allocate_state(batch_size=2)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param("sequence", id="sequence-must-be-positive-tile-multiple"),
        pytest.param("batch", id="forward-requires-batch-one"),
        pytest.param("hidden_width", id="hidden-width-must-match-config"),
        pytest.param("recurrent_shape", id="recurrent-state-shape"),
        pytest.param("recurrent_dtype", id="recurrent-state-dtype"),
        pytest.param("convolution_shape", id="convolution-state-shape"),
        pytest.param("convolution_dtype", id="convolution-state-dtype-and-layout"),
    ],
)
def test_layer_rejects_invalid_forward(case: str, device: ttnn.Device, expect_error) -> None:
    config = make_config()
    layer = ttKDA(device, config, random_weights(config))
    hidden_shape = (1, 32, config.hidden_size)
    state = layer.allocate_state()

    if case == "sequence":
        hidden_shape, error = (1, 4, config.hidden_size), r"requires local T .* divisible by 32"
    elif case == "batch":
        hidden_shape, error = (2, 32, config.hidden_size), "requires batch size 1"
    elif case == "hidden_width":
        hidden_shape, error = (1, 32, config.hidden_size + 32), "hidden_states shape"
    elif case == "recurrent_shape":
        state = replace(
            state,
            recurrent=ttnn.zeros(
                (1, config.num_heads, config.head_k_dim, config.head_v_dim + 32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        error = "recurrent state shape"
    elif case == "recurrent_dtype":
        state = replace(
            state,
            recurrent=ttnn.zeros(
                tuple(state.recurrent.shape),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        error = "recurrent state dtype"
    elif case == "convolution_shape":
        state = replace(
            state,
            convolution=ttnn.zeros(
                (1, config.conv_kernel_size, config.q_dim + config.k_dim + config.v_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        error = "convolution state shape"
    else:
        state = replace(
            state,
            convolution=ttnn.zeros(
                tuple(state.convolution.shape),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            ),
        )
        error = "convolution state must be BF16 row-major"

    hidden = torch.randn(*hidden_shape, generator=torch.Generator().manual_seed(45), dtype=torch.bfloat16)
    with expect_error(ValueError, error):
        _forward(layer, hidden, state)
