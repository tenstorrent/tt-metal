# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the stateless full-layer KDA reference transition."""

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.kda import kda_forward_reference, kda_forward_stages
from models.demos.deepseek_v3_d_p.reference.kda.ops import sigmoid_gated_rms_norm_reference
from models.demos.deepseek_v3_d_p.reference.kda.tests.helpers import make_config, random_weights
from tests.ttnn.unit_tests.operations.experimental.kda.kda_test_utils import (
    assert_accurate,
    assert_bit_identical,
    assert_equal,
)


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_split_forward_matches_full_forward_without_mutating_input_state(use_full_rank_gate: bool) -> None:
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    hidden_states = torch.randn(1, 7, config.hidden_size, generator=torch.Generator().manual_seed(31))

    full_output, full_state = kda_forward_reference(hidden_states, weights, config)
    first_output, first_state = kda_forward_reference(hidden_states[:, :5], weights, config)
    input_state_snapshot = tuple(tensor.clone() for tensor in first_state.__dict__.values())
    last_output, split_state = kda_forward_reference(hidden_states[:, 5:], weights, config, first_state)

    assert_accurate(
        full_output,
        torch.cat((first_output, last_output), dim=1),
        name="split layer output",
        pcc_threshold=0.99999,
    )
    for index, (full_tensor, split_tensor) in enumerate(
        zip(full_state.__dict__.values(), split_state.__dict__.values())
    ):
        assert_accurate(full_tensor, split_tensor, name=f"split layer state {index}", pcc_threshold=0.99999)
    for input_tensor, snapshot in zip(first_state.__dict__.values(), input_state_snapshot):
        assert_equal(snapshot, input_tensor, name="reference input state unchanged")


def test_reference_layer_is_bit_identical() -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 32, config.hidden_size, generator=torch.Generator().manual_seed(3031))
    expected_output, expected_state = kda_forward_reference(hidden_states, weights, config)
    for iteration in range(2):
        actual_output, actual_state = kda_forward_reference(hidden_states, weights, config)
        assert_bit_identical(expected_output, actual_output, name=f"output iteration {iteration}")
        for field in expected_state.__dataclass_fields__:
            assert_bit_identical(
                getattr(expected_state, field),
                getattr(actual_state, field),
                name=f"{field} iteration {iteration}",
            )


@pytest.mark.parametrize("use_full_rank_gate", [False, True])
def test_reference_layer_computes_in_requested_dtype(use_full_rank_gate: bool) -> None:
    config = make_config(use_full_rank_gate=use_full_rank_gate)
    weights = random_weights(config)
    hidden_states = torch.randn(1, 9, config.hidden_size, generator=torch.Generator().manual_seed(47)).bfloat16()

    default_output, default_state = kda_forward_reference(hidden_states, weights, config)
    explicit_output, explicit_state = kda_forward_reference(hidden_states, weights, config, dtype=torch.float32)
    golden_output, golden_state = kda_forward_reference(hidden_states, weights, config, dtype=torch.float64)

    assert_bit_identical(default_output, explicit_output, name="default dtype output")
    assert golden_output.dtype == torch.float64
    for field in default_state.__dataclass_fields__:
        assert_bit_identical(getattr(default_state, field), getattr(explicit_state, field), name=f"default {field}")
        assert getattr(golden_state, field).dtype == torch.float64, field
    # FP32 and FP64 compute the same function; they differ only by FP32 rounding.
    relative_error = (default_output.double() - golden_output).norm() / golden_output.norm()
    assert relative_error < 1e-5, relative_error


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_stages_expose_the_forward_intermediates(dtype: torch.dtype) -> None:
    config = make_config()
    weights = random_weights(config)
    hidden_states = torch.randn(1, 6, config.hidden_size, generator=torch.Generator().manual_seed(53)).bfloat16()

    output, state = kda_forward_reference(hidden_states, weights, config, dtype=dtype)
    stages = kda_forward_stages(hidden_states, weights, config, dtype=dtype)

    assert_bit_identical(output, stages.output, name="forward output")
    assert_bit_identical(state.recurrent, stages.state.recurrent, name="forward state")
    heads, k_dim, v_dim = config.num_heads, config.head_k_dim, config.head_v_dim
    expected_shapes = {
        "q": (1, 6, heads, k_dim),
        "k": (1, 6, heads, k_dim),
        "v": (1, 6, heads, v_dim),
        "gate": (1, 6, heads, k_dim),
        "beta": (1, 6, heads),
        "recurrence": (1, 6, heads, v_dim),
        "output_gate": (1, 6, heads, v_dim),
        "normalized": (1, 6, heads * v_dim),
    }
    for name, shape in expected_shapes.items():
        tensor = getattr(stages, name)
        assert tuple(tensor.shape) == shape and tensor.dtype == dtype, (name, tuple(tensor.shape), tensor.dtype)
    # The last two stages recompose the output exactly.
    normalized = sigmoid_gated_rms_norm_reference(
        stages.recurrence, stages.output_gate, weights["o_norm.weight"], config.norm_eps, dtype=dtype
    ).reshape(1, 6, heads * v_dim)
    assert_bit_identical(normalized, stages.normalized, name="normalized")
    assert_bit_identical(F.linear(normalized, weights["o_proj.weight"].to(dtype)), output, name="o_proj")


def test_reference_layer_validates_weights(expect_error) -> None:
    config = make_config()
    weights = random_weights(config)
    del weights["q_proj.weight"]

    with expect_error(ValueError, "missing KDA weight: q_proj.weight"):
        kda_forward_reference(torch.zeros(1, 1, config.hidden_size), weights, config)
