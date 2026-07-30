# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import fields, replace

import pytest

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    _decode_program,
    resolve_policy,
)


def test_optimized_class_owns_measured_entry_points():
    assert OptimizedDecoder is not FunctionalDecoder
    assert OptimizedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert OptimizedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert OptimizedDecoder._full_attention_decode is not FunctionalDecoder._full_attention_decode
    assert OptimizedDecoder._linear_attention_decode is not FunctionalDecoder._linear_attention_decode
    source = inspect.getsource(OptimizedDecoder.decode_forward)
    assert "FunctionalDecoder" not in source
    assert "super()" not in source


def test_linear_default_policy_is_reduced_precision_and_uses_selected_projection_forms():
    policy = resolve_policy("default", "linear_attention")
    assert policy.attention_weight_dtype == ttnn.bfloat4_b
    assert policy.mlp_gate_up_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_dtype == ttnn.bfloat4_b
    assert policy.cache_dtype == ttnn.bfloat8_b
    assert policy.attention_fidelity == ttnn.MathFidelity.LoFi
    assert policy.mlp_fidelity == ttnn.MathFidelity.LoFi
    assert policy.packed_qkv
    # The packed gate/up candidate is retained in the sweep, but split wins
    # at both B1 and B32 because it permits fused SiLU and has lower L1 demand.
    assert not policy.packed_gate_up
    assert policy.linear_packed_decode
    assert policy.linear_outer_product
    assert policy.linear_recurrent_program == "grid4_w4"
    assert policy.linear_recurrent_fidelity == ttnn.MathFidelity.HiFi2
    assert policy.linear_recurrent_state_dtype == ttnn.bfloat8_b
    assert policy.linear_input_weight_dtype == ttnn.bfloat4_b
    assert policy.linear_output_weight_dtype == ttnn.bfloat4_b
    assert policy.linear_input_fidelity == ttnn.MathFidelity.LoFi
    assert policy.linear_output_fidelity == ttnn.MathFidelity.LoFi
    assert policy.linear_packed_in0_block_w == 5
    assert policy.linear_out_in0_block_w == 12


@pytest.mark.parametrize(
    ("candidate", "changes"),
    (
        ("linear_input_bf16_lofi", {"linear_input_fidelity": ttnn.MathFidelity.LoFi}),
        ("linear_output_bf16_lofi", {"linear_output_fidelity": ttnn.MathFidelity.LoFi}),
        (
            "linear_both_bf16_lofi",
            {
                "linear_input_fidelity": ttnn.MathFidelity.LoFi,
                "linear_output_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "linear_input_bfp8_hifi2",
            {"linear_input_weight_dtype": ttnn.bfloat8_b},
        ),
        (
            "linear_input_bfp8_lofi",
            {
                "linear_input_weight_dtype": ttnn.bfloat8_b,
                "linear_input_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "linear_output_bfp8_hifi2",
            {"linear_output_weight_dtype": ttnn.bfloat8_b},
        ),
        (
            "linear_output_bfp8_lofi",
            {
                "linear_output_weight_dtype": ttnn.bfloat8_b,
                "linear_output_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "linear_input_bfp4_lofi",
            {
                "linear_input_weight_dtype": ttnn.bfloat4_b,
                "linear_input_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "linear_output_bfp4_lofi",
            {
                "linear_output_weight_dtype": ttnn.bfloat4_b,
                "linear_output_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
        (
            "linear_both_bfp4_lofi",
            {
                "linear_input_weight_dtype": ttnn.bfloat4_b,
                "linear_input_fidelity": ttnn.MathFidelity.LoFi,
                "linear_output_weight_dtype": ttnn.bfloat4_b,
                "linear_output_fidelity": ttnn.MathFidelity.LoFi,
            },
        ),
    ),
)
def test_linear_projection_candidates_are_independently_attributable(candidate, changes):
    selected = POLICIES["linear_proj_bf16_hifi2"]
    assert POLICIES[candidate] == replace(selected, **changes)


@pytest.mark.parametrize(
    ("candidate", "changes"),
    (
        ("linear_final_input_w1", {"linear_packed_in0_block_w": 1}),
        ("linear_final_input_w4", {"linear_packed_in0_block_w": 4}),
        ("linear_final_input_w5", {"linear_packed_in0_block_w": 5}),
        ("linear_final_input_w10", {"linear_packed_in0_block_w": 10}),
        ("linear_final_input_w20", {"linear_packed_in0_block_w": 20}),
        ("linear_final_output_w1", {"linear_out_in0_block_w": 1}),
        ("linear_final_output_w2", {"linear_out_in0_block_w": 2}),
        ("linear_final_output_w4", {"linear_out_in0_block_w": 4}),
        ("linear_final_output_w6", {"linear_out_in0_block_w": 6}),
        ("linear_final_output_w8", {"linear_out_in0_block_w": 8}),
        ("linear_final_output_w12", {"linear_out_in0_block_w": 12}),
        ("linear_final_output_w24", {"linear_out_in0_block_w": 24}),
        (
            "linear_final_input_w5_output_w8",
            {"linear_packed_in0_block_w": 5, "linear_out_in0_block_w": 8},
        ),
        (
            "linear_final_input_w5_output_w12",
            {"linear_packed_in0_block_w": 5, "linear_out_in0_block_w": 12},
        ),
        (
            "linear_final_input_w5_output_w24",
            {"linear_packed_in0_block_w": 5, "linear_out_in0_block_w": 24},
        ),
        ("linear_final_grid4", {"decode_storage_cores": 4}),
    ),
)
def test_linear_geometry_candidates_preserve_selected_precision(candidate, changes):
    selected = POLICIES["linear_both_bfp4_lofi"]
    assert POLICIES[candidate] == replace(selected, **changes)


@pytest.mark.parametrize(
    ("candidate", "state_dtype"),
    (
        ("linear_state_fp32", ttnn.float32),
        ("linear_state_bf16", ttnn.bfloat16),
        ("linear_state_bfp8", ttnn.bfloat8_b),
        ("linear_state_bfp4", ttnn.bfloat4_b),
    ),
)
def test_linear_recurrent_state_candidates_change_only_the_storage_boundary(candidate, state_dtype):
    selected = POLICIES["linear_proj_bf16_hifi2"]
    candidate_policy = resolve_policy(candidate, "linear_attention")
    assert candidate_policy.linear_recurrent_state_dtype == state_dtype
    assert candidate_policy == replace(selected, linear_recurrent_state_dtype=state_dtype)


def test_reduced_precision_recurrent_state_owns_prefill_and_decode_boundaries():
    prefill_source = inspect.getsource(OptimizedDecoder._linear_attention_prefill_chunk)
    decode_source = inspect.getsource(OptimizedDecoder._linear_attention_decode)
    assert "FunctionalDecoder._linear_attention_prefill_chunk" in prefill_source
    assert "ttnn.float32" in prefill_source
    assert "state_dtype" in prefill_source
    assert "ttnn.bfloat16" in decode_source
    assert "stored_recurrent" in decode_source


def test_full_attention_default_preserves_real_weight_accuracy_and_cache_capacity():
    policy = resolve_policy("default", "full_attention")
    assert policy.attention_weight_dtype == ttnn.bfloat16
    assert policy.mlp_gate_up_dtype == ttnn.bfloat4_b
    assert policy.mlp_down_dtype == ttnn.bfloat4_b
    assert policy.cache_dtype == ttnn.bfloat8_b
    assert policy.attention_fidelity == ttnn.MathFidelity.HiFi4
    assert policy.qkv_fidelity == ttnn.MathFidelity.HiFi2
    assert policy.o_fidelity == ttnn.MathFidelity.HiFi2
    assert policy.mlp_fidelity == ttnn.MathFidelity.LoFi


@pytest.mark.parametrize(
    ("candidate", "field_name", "value"),
    (
        ("final_cum_qkv_w4", "qkv_decode_in0_block_w", 4),
        ("final_cum_o_w4", "o_decode_in0_block_w", 4),
        ("final_cum_o_w6", "o_decode_in0_block_w", 6),
        ("final_cum_o_w8", "o_decode_in0_block_w", 8),
        ("final_cum_o_w12", "o_decode_in0_block_w", 12),
        ("final_cum_gate_w2", "mlp_gate_decode_in0_block_w", 2),
        ("final_cum_gate_w4", "mlp_gate_decode_in0_block_w", 4),
        ("final_cum_up_w2", "mlp_up_decode_in0_block_w", 2),
        ("final_cum_up_w4", "mlp_up_decode_in0_block_w", 4),
        ("final_cum_down_w4", "mlp_down_in0_block_w", 4),
        ("final_cum_down_w34", "mlp_down_in0_block_w", 34),
        ("final_cum_down_w68", "mlp_down_in0_block_w", 68),
    ),
)
def test_full_role_sweep_candidates_change_one_cumulative_field(candidate, field_name, value):
    baseline = POLICIES["final_cumulative"]
    policy = POLICIES[candidate]
    changed_fields = {
        field.name for field in fields(type(baseline)) if getattr(policy, field.name) != getattr(baseline, field.name)
    }
    assert changed_fields == {field_name}
    assert getattr(policy, field_name) == value


def test_optimized_class_owns_memory_bounded_long_prefill():
    source = inspect.getsource(OptimizedDecoder._full_attention_prefill_long)
    assert "chunked_scaled_dot_product_attention" in source
    assert "paged_fill_cache" in source
    assert "FunctionalDecoder" not in source
    assert "super()" not in source


@pytest.mark.parametrize("candidate", sorted(POLICIES))
def test_candidate_geometry_is_legal(candidate):
    policy = POLICIES[candidate]
    scale = 8 // policy.decode_storage_cores
    assert 20 * scale % policy.decode_in0_block_w == 0
    assert 20 * scale % (policy.qkv_decode_in0_block_w or policy.decode_in0_block_w) == 0
    assert 24 * scale % policy.o_decode_in0_block_w == 0
    assert 20 * scale % (policy.mlp_gate_decode_in0_block_w or policy.decode_in0_block_w) == 0
    assert 20 * scale % (policy.mlp_up_decode_in0_block_w or policy.decode_in0_block_w) == 0
    assert 68 * scale % policy.mlp_down_in0_block_w == 0
    assert 20 * scale % policy.linear_packed_in0_block_w == 0
    assert 24 * scale % policy.linear_out_in0_block_w == 0


def test_dram_sharded_program_contract_has_no_compute_grid_or_subblock_controls(expect_error):
    config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=2,
        per_core_M=1,
        per_core_N=56,
        fused_activation=None,
    )
    unavailable = (
        "compute_with_storage_grid_size",
        "out_subblock_h",
        "out_subblock_w",
        "out_block_h",
        "out_block_w",
        "allowed_worker_cores",
    )
    for field in unavailable:
        assert not hasattr(config, field)
        with expect_error(AttributeError, field):
            setattr(config, field, 1)
    with expect_error(TypeError, "incompatible function arguments"):
        ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=2,
            per_core_M=1,
            per_core_N=56,
            fused_activation=None,
            out_subblock_w=8,
        )


@pytest.mark.parametrize(
    ("k", "n", "in0_block_w", "per_core_n"),
    (
        (5120, 16480, 1, 65),
        (5120, 16480, 2, 65),
        (5120, 16480, 4, 65),
        (5120, 16480, 5, 65),
        (5120, 16480, 10, 65),
        (5120, 16480, 20, 65),
        (6144, 5120, 1, 20),
        (6144, 5120, 2, 20),
        (6144, 5120, 3, 20),
        (6144, 5120, 4, 20),
        (6144, 5120, 6, 20),
        (6144, 5120, 8, 20),
        (6144, 5120, 12, 20),
        (6144, 5120, 24, 20),
    ),
)
def test_linear_projection_geometry_lowers_to_exact_dram_program(k, n, in0_block_w, per_core_n):
    config = _decode_program(k=k, n=n, in0_block_w=in0_block_w, cores=8)
    assert config.in0_block_w == in0_block_w
    assert config.per_core_M == 1
    assert config.per_core_N == per_core_n


@pytest.mark.parametrize(("k", "n", "in0_block_w"), ((5120, 16480, 3), (6144, 5120, 5)))
def test_linear_projection_geometry_rejects_nondivisors(k, n, in0_block_w, expect_error):
    with expect_error(ValueError, rf"in0_block_w={in0_block_w} must divide"):
        _decode_program(k=k, n=n, in0_block_w=in0_block_w, cores=8)
