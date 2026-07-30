# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import inspect

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import POLICIES, OptimizedDecoder


def test_optimized_decoder_is_distinct_and_does_not_select_a_functional_fallback():
    assert OptimizedDecoder is not FusedDecoder
    assert OptimizedDecoder.__mro__[:3] == (OptimizedDecoder, FusedDecoder, FusedDecoder.__mro__[1])
    assert "_mlp" in OptimizedDecoder.__dict__
    assert "_linear" in OptimizedDecoder.__dict__


def test_default_policy_is_exercised_by_all_material_projection_methods():
    setup = inspect.getsource(OptimizedDecoder.from_state_dict)
    assert "bfp4_all_dram_w8" in setup
    assert "_linear" in inspect.getsource(OptimizedDecoder._mlp)
    assert "prefill_weights" in inspect.getsource(OptimizedDecoder._mlp_prefill)
    assert "compute_kernel_config" in inspect.getsource(OptimizedDecoder._linear)
    assert "compute_kernel_config" in inspect.getsource(OptimizedDecoder._optimized_prefill_linear)
    assert "compute_kernel_config" in inspect.getsource(OptimizedDecoder._optimized_decode_linear)
    for method, helper in (
        (OptimizedDecoder._full_attention_prefill, "_optimized_prefill_linear"),
        (OptimizedDecoder._full_attention_decode, "_optimized_decode_linear"),
        (OptimizedDecoder._linear_attention_prefill_chunk, "_optimized_prefill_linear"),
        (OptimizedDecoder._linear_attention_decode, "_optimized_decode_linear"),
    ):
        assert helper in inspect.getsource(method)


def test_fused_op_overrides_do_not_mutate_process_global_ttnn():
    runtime = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder._call_fused_with_scoped_ttnn,
            OptimizedDecoder._full_attention_prefill,
            OptimizedDecoder._full_attention_decode,
            OptimizedDecoder._linear_attention_prefill_chunk,
            OptimizedDecoder._linear_attention_decode,
        )
    )
    assert "ttnn.linear =" not in runtime
    assert "ttnn.concat =" not in runtime

    original_linear = ttnn.linear

    def probe(decoder):
        return ttnn.linear

    sentinel = object()
    assert (
        OptimizedDecoder._call_fused_with_scoped_ttnn(
            object(),
            probe,
            linear=sentinel,
        )
        is sentinel
    )
    assert ttnn.linear is original_linear


def test_required_precision_candidates_are_named_and_group_specific():
    assert {"bfp8_hifi2", "bfp8_lofi", "bfp4_mlp_lofi", "bfp4_all_lofi"} <= POLICIES.keys()
    assert POLICIES["bfp4_all_lofi"].attention_weight_dtype == ttnn.bfloat4_b
    assert POLICIES["bfp4_mlp_lofi"].attention_weight_dtype == ttnn.bfloat8_b
    assert POLICIES["bfp4_mlp_lofi"].mlp_gate_up_weight_dtype == ttnn.bfloat4_b
    assert POLICIES["bfp4_all_dram_w10"].max_in0_block_w == 10
    assert POLICIES["bfp4_all_dram_w8"].large_prefill_config
    assert not POLICIES["bfp4_all_dram_w8_default_prefill"].large_prefill_config


def test_packed_sources_are_released_and_selected_path_has_no_fallback_branch():
    setup = inspect.getsource(OptimizedDecoder.from_state_dict)
    assert "decoder.weights.pop(name, None)" in setup
    assert "FunctionalDecoder" not in setup
    assert OptimizedDecoder._largest_divisor_at_most(20, 8) == 5
    assert OptimizedDecoder._largest_divisor_at_most(68, 8) == 4
    prefill_config = inspect.getsource(OptimizedDecoder._prefill_program_config)
    assert "m_tiles < 10" in prefill_config
    assert "MatmulMultiCoreReuseMultiCastProgramConfig" in prefill_config


def test_optimized_runtime_has_no_host_tensor_transfer():
    runtime = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder._mlp,
            OptimizedDecoder._mlp_prefill,
            OptimizedDecoder._full_attention_prefill,
            OptimizedDecoder._full_attention_decode,
            OptimizedDecoder._linear_attention_prefill_chunk,
            OptimizedDecoder._linear_attention_decode,
            OptimizedDecoder.prefill_forward,
            OptimizedDecoder.decode_forward,
        )
    )
    for forbidden in ("torch.", "ttnn.from_torch", "ttnn.to_torch"):
        assert forbidden not in runtime
