# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
import time

import pytest
import torch
from tracy import signpost
from transformers import DynamicCache
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder import (
    EMITTED_CACHE_LENGTH,
    EMITTED_PREFILL_SEQUENCE,
    LAYER_IDX,
    _assert_pcc,
    _config,
    _hf_forward,
    _hf_layer,
    _position_tensor,
    _real_state_dict,
    _synthetic_state_dict,
    _to_torch,
    _to_tt,
)
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import FunctionalDecoder
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import (
    OptimizationConfig,
    OptimizedDecoder,
    OptimizedGPTOSSProgramConfig,
)


def _decoder(variant, state, config, mesh_device, *, max_cache_len=EMITTED_CACHE_LENGTH):
    decoder_cls = FunctionalDecoder if variant == "functional" else OptimizedDecoder
    kwargs = {}
    if decoder_cls is OptimizedDecoder:
        changes = {"use_decode_concat_heads": variant != "reshape_control"}
        if variant == "advisor":
            changes.update(
                use_shard_advisor_layouts=True,
                use_shard_advisor_moe_layouts=True,
                use_sparse_experts=False,
            )
        elif variant == "advisor_attention_only":
            changes.update(
                use_shard_advisor_layouts=True,
                use_shard_advisor_moe_layouts=False,
                use_sparse_experts=False,
            )
        elif variant == "advisor_attention_sparse":
            changes.update(
                use_shard_advisor_layouts=True,
                use_shard_advisor_moe_layouts=False,
            )
        elif variant == "advisor_moe_only":
            changes.update(
                use_shard_advisor_layouts=True,
                use_shard_advisor_attention_layouts=False,
                use_sparse_experts=False,
            )
        elif variant == "dense_control":
            changes.update(use_sparse_experts=False)
        elif variant == "sparse_bfp4":
            changes.update(expert_weight_dtype="bfloat4_b")
        elif variant == "sparse_bfp4_5x6":
            changes.update(
                expert_weight_dtype="bfloat4_b",
                expert_gate_up_cores=(5, 6),
                expert_down_cores=(5, 6),
            )
        elif variant == "sparse_bfp8":
            changes.update(expert_weight_dtype="bfloat8_b")
        elif variant == "sparse_bf16":
            changes.update(expert_weight_dtype="bfloat16")
        elif variant == "sparse_bfp8_8x8":
            changes.update(
                expert_weight_dtype="bfloat8_b",
                expert_gate_up_cores=(8, 8),
                expert_down_cores=(8, 8),
            )
        elif variant == "sparse_bfp8_5x6":
            changes.update(
                expert_weight_dtype="bfloat8_b",
                expert_gate_up_cores=(5, 6),
                expert_down_cores=(5, 6),
            )
        elif variant == "sparse_bfp8_5x6_subblock3":
            changes.update(
                expert_weight_dtype="bfloat8_b",
                expert_gate_up_cores=(5, 6),
                expert_down_cores=(5, 6),
                expert_gate_up_subblock_w=3,
                expert_down_subblock_w=3,
            )
        elif variant == "sparse_bfp8_9x10":
            changes.update(
                expert_weight_dtype="bfloat8_b",
                expert_gate_up_cores=(9, 10),
                expert_down_cores=(9, 10),
            )
        elif variant == "sparse_bfp8_lofi":
            changes.update(expert_weight_dtype="bfloat8_b", math_fidelity="lofi")
        elif variant == "sparse_bfp8_in0_45":
            changes.update(expert_gate_up_in0_block_w=45, expert_down_in0_block_w=45)
        elif variant == "sparse_bfp8_gate_in0_45":
            changes.update(expert_gate_up_in0_block_w=45)
        elif variant == "sparse_bfp8_down_in0_45":
            changes.update(expert_down_in0_block_w=45)
        elif variant == "sparse_bfp8_in0_90":
            changes.update(expert_gate_up_in0_block_w=90, expert_down_in0_block_w=90)
        elif variant == "sparse_bfp8_l1_input":
            changes.update(expert_input_l1=True)
        elif variant == "sparse_bfp8_no_sdpa_config":
            changes.update(expert_weight_dtype="bfloat8_b", explicit_sdpa_program_config=False)
        elif variant == "sparse_bfp8_explicit_sdpa":
            changes.update(expert_weight_dtype="bfloat8_b", explicit_sdpa_program_config=True)
        elif variant == "optimized_sliding_auto":
            changes.update(explicit_sliding_sdpa_program_config=False)
        elif variant == "optimized_sliding_k64":
            changes.update(sdpa_k_chunk_size=64)
        elif variant == "prefill_2d_8x4":
            changes.update(prefill_matmul_config="2d_8x4")
        elif variant == "prefill_2d_10x4":
            changes.update(prefill_matmul_config="2d_10x4")
        elif variant == "native_sliding_k32":
            changes.update(use_explicit_sliding_mask=False, explicit_sdpa_program_config=True)
        elif variant == "dram_bfp4_lofi":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat4_b",
                math_fidelity="lofi",
            )
        elif variant == "dram_bfp8_hifi2":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat8_b",
                math_fidelity="hifi2",
            )
        elif variant == "dram_bfp8_hifi2_kv_bfp8":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat8_b",
                math_fidelity="hifi2",
                kv_cache_dtype="bfloat8_b",
            )
        elif variant == "dram_bfp8_hifi2_sliding_auto":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat8_b",
                math_fidelity="hifi2",
                explicit_sliding_sdpa_program_config=False,
            )
        elif variant == "dram_bfp8_hifi2_sliding_k64":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat8_b",
                math_fidelity="hifi2",
                sdpa_k_chunk_size=64,
            )
        elif variant == "dram_bf16_hifi4":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat16",
                math_fidelity="hifi4",
            )
        elif variant == "dram_bf16_hifi4_sliding_auto":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat16",
                math_fidelity="hifi4",
                explicit_sliding_sdpa_program_config=False,
            )
        elif variant == "dram_bf16_hifi4_sliding_k64":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat16",
                math_fidelity="hifi4",
                sdpa_k_chunk_size=64,
            )
        elif variant == "dram_bf16_hifi4_query_dram":
            changes.update(
                use_dram_sharded_attention=True,
                dram_attention_weight_dtype="bfloat16",
                math_fidelity="hifi4",
                dram_attention_query_dram=True,
            )
        elif variant == "sparse_bfp8_kv_cache":
            changes.update(kv_cache_dtype="bfloat8_b")
        kwargs["optimization_config"] = OptimizationConfig().with_changes(**changes)
    return decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_cache_len=max_cache_len,
        **kwargs,
    )


def _host_hidden(hidden_states, mesh_device):
    return ttnn.from_torch(
        hidden_states.unsqueeze(0),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )


def _host_position(position, mesh_device):
    return ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )


def test_optimized_runtime_is_independent_and_host_fallback_free():
    policy = OptimizationConfig()
    assert policy.use_decode_concat_heads
    assert policy.use_sparse_experts
    assert policy.use_explicit_sliding_mask
    assert policy.explicit_sliding_sdpa_program_config
    assert policy.expert_weight_dtype == "bfloat8_b"
    assert policy.expert_gate_up_cores == (9, 10)
    assert policy.expert_down_cores == (9, 10)
    assert policy.expert_gate_up_in0_block_w == 45
    assert policy.expert_down_in0_block_w == 45
    assert policy.kv_cache_dtype == "bfloat16"
    assert policy.math_fidelity == "hifi4"
    assert policy.prefill_matmul_config == "auto"
    assert policy.use_shard_advisor_layouts
    assert policy.use_shard_advisor_attention_layouts
    assert not policy.use_shard_advisor_moe_layouts
    assert not policy.use_dram_sharded_attention
    assert not policy.explicit_sdpa_program_config

    runtime_methods = (
        OptimizedDecoder._prefill_attention,
        OptimizedDecoder._decode_attention,
        OptimizedDecoder._decode_sliding_attention_mask,
        OptimizedDecoder._route,
        OptimizedDecoder._sparse_moe_forward,
        OptimizedDecoder._moe_forward,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.forward,
    )
    forbidden = ("super().", "from_torch", "to_torch", "torch.", "untilize", "reshard")
    for method in runtime_methods:
        assert method.__module__.endswith("optimized_decoder")
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__
    assert "nlp_concat_heads_decode" in inspect.getsource(OptimizedDecoder._decode_attention)


def test_sparse_expert_subblock_geometry_is_validated(expect_error):
    policy = OptimizedGPTOSSProgramConfig(
        decode_gate_up_cores=(5, 6),
        decode_down_cores=(5, 6),
        decode_gate_up_subblock_w=3,
        decode_down_subblock_w=3,
    )
    gate = policy.get_decode_gate_up_config(1, 2880, k=2880)
    down = policy.get_decode_down_config(1, 2880, k=2880)
    for config in (gate, down):
        assert config.per_core_N == 3
        assert config.out_block_w == 3
        assert config.out_subblock_w == 3
    with expect_error(ValueError, "must divide per_core_N=1"):
        policy._build_matmul_config((9, 10), 1, 2880, out_subblock_w=3, k=2880)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_decode_concat_heads_graph_rewrite_matches_functional(mesh_device):
    config = _config()
    state = _real_state_dict()
    functional = _decoder("functional", state, config, mesh_device)
    # Hold MoE topology/dtype constant so this isolates reshape versus the
    # dedicated concat-heads rewrite rather than conflating BFP8 sparse MoE.
    optimized = _decoder("dense_control", state, config, mesh_device)
    generator = torch.Generator().manual_seed(20260716)
    prefill_hidden = torch.randn(1, EMITTED_PREFILL_SEQUENCE, config.hidden_size, generator=generator).to(
        torch.bfloat16
    )
    decode_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    functional_key, functional_value = functional.create_kv_cache()
    optimized_key, optimized_value = optimized.create_kv_cache()
    functional.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=functional_key,
        value_cache=functional_value,
    )
    optimized.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=optimized_key,
        value_cache=optimized_value,
    )
    functional_out = functional.decode_forward(
        _to_tt(decode_hidden, mesh_device),
        key_cache=functional_key,
        value_cache=functional_value,
        cache_position=EMITTED_PREFILL_SEQUENCE,
        cache_position_tensor=_position_tensor(EMITTED_PREFILL_SEQUENCE, mesh_device),
    )
    optimized_out = optimized.decode_forward(
        _to_tt(decode_hidden, mesh_device),
        key_cache=optimized_key,
        value_cache=optimized_value,
        cache_position=EMITTED_PREFILL_SEQUENCE,
        cache_position_tensor=_position_tensor(EMITTED_PREFILL_SEQUENCE, mesh_device),
    )
    _assert_pcc("graph-rewrite-decode-concat", _to_torch(functional_out), _to_torch(optimized_out), 0.999)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_synthetic_prefill_preserves_non_aligned_lengths(mesh_device):
    config = _config()
    state = _synthetic_state_dict(config)
    decoder = _decoder("optimized", state, config, mesh_device)
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(7070)

    for seq_len in (17, 33, EMITTED_CACHE_LENGTH):
        hidden = torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16)
        reference = _hf_forward(
            hf_layer,
            rotary,
            hidden,
            torch.arange(seq_len),
            DynamicCache(config=config),
        )
        key_cache, value_cache = decoder.create_kv_cache()
        actual = decoder.prefill_forward(
            _to_tt(hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
        )
        _assert_pcc(f"optimized-synthetic-prefill-{seq_len}", reference, _to_torch(actual), 0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_decoder_perf(mesh_device):
    if os.environ.get("RUN_OPTIMIZED_DECODER_PERF") != "1":
        pytest.skip("Set RUN_OPTIMIZED_DECODER_PERF=1 for warmed/traced performance evidence")
    variant = os.environ.get("OPTIMIZED_DECODER_PERF_VARIANT", "functional")
    replay_count = int(os.environ.get("OPTIMIZED_DECODER_TRACE_REPLAYS", "20"))
    prefill_repeats = int(os.environ.get("OPTIMIZED_DECODER_PREFILL_REPEATS", "3"))
    seq_len = int(os.environ.get("OPTIMIZED_DECODER_PERF_SEQ_LEN", str(EMITTED_PREFILL_SEQUENCE)))
    config = _config()
    state = _real_state_dict()
    decoder = _decoder(variant, state, config, mesh_device, max_cache_len=max(EMITTED_CACHE_LENGTH, seq_len + 32))
    key_cache, value_cache = decoder.create_kv_cache()
    generator = torch.Generator().manual_seed(8080)
    prefill_hidden = _to_tt(
        torch.randn(1, seq_len, config.hidden_size, generator=generator).to(torch.bfloat16),
        mesh_device,
    )
    decode_hidden = _to_tt(
        torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16),
        mesh_device,
    )
    position = _position_tensor(seq_len, mesh_device)

    decoder.prefill_forward(prefill_hidden, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    signpost(header="PERF_PREFILL")
    start = time.perf_counter()
    for _ in range(prefill_repeats):
        decoder.prefill_forward(prefill_hidden, key_cache=key_cache, value_cache=value_cache)
    ttnn.synchronize_device(mesh_device)
    prefill_ms = (time.perf_counter() - start) * 1000.0 / prefill_repeats
    signpost(header="PERF_PREFILL_END")

    decoder.decode_forward(
        decode_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=seq_len,
        cache_position_tensor=position,
    )
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        decode_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=seq_len,
        cache_position_tensor=position,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        signpost(header="PERF_DECODE")
        start = time.perf_counter()
        for _ in range(replay_count):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        decode_ms = (time.perf_counter() - start) * 1000.0 / replay_count
        signpost(header="PERF_DECODE_END")
        assert tuple(traced_output.shape) == (1, 1, 1, config.hidden_size)
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    print(
        "PERF_RESULT "
        f"variant={variant} seq={seq_len} "
        f"prefill_ms={prefill_ms:.6f} traced_decode_ms={decode_ms:.6f} "
        f"prefill_repeats={prefill_repeats} trace_replays={replay_count}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_real_weight_optimized_prefill_decode_pcc(mesh_device, layer_type):
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    variant = os.environ.get("OPTIMIZED_DECODER_CORRECTNESS_VARIANT", "optimized")
    decoder = _decoder(variant, state, config, mesh_device)
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(9090)
    prefill_hidden = torch.randn(1, EMITTED_PREFILL_SEQUENCE, config.hidden_size, generator=generator).to(
        torch.bfloat16
    )
    hf_cache = DynamicCache(config=config)
    prefill_reference = _hf_forward(
        hf_layer,
        rotary,
        prefill_hidden,
        torch.arange(EMITTED_PREFILL_SEQUENCE),
        hf_cache,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_actual = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc("optimized-real-prefill-17", prefill_reference, _to_torch(prefill_actual), 0.99)
    last_hidden = None
    last_actual = None
    for cache_position in range(EMITTED_PREFILL_SEQUENCE, EMITTED_PREFILL_SEQUENCE + 3):
        last_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
        decode_reference = _hf_forward(
            hf_layer,
            rotary,
            last_hidden,
            torch.tensor([cache_position]),
            hf_cache,
        )
        last_actual = decoder.decode_forward(
            _to_tt(last_hidden, mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            cache_position_tensor=_position_tensor(cache_position, mesh_device),
        )
        _assert_pcc(
            f"optimized-real-{layer_type}-decode-{cache_position}",
            decode_reference,
            _to_torch(last_actual),
            0.99,
        )

    # Rewriting the same paged-cache position with the same token must be
    # deterministic and leave every model-visible output bit unchanged.
    repeated = decoder.decode_forward(
        _to_tt(last_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=EMITTED_PREFILL_SEQUENCE + 2,
        cache_position_tensor=_position_tensor(EMITTED_PREFILL_SEQUENCE + 2, mesh_device),
    )
    assert torch.equal(_to_torch(last_actual), _to_torch(repeated))

    if variant != "functional":
        trace_hidden = _to_tt(last_hidden, mesh_device)
        trace_position = _position_tensor(EMITTED_PREFILL_SEQUENCE + 2, mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        traced_output = decoder.decode_forward(
            trace_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=EMITTED_PREFILL_SEQUENCE + 2,
            cache_position_tensor=trace_position,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        try:
            for _ in range(20):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            assert torch.equal(_to_torch(repeated), _to_torch(traced_output))
        finally:
            ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_traced_decode_updates_position_across_sliding_window_boundary(mesh_device, layer_type):
    config = _config()
    config.layer_types = list(config.layer_types)
    config.layer_types[LAYER_IDX] = layer_type
    state = _real_state_dict()
    variant = os.environ.get("OPTIMIZED_DECODER_BOUNDARY_VARIANT", "optimized")
    decoder = _decoder(variant, state, config, mesh_device, max_cache_len=256)
    hf_layer = _hf_layer(config, state)
    rotary = GptOssRotaryEmbedding(config)
    generator = torch.Generator().manual_seed(10010)
    prefill_len = config.sliding_window
    prefill_hidden = torch.randn(1, prefill_len, config.hidden_size, generator=generator).to(torch.bfloat16)
    hf_cache = DynamicCache(config=config)
    prefill_reference = _hf_forward(
        hf_layer,
        rotary,
        prefill_hidden,
        torch.arange(prefill_len),
        hf_cache,
    )
    key_cache, value_cache = decoder.create_kv_cache()
    prefill_actual = decoder.prefill_forward(
        _to_tt(prefill_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _assert_pcc(f"optimized-boundary-{layer_type}-prefill", prefill_reference, _to_torch(prefill_actual), 0.99)

    first_hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    trace_hidden = _to_tt(first_hidden, mesh_device)
    trace_position = _position_tensor(prefill_len, mesh_device)
    if layer_type == "sliding_attention":
        mask = decoder._decode_sliding_attention_mask(trace_position)
        mask_rows = ttnn.to_torch(ttnn.get_device_tensors(mask)[0])[0, 0]
        mask_row = mask_rows[0]
        assert (mask_row[:1] <= -1_000).all()
        assert torch.equal(mask_row[1 : prefill_len + 1], torch.zeros(prefill_len, dtype=mask_row.dtype))
        assert (mask_row[prefill_len + 1 :] <= -1_000).all()
    reference_history = torch.cat([prefill_hidden, first_hidden], dim=1)
    first_reference = _hf_forward(
        hf_layer,
        rotary,
        reference_history,
        torch.arange(reference_history.shape[1]),
        DynamicCache(),
    )[:, -1:]
    # Compile and capture with identical inputs. The repeated write to the same
    # KV-cache row is idempotent and keeps stable trace input addresses.
    compile_output = decoder.decode_forward(
        trace_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.synchronize_device(mesh_device)
    _assert_pcc(
        f"optimized-boundary-{layer_type}-compile-decode-{prefill_len}",
        first_reference,
        _to_torch(compile_output),
        0.99,
    )
    if os.environ.get("OPTIMIZED_DECODER_BOUNDARY_EAGER") == "1":
        for position in range(prefill_len + 1, prefill_len + 3):
            hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
            reference_history = torch.cat([reference_history, hidden], dim=1)
            reference = _hf_forward(
                hf_layer,
                rotary,
                reference_history,
                torch.arange(reference_history.shape[1]),
                DynamicCache(),
            )[:, -1:]
            if layer_type == "sliding_attention" and os.environ.get("OPTIMIZED_DECODER_CHECK_MASK_EACH") == "1":
                position_tensor = _position_tensor(position, mesh_device)
                mask = decoder._decode_sliding_attention_mask(position_tensor)
                mask_row = ttnn.to_torch(ttnn.get_device_tensors(mask)[0])[0, 0, 0]
                window_start = position - config.sliding_window + 1
                assert (mask_row[:window_start] <= -1_000).all()
                assert torch.equal(
                    mask_row[window_start : position + 1],
                    torch.zeros(config.sliding_window, dtype=mask_row.dtype),
                )
                assert (mask_row[position + 1 :] <= -1_000).all()
            actual = decoder.decode_forward(
                _to_tt(hidden, mesh_device),
                key_cache=key_cache,
                value_cache=value_cache,
                cache_position=position,
                cache_position_tensor=_position_tensor(position, mesh_device),
            )
            _assert_pcc(
                f"optimized-boundary-eager-{layer_type}-decode-{position}",
                reference,
                _to_torch(actual),
                0.99,
            )
        return
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    traced_output = decoder.decode_forward(
        trace_hidden,
        key_cache=key_cache,
        value_cache=value_cache,
        cache_position=prefill_len,
        cache_position_tensor=trace_position,
    )
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-boundary-{layer_type}-decode-{prefill_len}",
            first_reference,
            _to_torch(traced_output),
            0.99,
        )
        for position in range(prefill_len + 1, prefill_len + 3):
            hidden = torch.randn(1, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
            reference_history = torch.cat([reference_history, hidden], dim=1)
            reference = _hf_forward(
                hf_layer,
                rotary,
                reference_history,
                torch.arange(reference_history.shape[1]),
                DynamicCache(),
            )[:, -1:]
            host_hidden = _host_hidden(hidden, mesh_device)
            host_position = _host_position(position, mesh_device)
            ttnn.copy_host_to_device_tensor(host_hidden, trace_hidden)
            ttnn.copy_host_to_device_tensor(host_position, trace_position)
            ttnn.synchronize_device(mesh_device)
            assert torch.equal(_to_torch(trace_hidden), hidden)
            assert int(ttnn.to_torch(ttnn.get_device_tensors(trace_position)[0]).item()) == position
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            _assert_pcc(
                f"optimized-boundary-{layer_type}-decode-{position}",
                reference,
                _to_torch(traced_output),
                0.99,
            )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
