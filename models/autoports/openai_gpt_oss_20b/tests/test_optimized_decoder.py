# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import os
import statistics
import time

import pytest
import torch
from tracy import signpost

import models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.openai_gpt_oss_20b.tests.test_fused_decoder import _functional_helpers_for_layer
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import EMITTED_PREFILL_SEQUENCE, FunctionalDecoder
from models.autoports.openai_gpt_oss_20b.tt.fused_decoder import FusedDecoder
from models.autoports.openai_gpt_oss_20b.tt.optimized_decoder import OptimizationConfig, OptimizedDecoder

SLIDING_LAYER = 12
FULL_LAYER = 13


def _optimization_config(variant: str | None = None) -> OptimizationConfig:
    variant = variant or os.environ.get("OPTIMIZED_DECODER_VARIANT", "optimized")
    base = OptimizationConfig()
    variants = {
        "optimized": base,
        "no_advisor": base.with_changes(
            use_shard_advisor_attention_layouts=False,
            use_shard_advisor_router_layouts=False,
        ),
        "advisor_attention": base.with_changes(
            use_shard_advisor_attention_layouts=True,
            use_shard_advisor_router_layouts=False,
        ),
        "advisor_attention_router": base.with_changes(
            use_shard_advisor_attention_layouts=True,
            use_shard_advisor_router_layouts=True,
        ),
        "attention_lofi": base.with_changes(attention_math_fidelity="lofi"),
        "attention_hifi2": base.with_changes(attention_math_fidelity="hifi2"),
        "dram_attention_bfp4": base.with_changes(
            use_dram_sharded_attention=True,
            dram_attention_weight_dtype="bfloat4_b",
            attention_math_fidelity="lofi",
        ),
        "dram_attention_bfp4_64": base.with_changes(
            use_dram_sharded_attention=True,
            dram_attention_weight_dtype="bfloat4_b",
            dram_attention_core_limit=64,
            attention_math_fidelity="lofi",
        ),
        "dram_attention_bfp8": base.with_changes(
            use_dram_sharded_attention=True,
            dram_attention_weight_dtype="bfloat8_b",
            attention_math_fidelity="hifi2",
        ),
        "dram_attention_bf16": base.with_changes(
            use_dram_sharded_attention=True,
            dram_attention_weight_dtype="bfloat16",
            attention_math_fidelity="hifi4",
        ),
        "dram_attention_bf16_lofi": base.with_changes(
            use_dram_sharded_attention=True,
            dram_attention_weight_dtype="bfloat16",
            attention_math_fidelity="lofi",
        ),
        "kv_bfp8": base.with_changes(kv_cache_dtype="bfloat8_b"),
        "kv_bf16": base.with_changes(kv_cache_dtype="bfloat16"),
        "kv_bfp8_sdpa_k32": base.with_changes(
            kv_cache_dtype="bfloat8_b",
            explicit_sdpa_program_config=True,
        ),
        "sdpa_8x8_k32": base.with_changes(explicit_sdpa_program_config=True),
        "sdpa_default": base.with_changes(explicit_sdpa_program_config=False),
        "sdpa_8x8_k64": base.with_changes(
            explicit_sdpa_program_config=True,
            sdpa_k_chunk_size=64,
        ),
        "prefill_auto": base.with_changes(prefill_matmul_config="auto"),
        "prefill_2d_8x4": base.with_changes(prefill_matmul_config="2d_8x4"),
        "prefill_2d_10x4": base.with_changes(prefill_matmul_config="2d_10x4"),
        "prefill_sdpa_32": base.with_changes(
            prefill_sdpa_chunk_size=32,
            use_manual_prefill_attention=False,
            use_dense_long_prefill=False,
        ),
        "prefill_sdpa_64": base.with_changes(
            prefill_sdpa_chunk_size=64,
            use_manual_prefill_attention=False,
            use_dense_long_prefill=False,
        ),
        "manual_prefill_attention": base.with_changes(use_manual_prefill_attention=True),
        "manual_prefill_attention_bf16_experts": base.with_changes(
            use_manual_prefill_attention=True,
            expert_weight_dtype="bfloat16",
        ),
        "manual_dense_long_prefill": base.with_changes(
            use_manual_prefill_attention=True,
            use_dense_long_prefill=True,
        ),
        "dense": base.with_changes(use_sparse_experts=False),
        "dense_prefill_auto": base.with_changes(use_sparse_experts=False, prefill_matmul_config="auto"),
        "advisor_dense_moe": base.with_changes(
            use_sparse_experts=False,
            use_shard_advisor_dense_moe_layouts=True,
        ),
        "sparse_bf16": base.with_changes(expert_weight_dtype="bfloat16"),
        "sparse_hifi2": base.with_changes(expert_math_fidelity="hifi2"),
        "sparse_prefill_shared": base.with_changes(
            use_precise_sparse_prefill=False,
            use_manual_prefill_attention=False,
            use_dense_long_prefill=False,
        ),
        "sparse_prefill_bfp8_lofi": base.with_changes(
            prefill_expert_output_dtype="bfloat8_b",
            prefill_expert_math_fidelity="lofi",
            use_manual_prefill_attention=False,
            use_dense_long_prefill=False,
        ),
        "sparse_packed_gate_up": base.with_changes(use_packed_sparse_gate_up=True),
        "expert_input_l1": base.with_changes(expert_input_l1=True),
        "expert_input_dram": base.with_changes(expert_input_l1=False),
        "sparse_bfp4": base.with_changes(expert_weight_dtype="bfloat4_b"),
        "sparse_bfp4_5x6": base.with_changes(
            expert_weight_dtype="bfloat4_b",
            expert_gate_up_cores=(5, 6),
            expert_down_cores=(5, 6),
        ),
        "sparse_5x6": base.with_changes(expert_gate_up_cores=(5, 6), expert_down_cores=(5, 6)),
        "sparse_8x8": base.with_changes(expert_gate_up_cores=(8, 8), expert_down_cores=(8, 8)),
        "sparse_9x5": base.with_changes(expert_gate_up_cores=(9, 5), expert_down_cores=(9, 5)),
        "sparse_3x4_5x6": base.with_changes(expert_gate_up_cores=(3, 4), expert_down_cores=(5, 6)),
        "sparse_block30": base.with_changes(expert_gate_up_in0_block_w=30, expert_down_in0_block_w=30),
        "sparse_gate_block30": base.with_changes(expert_gate_up_in0_block_w=30),
        "sparse_down_block30": base.with_changes(expert_down_in0_block_w=30),
        "sparse_block90": base.with_changes(expert_gate_up_in0_block_w=90, expert_down_in0_block_w=90),
        "sparse_subblock3_5x6": base.with_changes(
            expert_gate_up_cores=(5, 6),
            expert_down_cores=(5, 6),
            expert_gate_up_subblock_w=3,
            expert_down_subblock_w=3,
        ),
    }
    try:
        return variants[variant]
    except KeyError as error:
        raise ValueError(f"unknown OPTIMIZED_DECODER_VARIANT={variant!r}; expected {tuple(variants)}") from error


def _decoder(
    state,
    config,
    mesh_device,
    *,
    layer_idx: int = SLIDING_LAYER,
    batch: int = 1,
    max_cache_len: int = 128,
    variant: str | None = None,
):
    return OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=max_cache_len,
        optimization_config=_optimization_config(variant),
    )


def _empty_caches_for_decoder(config, mesh_device, decoder, *, batch=1):
    if isinstance(decoder, OptimizedDecoder):
        return decoder.create_kv_cache()
    return functional_test._empty_caches(config, mesh_device, batch=batch)


def test_optimized_runtime_is_distinct_and_has_no_host_fallback():
    policy = OptimizationConfig()
    assert policy.use_sparse_experts
    assert policy.expert_math_fidelity == "lofi"
    assert policy.expert_input_l1
    assert policy.kv_cache_dtype == "bfloat8_b"
    assert policy.prefill_matmul_config == "auto"
    assert policy.explicit_sdpa_program_config
    assert OptimizedDecoder.prefill_forward is not FusedDecoder.prefill_forward
    assert OptimizedDecoder.decode_forward is not FusedDecoder.decode_forward
    assert OptimizedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert OptimizedDecoder.decode_forward is not FunctionalDecoder.decode_forward

    runtime_methods = (
        OptimizedDecoder._route,
        OptimizedDecoder._apply_fused_swiglu,
        OptimizedDecoder._sparse_decode_experts,
        OptimizedDecoder._sparse_prefill_chunk,
        OptimizedDecoder._sparse_prefill_experts,
        OptimizedDecoder._dense_reference_expert_chunk,
        OptimizedDecoder._dense_reference_experts,
        OptimizedDecoder._optimized_moe_forward,
        OptimizedDecoder._bounded_prefill_linear,
        OptimizedDecoder._manual_prefill_attention,
        OptimizedDecoder._manual_full_decode_attention,
        OptimizedDecoder._prefill_attention,
        OptimizedDecoder._decode_attention,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
    )
    forbidden = (
        "torch",
        "from_torch",
        "to_torch",
        "all_reduce",
        "all_gather",
        "reduce_scatter",
        "mesh_partition",
        "tilize",
        "untilize",
        "reshard",
        "super().prefill_forward",
        "super().decode_forward",
    )
    for method in runtime_methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"

    assert "self.experts(" in inspect.getsource(OptimizedDecoder._optimized_moe_forward)
    assert "scaled_dot_product_attention(" in inspect.getsource(OptimizedDecoder._prefill_attention)
    assert "scaled_dot_product_attention_decode(" in inspect.getsource(OptimizedDecoder._decode_attention)
    assert "paged_update_cache(" in inspect.getsource(OptimizedDecoder._decode_attention)
    assert "nlp_concat_heads_decode(" in inspect.getsource(OptimizedDecoder._decode_attention)


def _boundary_seq_lengths() -> tuple[int, ...]:
    requested = os.environ.get("OPTIMIZED_DECODER_BOUNDARY_SEQ_LEN")
    return (int(requested),) if requested is not None else (128, 129)


@pytest.mark.parametrize("layer_idx,expected_window", [(SLIDING_LAYER, 128), (FULL_LAYER, None)])
@pytest.mark.parametrize("prefill_len", _boundary_seq_lengths())
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_layer_kind_boundary_beyond_emitted_cache(mesh_device, layer_idx, expected_window, prefill_len):
    """Cross the sliding-window boundary and the functional stage's cache extent."""

    cache_len = int(os.environ.get("OPTIMIZED_DECODER_BOUNDARY_CACHE_LEN", str(max(256, prefill_len + 1))))
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(state, config, mesh_device, layer_idx=layer_idx, max_cache_len=cache_len)
        reference_layer = functional_test._hf_layer(state, config)
        key_cache, value_cache = decoder.create_kv_cache()
        boundary_seed = int(os.environ.get("OPTIMIZED_DECODER_BOUNDARY_SEED", str(16100 + layer_idx)))
        generator = torch.Generator().manual_seed(boundary_seed)
        hidden = torch.randn((1, 1, prefill_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference_prefill, reference_key, reference_value, reference_cache = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
        )
        actual_prefill = decoder.prefill_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
        )
        functional_test._assert_pcc(
            reference_prefill,
            functional_test._to_host(actual_prefill),
            0.99,
            f"layer{layer_idx} prefill S={prefill_len}",
        )
        reference_cache_len = reference_key.shape[-2]
        cache_start = prefill_len - reference_cache_len
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, cache_start:prefill_len, :],
            0.99,
            f"layer{layer_idx} prefill boundary key",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, cache_start:prefill_len, :],
            0.99,
            f"layer{layer_idx} prefill boundary value",
        )

        assert decoder.max_cache_len == cache_len
        assert decoder.attention_window == expected_window

        decode_seed = int(os.environ.get("OPTIMIZED_DECODER_BOUNDARY_DECODE_SEED", "26014"))
        decode_generator = torch.Generator().manual_seed(decode_seed)
        decode_hidden = torch.randn((1, 1, 1, config.hidden_size), generator=decode_generator, dtype=torch.bfloat16)
        # Full attention uses the incremental HF cache contract. Recompute the
        # sliding reference so its 128-entry window is explicit at the boundary.
        reference_history = torch.cat((hidden, decode_hidden), dim=2)
        if expected_window is None:
            reference_decode, decode_key, decode_value, _ = functional_test._reference_layer(
                reference_layer,
                decode_hidden,
                config,
                start_pos=prefill_len,
                cache=reference_cache,
            )
        else:
            reference_decode, decode_key, decode_value, _ = functional_test._reference_layer(
                reference_layer,
                reference_history,
                config,
            )
        actual_decode = decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=prefill_len,
        )
        functional_test._assert_pcc(
            decode_key[:, :, -1:, :],
            functional_test._to_host(key_cache)[:, :, prefill_len : prefill_len + 1, :],
            0.99,
            f"layer{layer_idx} boundary key",
        )
        functional_test._assert_pcc(
            decode_value[:, :, -1:, :],
            functional_test._to_host(value_cache)[:, :, prefill_len : prefill_len + 1, :],
            0.99,
            f"layer{layer_idx} boundary value",
        )
        functional_test._assert_pcc(
            reference_decode[:, :, -1:, :],
            functional_test._to_host(actual_decode),
            0.99,
            f"layer{layer_idx} decode position={prefill_len}",
        )
        del decoder, reference_layer, state
        gc.collect()


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_full_layer_capacity_probe(mesh_device):
    """Opt-in real-weight forward probe without an O(sequence^2) CPU oracle."""

    raw_seq_len = os.environ.get("OPTIMIZED_DECODER_CAPACITY_SEQ_LEN")
    if raw_seq_len is None:
        pytest.skip("set OPTIMIZED_DECODER_CAPACITY_SEQ_LEN to run the hardware capacity probe")
    seq_len = int(raw_seq_len)
    if seq_len < 1:
        raise ValueError(f"OPTIMIZED_DECODER_CAPACITY_SEQ_LEN must be positive, got {seq_len}")

    with _functional_helpers_for_layer(FULL_LAYER):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(state, config, mesh_device, layer_idx=FULL_LAYER, max_cache_len=seq_len)
        key_cache, value_cache = decoder.create_kv_cache()
        generator = torch.Generator().manual_seed(28113)
        hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        output = decoder.prefill_forward(
            functional_test._tt_tensor(hidden, mesh_device),
            key_cache,
            value_cache,
        )
        ttnn.synchronize_device(mesh_device)
        assert tuple(output.shape) == (1, 1, seq_len, config.hidden_size)
        edge = ttnn.concat(
            [
                ttnn.slice(output, [0, 0, 0, 0], [1, 1, 1, config.hidden_size], [1, 1, 1, 1]),
                ttnn.slice(
                    output,
                    [0, 0, seq_len - 1, 0],
                    [1, 1, seq_len, config.hidden_size],
                    [1, 1, 1, 1],
                ),
            ],
            dim=2,
        )
        assert torch.isfinite(functional_test._to_host(edge)).all()
        print(f"real-weight full-layer capacity S={seq_len}: PASS")
        del decoder, state
        gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_large_prefill_uses_correct_auto_program(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
    generator = torch.Generator().manual_seed(12810)
    hidden = torch.randn((1, 1, 128, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference, reference_key, reference_value, _ = functional_test._reference_layer(reference_layer, hidden, config)
    actual = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache)
    functional_test._assert_pcc(reference, functional_test._to_host(actual), 0.99, "optimized prefill S=128")
    reference_cache_len = reference_key.shape[-2]
    cache_start = 128 - reference_cache_len
    functional_test._assert_pcc(
        reference_key,
        functional_test._to_host(key_cache)[:, :, cache_start:128, :],
        0.99,
        "optimized key S=128",
    )
    functional_test._assert_pcc(
        reference_value,
        functional_test._to_host(value_cache)[:, :, cache_start:128, :],
        0.99,
        "optimized value S=128",
    )
    assert decoder.prefill_qkv_program_config is None
    assert decoder.prefill_o_program_config is None
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_non_aligned_prefill_and_determinism(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    repeat_outputs = []
    repeat_caches = []

    for seq_len in (3, EMITTED_PREFILL_SEQUENCE, 33):
        generator = torch.Generator().manual_seed(1200 + seq_len)
        hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference, reference_key, reference_value, _ = functional_test._reference_layer(reference_layer, hidden, config)
        key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
        actual = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache)
        functional_test._assert_pcc(reference, functional_test._to_host(actual), 0.99, f"optimized prefill S={seq_len}")
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, :seq_len, :],
            0.99,
            f"optimized key S={seq_len}",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, :seq_len, :],
            0.99,
            f"optimized value S={seq_len}",
        )
        if seq_len == EMITTED_PREFILL_SEQUENCE:
            repeat_outputs.append(functional_test._to_host(actual))
            repeat_caches.append((functional_test._to_host(key_cache), functional_test._to_host(value_cache)))
            second_key, second_value = _empty_caches_for_decoder(config, mesh_device, decoder)
            second = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), second_key, second_value)
            repeat_outputs.append(functional_test._to_host(second))
            repeat_caches.append((functional_test._to_host(second_key), functional_test._to_host(second_value)))

    assert torch.equal(repeat_outputs[0], repeat_outputs[1])
    assert torch.equal(repeat_caches[0][0], repeat_caches[1][0])
    assert torch.equal(repeat_caches[0][1], repeat_caches[1][1])
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch_two_prefill_and_paged_decode(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(state, config, mesh_device, batch=2)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder, batch=2)
    generator = torch.Generator().manual_seed(2203)

    prefill_hidden = torch.randn((1, 2, 3, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_prefill, reference_key, reference_value, reference_cache = functional_test._reference_layer(
        reference_layer, prefill_hidden, config
    )
    actual_prefill = decoder.prefill_forward(
        functional_test._tt_tensor(prefill_hidden, mesh_device), key_cache, value_cache
    )
    functional_test._assert_pcc(reference_prefill, functional_test._to_host(actual_prefill), 0.99, "batch2 prefill")
    functional_test._assert_pcc(reference_key, functional_test._to_host(key_cache)[:, :, :3, :], 0.99, "batch2 key")
    functional_test._assert_pcc(
        reference_value, functional_test._to_host(value_cache)[:, :, :3, :], 0.99, "batch2 value"
    )

    decode_hidden = torch.randn((1, 2, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference_decode, decode_key, decode_value, _ = functional_test._reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=3,
        cache=reference_cache,
    )
    actual_decode = decoder.decode_forward(
        functional_test._tt_tensor(decode_hidden, mesh_device), key_cache, value_cache, current_pos=3
    )
    functional_test._assert_pcc(reference_decode, functional_test._to_host(actual_decode), 0.99, "batch2 decode")
    functional_test._assert_pcc(decode_key, functional_test._to_host(key_cache)[:, :, 3:4, :], 0.99, "batch2 key")
    functional_test._assert_pcc(decode_value, functional_test._to_host(value_cache)[:, :, 3:4, :], 0.99, "batch2 value")
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("layer_idx,seed", [(SLIDING_LAYER, 9090), (FULL_LAYER, 9249)])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_layer_kind_prefill_and_decode(mesh_device, layer_idx, seed):
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(state, config, mesh_device, layer_idx=layer_idx)
        reference_layer = functional_test._hf_layer(state, config)
        key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
        generator = torch.Generator().manual_seed(seed)

        hidden = torch.randn((1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size), generator=generator).to(
            torch.bfloat16
        )
        reference, reference_key, reference_value, reference_cache = functional_test._reference_layer(
            reference_layer, hidden, config
        )
        actual = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache)
        functional_test._assert_pcc(reference, functional_test._to_host(actual), 0.99, f"layer{layer_idx} prefill")
        functional_test._assert_pcc(
            reference_key,
            functional_test._to_host(key_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
            0.99,
            f"layer{layer_idx} key",
        )
        functional_test._assert_pcc(
            reference_value,
            functional_test._to_host(value_cache)[:, :, :EMITTED_PREFILL_SEQUENCE, :],
            0.99,
            f"layer{layer_idx} value",
        )

        decode_hidden = torch.randn((1, 1, 1, config.hidden_size), generator=generator).to(torch.bfloat16)
        reference_decode, decode_key, decode_value, _ = functional_test._reference_layer(
            reference_layer,
            decode_hidden,
            config,
            start_pos=EMITTED_PREFILL_SEQUENCE,
            cache=reference_cache,
        )
        actual_decode = decoder.decode_forward(
            functional_test._tt_tensor(decode_hidden, mesh_device),
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        functional_test._assert_pcc(
            reference_decode, functional_test._to_host(actual_decode), 0.99, f"layer{layer_idx} decode"
        )
        functional_test._assert_pcc(
            decode_key,
            functional_test._to_host(key_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
            0.99,
            f"layer{layer_idx} decode key",
        )
        functional_test._assert_pcc(
            decode_value,
            functional_test._to_host(value_cache)[:, :, EMITTED_PREFILL_SEQUENCE : EMITTED_PREFILL_SEQUENCE + 1, :],
            0.99,
            f"layer{layer_idx} decode value",
        )
        if decoder.optimization_config.use_sparse_experts:
            assert decoder.experts is not None
            retains_dense_long_prefill = (
                decoder.optimization_config.use_dense_long_prefill and decoder.attention_window is None
            )
            if retains_dense_long_prefill:
                assert decoder.gate_up_weight is None
                assert decoder.gate_weight is not None
                assert decoder.up_weight is not None
                assert decoder.down_weight is not None
            else:
                assert decoder.gate_up_weight is None
                assert decoder.down_weight is None
        del decoder, reference_layer, state
        gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_repeated_paged_decode_stress(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
    generator = torch.Generator().manual_seed(3301)

    prefill = torch.randn((1, 1, 3, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    _, _, _, reference_cache = functional_test._reference_layer(reference_layer, prefill, config)
    decoder.prefill_forward(functional_test._tt_tensor(prefill, mesh_device), key_cache, value_cache)
    untouched_key = functional_test._to_host(key_cache)
    untouched_value = functional_test._to_host(value_cache)
    for current_pos in range(3, 19):
        hidden = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference, _, _, reference_cache = functional_test._reference_layer(
            reference_layer,
            hidden,
            config,
            start_pos=current_pos,
            cache=reference_cache,
        )
        actual = decoder.decode_forward(
            functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache, current_pos=current_pos
        )
        key_host = functional_test._to_host(key_cache)
        value_host = functional_test._to_host(value_cache)
        full_reference_key = reference_cache.layers[SLIDING_LAYER].keys[:, :, : current_pos + 1, :]
        full_reference_value = reference_cache.layers[SLIDING_LAYER].values[:, :, : current_pos + 1, :]
        functional_test._assert_pcc(
            full_reference_key, key_host[:, :, : current_pos + 1, :], 0.99, f"full key prefix pos={current_pos}"
        )
        functional_test._assert_pcc(
            full_reference_value,
            value_host[:, :, : current_pos + 1, :],
            0.99,
            f"full value prefix pos={current_pos}",
        )
        assert torch.equal(key_host[:, :, current_pos + 1 :, :], untouched_key[:, :, current_pos + 1 :, :])
        assert torch.equal(value_host[:, :, current_pos + 1 :, :], untouched_value[:, :, current_pos + 1 :, :])
        functional_test._assert_pcc(reference, functional_test._to_host(actual), 0.99, f"decode pos={current_pos}")
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 128_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_traced_decode_output_and_full_cache_integrity(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
    generator = torch.Generator().manual_seed(3391)
    prefill = torch.randn(
        (1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size), generator=generator, dtype=torch.bfloat16
    )
    _, _, _, reference_cache = functional_test._reference_layer(reference_layer, prefill, config)
    decoder.prefill_forward(functional_test._tt_tensor(prefill, mesh_device), key_cache, value_cache)
    post_prefill_key = functional_test._to_host(key_cache)
    post_prefill_value = functional_test._to_host(value_cache)
    decode_hidden = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    reference, _, _, reference_cache = functional_test._reference_layer(
        reference_layer,
        decode_hidden,
        config,
        start_pos=EMITTED_PREFILL_SEQUENCE,
        cache=reference_cache,
    )
    decode_input = functional_test._tt_tensor(decode_hidden, mesh_device)
    warm = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    first_output = functional_test._to_host(trace_output)
    first_key = functional_test._to_host(key_cache)
    first_value = functional_test._to_host(value_cache)
    for _ in range(9):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    final_output = functional_test._to_host(trace_output)
    final_key = functional_test._to_host(key_cache)
    final_value = functional_test._to_host(value_cache)
    prefix_len = EMITTED_PREFILL_SEQUENCE + 1
    functional_test._assert_pcc(reference, final_output, 0.99, "traced optimized decode")
    functional_test._assert_pcc(
        reference_cache.layers[SLIDING_LAYER].keys[:, :, :prefix_len, :],
        final_key[:, :, :prefix_len, :],
        0.99,
        "traced key prefix",
    )
    functional_test._assert_pcc(
        reference_cache.layers[SLIDING_LAYER].values[:, :, :prefix_len, :],
        final_value[:, :, :prefix_len, :],
        0.99,
        "traced value prefix",
    )
    assert torch.equal(final_key[:, :, prefix_len:, :], post_prefill_key[:, :, prefix_len:, :])
    assert torch.equal(final_value[:, :, prefix_len:, :], post_prefill_value[:, :, prefix_len:, :])
    assert torch.equal(first_output, final_output)
    assert torch.equal(first_key, final_key)
    assert torch.equal(first_value, final_value)
    ttnn.release_trace(mesh_device, trace_id)
    trace_output.deallocate(True)
    del decoder, reference_layer, state
    gc.collect()


def _measure_decoder(decoder_cls, state, config, mesh_device, *, optimization_config=None):
    kwargs = {}
    if optimization_config is not None:
        kwargs["optimization_config"] = optimization_config
    decoder = decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=SLIDING_LAYER,
        mesh_device=mesh_device,
        **kwargs,
    )
    generator = torch.Generator().manual_seed(4401)
    seq_len = int(os.environ.get("OPTIMIZED_DECODER_PERF_SEQ_LEN", str(EMITTED_PREFILL_SEQUENCE)))
    prefill_input = functional_test._tt_tensor(
        torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    decode_input = functional_test._tt_tensor(
        torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
    output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    output.deallocate(True)
    prefill_repeats = int(os.environ.get("OPTIMIZED_DECODER_PREFILL_REPEATS", "10"))
    decode_replays = int(os.environ.get("OPTIMIZED_DECODER_TRACE_REPLAYS", "100"))
    prefill_ms = []
    for _ in range(prefill_repeats):
        start = time.perf_counter()
        output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        prefill_ms.append((time.perf_counter() - start) * 1000)
        output.deallocate(True)

    output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.synchronize_device(mesh_device)
    output.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(decode_input, key_cache, value_cache, current_pos=EMITTED_PREFILL_SEQUENCE)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    decode_ms = []
    for _ in range(decode_replays):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_ms.append((time.perf_counter() - start) * 1000)
    ttnn.release_trace(mesh_device, trace_id)
    trace_output.deallocate(True)
    result = {
        "prefill_seq_len": seq_len,
        "prefill_mean_ms": statistics.mean(prefill_ms),
        "prefill_median_ms": statistics.median(prefill_ms),
        "prefill_stdev_ms": statistics.pstdev(prefill_ms),
        "prefill_min_ms": min(prefill_ms),
        "prefill_max_ms": max(prefill_ms),
        "decode_traced_mean_ms": statistics.mean(decode_ms),
        "decode_traced_median_ms": statistics.median(decode_ms),
        "decode_traced_stdev_ms": statistics.pstdev(decode_ms),
        "decode_traced_min_ms": min(decode_ms),
        "decode_traced_max_ms": max(decode_ms),
    }
    del decoder
    gc.collect()
    return result


@pytest.mark.skipif(
    os.environ.get("RUN_OPTIMIZED_DECODER_PROFILE") != "1",
    reason="opt-in Tracy/device-profiler window",
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 128_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_profile_optimized_warmed_windows(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._real_state()
    generator = torch.Generator().manual_seed(5501)

    def profile_decoder(decoder, prefix: str):
        prefill_input = functional_test._tt_tensor(
            torch.randn(
                (1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size),
                generator=generator,
                dtype=torch.bfloat16,
            ),
            mesh_device,
        )
        decode_input = functional_test._tt_tensor(
            torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16),
            mesh_device,
        )
        key_cache, value_cache = _empty_caches_for_decoder(config, mesh_device, decoder)
        output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        output.deallocate(True)
        signpost(header=f"{prefix}_PREFILL")
        prefill_start = time.perf_counter()
        output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
        ttnn.synchronize_device(mesh_device)
        profiled_prefill_ms = (time.perf_counter() - prefill_start) * 1000
        signpost(header=f"{prefix}_PREFILL_END")
        output.deallocate(True)

        output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.synchronize_device(mesh_device)
        output.deallocate(True)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        trace_output = decoder.decode_forward(
            decode_input,
            key_cache,
            value_cache,
            current_pos=EMITTED_PREFILL_SEQUENCE,
        )
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        signpost(header=f"{prefix}_DECODE")
        decode_start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        profiled_decode_ms = (time.perf_counter() - decode_start) * 1000
        signpost(header=f"{prefix}_DECODE_END")
        print(
            f"{prefix.lower()}_profiled_wall_ms="
            f"{{'prefill': {profiled_prefill_ms}, 'traced_decode': {profiled_decode_ms}}}"
        )
        assert tuple(trace_output.shape) == (1, 1, 1, config.hidden_size)
        ttnn.release_trace(mesh_device, trace_id)
        trace_output.deallocate(True)

    baseline = FusedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=SLIDING_LAYER,
        mesh_device=mesh_device,
    )
    profile_decoder(baseline, "FUSED")
    del baseline
    gc.collect()

    decoder = _decoder(state, config, mesh_device, layer_idx=SLIDING_LAYER)
    profile_decoder(decoder, "OPTIMIZED")
    prefill_128 = functional_test._tt_tensor(
        torch.randn((1, 1, 128, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    key_cache_128, value_cache_128 = decoder.create_kv_cache()
    output = decoder.prefill_forward(prefill_128, key_cache_128, value_cache_128)
    ttnn.synchronize_device(mesh_device)
    output.deallocate(True)
    signpost(header="OPTIMIZED_PREFILL_128")
    output = decoder.prefill_forward(prefill_128, key_cache_128, value_cache_128)
    ttnn.synchronize_device(mesh_device)
    signpost(header="OPTIMIZED_PREFILL_128_END")
    output.deallocate(True)
    del decoder, state
    gc.collect()


@pytest.mark.skipif(os.environ.get("RUN_OPTIMIZED_DECODER_PERF") != "1", reason="opt-in hardware performance gate")
@pytest.mark.parametrize("device_params", [{"trace_region_size": 128_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_optimized_beats_fused_warmed_prefill_and_traced_decode(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._real_state()
    baseline = _measure_decoder(FusedDecoder, state, config, mesh_device)
    optimized_config = _optimization_config(os.environ.get("OPTIMIZED_DECODER_PERF_VARIANT", "optimized"))
    optimized = _measure_decoder(
        OptimizedDecoder,
        state,
        config,
        mesh_device,
        optimization_config=optimized_config,
    )
    print(f"optimization_config={optimized_config}")
    print(f"fused_perf={baseline}")
    print(f"optimized_perf={optimized}")
    if optimized["prefill_seq_len"] == EMITTED_PREFILL_SEQUENCE:
        assert optimized["prefill_mean_ms"] < baseline["prefill_mean_ms"]
    assert optimized["decode_traced_mean_ms"] < baseline["decode_traced_mean_ms"]
    del state
    gc.collect()
