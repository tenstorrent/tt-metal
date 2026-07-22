# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import inspect
import time
from contextlib import contextmanager

import pytest
import torch

import models.autoports.openai_gpt_oss_20b.tests.test_functional_decoder as functional_test
import ttnn
from models.autoports.openai_gpt_oss_20b.tt.functional_decoder import EMITTED_PREFILL_SEQUENCE, FunctionalDecoder
from models.autoports.openai_gpt_oss_20b.tt.fused_decoder import FusedDecoder

SLIDING_LAYER = 12
FULL_LAYER = 13


@contextmanager
def _functional_helpers_for_layer(layer_idx: int):
    previous = functional_test.LAYER_IDX
    functional_test.LAYER_IDX = layer_idx
    try:
        yield
    finally:
        functional_test.LAYER_IDX = previous


def _decoder(decoder_cls, state, config, mesh_device, *, layer_idx=SLIDING_LAYER, batch=1):
    return decoder_cls.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
    )


def test_fused_runtime_is_distinct_and_has_no_host_fallback():
    assert FusedDecoder.prefill_forward is not FunctionalDecoder.prefill_forward
    assert FusedDecoder.decode_forward is not FunctionalDecoder.decode_forward
    assert FusedDecoder._moe_forward is not FunctionalDecoder._moe_forward

    methods = (
        FusedDecoder._moe_forward,
        FusedDecoder._sparse_moe_forward,
        FusedDecoder.prefill_forward,
        FusedDecoder.decode_forward,
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
    for method in methods:
        source = inspect.getsource(method)
        for token in forbidden:
            assert token not in source, f"{method.__name__} contains forbidden runtime token {token!r}"

    prefill_source = inspect.getsource(FusedDecoder.prefill_forward)
    decode_source = inspect.getsource(FusedDecoder.decode_forward)
    moe_source = inspect.getsource(FusedDecoder._moe_forward)
    sparse_moe_source = inspect.getsource(FusedDecoder._sparse_moe_forward)
    assert "scaled_dot_product_attention(" in prefill_source
    assert "scaled_dot_product_attention_decode(" in decode_source
    assert "paged_update_cache(" in decode_source
    assert "nlp_concat_heads_decode(" in decode_source
    assert "repeat_interleave" not in prefill_source
    assert "_sparse_moe_forward" in moe_source
    assert sparse_moe_source.count("sparse_matmul(") == 4
    # GQA SDPA rejects sharded output; this is the sole required runtime
    # interleaved-to-sharded transfer feeding nlp_concat_heads_decode.
    assert decode_source.count("to_memory_config(") == 1


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_non_aligned_prefill_and_determinism(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)

    deterministic_outputs = []
    deterministic_caches = []
    for seq_len in (3, EMITTED_PREFILL_SEQUENCE, 33):
        generator = torch.Generator().manual_seed(1200 + seq_len)
        hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
        reference, reference_key, reference_value, _ = functional_test._reference_layer(reference_layer, hidden, config)
        key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
        actual = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), key_cache, value_cache)

        functional_test._assert_pcc(reference, functional_test._to_host(actual), 0.99, f"fused prefill S={seq_len}")
        key_host = functional_test._to_host(key_cache)[:, :, :seq_len, :]
        value_host = functional_test._to_host(value_cache)[:, :, :seq_len, :]
        functional_test._assert_pcc(reference_key, key_host, 0.99, f"fused key S={seq_len}")
        functional_test._assert_pcc(reference_value, value_host, 0.99, f"fused value S={seq_len}")
        if seq_len == EMITTED_PREFILL_SEQUENCE:
            deterministic_outputs.append(functional_test._to_host(actual))
            deterministic_caches.append((key_host, value_host))
            second_key, second_value = functional_test._empty_caches(config, mesh_device)
            second = decoder.prefill_forward(functional_test._tt_tensor(hidden, mesh_device), second_key, second_value)
            deterministic_outputs.append(functional_test._to_host(second))
            deterministic_caches.append(
                (
                    functional_test._to_host(second_key)[:, :, :seq_len, :],
                    functional_test._to_host(second_value)[:, :, :seq_len, :],
                )
            )
            decode_hidden = torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16)
            first_decode = decoder.decode_forward(
                functional_test._tt_tensor(decode_hidden, mesh_device),
                key_cache,
                value_cache,
                current_pos=seq_len,
            )
            second_decode = decoder.decode_forward(
                functional_test._tt_tensor(decode_hidden, mesh_device),
                second_key,
                second_value,
                current_pos=seq_len,
            )
            assert torch.equal(functional_test._to_host(first_decode), functional_test._to_host(second_decode))
            assert torch.equal(
                functional_test._to_host(key_cache)[:, :, seq_len : seq_len + 1, :],
                functional_test._to_host(second_key)[:, :, seq_len : seq_len + 1, :],
            )
            assert torch.equal(
                functional_test._to_host(value_cache)[:, :, seq_len : seq_len + 1, :],
                functional_test._to_host(second_value)[:, :, seq_len : seq_len + 1, :],
            )

    assert torch.equal(deterministic_outputs[0], deterministic_outputs[1])
    assert torch.equal(deterministic_caches[0][0], deterministic_caches[1][0])
    assert torch.equal(deterministic_caches[0][1], deterministic_caches[1][1])
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_batch_two_prefill_and_paged_decode(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device, batch=2)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device, batch=2)
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
    functional_test._assert_pcc(
        decode_key, functional_test._to_host(key_cache)[:, :, 3:4, :], 0.99, "batch2 decode key"
    )
    functional_test._assert_pcc(
        decode_value, functional_test._to_host(value_cache)[:, :, 3:4, :], 0.99, "batch2 decode value"
    )
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("layer_idx,seed", [(SLIDING_LAYER, 9090), (FULL_LAYER, 9249)])
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_layer_kind_prefill_and_decode(mesh_device, layer_idx, seed):
    with _functional_helpers_for_layer(layer_idx):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(FusedDecoder, state, config, mesh_device, layer_idx=layer_idx)
        reference_layer = functional_test._hf_layer(state, config)
        key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
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
        del decoder, reference_layer, state
        gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_real_weight_non_aligned_prefill_moe_matches_wide_control(mesh_device):
    """Prove the split prefill projection with nonzero expert matrices."""

    with _functional_helpers_for_layer(SLIDING_LAYER):
        config = functional_test._config()
        state = functional_test._real_state()
        decoder = _decoder(FusedDecoder, state, config, mesh_device)
        generator = torch.Generator().manual_seed(9441)
        for seq_len in (3, EMITTED_PREFILL_SEQUENCE, 33):
            hidden = torch.randn((1, 1, seq_len, config.hidden_size), generator=generator, dtype=torch.bfloat16)
            wide_input = functional_test._tt_tensor(hidden, mesh_device)
            split_input = functional_test._tt_tensor(hidden, mesh_device)
            decoder.moe_policy = "wide"
            wide = functional_test._to_host(decoder._moe_forward(wide_input, seq_len))
            decoder.moe_policy = "auto"
            split = functional_test._to_host(decoder._moe_forward(split_input, seq_len))
            functional_test._assert_pcc(wide, split, 0.999, f"real MoE split-vs-wide S={seq_len}")
        del decoder, state
        gc.collect()


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_repeated_paged_decode_stress(mesh_device):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
    generator = torch.Generator().manual_seed(3301)

    prefill = torch.randn((1, 1, 3, config.hidden_size), generator=generator, dtype=torch.bfloat16)
    _, _, _, reference_cache = functional_test._reference_layer(reference_layer, prefill, config)
    decoder.prefill_forward(functional_test._tt_tensor(prefill, mesh_device), key_cache, value_cache)
    post_prefill_key = functional_test._to_host(key_cache)
    post_prefill_value = functional_test._to_host(value_cache)
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
            full_reference_key,
            key_host[:, :, : current_pos + 1, :],
            0.99,
            f"repeated full key prefix pos={current_pos}",
        )
        functional_test._assert_pcc(
            full_reference_value,
            value_host[:, :, : current_pos + 1, :],
            0.99,
            f"repeated full value prefix pos={current_pos}",
        )
        assert torch.equal(key_host[:, :, current_pos + 1 :, :], post_prefill_key[:, :, current_pos + 1 :, :])
        assert torch.equal(value_host[:, :, current_pos + 1 :, :], post_prefill_value[:, :, current_pos + 1 :, :])
        functional_test._assert_pcc(
            reference, functional_test._to_host(actual), 0.99, f"repeated decode pos={current_pos}"
        )
    del decoder, reference_layer, state
    gc.collect()


@pytest.mark.parametrize("device_params", [{"trace_region_size": 128_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_traced_decode_output_and_full_cache_integrity(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    decoder = _decoder(FusedDecoder, state, config, mesh_device)
    reference_layer = functional_test._hf_layer(state, config)
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device)
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

    # Warm the exact fixed-position microtrace and its lazy position views.
    warm = decoder.decode_forward(
        decode_input,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
    ttnn.synchronize_device(mesh_device)
    warm.deallocate(True)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(
        decode_input,
        key_cache,
        value_cache,
        current_pos=EMITTED_PREFILL_SEQUENCE,
    )
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
    functional_test._assert_pcc(reference, final_output, 0.99, "traced fused decode")
    functional_test._assert_pcc(
        reference_cache.layers[SLIDING_LAYER].keys[:, :, :prefix_len, :],
        final_key[:, :, :prefix_len, :],
        0.99,
        "traced full key prefix",
    )
    functional_test._assert_pcc(
        reference_cache.layers[SLIDING_LAYER].values[:, :, :prefix_len, :],
        final_value[:, :, :prefix_len, :],
        0.99,
        "traced full value prefix",
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


def _measure_decoder(decoder_cls, state, config, mesh_device):
    decoder = _decoder(decoder_cls, state, config, mesh_device)
    generator = torch.Generator().manual_seed(4401)
    prefill_input = functional_test._tt_tensor(
        torch.randn((1, 1, EMITTED_PREFILL_SEQUENCE, config.hidden_size), generator=generator, dtype=torch.bfloat16),
        mesh_device,
    )
    decode_input = functional_test._tt_tensor(
        torch.randn((1, 1, 1, config.hidden_size), generator=generator, dtype=torch.bfloat16), mesh_device
    )
    key_cache, value_cache = functional_test._empty_caches(config, mesh_device)

    output = decoder.prefill_forward(prefill_input, key_cache, value_cache)
    ttnn.synchronize_device(mesh_device)
    output.deallocate(True)
    prefill_ms = []
    for _ in range(20):
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
    for _ in range(200):
        start = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        decode_ms.append((time.perf_counter() - start) * 1000)
    ttnn.release_trace(mesh_device, trace_id)
    trace_output.deallocate(True)
    result = {
        "prefill_mean_ms": sum(prefill_ms) / len(prefill_ms),
        "prefill_min_ms": min(prefill_ms),
        "decode_traced_mean_ms": sum(decode_ms) / len(decode_ms),
        "decode_traced_min_ms": min(decode_ms),
    }
    del decoder
    gc.collect()
    return result


@pytest.mark.parametrize("device_params", [{"trace_region_size": 128_000_000}], indirect=True)
@pytest.mark.parametrize("mesh_device", [1], indirect=True)
def test_fused_beats_functional_warmed_prefill_and_traced_decode(mesh_device, device_params):
    config = functional_test._config()
    state = functional_test._synthetic_state(config)
    baseline = _measure_decoder(FunctionalDecoder, state, config, mesh_device)
    fused = _measure_decoder(FusedDecoder, state, config, mesh_device)
    print(f"functional_perf={baseline}")
    print(f"fused_perf={fused}")
    assert fused["prefill_mean_ms"] < baseline["prefill_mean_ms"]
    assert fused["decode_traced_mean_ms"] < baseline["decode_traced_mean_ms"]
    del state
    gc.collect()
