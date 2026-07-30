# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math

import pytest
import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tests.test_functional_decoder import (
    LAYER_IDX,
    _assert_pcc,
    _config,
    _page_table,
    _positions,
    _real_state,
    _reference_decode,
    _reference_decode_zero_prefix,
    _reference_prefill,
    _reference_prefill_last_token,
    _synthetic_state,
    _to_torch_decode,
    _to_torch_prefill,
    _to_tt_decode,
    _to_tt_prefill,
    _zero_state,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    DEFAULT_OPTIMIZATION_POLICY,
    OptimizationPolicy,
    OptimizedDecoder,
)


def test_optimized_static_contract_and_runtime_fallback_audit():
    assert OptimizedDecoder.decode_forward is not OptimizedDecoder.__mro__[1].decode_forward
    assert OptimizedDecoder.prefill_forward is not OptimizedDecoder.__mro__[1].prefill_forward
    runtime = (
        OptimizedDecoder._decode_norm,
        OptimizedDecoder._decode_linear,
        OptimizedDecoder._decode_mlp,
        OptimizedDecoder._prefill_linear,
        OptimizedDecoder._prefill_mlp,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
    )
    for method in runtime:
        source = inspect.getsource(method)
        for forbidden in ("torch", "from_torch", "to_torch", ".cpu(", "all_gather", "all_reduce"):
            assert forbidden not in source, (method.__name__, forbidden)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [33, 65])
def test_optimized_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config)
    max_context = math.ceil(seq_len / 32) * 32
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=max_context,
    )
    hidden = torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(seq_len)).to(
        torch.bfloat16
    )
    reference, _ = _reference_prefill(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    output = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, max_context, mesh_device, permute=True),
    )
    _assert_pcc(f"optimized_prefill_s{seq_len}", reference, _to_torch_prefill(output))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_decode_real_weights_and_trace_replay(mesh_device, batch):
    config = _config()
    state = _real_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    generator = torch.Generator().manual_seed(700 + batch)
    hidden = torch.randn(batch, 1, config.hidden_size, generator=generator).to(torch.bfloat16)
    positions = list(range(1, batch + 1)) if batch > 1 else [33]
    reference = _reference_decode_zero_prefix(config, state, hidden, positions)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    current_positions = _positions(positions, mesh_device)
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()

    def run_decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    run_decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run_decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        replay_outputs = []
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_outputs.append(_to_torch_decode(output))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    _assert_pcc(f"optimized_decode_real_b{batch}", reference, replay_outputs[0])
    assert all(torch.equal(replay_outputs[0], replay) for replay in replay_outputs[1:])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_paged_prefill_then_decode(mesh_device):
    config = _config()
    state = _real_state()
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=64
    )
    page_table = _page_table(1, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(3500)
    prefill = (torch.randn(1, 33, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    prefill_reference, past = _reference_prefill(config, state, prefill)
    prefill_actual = decoder.prefill_forward(
        _to_tt_prefill(prefill, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_pcc("optimized-real-prefill-33", prefill_reference, _to_torch_prefill(prefill_actual))
    hidden = (torch.randn(1, 1, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    decode_reference = _reference_decode(config, state, hidden, 33, past)
    decode_actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([33], mesh_device),
        use_long_rope=False,
    )
    _assert_pcc("optimized-real-decode-after-prefill-33", decode_reference, _to_torch_decode(decode_actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_decode_at_advertised_context(mesh_device):
    config = _config()
    state = _real_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=config.max_position_embeddings,
    )
    page_table = _page_table(1, config.max_position_embeddings, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    position = config.max_position_embeddings - 1
    hidden = (torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(131072)) * 0.2).to(
        torch.bfloat16
    )
    reference = _reference_decode_zero_prefix(config, state, hidden, position)
    actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([position], mesh_device),
        use_long_rope=True,
    )
    _assert_pcc("optimized-real-decode-context-131072", reference, _to_torch_decode(actual))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_long_rope_traced_decode_matches_reference(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    position = config.original_max_position_embeddings
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=position + 1
    )
    page_table = _page_table(1, position + 1, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(position)).to(torch.bfloat16)
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    current_positions = _positions([position], mesh_device)

    def run_decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=True,
        )

    run_decode()
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = run_decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        actual = _to_torch_decode(output)
    finally:
        ttnn.release_trace(mesh_device, trace_id)
    reference = _reference_decode_zero_prefix(config, state, hidden, position)
    _assert_pcc("optimized-trace-decode-long-rope-position-4096", reference, actual)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [131_071, 131_072])
def test_optimized_paged_prefill_advertised_context(mesh_device, seq_len):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _zero_state(config), hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=seq_len
    )
    hidden = torch.zeros(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    output = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    actual = _to_torch_prefill(output)
    assert tuple(actual.shape) == tuple(hidden.shape)
    assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_nonzero_prefill_crosses_chunk_boundary(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    seq_len = 32_769
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=LAYER_IDX, mesh_device=mesh_device, max_context=seq_len
    )
    hidden = (torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(seq_len)) * 0.02).to(
        torch.bfloat16
    )
    reference = _reference_prefill_last_token(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    output = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    _assert_pcc("optimized-prefill-nonzero-32769-last-token", reference, _to_torch_prefill(output)[:, -1:, :])


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_policy_is_materialized(mesh_device):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=64,
    )
    assert decoder.weights["qkv"].dtype == DEFAULT_OPTIMIZATION_POLICY.attention_weight_dtype
    assert decoder.weights["gate_up"].dtype == DEFAULT_OPTIMIZATION_POLICY.gate_up_weight_dtype
    assert decoder.weights["down"].dtype == DEFAULT_OPTIMIZATION_POLICY.down_weight_dtype
    assert decoder.create_paged_kv_cache()[0].dtype == DEFAULT_OPTIMIZATION_POLICY.kv_cache_dtype
    assert decoder.decode_programs["qkv"].in0_block_w == 6
    assert decoder.decode_programs["down"].in0_block_w == 16
    assert "gate_proj" in decoder.weights and "up_proj" in decoder.weights


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_split_qkv_candidate_matches_reference(mesh_device, batch):
    config = _config()
    state = _real_state()
    policy = OptimizationPolicy(split_decode_qkv=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
        optimization_policy=policy,
    )
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(9100 + batch)).to(
        torch.bfloat16
    )
    positions = [33] * batch
    reference = _reference_decode_zero_prefix(config, state, hidden, positions)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    output = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(batch, 128, mesh_device, permute=True),
        current_positions=_positions(positions, mesh_device),
        use_long_rope=False,
    )
    _assert_pcc(f"optimized-split-qkv-b{batch}", reference, _to_torch_decode(output))


def bfp4_attention_policy():
    """Candidate factory used by the optimization sweep harness."""
    return OptimizationPolicy(attention_weight_dtype=ttnn.bfloat4_b)
