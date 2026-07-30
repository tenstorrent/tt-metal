# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness and contract gates for the single-device optimized decoder."""

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
    _finish,
    _page_table,
    _positions,
    _project_qkv,
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
    PCC_ACCEPTANCE,
    OptimizationPolicy,
    OptimizedDecoder,
)
from models.common.lightweightmodule import LightweightModule


def test_optimized_path_and_policy_contract():
    assert issubclass(OptimizedDecoder, LightweightModule)
    assert "FunctionalDecoder" not in inspect.getsource(OptimizedDecoder)
    policy = OptimizationPolicy()
    assert policy.attention_weight_dtype == ttnn.bfloat4_b
    assert policy.gate_up_weight_dtype == ttnn.bfloat4_b
    assert policy.down_weight_dtype == ttnn.bfloat4_b
    assert policy.kv_cache_dtype == ttnn.bfloat8_b
    assert policy.attention_math_fidelity == ttnn.MathFidelity.LoFi
    assert policy.mlp_math_fidelity == ttnn.MathFidelity.LoFi
    assert policy.decode_core_grid == (8, 1)
    assert policy.fused_paged_cache_update
    assert not policy.explicit_decode_sdpa
    assert not policy.fused_rope
    assert not policy.fused_prefill_rope
    assert policy.prefill_qkv_in0_block_w == 2
    assert policy.prefill_gate_up_in0_block_w == 2
    assert "qkv_prefill" in inspect.getsource(OptimizedDecoder.prefill_forward)

    runtime = (
        OptimizedDecoder._norm_prefill,
        OptimizedDecoder._norm_decode,
        OptimizedDecoder._linear_prefill,
        OptimizedDecoder._linear_decode,
        OptimizedDecoder._mlp_prefill,
        OptimizedDecoder._mlp_decode,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
        OptimizedDecoder.forward,
    )
    for method in runtime:
        source = inspect.getsource(method)
        for forbidden in ("torch", "from_torch", "to_torch", ".cpu(", "FunctionalDecoder"):
            assert forbidden not in source, (method.__name__, forbidden)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [31, 32, 33, 63, 64, 65])
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
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, max_context, mesh_device, permute=True),
    )
    _assert_pcc(f"optimized-prefill-{seq_len}", reference, _to_torch_prefill(actual), PCC_ACCEPTANCE)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_real_weight_prefill_and_decode(mesh_device):
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
    _assert_pcc("optimized-real-prefill-33", prefill_reference, _to_torch_prefill(prefill_actual), PCC_ACCEPTANCE)

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
    _assert_pcc("optimized-real-decode-33", decode_reference, _to_torch_decode(decode_actual), PCC_ACCEPTANCE)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_multi_user_paged_prefill_routes_cache(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    batch = 2
    seq_len = 33
    max_context = 64
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=max_context,
    )
    hidden = torch.randn(batch, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(233)).to(
        torch.bfloat16
    )
    reference, past = _reference_prefill(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _page_table(batch, max_context, mesh_device, permute=True)
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
    )
    _assert_pcc("optimized-prefill-batch2-33", reference, _to_torch_prefill(actual), PCC_ACCEPTANCE)

    # Consume each user's distinct, permuted physical cache blocks.  Prefill
    # output alone never reads the paged cache and therefore cannot prove fill
    # routing.
    decode_hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(234)).to(
        torch.bfloat16
    )
    decode_reference = _reference_decode(config, state, decode_hidden, seq_len, past)
    decode_actual = decoder.decode_forward(
        _to_tt_decode(decode_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([seq_len] * batch, mesh_device),
        use_long_rope=False,
    )
    _assert_pcc(
        "optimized-prefill-cache-consume-batch2-33",
        decode_reference,
        _to_torch_decode(decode_actual),
        PCC_ACCEPTANCE,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_nonzero_prefill_chunk_boundary_last_token(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    seq_len = 32_769
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=seq_len,
    )
    hidden = (torch.randn(1, seq_len, config.hidden_size, generator=torch.Generator().manual_seed(seq_len)) * 0.02).to(
        torch.bfloat16
    )
    reference = _reference_prefill_last_token(config, state, hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    _assert_pcc(
        "optimized-prefill-nonzero-32769-last-token",
        reference,
        _to_torch_prefill(actual)[:, -1:, :],
        PCC_ACCEPTANCE,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_decode_trace_replay_is_deterministic(mesh_device, batch):
    config = _config()
    state = _synthetic_state(config)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=64,
    )
    page_table = _page_table(batch, 64, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(batch, 1, config.hidden_size, generator=torch.Generator().manual_seed(100 + batch)).to(
        torch.bfloat16
    )
    tt_hidden = _to_tt_decode(hidden, mesh_device)
    positions = [33] if batch == 1 else list(range(1, batch + 1))
    current_positions = _positions(positions, mesh_device)

    def decode():
        return decoder.decode_forward(
            tt_hidden,
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current_positions,
            use_long_rope=False,
        )

    decode()
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decode()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    replayed = []
    try:
        for _ in range(10):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replayed.append(ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).clone())
    finally:
        ttnn.release_trace(mesh_device, trace_id)

    assert all(torch.equal(replayed[0], value) for value in replayed[1:])
    reference = _reference_decode_zero_prefix(config, state, hidden, positions, use_long=False)
    _assert_pcc(
        f"optimized-trace-decode-reference-batch{batch}",
        reference,
        replayed[0].squeeze(0).transpose(0, 1),
        PCC_ACCEPTANCE,
    )
    print(f"TRACE_RESULT path=optimized batch={batch} replays=10 bitwise_deterministic=true")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_varied_positions_consume_nonzero_decode_cache(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    batch = 32
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=batch,
        max_context=128,
    )
    page_table = _page_table(batch, 128, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    generator = torch.Generator().manual_seed(3550)
    first_hidden = (torch.randn(batch, 1, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    second_hidden = (torch.randn(batch, 1, config.hidden_size, generator=generator) * 0.2).to(torch.bfloat16)
    first_positions = list(range(32, 64))
    second_positions = [position + 1 for position in first_positions]

    decoder.decode_forward(
        _to_tt_decode(first_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions(first_positions, mesh_device),
        use_long_rope=False,
    )
    actual = decoder.decode_forward(
        _to_tt_decode(second_hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions(second_positions, mesh_device),
        use_long_rope=False,
    )

    references = []
    scale = math.sqrt(config.hidden_size // config.num_attention_heads)
    for index, first_position in enumerate(first_positions):
        _, first_key, first_value = _project_qkv(
            config,
            state,
            first_hidden[index : index + 1],
            torch.tensor([first_position]),
            use_long=False,
        )
        second_query, second_key, second_value = _project_qkv(
            config,
            state,
            second_hidden[index : index + 1],
            torch.tensor([first_position + 1]),
            use_long=False,
        )
        first_score = (second_query.float() * first_key.float()).sum(-1) / scale
        second_score = (second_query.float() * second_key.float()).sum(-1) / scale
        denominator = first_position + torch.exp(first_score) + torch.exp(second_score)
        attended = (
            torch.exp(first_score).unsqueeze(-1) * first_value.float()
            + torch.exp(second_score).unsqueeze(-1) * second_value.float()
        ) / denominator.unsqueeze(-1)
        references.append(
            _finish(
                config,
                state,
                second_hidden[index : index + 1],
                attended.to(second_value.dtype),
            )
        )
    reference = torch.cat(references, dim=0)
    _assert_pcc(
        "optimized-varied-position-nonzero-cache-batch32",
        reference,
        _to_torch_decode(actual),
        PCC_ACCEPTANCE,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_long_rope_decode_at_advertised_context(mesh_device):
    config = _config()
    state = _synthetic_state(config)
    position = config.max_position_embeddings - 1
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=config.max_position_embeddings,
    )
    page_table = _page_table(1, config.max_position_embeddings, mesh_device, permute=True)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    hidden = torch.randn(1, 1, config.hidden_size, generator=torch.Generator().manual_seed(position)).to(torch.bfloat16)
    actual = decoder.decode_forward(
        _to_tt_decode(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=_positions([position], mesh_device),
        use_long_rope=True,
    )
    reference = _reference_decode_zero_prefix(config, state, hidden, position)
    _assert_pcc("optimized-decode-context-131072", reference, _to_torch_decode(actual), PCC_ACCEPTANCE)
    print("CONTEXT_RESULT path=optimized mode=decode batch=1 context=131072")


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [131_071, 131_072])
def test_optimized_advertised_context_prefill(mesh_device, seq_len):
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _zero_state(config),
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        max_context=seq_len,
    )
    hidden = torch.zeros(1, seq_len, config.hidden_size, dtype=torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    actual = decoder.prefill_forward(
        _to_tt_prefill(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=_page_table(1, seq_len, mesh_device, permute=True),
    )
    result = _to_torch_prefill(actual)
    assert tuple(result.shape) == tuple(hidden.shape)
    assert torch.count_nonzero(result) == 0
    print(
        f"CONTEXT_RESULT path=optimized mode=prefill batch=1 context={seq_len} "
        f"non_aligned={str(seq_len % 32 != 0).lower()}"
    )
