# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import math
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    REPRESENTATIVE_LAYERS,
    _assert_pcc,
    _config,
    _decode_inputs,
    _dense_reference,
    _normalized,
    _page_table,
    _project_split_qkv,
    _randn,
    _real_layer_one_state,
    _rope_interleaved,
    _sparse_moe_reference,
    _synthetic_state,
    _to_host_tt,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import (
    ADVERTISED_CONTEXT,
    FunctionalDecoder,
    _load_expert_weights,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import (
    POLICIES,
    OptimizedDecoder,
    _decode_sparse_nnz,
)
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import comp_pcc


def _candidate():
    return os.environ.get("NORTH_MINI_OPT_CANDIDATE", "default")


def _real_dense_layer_zero_state():
    """Load the official dense layer available in the repo-local partial checkpoint."""
    explicit = os.environ.get("NORTH_MINI_REAL_WEIGHT_DIR")
    roots = [Path(explicit)] if explicit else []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.extend((Path(hf_home), Path(hf_home) / "hub"))
    roots.append(Path("/huggingface/hub"))
    snapshot = next(
        (
            path
            for root in roots
            for path in root.glob(f"models--CohereLabs--North-Mini-Code-1.0/snapshots/{REAL_REVISION}")
            if path.is_dir()
        ),
        None,
    )
    if snapshot is None:
        pytest.skip("North-Mini checkpoint not cached; set NORTH_MINI_REAL_WEIGHT_DIR")
    shard = snapshot / "model-00001-of-00049.safetensors"
    if not shard.is_file():
        pytest.skip("North-Mini dense layer-0 shard 1 is not cached")
    prefix = "model.layers.0."
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}


def _isolated_candidate(candidate):
    """Return a candidate whose delta from production is asserted in the test."""
    policy = POLICIES[candidate]
    default = POLICIES["default"]
    changed_fields = {name for name in default.__dataclass_fields__ if getattr(policy, name) != getattr(default, name)}
    expected_fields = {
        "sparse_bfp4_bf16_cache_selected_decode": {
            "cache_dtype",
            "decode_expert_gate_up_dtype",
            "decode_expert_down_dtype",
            "decode_expert_gate_up_fidelity",
            "decode_expert_down_fidelity",
        },
        "cache_bf16_selected_decode": {"cache_dtype"},
        "bfp4_attention_selected_decode": {"attention_weight_dtype"},
    }
    assert changed_fields == expected_fields[candidate]
    return candidate


def _sparse_moe_reference_any_layout(normalized, state, config, layer_idx, *, flat_indices=None):
    """Evaluate routed experts from either packed test weights or hub per-expert keys."""
    prefix = f"model.layers.{layer_idx}."
    packed_key = prefix + "mlp.experts.gate_up_proj"
    if packed_key in state:
        return _sparse_moe_reference(
            normalized,
            state,
            config,
            layer_idx,
            flat_indices=flat_indices,
        )

    flat = normalized.reshape(-1, config.hidden_size)
    if flat_indices is not None:
        flat = flat[torch.as_tensor(flat_indices)]
    logits = F.linear(flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    result = torch.zeros_like(flat)
    for token in range(flat.shape[0]):
        for route in range(config.num_experts_per_tok):
            expert = int(experts[token, route])
            expert_prefix = prefix + f"mlp.experts.{expert}."
            gate = F.linear(flat[token], state[expert_prefix + "gate_proj.weight"])
            up = F.linear(flat[token], state[expert_prefix + "up_proj.weight"])
            contribution = F.linear(F.silu(gate) * up, state[expert_prefix + "down_proj.weight"])
            result[token] += contribution * scores[token, route]
    return result, experts


def test_optimized_contract_and_no_runtime_functional_fallback():
    assert issubclass(OptimizedDecoder, LightweightModule)
    assert POLICIES["default"].sparse_experts
    assert POLICIES["default"].decode_expert_gate_up_dtype == ttnn.bfloat8_b
    assert POLICIES["default"].decode_expert_down_dtype == ttnn.bfloat8_b
    assert "ttnn.sparse_matmul" in inspect.getsource(OptimizedDecoder._sparse_expert_moe)
    assert OptimizedDecoder.prefill_forward.__qualname__.startswith("OptimizedDecoder.")
    assert OptimizedDecoder.decode_forward.__qualname__.startswith("OptimizedDecoder.")

    runtime_methods = (
        OptimizedDecoder._qkv_prefill,
        OptimizedDecoder._attention_prefill,
        OptimizedDecoder._qkv_decode,
        OptimizedDecoder._attention_decode,
        OptimizedDecoder._dense_mlp,
        OptimizedDecoder._routing,
        OptimizedDecoder._dense_expert_moe,
        OptimizedDecoder._sparse_expert_moe,
        OptimizedDecoder._sparse_moe_chunk,
        OptimizedDecoder._sparse_moe,
        OptimizedDecoder.prefill_forward,
        OptimizedDecoder.decode_forward,
    )
    forbidden = ("import torch", "from_torch", "to_torch", ".cpu(", ".numpy(")
    for method in runtime_methods:
        source = inspect.getsource(method)
        assert all(token not in source for token in forbidden), method.__name__


def test_batch1_exact_nnz_candidate_has_an_exact_presence_invariant():
    default = POLICIES["default"]
    candidate = POLICIES["batch1_exact_nnz8"]
    control = POLICIES["batch1_dynamic_nnz_control"]
    candidate_changed_fields = {
        name for name in default.__dataclass_fields__ if getattr(default, name) != getattr(candidate, name)
    }
    control_changed_fields = {
        name for name in default.__dataclass_fields__ if getattr(default, name) != getattr(control, name)
    }
    assert candidate_changed_fields == set()
    assert control_changed_fields == {"decode_exact_nnz"}
    assert default.decode_exact_nnz == 8
    assert candidate.decode_exact_nnz == 8
    assert control.decode_exact_nnz is None

    # Only one-token decode has a compile-time exact union. Batch-32 decode
    # and every prefill retain device-side count inference.
    assert _decode_sparse_nnz(default, 1, prefill=False) == 8
    assert _decode_sparse_nnz(default, 32, prefill=False) is None
    assert _decode_sparse_nnz(default, 1, prefill=True) is None
    assert _decode_sparse_nnz(control, 1, prefill=False) is None

    routing_source = inspect.getsource(OptimizedDecoder._routing)
    sparse_source = inspect.getsource(OptimizedDecoder._sparse_expert_moe)
    assert "src=ttnn.ones_like(top_values)" in routing_source
    assert "route_presence if route_presence is not None else routing" in sparse_source
    assert sparse_source.count("nnz=nnz") == 4


def test_reviewer_followup_candidates_and_selected_policy_are_wired():
    """Keep rejected experiments explicit and bind the measured production chain."""

    default = POLICIES["default"]
    assert not default.packed_dense_large_prefill
    assert default.decode_sharded_residual
    assert default.attention_dram_sharded
    assert not default.attention_dram_sharded_serving
    assert default.decode_router_grid is None
    assert default.decode_exact_nnz == 8
    assert default.prefill_functional_router_compute
    assert default.dense_gate_up_dtype == ttnn.bfloat4_b
    assert default.dense_down_dtype == ttnn.bfloat8_b
    assert default.dense_gate_up_fidelity == ttnn.MathFidelity.LoFi
    assert default.dense_down_fidelity == ttnn.MathFidelity.LoFi
    assert default.packed_dense_mlp
    assert default.dense_decode_unpacked_batch1
    assert default.dense_decode_lofi_batch32
    assert default.explicit_dense_decode_programs
    assert default.dense_decode_down_dram_sharded
    assert not default.dense_decode_down_dram_sharded_batch1

    dense_bfp4_control = POLICIES["dense_bfp4_lofi_control"]
    assert dense_bfp4_control.dense_gate_up_dtype == ttnn.bfloat4_b
    assert dense_bfp4_control.dense_down_dtype == ttnn.bfloat8_b
    assert dense_bfp4_control.dense_gate_up_fidelity == ttnn.MathFidelity.LoFi
    assert dense_bfp4_control.dense_down_fidelity == ttnn.MathFidelity.LoFi

    dense_bfp8_lofi_control = POLICIES["dense_bfp8_lofi_control"]
    assert dense_bfp8_lofi_control.dense_gate_up_dtype == ttnn.bfloat8_b
    assert dense_bfp8_lofi_control.dense_down_dtype == ttnn.bfloat8_b
    assert dense_bfp8_lofi_control.dense_gate_up_fidelity == ttnn.MathFidelity.LoFi
    assert dense_bfp8_lofi_control.dense_down_fidelity == ttnn.MathFidelity.LoFi
    assert dense_bfp8_lofi_control.packed_dense_mlp

    dense_unpacked_control = POLICIES["dense_unpacked_bfp8_hifi2_control"]
    assert not dense_unpacked_control.packed_dense_mlp
    assert dense_unpacked_control.explicit_dense_decode_programs
    assert dense_unpacked_control.dense_decode_gate_up_grid == (8, 6)
    assert dense_unpacked_control.dense_decode_gate_up_out_block_w == 2
    assert dense_unpacked_control.dense_decode_gate_up_subblock_w == 2

    dense_packed_batch1 = POLICIES["dense_packed_batch1_control"]
    assert dense_packed_batch1.packed_dense_mlp
    assert not dense_packed_batch1.dense_decode_unpacked_batch1

    dense_hifi2_batch32 = POLICIES["dense_hifi2_decode_batch32_control"]
    assert not dense_hifi2_batch32.dense_decode_lofi_batch32
    assert dense_hifi2_batch32.dense_gate_up_fidelity == ttnn.MathFidelity.HiFi2
    assert dense_hifi2_batch32.dense_down_fidelity == ttnn.MathFidelity.HiFi2
    assert POLICIES["dense_bfp8_hifi2_control"] is dense_hifi2_batch32

    for in0_block_w in (4, 8, 16, 32):
        packed = POLICIES[f"dense_packed_gate_up_block{in0_block_w}"]
        unpacked = POLICIES[f"dense_unpacked_gate_up_block{in0_block_w}"]
        assert packed.packed_dense_mlp
        assert not packed.dense_decode_unpacked_batch1
        assert packed.dense_decode_gate_up_in0_block_w == in0_block_w
        assert packed.dense_decode_gate_up_interleaved_input
        assert not unpacked.packed_dense_mlp
        assert unpacked.dense_decode_gate_up_grid == (8, 6)
        assert unpacked.dense_decode_gate_up_in0_block_w == in0_block_w
        assert unpacked.dense_decode_gate_up_interleaved_input

    packed = POLICIES["dense_prefill_packed_2d_g8x8"]
    assert packed.packed_dense_large_prefill
    assert not packed.decode_sharded_residual

    residual = POLICIES["decode_sharded_residual_chain"]
    assert residual.decode_sharded_residual
    assert not residual.attention_dram_sharded

    attention = POLICIES["attention_dram_sharded_chain"]
    assert attention.decode_sharded_residual
    assert attention.attention_dram_sharded

    router = POLICIES["router_decode_g2_block8_subblock2"]
    assert router.decode_router_grid == (2, 1)
    assert router.decode_router_in0_block_w == 8
    assert router.decode_router_out_block_w == 2
    assert router.decode_router_subblock_w == 2

    functional_router = POLICIES["prefill_functional_router_m1024"]
    assert functional_router.prefill_functional_router_compute
    assert functional_router.dense_large_prefill_chunk_size == 1024
    assert not functional_router.dense_large_prefill_functional_compute

    hifi4_router = POLICIES["prefill_hifi4_router_control"]
    assert not hifi4_router.prefill_functional_router_compute

    functional_m32 = POLICIES["prefill_functional_m32_control"]
    assert functional_m32.prefill_functional_router_compute
    assert functional_m32.dense_large_prefill_chunk_size == 32
    assert functional_m32.dense_large_prefill_functional_compute

    assert "expert_gate_up_dense_prefill" in inspect.getsource(OptimizedDecoder._dense_expert_moe)
    assert "decode_norm_program_config" in inspect.getsource(OptimizedDecoder.decode_forward)
    assert "decode_qkv_program_config" in inspect.getsource(OptimizedDecoder._qkv_decode)
    assert "decode_router_grid" in inspect.getsource(OptimizedDecoder._routing)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [1, 31, 33, 65])
def test_optimized_dense_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=96,
        candidate=_candidate(),
    )
    generator = torch.Generator().manual_seed(23000 + seq_len)
    hidden = _randn(generator, 1, seq_len, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(hidden, torch.arange(seq_len).reshape(1, -1), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 3), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    _assert_pcc(f"optimized-dense-prefill-{seq_len}", reference, ttnn.to_torch(actual).squeeze(0))


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_nonzero_paged_cache_slots_and_determinism(mesh_device):
    config = _config()
    state = _synthetic_state(config, 0)
    batch, positions = 4, [5, 17, 31, 63]
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=64,
        candidate=_candidate(),
    )
    generator = torch.Generator().manual_seed(23363)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    table = _page_table(batch, 2)
    page_table = _to_tt(table, mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    first = ttnn.to_torch(decoder.decode_forward(hidden_tt, **kwargs))
    second = ttnn.to_torch(decoder.decode_forward(hidden_tt, **kwargs))
    assert torch.equal(first, second)
    assert key_cache.dtype == decoder.policy.cache_dtype
    assert value_cache.dtype == decoder.policy.cache_dtype

    prefix = "model.layers.0."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    expected_key = F.linear(normalized, state[prefix + "self_attn.k_proj.weight"])
    expected_key = expected_key.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    expected_key = _rope_interleaved(expected_key, torch.tensor(positions).reshape(batch, 1), config)
    expected_key = torch.cat((expected_key[..., ::2], expected_key[..., 1::2]), dim=-1)
    expected_value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    expected_value = expected_value.view(batch, 1, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    physical_key = ttnn.to_torch(key_cache)
    physical_value = ttnn.to_torch(value_cache)
    for user, position in enumerate(positions):
        physical_block = int(table[user, position // decoder.page_size])
        slot = position % decoder.page_size
        _assert_pcc(
            f"optimized-key-slot-{user}",
            expected_key[user, :, 0],
            physical_key[physical_block, :, slot],
            threshold=0.99,
        )
        _assert_pcc(
            f"optimized-value-slot-{user}",
            expected_value[user, :, 0],
            physical_value[physical_block, :, slot],
            threshold=0.99,
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,sequence,selected_tokens",
    [
        (1, 33, [0, 31, 32]),
        (4, 33, [0, 31, 32]),
        (1, 65, [0, 31, 32, 63, 64]),
        (1, 1025, [0, 1023, 1024]),
    ],
)
def test_optimized_sparse_non_aligned_prefill_crosses_internal_chunks(
    mesh_device, layer_idx, sequence, selected_tokens
):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    generator = torch.Generator().manual_seed(23500 + layer_idx + sequence)
    hidden = _randn(generator, 1, sequence, config.hidden_size, scale=0.02)
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx, flat_indices=selected_tokens)
    reference = hidden[:, selected_tokens] + reference_moe.reshape(1, len(selected_tokens), -1)

    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=sequence,
        candidate=_candidate(),
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(1, math.ceil(sequence / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    actual = ttnn.to_torch(actual).squeeze(0)[:, selected_tokens]
    _assert_pcc(f"optimized-sparse-layer-{layer_idx}-prefill-{sequence}", reference, actual)


def _run_optimized_sparse_prefill_batch32_sequence128_exercises_large_dense_path(mesh_device, monkeypatch):
    """Cover the exact serving-prefill shape and prove it takes the optimized large-M composite."""
    config = _config()
    layer_idx, batch, sequence = 1, 32, 128
    state = _real_layer_one_state()
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()

    generator = torch.Generator().manual_seed(32128)
    hidden = _randn(generator, batch, sequence, config.hidden_size, scale=0.02)
    normalized = _normalized(hidden, state, config, layer_idx)
    selected_tokens = sorted(
        {
            *(user * sequence + (37 * user) % sequence for user in range(batch)),
            0,
            1023,
            1024,
            2047,
            2048,
            3071,
            3072,
            batch * sequence - 1,
        }
    )
    selected_flat = normalized.reshape(-1, config.hidden_size)[torch.as_tensor(selected_tokens)]
    logits = F.linear(selected_flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    gate_weights, up_weights, down_weights = _load_expert_weights(
        state, layer_idx, config.num_experts, config.intermediate_size
    )
    reference_moe = torch.zeros_like(selected_flat)
    for token in range(selected_flat.shape[0]):
        for route in range(config.num_experts_per_tok):
            expert = int(experts[token, route])
            gate = selected_flat[token].float() @ gate_weights[expert].float()
            up = selected_flat[token].float() @ up_weights[expert].float()
            contribution = (F.silu(gate) * up) @ down_weights[expert].float()
            reference_moe[token] += (contribution * scores[token, route]).to(reference_moe.dtype)
    del gate_weights, up_weights, down_weights
    reference = hidden.reshape(-1, config.hidden_size)[selected_tokens] + reference_moe

    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=sequence,
        candidate=_candidate(),
    )
    _, routed_tt, _ = decoder._routing(
        _to_tt(normalized.reshape(1, 1, batch * sequence, config.hidden_size), mesh_device),
        batch * sequence,
        prefill=True,
    )
    routed = ttnn.to_torch(routed_tt)[selected_tokens]
    routed_experts = torch.topk(routed, config.num_experts_per_tok, dim=-1).indices
    route_agreement = (
        (torch.sort(routed_experts, dim=-1).values == torch.sort(experts, dim=-1).values)
        .all(dim=-1)
        .float()
        .mean()
        .item()
    )
    print(f"optimized-sparse-prefill-batch32-route-agreement: {route_agreement}")
    dense_calls = []
    dense_expert_moe = decoder._dense_expert_moe

    def tracked_dense_expert_moe(flat, routing, token_count, *, prefill=False):
        dense_calls.append((token_count, prefill))
        return dense_expert_moe(flat, routing, token_count, prefill=prefill)

    def unexpected_sparse_expert_moe(*args, **kwargs):
        pytest.fail("batch-32 sequence-128 prefill incorrectly entered the small-M sparse expert path")

    monkeypatch.setattr(decoder, "_dense_expert_moe", tracked_dense_expert_moe)
    monkeypatch.setattr(decoder, "_sparse_expert_moe", unexpected_sparse_expert_moe)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(batch, math.ceil(sequence / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    chunk_size = decoder.policy.dense_large_prefill_chunk_size
    assert dense_calls == [
        (min(chunk_size, batch * sequence - start), True) for start in range(0, batch * sequence, chunk_size)
    ]
    actual = ttnn.to_torch(actual).squeeze(0).reshape(-1, config.hidden_size)[selected_tokens]
    # `monkeypatch` would otherwise restore bound methods onto the instance,
    # creating a decoder -> bound-method -> decoder cycle that retains several
    # GiB of expert tensors until a later GC.  Release the instrumentation and
    # device tensors before the functional control is constructed.
    monkeypatch.undo()
    decoder.__dict__.pop("_dense_expert_moe", None)
    decoder.__dict__.pop("_sparse_expert_moe", None)
    del dense_expert_moe, tracked_dense_expert_moe, unexpected_sparse_expert_moe
    del decoder, key_cache, value_cache, routed_tt
    _, optimized_hf_pcc = comp_pcc(reference.float(), actual.float(), pcc=0.0)
    print(f"optimized-sparse-prefill-batch32-vs-hf: {optimized_hf_pcc}")

    # The official router has close top-8 boundaries: TTNN and CPU select
    # different eighth experts for a minority of tokens.  Preserve the
    # functional decoder's device semantics as the stage correctness bar.
    functional = FunctionalDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=sequence,
    )
    functional_key, functional_value = functional.create_paged_kv_cache()
    functional_actual = functional.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=functional_key,
        value_cache=functional_value,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    functional_actual = ttnn.to_torch(functional_actual).squeeze(0).reshape(-1, config.hidden_size)[selected_tokens]
    _, functional_hf_pcc = comp_pcc(reference.float(), functional_actual.float(), pcc=0.0)
    print(f"functional-sparse-prefill-batch32-vs-hf: {functional_hf_pcc}")
    direct_passed, optimized_functional_pcc = comp_pcc(
        functional_actual.float(),
        actual.float(),
        pcc=0.995,
    )
    print(f"optimized-sparse-prefill-batch32-vs-functional: {optimized_functional_pcc}")
    assert optimized_hf_pcc >= functional_hf_pcc
    assert direct_passed, optimized_functional_pcc


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx,batch", [(1, 1), (4, 1), (1, 32)])
def test_optimized_sparse_traced_decode_matches_active_expert_reference(mesh_device, layer_idx, batch):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        candidate=_candidate(),
    )
    generator = torch.Generator().manual_seed(24000 + layer_idx + batch)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    replay_hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    prefix = f"model.layers.{layer_idx}."
    normalized = (
        replay_hidden.float() * torch.rsqrt(replay_hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)
    ).to(torch.bfloat16)
    normalized *= state[prefix + "input_layernorm.weight"]
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(batch, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        batch, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = replay_hidden + attention + moe.reshape_as(replay_hidden)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos if decoder.use_rope else None,
        position_sin=sin if decoder.use_rope else None,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        replay_hidden_host = _to_host_tt(replay_hidden.unsqueeze(0), mesh_device)
        ttnn.copy_host_to_device_tensor(replay_hidden_host, hidden_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-sparse-layer-{layer_idx}-batch-{batch}-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_dense_traced_decode_batch_1_and_serving_batch_matches_reference(mesh_device, batch):
    """Exercise the optimized dense decode path under both required batch shapes."""
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        candidate=_candidate(),
    )
    generator = torch.Generator().manual_seed(26000 + batch)
    hidden_a = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(
        hidden_b,
        torch.zeros(batch, 1, dtype=torch.long),
        state,
        config,
    )

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        hidden_b_host = _to_host_tt(hidden_b.unsqueeze(0), mesh_device)
        ttnn.copy_host_to_device_tensor(hidden_b_host, hidden_tt)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-dense-batch-{batch}-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("mode,batch", [("prefill", 1), ("decode", 1), ("decode", 32)])
def test_optimized_real_weight_dense_precision_policy(mesh_device, monkeypatch, mode, batch):
    """Gate the conservative dense policy on official layer-0 weights."""
    state = _real_dense_layer_zero_state()
    monkeypatch.setattr(
        "models.autoports.coherelabs_north_mini_code_1_0.tests.test_optimized_decoder._synthetic_state",
        lambda *args, **kwargs: state,
    )
    if mode == "prefill":
        test_optimized_dense_non_aligned_prefill_matches_reference(mesh_device, 65)
    else:
        test_optimized_dense_traced_decode_batch_1_and_serving_batch_matches_reference(mesh_device, batch)


def _run_sliding_moe_populated_history_dynamic_trace_replay(mesh_device, candidate, *, real_weights=False):
    """Replay updated stable inputs after a nonzero 4096-token paged history."""
    config = _config()
    layer_idx, history = 1, config.sliding_window
    state = _real_layer_one_state() if real_weights else _synthetic_state(config, layer_idx, sparse_weights=True)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=history + 2,
        candidate=candidate,
    )
    generator = torch.Generator().manual_seed(274096)
    past_key = _randn(generator, 1, config.num_key_value_heads, history, config.head_dim, scale=0.01)
    past_value = _randn(generator, 1, config.num_key_value_heads, history, config.head_dim, scale=0.01)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil((history + 2) / decoder.page_size)
    page_table = _to_tt(
        _page_table(1, blocks),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    past_key_tt = _to_tt(past_key, mesh_device, dtype=decoder.policy.cache_dtype)
    past_value_tt = _to_tt(past_value, mesh_device, dtype=decoder.policy.cache_dtype)
    past_key = ttnn.to_torch(past_key_tt)
    past_value = ttnn.to_torch(past_value_tt)
    ttnn.experimental.paged_fill_cache(key_cache, past_key_tt, page_table, batch_idx=0)
    ttnn.experimental.paged_fill_cache(value_cache, past_value_tt, page_table, batch_idx=0)

    hidden_a = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [history])
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    try:
        hidden_b_host = _to_host_tt(hidden_b.unsqueeze(0), mesh_device)
        current_host = _to_host_tt(
            torch.tensor([history + 1], dtype=torch.int32),
            mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(hidden_b_host, hidden_tt)
        ttnn.copy_host_to_device_tensor(
            current_host,
            current,
        )
        next_cos, next_sin = decoder.build_rope_rows([history + 1], hf_config=config, decode=True)
        cos_host = _to_host_tt(next_cos, mesh_device)
        sin_host = _to_host_tt(next_sin, mesh_device)
        ttnn.copy_host_to_device_tensor(cos_host, cos)
        ttnn.copy_host_to_device_tensor(sin_host, sin)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)

        _, query_b, key_b, value_b = _project_split_qkv(
            hidden_b,
            torch.tensor([[history + 1]]),
            state,
            config,
            layer_idx,
        )
        _, _, key_a, value_a = _project_split_qkv(
            hidden_a,
            torch.tensor([[history]]),
            state,
            config,
            layer_idx,
        )
        all_key = torch.cat((past_key, key_a, key_b), dim=2)[:, :, -config.sliding_window :]
        all_value = torch.cat((past_value, value_a, value_b), dim=2)[:, :, -config.sliding_window :]
        repeated_key = all_key.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        repeated_value = all_value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=1)
        scores = torch.matmul(query_b.float(), repeated_key.float().transpose(-2, -1))
        scores /= math.sqrt(config.head_dim)
        probabilities = torch.softmax(scores, dim=-1)
        attention = (
            torch.matmul(probabilities, repeated_value.float()).to(torch.bfloat16).transpose(1, 2).reshape(1, 1, -1)
        )
        attention = F.linear(attention, state["model.layers.1.self_attn.o_proj.weight"])
        normalized_b = _normalized(hidden_b, state, config, layer_idx)
        moe, _ = _sparse_moe_reference_any_layout(normalized_b, state, config, layer_idx)
        reference = hidden_b + attention + moe.reshape_as(hidden_b)
        _assert_pcc(
            "optimized-sliding-moe-populated-history-dynamic-trace",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.skipif(
    os.environ.get("NORTH_MINI_LONG_HISTORY_TRACE") != "1",
    reason="opt in with NORTH_MINI_LONG_HISTORY_TRACE=1",
)
def test_optimized_sliding_moe_populated_history_dynamic_trace_replay_matches_reference(mesh_device):
    _run_sliding_moe_populated_history_dynamic_trace_replay(mesh_device, _candidate())


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_sparse_prefill_batch32_sequence128_exercises_large_dense_path(mesh_device, monkeypatch):
    """Run large-prefill coverage after trace-sensitive decode tests."""
    _run_optimized_sparse_prefill_batch32_sequence128_exercises_large_dense_path(mesh_device, monkeypatch)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(
            "sparse_bfp4_bf16_cache_selected_decode",
            id="sparse_bfp4_bf16_cache_selected_decode",
            marks=pytest.mark.skipif(
                os.environ.get("NORTH_MINI_POPULATED_HISTORY_CANDIDATE") != "sparse_bfp4_bf16_cache_selected_decode",
                reason=(
                    "set NORTH_MINI_POPULATED_HISTORY_CANDIDATE=sparse_bfp4_bf16_cache_selected_decode "
                    "to run this isolated probe"
                ),
            ),
        ),
        pytest.param(
            "cache_bf16_selected_decode",
            id="cache_bf16_selected_decode",
            marks=pytest.mark.skipif(
                os.environ.get("NORTH_MINI_POPULATED_HISTORY_CANDIDATE") != "cache_bf16_selected_decode",
                reason="set NORTH_MINI_POPULATED_HISTORY_CANDIDATE=cache_bf16_selected_decode to run this isolated probe",
            ),
        ),
        pytest.param(
            "bfp4_attention_selected_decode",
            id="bfp4_attention_selected_decode",
            marks=pytest.mark.skipif(
                os.environ.get("NORTH_MINI_POPULATED_HISTORY_CANDIDATE") != "bfp4_attention_selected_decode",
                reason=(
                    "set NORTH_MINI_POPULATED_HISTORY_CANDIDATE=bfp4_attention_selected_decode "
                    "to run this isolated probe"
                ),
            ),
        ),
    ],
)
def test_optimized_populated_history_isolated_candidate(mesh_device, candidate):
    _run_sliding_moe_populated_history_dynamic_trace_replay(
        mesh_device,
        _isolated_candidate(candidate),
        real_weights=True,
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_advertised_context_cache_capacity(mesh_device, batch):
    """Allocate the exact optimized cache volume promised by the context contract."""
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config, 0),
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=ADVERTISED_CONTEXT,
        candidate=_candidate(),
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    expected_blocks = batch * math.ceil(ADVERTISED_CONTEXT / decoder.page_size)
    expected_shape = (expected_blocks, config.num_key_value_heads, decoder.page_size, config.head_dim)
    assert tuple(key_cache.shape) == expected_shape
    assert tuple(value_cache.shape) == expected_shape
    assert key_cache.dtype == decoder.policy.cache_dtype == ttnn.bfloat8_b
    assert value_cache.dtype == decoder.policy.cache_dtype

    elements_per_cache = math.prod(expected_shape)
    assert (
        2 * elements_per_cache
        == batch
        * math.ceil(ADVERTISED_CONTEXT / decoder.page_size)
        * decoder.page_size
        * 2
        * config.num_key_value_heads
        * config.head_dim
    )
    assert 2 * elements_per_cache == batch * 512_000_000


@pytest.mark.skipif(
    os.environ.get("NORTH_MINI_NEAR_LIMIT_PREFILL") != "1",
    reason="set NORTH_MINI_NEAR_LIMIT_PREFILL=1 for the costly advertised-context execution probe",
)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_advertised_context_prefill_executes_with_finite_output(mesh_device):
    """Execute the optimized path at the exact advertised logical prefill length."""
    config = _config()
    sequence = ADVERTISED_CONTEXT
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config, 0),
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=sequence,
        candidate="default",
    )
    hidden = torch.zeros(1, 1, sequence, config.hidden_size, dtype=torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(
        _page_table(1, math.ceil(sequence / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    cos, sin = decoder.build_rope_rows(torch.arange(sequence), hf_config=config)
    output = decoder.prefill_forward(
        _to_tt(hidden, mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device),
        position_sin=_to_tt(sin, mesh_device),
    )
    output = ttnn.to_torch(output)
    assert output.shape == hidden.shape
    assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_optimized_sparse_weights_coexist_with_serving_cache_capacity(mesh_device, layer_idx):
    """Sparse layer weights and the exact batch-32 serving cache must coexist."""
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config, layer_idx),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=32,
        max_cache_len=ADVERTISED_CONTEXT,
        candidate="default",
    )

    # Small prefill and decode share the BFP8 down buffer. Large prefill owns
    # BF16 matrices because the official-weight serving-prefill PCC requires
    # them; the selected BFP8 cache leaves room for both representations.
    assert decoder.weights["expert_down_prefill"] is decoder.weights["expert_down"]
    expert_names = (
        "expert_gate_up",
        "expert_gate_prefill",
        "expert_up_prefill",
        "expert_down",
        "expert_down_prefill",
        "expert_gate_dense_prefill",
        "expert_up_dense_prefill",
        "expert_down_dense_prefill",
    )
    unique_expert_tensors = {id(decoder.weights[name]): decoder.weights[name] for name in expert_names}
    unit_elements = config.num_experts * config.hidden_size * config.intermediate_size
    assert sum(math.prod(tensor.shape) for tensor in unique_expert_tensors.values()) == 8 * unit_elements

    # BFP8 tiles occupy 1088 bytes for 1024 elements, including headers.  All
    # model dimensions are tile aligned, so this is exact rather than a logical
    # element estimate.
    bfp8_bytes_per_tile, elements_per_tile = 1088, 1024
    expert_bytes = 5 * unit_elements * bfp8_bytes_per_tile // elements_per_tile + 3 * unit_elements * 2
    nonexpert_weight_bytes = (
        (config.hidden_size * (config.num_attention_heads + 2 * config.num_key_value_heads) * config.head_dim)
        * bfp8_bytes_per_tile
        // elements_per_tile
        + (config.hidden_size * config.hidden_size) * bfp8_bytes_per_tile // elements_per_tile
        + config.hidden_size * 2
        + config.hidden_size * config.num_experts * 2
    )
    cache_elements = 32 * 512_000_000
    cache_bytes = cache_elements * bfp8_bytes_per_tile // elements_per_tile
    device_dram_bytes = 8 * 4_272_341_376
    assert cache_bytes + expert_bytes + nonexpert_weight_bytes <= device_dram_bytes
    bf16_cache_bytes = cache_elements * 2
    assert bf16_cache_bytes + expert_bytes + nonexpert_weight_bytes > device_dram_bytes

    key_cache, value_cache = decoder.create_paged_kv_cache()
    expected_blocks = 32 * math.ceil(ADVERTISED_CONTEXT / decoder.page_size)
    expected_shape = (expected_blocks, config.num_key_value_heads, decoder.page_size, config.head_dim)
    assert tuple(key_cache.shape) == expected_shape
    assert tuple(value_cache.shape) == expected_shape
    assert key_cache.dtype == value_cache.dtype == ttnn.bfloat8_b

    # Exercise the sparse decoder at the last advertised logical position
    # while the exact serving cache and all optimized weights coexist.  Cache
    # contents are intentionally unspecified here; semantic cache reads are
    # covered by the populated-history tests.
    positions = [ADVERTISED_CONTEXT - 1] * 32
    hidden = _randn(torch.Generator().manual_seed(24600 + layer_idx), 32, 1, config.hidden_size, scale=0.02)
    page_table = _to_tt(
        _page_table(32, math.ceil(ADVERTISED_CONTEXT / decoder.page_size)),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, positions)
    output = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    assert tuple(output.shape) == (1, 32, 1, config.hidden_size)
    ttnn.synchronize_device(mesh_device)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", REPRESENTATIVE_LAYERS)
def test_optimized_advertised_context_last_position_decode_executes(mesh_device, layer_idx):
    """Probe the last advertised logical position through every optimized layer kind."""
    config = _config()
    decoder = OptimizedDecoder.from_state_dict(
        _synthetic_state(config, layer_idx, sparse_weights=layer_idx != 0),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=ADVERTISED_CONTEXT,
        candidate=_candidate(),
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(ADVERTISED_CONTEXT / decoder.page_size)
    page_table = _to_tt(
        _page_table(1, blocks),
        mesh_device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    hidden = _randn(
        torch.Generator().manual_seed(28000 + layer_idx),
        1,
        1,
        config.hidden_size,
        scale=0.02,
    )
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [ADVERTISED_CONTEXT - 1])
    actual = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos if decoder.use_rope else None,
        position_sin=sin if decoder.use_rope else None,
    )
    output = ttnn.to_torch(actual)
    assert output.shape == (1, 1, 1, config.hidden_size)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [1, 32])
def test_optimized_real_weight_sparse_decode_and_repeated_trace(mesh_device, batch):
    config = _config()
    state = _real_layer_one_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
        candidate=_candidate(),
    )
    generator = torch.Generator().manual_seed(123 + batch)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    prefix = "model.layers.1."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(batch, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        batch, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    normalized_flat = normalized.reshape(batch, config.hidden_size)
    logits = F.linear(normalized_flat, state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(normalized_flat)
    for token in range(batch):
        for topk_index, expert in enumerate(experts[token].tolist()):
            token_hidden = normalized_flat[token : token + 1]
            gate = F.linear(token_hidden, state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"])
            up = F.linear(token_hidden, state[f"{prefix}mlp.experts.{expert}.up_proj.weight"])
            contribution = F.linear(
                F.silu(gate) * up,
                state[f"{prefix}mlp.experts.{expert}.down_proj.weight"],
            )
            moe[token] += contribution.squeeze(0) * scores[token, topk_index]
    print(f"optimized-real-selected-expert-union: {torch.unique(experts).numel()}")
    reference = hidden + attention + moe.reshape_as(hidden)

    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden.unsqueeze(0), mesh_device)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    actual = decoder.decode_forward(hidden_tt, **kwargs)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    try:
        captures = []
        for _ in range(5):
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            captures.append(ttnn.to_torch(actual).clone())
        assert all(torch.equal(captures[0], item) for item in captures[1:])
        _assert_pcc(f"optimized-real-sparse-trace-batch-{batch}", reference, captures[0].squeeze(0))
    finally:
        ttnn.release_trace(mesh_device, trace_id)
        ttnn.synchronize_device(mesh_device)
