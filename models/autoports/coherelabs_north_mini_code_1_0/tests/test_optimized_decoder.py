# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Correctness floor for the North-Mini optimized decoder path."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

import ttnn
from models.autoports.coherelabs_north_mini_code_1_0.tests.test_functional_decoder import (
    REAL_REVISION,
    _assert_pcc,
    _config,
    _decode_inputs,
    _dense_reference,
    _page_table,
    _randn,
    _real_layer_one_state,
    _sparse_moe_reference,
    _synthetic_state,
    _to_host_tt,
    _to_tt,
)
from models.autoports.coherelabs_north_mini_code_1_0.tt.functional_decoder import FunctionalDecoder
from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder


def _real_layer_zero_state():
    snapshot = Path("/huggingface/hub/models--CohereLabs--North-Mini-Code-1.0") / "snapshots" / REAL_REVISION
    shard = snapshot / "model-00001-of-00049.safetensors"
    if not shard.is_file():
        pytest.skip("North-Mini official layer-0 shard is not cached")
    prefix = "model.layers.0."
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(prefix)}


def test_optimized_path_is_materially_overridden_and_runtime_clean():
    assert issubclass(OptimizedDecoder, FunctionalDecoder)
    assert OptimizedDecoder._dense_mlp is not FunctionalDecoder._dense_mlp
    source = "\n".join(
        inspect.getsource(method)
        for method in (
            OptimizedDecoder._dense_mlp,
            OptimizedDecoder._sparse_moe,
            OptimizedDecoder._sparse_moe_chunk,
            OptimizedDecoder._sparse_moe_decode,
            OptimizedDecoder.prefill_forward,
            OptimizedDecoder.decode_forward,
        )
    )
    for forbidden in ("import torch", "from_torch", "to_torch", "tilize", "untilize"):
        assert forbidden not in source
    assert 'self.weights["gate_up"]' in inspect.getsource(OptimizedDecoder._dense_mlp)
    assert 'self.weights["gate_proj"]' not in inspect.getsource(OptimizedDecoder._dense_mlp)
    assert "super()._sparse_moe" not in source


@pytest.mark.parametrize("out_subblock_w", [0, 2, 3])
def test_sparse_program_rejects_invalid_output_subblocks_before_device(out_subblock_w, expect_error):
    with expect_error(ValueError, r"must be positive, no greater than, and divide out_block_w \(1\)"):
        OptimizedDecoder._sparse_matmul_program(
            None,
            n=3072,
            k=2048,
            cores=11,
            in0_block_w=32,
            out_subblock_w=out_subblock_w,
        )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [33, 65])
def test_optimized_dense_non_aligned_prefill_matches_reference(mesh_device, seq_len):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state, hf_config=config, layer_idx=0, mesh_device=mesh_device, batch=1, max_cache_len=96
    )
    generator = torch.Generator().manual_seed(21000 + seq_len)
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
@pytest.mark.parametrize(
    "dense_decode_variant",
    [
        "packed_interleaved",
        "advisor_dram_sharded",
        "advisor_dram_sharded_bfp4_gate_up",
        "advisor_dram_sharded_bfp4_all",
        "separate_dram_sharded_bfp4",
    ],
)
def test_optimized_dense_paged_decode_trace_replay_matches_reference(mesh_device, dense_decode_variant):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=64,
        dense_decode_variant=dense_decode_variant,
    )
    generator = torch.Generator().manual_seed(22001)
    hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(hidden, torch.zeros(1, 1, dtype=torch.long), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 2), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
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
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-dense-traced-decode-{dense_decode_variant}",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "dense_decode_variant",
    [
        "advisor_dram_sharded",
        "advisor_dram_sharded_bfp4_gate_up",
        "advisor_dram_sharded_bfp4_all",
    ],
)
def test_real_weight_dense_decode_precision_candidates(mesh_device, dense_decode_variant):
    config = _config()
    state = _real_layer_zero_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        dense_decode_variant=dense_decode_variant,
    )
    generator = torch.Generator().manual_seed(24001)
    hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    reference, _ = _dense_reference(hidden, torch.zeros(1, 1, dtype=torch.long), state, config)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
    actual = decoder.decode_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=current,
        position_cos=cos,
        position_sin=sin,
    )
    _assert_pcc(
        f"real-layer0-{dense_decode_variant}",
        reference,
        ttnn.to_torch(actual).squeeze(0),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("layer_idx", [1, 4])
def test_optimized_routed_sparse_decode_trace_matches_active_expert_reference(mesh_device, layer_idx):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        sparse_weight_dtype="bfp8",
    )
    generator = torch.Generator().manual_seed(25000 + layer_idx)
    hidden = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = hidden + reference_moe.reshape_as(hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
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
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-routed-sparse-layer-{layer_idx}",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize(
    "layer_idx,seq_len,selected_tokens",
    [(1, 1025, [0, 1023, 1024]), (4, 33, [0, 16, 32])],
)
def test_optimized_sparse_non_aligned_prefill_matches_reference(mesh_device, layer_idx, seq_len, selected_tokens):
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        state[prefix + f"self_attn.{projection}.weight"].zero_()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=seq_len,
    )
    generator = torch.Generator().manual_seed(16000 + layer_idx + seq_len)
    hidden = _randn(generator, 1, seq_len, config.hidden_size, scale=0.02)
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx, flat_indices=selected_tokens)
    reference = hidden[:, selected_tokens] + reference_moe.reshape(1, len(selected_tokens), -1)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = (seq_len + decoder.page_size - 1) // decoder.page_size
    page_table = _to_tt(_page_table(1, blocks), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cos, sin = decoder.build_rope_rows(torch.arange(seq_len), hf_config=config)
    actual = decoder.prefill_forward(
        _to_tt(hidden.unsqueeze(0), mesh_device),
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        position_cos=_to_tt(cos, mesh_device) if decoder.use_rope else None,
        position_sin=_to_tt(sin, mesh_device) if decoder.use_rope else None,
    )
    _assert_pcc(
        f"optimized-sparse-prefill-layer-{layer_idx}",
        reference,
        ttnn.to_torch(actual).squeeze(0)[:, selected_tokens],
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("sparse_weight_dtype", ["bfp8", "bfp4"])
def test_real_weight_optimized_sliding_moe_decode_matches_reference(mesh_device, sparse_weight_dtype):
    config = _config()
    state = _real_layer_one_state()
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=1,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        sparse_weight_dtype=sparse_weight_dtype,
    )
    hidden = _randn(torch.Generator().manual_seed(123), 1, 1, config.hidden_size, scale=0.02)
    prefix = "model.layers.1."
    normalized = (hidden.float() * torch.rsqrt(hidden.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps)).to(
        torch.bfloat16
    )
    normalized *= state[prefix + "input_layernorm.weight"]
    value = F.linear(normalized, state[prefix + "self_attn.v_proj.weight"])
    value = value.view(1, 1, config.num_key_value_heads, config.head_dim)
    attention = value.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, dim=2).reshape(
        1, 1, -1
    )
    attention = F.linear(attention, state[prefix + "self_attn.o_proj.weight"])
    logits = F.linear(normalized.reshape(1, -1), state[prefix + "mlp.gate.weight"])
    scores, experts = torch.topk(logits, config.num_experts_per_tok, dim=-1)
    scores = torch.sigmoid(scores)
    moe = torch.zeros_like(normalized.reshape(1, -1))
    for topk_index, expert in enumerate(experts[0].tolist()):
        gate = F.linear(normalized.reshape(1, -1), state[f"{prefix}mlp.experts.{expert}.gate_proj.weight"])
        up = F.linear(normalized.reshape(1, -1), state[f"{prefix}mlp.experts.{expert}.up_proj.weight"])
        moe += (
            F.linear(
                F.silu(gate) * up,
                state[f"{prefix}mlp.experts.{expert}.down_proj.weight"],
            )
            * scores[0, topk_index]
        )
    reference = hidden + attention + moe.reshape_as(hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0])
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
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"real-layer1-optimized-decode-{sparse_weight_dtype}",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_optimized_sparse_serving_batch_trace_matches_reference(mesh_device):
    batch = 32
    layer_idx = 1
    config = _config()
    state = _synthetic_state(config, layer_idx, sparse_weights=True)
    prefix = f"model.layers.{layer_idx}."
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=batch,
        max_cache_len=32,
    )
    generator = torch.Generator().manual_seed(18000 + layer_idx + batch)
    hidden_a = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
    hidden = _randn(generator, batch, 1, config.hidden_size, scale=0.02)
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
    reference_moe, _ = _sparse_moe_reference(normalized, state, config, layer_idx)
    reference = hidden + attention + reference_moe.reshape_as(hidden)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(batch, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    current, cos, sin = _decode_inputs(decoder, config, mesh_device, [0] * batch)
    hidden_tt = _to_tt(hidden_a.unsqueeze(0), mesh_device)
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
        ttnn.copy_host_to_device_tensor(_to_host_tt(hidden.unsqueeze(0), mesh_device), hidden_tt)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        _assert_pcc(
            f"optimized-serving-batch-layer-{layer_idx}",
            reference,
            ttnn.to_torch(actual).squeeze(0),
        )
    finally:
        ttnn.release_trace(mesh_device, trace_id)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("kv_cache_dtype", ["bf16", "bfp8"])
def test_optimized_multi_position_paged_cache_and_determinism(mesh_device, kv_cache_dtype):
    config = _config()
    state = _synthetic_state(config, 0)
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=1,
        max_cache_len=32,
        kv_cache_dtype=kv_cache_dtype,
    )
    generator = torch.Generator().manual_seed(28000)
    hidden_a = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    hidden_b = _randn(generator, 1, 1, config.hidden_size, scale=0.02)
    _, cache = _dense_reference(hidden_a, torch.zeros(1, 1, dtype=torch.long), state, config)
    reference_b, _ = _dense_reference(hidden_b, torch.ones(1, 1, dtype=torch.long), state, config, cache=cache)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = _to_tt(_page_table(1, 1), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    def run(hidden, position):
        current, cos, sin = _decode_inputs(decoder, config, mesh_device, [position])
        return decoder.decode_forward(
            _to_tt(hidden.unsqueeze(0), mesh_device),
            key_cache=key_cache,
            value_cache=value_cache,
            page_table=page_table,
            current_positions=current,
            position_cos=cos,
            position_sin=sin,
        )

    run(hidden_a, 0)
    actual_b = run(hidden_b, 1)
    actual_b_repeat = run(hidden_b, 1)
    actual_host = ttnn.to_torch(actual_b).squeeze(0)
    repeated_host = ttnn.to_torch(actual_b_repeat).squeeze(0)
    _assert_pcc(f"optimized-cache-{kv_cache_dtype}-position-1", reference_b, actual_host)
    torch.testing.assert_close(actual_host, repeated_host, rtol=0, atol=0)
