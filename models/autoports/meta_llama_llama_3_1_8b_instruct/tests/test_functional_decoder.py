# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig
from transformers.cache_utils import StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_DECODE_CACHE_LEN,
    HF_MODEL_ID,
    FunctionalDecoder,
    build_decode_mask,
    build_rope_tables,
    build_update_indices,
)
from models.common.utility_functions import comp_pcc

SMALL_SEQ_LEN = 16
NON_ALIGNED_SEQ_LEN = 17
LARGE_SEQ_LEN = 64
DECODE_CACHE_POSITION = 7

TARGET_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": [128001, 128008, 128009],
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 131072,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1.0e-5,
    "rope_scaling": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3",
    },
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "vocab_size": 128256,
}


@pytest.fixture(scope="module")
def hf_config():
    local_path = os.environ.get("LLAMA31_8B_INSTRUCT_HF_PATH")
    if local_path:
        return AutoConfig.from_pretrained(local_path, local_files_only=Path(local_path).is_dir())
    try:
        return AutoConfig.from_pretrained(HF_MODEL_ID, local_files_only=True)
    except Exception:
        return LlamaConfig.from_dict(TARGET_CONFIG)


@pytest.fixture()
def mesh_device():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def _tt_tensor(tensor: torch.Tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _synthetic_state_dict(hf_config, layer_idx=0):
    torch.manual_seed(20260713)
    hidden = hf_config.hidden_size
    q_width = hf_config.num_attention_heads * hf_config.head_dim
    kv_width = hf_config.num_key_value_heads * hf_config.head_dim
    inter = hf_config.intermediate_size
    prefix = f"model.layers.{layer_idx}"
    scale = 0.02
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.randn(q_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv_width, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(hidden, q_width, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.bfloat16) * scale,
        f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _layer_local_state_dict(state_dict, layer_idx=0):
    prefix = f"model.layers.{layer_idx}."
    return {key.removeprefix(prefix): value for key, value in state_dict.items() if key.startswith(prefix)}


def _causal_mask(seq_len):
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def _decode_mask(batch, cache_position, cache_len):
    mask = torch.full((1, 1, batch, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask[:, :, :, : cache_position + 1] = 0.0
    return mask


def _hf_static_prefill_mask(seq_len, cache_len):
    mask = torch.full((1, 1, seq_len, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    for row in range(seq_len):
        mask[:, :, row, : row + 1] = 0.0
    return mask


def _hf_static_decode_mask(cache_position, cache_len):
    mask = torch.full((1, 1, 1, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask[:, :, :, : cache_position + 1] = 0.0
    return mask


def _run_reference_prefill(layer, rotary_emb, hidden_states, seq_len):
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    with torch.no_grad():
        return layer(
            hidden_states,
            attention_mask=_causal_mask(seq_len),
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )


def _run_reference_decode(layer, rotary_emb, hf_config, prefix_hidden, decode_hidden):
    cache = StaticCache(config=hf_config, max_cache_len=EMITTED_DECODE_CACHE_LEN)
    cache.early_initialization(
        batch_size=EMITTED_BATCH_SIZE,
        num_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    prefix_len = prefix_hidden.shape[1]
    prefix_positions = torch.arange(prefix_len, dtype=torch.long)
    with torch.no_grad():
        layer(
            prefix_hidden,
            attention_mask=_hf_static_prefill_mask(prefix_len, EMITTED_DECODE_CACHE_LEN),
            position_ids=prefix_positions.unsqueeze(0),
            position_embeddings=rotary_emb(prefix_hidden, prefix_positions.unsqueeze(0)),
            past_key_values=cache,
            use_cache=True,
            cache_position=prefix_positions,
        )
        decode_position = torch.tensor([prefix_len], dtype=torch.long)
        reference = layer(
            decode_hidden,
            attention_mask=_hf_static_decode_mask(prefix_len, EMITTED_DECODE_CACHE_LEN),
            position_ids=decode_position.unsqueeze(0),
            position_embeddings=rotary_emb(decode_hidden, decode_position.unsqueeze(0)),
            past_key_values=cache,
            use_cache=True,
            cache_position=decode_position,
        )
    return reference, cache.layers[0].keys, cache.layers[0].values


def _run_tt_prefill(decoder, hidden_states, mesh_device):
    tt_input = _tt_tensor(
        hidden_states.reshape(1, EMITTED_BATCH_SIZE, hidden_states.shape[1], hidden_states.shape[2]), mesh_device
    )
    tt_output = decoder.prefill_forward(tt_input)
    return ttnn.to_torch(tt_output).reshape_as(hidden_states).to(torch.float32)


def _assert_pcc(reference, actual, threshold):
    passing, pcc = comp_pcc(reference.to(torch.float32), actual.to(torch.float32), threshold)
    print(f"PCC={pcc}")
    assert passing, f"PCC {pcc} below threshold {threshold}"
    return float(pcc)


@pytest.mark.parametrize("seq_len", [SMALL_SEQ_LEN, NON_ALIGNED_SEQ_LEN, LARGE_SEQ_LEN])
def test_synthetic_weight_prefill_matches_hf_layer_at_emitted_batch(hf_config, mesh_device, seq_len):
    layer = LlamaDecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict(_layer_local_state_dict(state_dict))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    hidden_states = torch.randn(EMITTED_BATCH_SIZE, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=LARGE_SEQ_LEN,
    )

    reference = _run_reference_prefill(layer, rotary_emb, hidden_states, seq_len)
    actual = _run_tt_prefill(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.99)


def _load_real_model_or_skip():
    model_source = os.environ.get("LLAMA31_8B_INSTRUCT_HF_PATH", HF_MODEL_ID)
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype=torch.bfloat16,
            local_files_only=Path(model_source).is_dir(),
        ).eval()
    except Exception as exc:
        pytest.skip(
            f"real HF weights unavailable for {HF_MODEL_ID}; set LLAMA31_8B_INSTRUCT_HF_PATH to a "
            f"canonical local HF checkout or provide HF auth: {exc}"
        )


def test_real_weight_single_layer_prefill_matches_hf_at_emitted_batch():
    seq_len = SMALL_SEQ_LEN
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    rotary_emb = model.model.rotary_emb
    torch.manual_seed(20260713)
    hidden_states = torch.randn(EMITTED_BATCH_SIZE, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            batch=EMITTED_BATCH_SIZE,
            max_seq_len=seq_len,
        )

        reference = _run_reference_prefill(model.model.layers[0], rotary_emb, hidden_states, seq_len)
        actual = _run_tt_prefill(decoder, hidden_states, mesh)
        _assert_pcc(reference, actual, 0.99)
    finally:
        ttnn.close_mesh_device(mesh)


def test_synthetic_weight_decode_matches_hf_one_step_at_emitted_batch(hf_config, mesh_device):
    layer = LlamaDecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict(_layer_local_state_dict(state_dict))
    rotary_emb = LlamaRotaryEmbedding(hf_config)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache = _run_reference_decode(
        layer, rotary_emb, hf_config, prefix_hidden, decode_hidden
    )

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        max_seq_len=EMITTED_DECODE_CACHE_LEN,
    )
    position_cos, position_sin = build_rope_tables(
        hf_config,
        1,
        mesh_device,
        start_pos=DECODE_CACHE_POSITION,
    )
    tt_output = decoder.decode_forward(
        _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh_device),
        key_cache=_tt_tensor(key_cache, mesh_device),
        value_cache=_tt_tensor(value_cache, mesh_device),
        update_idxs_tensor=build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh_device),
        position_cos=position_cos,
        position_sin=position_sin,
        attention_mask=build_decode_mask(
            EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh_device
        ),
    )
    actual = ttnn.to_torch(tt_output).reshape_as(reference).to(torch.float32)
    _assert_pcc(reference, actual, 0.99)


def test_real_weight_single_layer_decode_matches_hf_at_emitted_batch():
    model = _load_real_model_or_skip()
    hf_config = model.config
    state_dict = model.state_dict()
    torch.manual_seed(20260714)
    prefix_hidden = torch.randn(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, hf_config.hidden_size, dtype=torch.bfloat16)
    decode_hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    reference, key_cache, value_cache = _run_reference_decode(
        model.model.layers[0],
        model.model.rotary_emb,
        hf_config,
        prefix_hidden,
        decode_hidden,
    )

    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        decoder = FunctionalDecoder.from_state_dict(
            state_dict,
            hf_config=hf_config,
            layer_idx=0,
            mesh_device=mesh,
            batch=EMITTED_BATCH_SIZE,
            max_seq_len=EMITTED_DECODE_CACHE_LEN,
        )
        position_cos, position_sin = build_rope_tables(
            hf_config,
            1,
            mesh,
            start_pos=DECODE_CACHE_POSITION,
        )
        tt_output = decoder.decode_forward(
            _tt_tensor(decode_hidden.reshape(1, EMITTED_BATCH_SIZE, 1, hf_config.hidden_size), mesh),
            key_cache=_tt_tensor(key_cache, mesh),
            value_cache=_tt_tensor(value_cache, mesh),
            update_idxs_tensor=build_update_indices(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, mesh),
            position_cos=position_cos,
            position_sin=position_sin,
            attention_mask=build_decode_mask(EMITTED_BATCH_SIZE, DECODE_CACHE_POSITION, EMITTED_DECODE_CACHE_LEN, mesh),
        )
        actual = ttnn.to_torch(tt_output).reshape_as(reference).to(torch.float32)
        _assert_pcc(reference, actual, 0.99)
    finally:
        ttnn.close_mesh_device(mesh)


def test_runtime_forwards_have_no_host_fallback():
    forbidden = ("torch", "from_torch", "to_torch")
    methods = [
        FunctionalDecoder._split_emit_qkv,
        FunctionalDecoder._prepare_qkv,
        FunctionalDecoder._attention_mlp,
        FunctionalDecoder.prefill_forward,
        FunctionalDecoder.decode_forward,
    ]
    hits = []
    for method in methods:
        source = inspect.getsource(method)
        hits.extend(f"{method.__name__}:{term}" for term in forbidden if term in source)
    assert hits == []
