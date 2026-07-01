# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID, PENDING_DECODE_MESSAGE, FunctionalDecoder
from models.common.utility_functions import comp_pcc

SMALL_SEQ_LEN = 16
LARGE_SEQ_LEN = 64


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(HF_MODEL_ID)


@pytest.fixture()
def mesh_device():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def _tt_tensor(tensor, mesh_device):
    return ttnn.from_torch(
        tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _synthetic_state_dict(hf_config, layer_idx=0):
    torch.manual_seed(20260701)
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
        f"{prefix}.self_attn.q_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.self_attn.k_norm.weight": torch.ones(hf_config.head_dim, dtype=torch.bfloat16),
        f"{prefix}.mlp.gate_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.up_proj.weight": torch.randn(inter, hidden, dtype=torch.bfloat16) * scale,
        f"{prefix}.mlp.down_proj.weight": torch.randn(hidden, inter, dtype=torch.bfloat16) * scale,
        f"{prefix}.input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }


def _causal_mask(seq_len):
    mask = torch.full((1, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(mask, diagonal=1)


def _run_reference_layer(layer, rotary_emb, hidden_states, seq_len):
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


def _run_tt_layer(decoder, hidden_states, mesh_device):
    tt_input = _tt_tensor(hidden_states.reshape(1, 1, hidden_states.shape[1], hidden_states.shape[2]), mesh_device)
    tt_output = decoder.prefill_forward(tt_input)
    return ttnn.to_torch(tt_output).reshape_as(hidden_states).to(torch.float32)


def _assert_pcc(reference, actual, threshold):
    passing, pcc = comp_pcc(reference.to(torch.float32), actual.to(torch.float32), threshold)
    print(f"PCC={pcc}")
    assert passing, f"PCC {pcc} below threshold {threshold}"
    return float(pcc)


@pytest.mark.parametrize("seq_len", [SMALL_SEQ_LEN, LARGE_SEQ_LEN])
def test_synthetic_weight_prefill_matches_hf_layer(hf_config, mesh_device, seq_len):
    layer = Qwen3DecoderLayer(hf_config, layer_idx=0).to(dtype=torch.bfloat16).eval()
    state_dict = _synthetic_state_dict(hf_config)
    layer.load_state_dict({key.removeprefix("model.layers.0."): value for key, value in state_dict.items()})
    rotary_emb = Qwen3RotaryEmbedding(hf_config)
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = FunctionalDecoder.from_state_dict(
        state_dict, hf_config=hf_config, layer_idx=0, mesh_device=mesh_device, max_seq_len=seq_len
    )

    reference = _run_reference_layer(layer, rotary_emb, hidden_states, seq_len)
    actual = _run_tt_layer(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.99)


def test_real_weight_single_layer_prefill_matches_hf(hf_config, mesh_device):
    seq_len = SMALL_SEQ_LEN
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16).eval()
    state_dict = model.state_dict()
    layer = model.model.layers[0]
    rotary_emb = model.model.rotary_emb
    hidden_states = torch.randn(1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    decoder = FunctionalDecoder.from_state_dict(
        state_dict, hf_config=hf_config, layer_idx=0, mesh_device=mesh_device, max_seq_len=seq_len
    )

    reference = _run_reference_layer(layer, rotary_emb, hidden_states, seq_len)
    actual = _run_tt_layer(decoder, hidden_states, mesh_device)
    _assert_pcc(reference, actual, 0.99)


def test_decode_forward_documents_pending_path(hf_config, mesh_device, expect_error):
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state_dict(hf_config),
        hf_config=hf_config,
        layer_idx=0,
        mesh_device=mesh_device,
        max_seq_len=SMALL_SEQ_LEN,
    )
    with expect_error(NotImplementedError, PENDING_DECODE_MESSAGE):
        decoder.decode_forward(None)


def test_prefill_runtime_has_no_host_fallback():
    source = inspect.getsource(FunctionalDecoder.prefill_forward)
    forbidden = ("torch", "from_torch", "to_torch")
    hits = [term for term in forbidden if term in source]
    assert hits == []
