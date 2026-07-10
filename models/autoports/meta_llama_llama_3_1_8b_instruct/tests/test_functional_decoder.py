# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import FunctionalDecoder

HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SNAPSHOT = Path(
    "/home/mvasiljevic/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/"
    "snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
)
EMITTED_BATCH = 32
LAYER_IDX = 0


@pytest.fixture(scope="module")
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


@pytest.fixture(scope="module")
def hf_config():
    return AutoConfig.from_pretrained(HF_MODEL, local_files_only=True)


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected = expected.float().flatten()
    actual = actual.float().flatten()
    expected = expected - expected.mean()
    actual = actual - actual.mean()
    denom = torch.sqrt(torch.sum(expected * expected) * torch.sum(actual * actual))
    return float(torch.sum(expected * actual) / denom)


def _synthetic_state_dict(config, layer_idx=LAYER_IDX):
    g = torch.Generator().manual_seed(20260708)
    prefix = f"model.layers.{layer_idx}."

    def randn(*shape):
        return (torch.randn(*shape, generator=g) * 0.02).to(torch.bfloat16)

    return {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "self_attn.k_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.v_proj.weight": randn(config.num_key_value_heads * config.head_dim, config.hidden_size),
        prefix + "self_attn.o_proj.weight": randn(config.hidden_size, config.hidden_size),
        prefix + "mlp.gate_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.up_proj.weight": randn(config.intermediate_size, config.hidden_size),
        prefix + "mlp.down_proj.weight": randn(config.hidden_size, config.intermediate_size),
    }


def _real_layer_state_dict(layer_idx=LAYER_IDX):
    index_path = MODEL_SNAPSHOT / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]
    keys = [
        f"model.layers.{layer_idx}.input_layernorm.weight",
        f"model.layers.{layer_idx}.post_attention_layernorm.weight",
        f"model.layers.{layer_idx}.self_attn.q_proj.weight",
        f"model.layers.{layer_idx}.self_attn.k_proj.weight",
        f"model.layers.{layer_idx}.self_attn.v_proj.weight",
        f"model.layers.{layer_idx}.self_attn.o_proj.weight",
        f"model.layers.{layer_idx}.mlp.gate_proj.weight",
        f"model.layers.{layer_idx}.mlp.up_proj.weight",
        f"model.layers.{layer_idx}.mlp.down_proj.weight",
    ]
    loaded = {}
    for shard in sorted({weight_map[key] for key in keys}):
        loaded.update(load_file(str(MODEL_SNAPSHOT / shard)))
    return {key: loaded[key].to(torch.bfloat16) for key in keys}


def _reference_layer(config, state_dict, layer_idx=LAYER_IDX):
    layer = LlamaDecoderLayer(config, layer_idx=layer_idx).to(dtype=torch.bfloat16)
    local = {key.removeprefix(f"model.layers.{layer_idx}."): value for key, value in state_dict.items()}
    layer.load_state_dict(local, strict=True)
    layer.eval()
    return layer


def _rope(config, hidden_states, seq_len):
    rotary = LlamaRotaryEmbedding(config).to(dtype=torch.bfloat16)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(hidden_states.shape[0], -1)
    return rotary(hidden_states, position_ids)


def _run_compare(config, state_dict, mesh_device, seq_len, pcc_threshold):
    torch.manual_seed(1234 + seq_len)
    hidden = torch.randn(EMITTED_BATCH, seq_len, config.hidden_size, dtype=torch.bfloat16)
    cos, sin = _rope(config, hidden, seq_len)

    ref = _reference_layer(config, state_dict)
    attention_mask = torch.full((EMITTED_BATCH, 1, seq_len, seq_len), torch.finfo(torch.float32).min)
    attention_mask = torch.triu(attention_mask, diagonal=1).to(torch.bfloat16)
    with torch.no_grad():
        expected = ref(hidden, attention_mask=attention_mask, position_embeddings=(cos, sin))

    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
    )
    tt_hidden = FunctionalDecoder.prepare_inputs(hidden, mesh_device)
    tt_cos, tt_sin = FunctionalDecoder.prepare_rope(cos, sin, mesh_device)
    actual = ttnn.to_torch(decoder.prefill_forward(tt_hidden, position_cos=tt_cos, position_sin=tt_sin)).squeeze(0)
    measured = _pcc(expected, actual)
    assert measured >= pcc_threshold, f"PCC {measured:.6f} below required {pcc_threshold}"
    return measured


@pytest.mark.parametrize("seq_len", [1, 8])
def test_synthetic_weight_prefill_real_shapes(hf_config, mesh_device, seq_len):
    measured = _run_compare(hf_config, _synthetic_state_dict(hf_config), mesh_device, seq_len, 0.99)
    print(f"SYNTHETIC_PREFILL_SEQ_{seq_len}_PCC={measured:.6f}")


@pytest.mark.parametrize("seq_len", [4])
def test_real_weight_single_layer_prefill_pcc(hf_config, mesh_device, seq_len):
    measured = _run_compare(hf_config, _real_layer_state_dict(), mesh_device, seq_len, 0.99)
    print(f"REAL_WEIGHT_PREFILL_SEQ_{seq_len}_PCC={measured:.6f}")


def test_decode_forward_documents_pending_emit(hf_config, mesh_device, expect_error):
    decoder = FunctionalDecoder.from_state_dict(
        _synthetic_state_dict(hf_config),
        hf_config=hf_config,
        layer_idx=LAYER_IDX,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH,
    )
    with expect_error(NotImplementedError, "pending emitted-decode forge version"):
        decoder.decode_forward(None)


def test_prefill_forward_has_no_runtime_host_fallbacks():
    source = inspect.getsource(FunctionalDecoder.prefill_forward)
    forbidden = ("torch", "from_torch", "to_torch")
    assert not any(token in source for token in forbidden)
