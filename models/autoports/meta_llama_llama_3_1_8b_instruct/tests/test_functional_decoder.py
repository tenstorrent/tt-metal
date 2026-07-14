# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_CACHE_LEN,
    MODEL_ID,
    PREFILL_NOT_EMITTED_MESSAGE,
    FunctionalDecoder,
    build_decode_attention_mask,
    build_decode_rope,
    build_decode_rope_torch,
    build_llama31_inv_freq,
)

AUTOport_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = AUTOport_DIR / "doc" / "functional_decoder"
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def hf_config():
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0, physical_device_ids=[0])
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def test_runtime_forward_sources_do_not_call_host_fallback_boundaries():
    runtime_source = "\n".join(
        [
            inspect.getsource(FunctionalDecoder.prefill_forward),
            inspect.getsource(FunctionalDecoder.decode_forward),
            inspect.getsource(FunctionalDecoder.forward),
        ]
    )
    for token in ("torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch"):
        assert token not in runtime_source


def test_prefill_stub_documents_emit_absence(hf_config, expect_error):
    decoder = FunctionalDecoder.__new__(FunctionalDecoder)
    with expect_error(NotImplementedError, "did not ship a prefill graph"):
        decoder.prefill_forward(None)
    assert "single-token decode graph" in PREFILL_NOT_EMITTED_MESSAGE


def test_decode_rope_matches_hf_scaled_inv_freq(hf_config):
    cache_position = EMITTED_CACHE_LEN - 1
    rotary = LlamaRotaryEmbedding(hf_config)
    position_ids = torch.full((EMITTED_BATCH_SIZE, 1), cache_position, dtype=torch.long)
    hidden = torch.zeros(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16)
    hf_cos, hf_sin = rotary(hidden, position_ids)
    tt_cos, tt_sin = build_decode_rope_torch(hf_config, cache_position)

    assert torch.equal(build_llama31_inv_freq(hf_config), rotary.inv_freq.detach().cpu())
    expected_cos = tt_cos.reshape(1, 1, hf_config.head_dim).expand_as(hf_cos)
    expected_sin = tt_sin.reshape(1, 1, hf_config.head_dim).expand_as(hf_sin)
    assert torch.equal(expected_cos.to(torch.bfloat16), hf_cos)
    assert torch.equal(expected_sin.to(torch.bfloat16), hf_sin)


def test_decode_forward_synthetic_weights(mesh_device, hf_config):
    result = _run_decode_case(mesh_device, hf_config, _synthetic_state_dict(hf_config), "synthetic", scale=0.02)
    assert result["pcc"] >= PCC_THRESHOLD
    _write_result("test_results_decode_synthetic.json", result)


def test_decode_forward_real_weights(mesh_device, hf_config):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    model.eval()
    result = _run_decode_case(mesh_device, hf_config, model.state_dict(), "real", scale=0.01)
    assert result["pcc"] >= PCC_THRESHOLD
    _write_result("test_results_decode_real.json", result)


def _run_decode_case(mesh_device, hf_config, state_dict, weight_label: str, *, scale: float):
    torch.manual_seed(20260714 if weight_label == "real" else 20260715)
    layer_idx = 0
    cache_position = EMITTED_CACHE_LEN - 1
    hidden = torch.randn(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size, dtype=torch.bfloat16) * scale
    prefill_hidden = (
        torch.randn(EMITTED_BATCH_SIZE, EMITTED_CACHE_LEN - 1, hf_config.hidden_size, dtype=torch.bfloat16) * scale
    )

    hf_layer = LlamaDecoderLayer(hf_config, layer_idx=layer_idx).to(dtype=torch.bfloat16).eval()
    hf_layer.load_state_dict(_layer_state_dict(state_dict, layer_idx), strict=True)
    rotary = LlamaRotaryEmbedding(hf_config)
    cache = _prefill_hf_cache(hf_config, hf_layer, rotary, prefill_hidden, layer_idx)

    with torch.no_grad():
        position_ids = torch.full((EMITTED_BATCH_SIZE, 1), cache_position, dtype=torch.long)
        position_embeddings = rotary(hidden, position_ids)
        hf_out = hf_layer(
            hidden,
            attention_mask=_hf_decode_mask(EMITTED_BATCH_SIZE, EMITTED_CACHE_LEN, cache_position),
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=position_embeddings,
        )

    key_cache_pt = cache.layers[layer_idx].keys.detach().contiguous()
    value_cache_pt = cache.layers[layer_idx].values.detach().contiguous()
    decoder = FunctionalDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        cache_len=EMITTED_CACHE_LEN,
    )
    tt_hidden = _tt(hidden.reshape(1, 1, EMITTED_BATCH_SIZE, hf_config.hidden_size), mesh_device)
    tt_key_cache = _tt(key_cache_pt, mesh_device)
    tt_value_cache = _tt(value_cache_pt, mesh_device)
    tt_pos = _tt(
        torch.tensor([cache_position], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    cos, sin = build_decode_rope(hf_config, cache_position, mesh_device)
    tt_mask = build_decode_attention_mask(cache_position, EMITTED_CACHE_LEN, mesh_device)

    tt_out = decoder.decode_forward(
        tt_hidden,
        key_cache=tt_key_cache,
        value_cache=tt_value_cache,
        cache_position=tt_pos,
        cos=cos,
        sin=sin,
        attention_mask=tt_mask,
    )
    tt_out_torch = ttnn.to_torch(tt_out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
    pcc = _pcc(hf_out.float(), tt_out_torch.float())
    return {
        "test": f"test_decode_forward_{weight_label}_weights",
        "weights": weight_label,
        "batch": EMITTED_BATCH_SIZE,
        "cache_len": EMITTED_CACHE_LEN,
        "cache_position": cache_position,
        "pcc": pcc,
        "pcc_threshold": PCC_THRESHOLD,
    }


def _prefill_hf_cache(hf_config, hf_layer, rotary, hidden, layer_idx: int):
    cache = StaticCache(
        config=hf_config,
        max_batch_size=EMITTED_BATCH_SIZE,
        max_cache_len=EMITTED_CACHE_LEN,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache.early_initialization(
        batch_size=EMITTED_BATCH_SIZE,
        num_heads=hf_config.num_key_value_heads,
        head_dim=hf_config.head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    with torch.no_grad():
        pos = torch.arange(0, EMITTED_CACHE_LEN - 1, dtype=torch.long)
        position_ids = pos.unsqueeze(0).expand(EMITTED_BATCH_SIZE, -1)
        hf_layer(
            hidden,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            position_embeddings=rotary(hidden, position_ids),
        )
    return cache


def _hf_decode_mask(batch: int, cache_len: int, cache_position: int) -> torch.Tensor:
    mask = torch.full((batch, 1, 1, cache_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    mask[:, :, :, : cache_position + 1] = 0
    return mask


def _synthetic_state_dict(hf_config):
    torch.manual_seed(2026)
    h = hf_config.hidden_size
    kv = hf_config.num_key_value_heads * hf_config.head_dim
    return {
        "model.layers.0.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.self_attn.k_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.self_attn.v_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.self_attn.o_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.mlp.gate_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.mlp.up_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        "model.layers.0.mlp.down_proj.weight": torch.randn(h, hf_config.intermediate_size, dtype=torch.bfloat16) * 0.01,
    }


def _layer_state_dict(state_dict, layer_idx: int):
    prefix = f"model.layers.{layer_idx}."
    out = {}
    for key, tensor in state_dict.items():
        if key.startswith(prefix):
            out[key[len(prefix) :]] = tensor.detach().cpu()
    return out


def _tt(tensor, mesh_device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _pcc(expected, actual) -> float:
    x = expected.reshape(-1).to(torch.float32)
    y = actual.reshape(-1).to(torch.float32)
    if torch.allclose(x, y):
        return 1.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy))
    if denom == 0:
        return 0.0
    return float(torch.sum(vx * vy) / denom)


def _write_result(name: str, data: dict):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / name).write_text(json.dumps(data, indent=2) + "\n")
