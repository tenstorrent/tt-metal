# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost
from transformers import AutoConfig, AutoModelForCausalLM, StaticCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    EMITTED_BATCH_SIZE,
    EMITTED_CACHE_LEN,
    MODEL_ID,
    PREFILL_NOT_EMITTED_MESSAGE,
    build_decode_attention_mask,
    build_decode_rope,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    OptimizedDecoderPolicy,
)

AUTOPORT_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = AUTOPORT_DIR / "doc" / "optimized_decoder"
PCC_THRESHOLD = 0.99


@pytest.fixture(scope="module")
def hf_config():
    config = AutoConfig.from_pretrained(MODEL_ID, local_files_only=True)
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def mesh_device():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=8 << 20, physical_device_ids=[0])
    try:
        yield mesh
    finally:
        ttnn.close_mesh_device(mesh)


def test_runtime_forward_sources_do_not_call_functional_or_host_fallback_boundaries():
    runtime_source = "\n".join(
        [
            inspect.getsource(OptimizedDecoder.prefill_forward),
            inspect.getsource(OptimizedDecoder.decode_forward),
            inspect.getsource(OptimizedDecoder.forward),
            inspect.getsource(OptimizedDecoder._matmul),
            inspect.getsource(OptimizedDecoder._create_qkv_heads),
        ]
    )
    for token in ("FunctionalDecoder", "torch.", "import torch", "ttnn.from_torch", "ttnn.to_torch"):
        assert token not in runtime_source


def test_prefill_stub_preserves_functional_contract(expect_error):
    decoder = OptimizedDecoder.__new__(OptimizedDecoder)
    with expect_error(NotImplementedError, "did not ship a prefill graph"):
        decoder.prefill_forward(None)
    assert "single-token decode graph" in PREFILL_NOT_EMITTED_MESSAGE


def test_optimized_decode_synthetic_layer0(mesh_device, hf_config):
    result = _run_decode_case(mesh_device, hf_config, _synthetic_state_dict(hf_config, layer_idx=0), 0, "synthetic")
    assert result["pcc"] >= PCC_THRESHOLD
    _write_result("test_results_decode_synthetic_layer0.json", result)


def test_optimized_decode_synthetic_layer31_reorder(mesh_device, hf_config):
    result = _run_decode_case(mesh_device, hf_config, _synthetic_state_dict(hf_config, layer_idx=31), 31, "synthetic")
    assert result["pcc"] >= PCC_THRESHOLD
    _write_result("test_results_decode_synthetic_layer31.json", result)


def test_optimized_decode_real_weights(mesh_device, hf_config):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, local_files_only=True)
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    model.eval()
    result = _run_decode_case(mesh_device, hf_config, model.state_dict(), 0, "real")
    assert result["pcc"] >= PCC_THRESHOLD
    _write_result("test_results_decode_real_layer0.json", result)


def test_optimized_decode_trace_replay_and_warmed_latency(mesh_device, hf_config):
    state_dict = _synthetic_state_dict(hf_config, layer_idx=0)
    case = _build_decode_case(mesh_device, hf_config, state_dict, 0, "trace", scale=0.02)
    decoder = case["decoder"]

    for _ in range(2):
        decoder.decode_forward(**case["tt_inputs"])
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    decoder.decode_forward(**case["tt_inputs"])
    ttnn.synchronize_device(mesh_device)
    eager_ms = (time.perf_counter() - start) * 1000.0

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = decoder.decode_forward(**case["tt_inputs"])
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(mesh_device)

    start = time.perf_counter()
    signpost("PERF_DECODE")
    for _ in range(5):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
    signpost("PERF_DECODE_END")
    ttnn.synchronize_device(mesh_device)
    traced_ms = (time.perf_counter() - start) * 1000.0 / 5.0
    ttnn.release_trace(mesh_device, trace_id)

    tt_out_torch = ttnn.to_torch(trace_output).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
    pcc = _pcc(case["hf_out"].float(), tt_out_torch.float())
    tt_key_cache = ttnn.to_torch(case["tt_inputs"]["key_cache"])
    tt_value_cache = ttnn.to_torch(case["tt_inputs"]["value_cache"])
    cache_position = EMITTED_CACHE_LEN - 1
    key_cache_pcc = _pcc(
        case["hf_key_cache"][:, :, cache_position : cache_position + 1, :].float(),
        tt_key_cache[:, :, cache_position : cache_position + 1, :].float(),
    )
    value_cache_pcc = _pcc(
        case["hf_value_cache"][:, :, cache_position : cache_position + 1, :].float(),
        tt_value_cache[:, :, cache_position : cache_position + 1, :].float(),
    )
    result = {
        "test": "test_optimized_decode_trace_replay_and_warmed_latency",
        "weights": "synthetic",
        "batch": EMITTED_BATCH_SIZE,
        "cache_len": EMITTED_CACHE_LEN,
        "eager_ms": eager_ms,
        "traced_ms_per_replay": traced_ms,
        "pcc_after_trace_replay": pcc,
        "key_cache_pcc_after_trace_replay": key_cache_pcc,
        "value_cache_pcc_after_trace_replay": value_cache_pcc,
        "pcc_threshold": PCC_THRESHOLD,
    }
    assert pcc >= PCC_THRESHOLD
    assert key_cache_pcc >= PCC_THRESHOLD
    assert value_cache_pcc >= PCC_THRESHOLD
    assert traced_ms > 0.0
    _write_result("test_results_trace_latency.json", result)


def test_optimized_decode_repeated_stress(mesh_device, hf_config):
    state_dict = _synthetic_state_dict(hf_config, layer_idx=0)
    case = _build_decode_case(mesh_device, hf_config, state_dict, 0, "stress", scale=0.02)
    decoder = case["decoder"]
    pccs = []
    for _ in range(3):
        out = decoder.decode_forward(**case["tt_inputs"])
        tt_out_torch = ttnn.to_torch(out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
        pccs.append(_pcc(case["hf_out"].float(), tt_out_torch.float()))
    result = {
        "test": "test_optimized_decode_repeated_stress",
        "weights": "synthetic",
        "iterations": len(pccs),
        "pccs": pccs,
        "pcc_threshold": PCC_THRESHOLD,
    }
    assert min(pccs) >= PCC_THRESHOLD
    _write_result("test_results_repeated_stress.json", result)


def _run_decode_case(
    mesh_device,
    hf_config,
    state_dict,
    layer_idx: int,
    weight_label: str,
    *,
    policy: OptimizedDecoderPolicy | None = None,
):
    case = _build_decode_case(
        mesh_device,
        hf_config,
        state_dict,
        layer_idx,
        weight_label,
        scale=0.01 if weight_label == "real" else 0.02,
        policy=policy,
    )
    tt_out = case["decoder"].decode_forward(**case["tt_inputs"])
    tt_out_torch = ttnn.to_torch(tt_out).reshape(EMITTED_BATCH_SIZE, 1, hf_config.hidden_size)
    pcc = _pcc(case["hf_out"].float(), tt_out_torch.float())
    return {
        "test": f"test_optimized_decode_{weight_label}_layer{layer_idx}",
        "weights": weight_label,
        "layer_idx": layer_idx,
        "batch": EMITTED_BATCH_SIZE,
        "cache_len": EMITTED_CACHE_LEN,
        "cache_position": EMITTED_CACHE_LEN - 1,
        "pcc": pcc,
        "pcc_threshold": PCC_THRESHOLD,
        "policy": _policy_dict(case["decoder"].policy),
    }


def _build_decode_case(
    mesh_device,
    hf_config,
    state_dict,
    layer_idx: int,
    weight_label: str,
    *,
    scale: float,
    policy: OptimizedDecoderPolicy | None = None,
):
    torch.manual_seed(20260714 + layer_idx if weight_label == "real" else 20260715 + layer_idx)
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
    decoder = OptimizedDecoder.from_state_dict(
        state_dict,
        hf_config=hf_config,
        layer_idx=layer_idx,
        mesh_device=mesh_device,
        batch=EMITTED_BATCH_SIZE,
        cache_len=EMITTED_CACHE_LEN,
        policy=policy or OptimizedDecoderPolicy(),
    )
    tt_hidden = _tt(hidden.reshape(1, 1, EMITTED_BATCH_SIZE, hf_config.hidden_size), mesh_device)
    tt_key_cache = _tt(key_cache_pt, mesh_device)
    tt_value_cache = _tt(value_cache_pt, mesh_device)
    tt_pos = _tt(
        torch.tensor([cache_position], dtype=torch.int32), mesh_device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    cos, sin = build_decode_rope(hf_config, cache_position, mesh_device)
    tt_mask = build_decode_attention_mask(cache_position, EMITTED_CACHE_LEN, mesh_device)
    tt_inputs = {
        "hidden_states": tt_hidden,
        "key_cache": tt_key_cache,
        "value_cache": tt_value_cache,
        "cache_position": tt_pos,
        "cos": cos,
        "sin": sin,
        "attention_mask": tt_mask,
    }
    last_output = decoder.decode_forward(**tt_inputs)
    tt_inputs["hidden_states"] = tt_hidden
    return {
        "decoder": decoder,
        "tt_inputs": tt_inputs,
        "hf_out": hf_out,
        "hf_key_cache": key_cache_pt,
        "hf_value_cache": value_cache_pt,
        "last_output": last_output,
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


def _synthetic_state_dict(hf_config, *, layer_idx: int):
    torch.manual_seed(2026 + layer_idx)
    h = hf_config.hidden_size
    kv = hf_config.num_key_value_heads * hf_config.head_dim
    prefix = f"model.layers.{layer_idx}"
    return {
        f"{prefix}.input_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(h, dtype=torch.bfloat16),
        f"{prefix}.self_attn.q_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.k_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.v_proj.weight": torch.randn(kv, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(h, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.gate_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.up_proj.weight": torch.randn(hf_config.intermediate_size, h, dtype=torch.bfloat16) * 0.01,
        f"{prefix}.mlp.down_proj.weight": torch.randn(h, hf_config.intermediate_size, dtype=torch.bfloat16) * 0.01,
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


def _policy_dict(policy: OptimizedDecoderPolicy) -> dict[str, str | bool]:
    return {
        "attention_weight_dtype": str(policy.attention_weight_dtype),
        "mlp_weight_dtype": str(policy.mlp_weight_dtype),
        "output_weight_dtype": str(policy.output_weight_dtype),
        "norm_weight_dtype": str(policy.norm_weight_dtype),
        "activation_dtype": str(policy.activation_dtype),
        "attention_math_fidelity": str(policy.attention_math_fidelity),
        "mlp_math_fidelity": str(policy.mlp_math_fidelity),
        "qvk_in0_block_w": policy.qvk_in0_block_w,
        "o_in0_block_w": policy.o_in0_block_w,
        "gate_in0_block_w": policy.gate_in0_block_w,
        "up_in0_block_w": policy.up_in0_block_w,
        "down_in0_block_w": policy.down_in0_block_w,
        "qvk_l1_core_count": policy.qvk_l1_core_count,
        "o_l1_core_count": policy.o_l1_core_count,
        "gate_l1_core_count": policy.gate_l1_core_count,
        "up_l1_core_count": policy.up_l1_core_count,
        "down_l1_core_count": policy.down_l1_core_count,
        "use_dram_sharded_weights": policy.use_dram_sharded_weights,
        "use_decode_create_heads": policy.use_decode_create_heads,
        "use_explicit_sdpa_program_config": policy.use_explicit_sdpa_program_config,
        "use_packed_gate_up_projection": policy.use_packed_gate_up_projection,
    }


def _write_result(name: str, data: dict):
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / name).write_text(json.dumps(data, indent=2) + "\n")
