# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import textwrap
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tt import functional_decoder as functional_decoder_module
from models.autoports.qwen_qwen3_6_35b_a3b.tt.functional_decoder import FunctionalDecoder
from models.common.utility_functions import comp_pcc

MODEL_ID = "Qwen/Qwen3.6-35B-A3B"
PCC_BAR = 0.995
BLOCK_SIZE = 32


class _LinearLayerState:
    def __init__(self):
        self.conv_states = None
        self.recurrent_states = None


class _LinearCache:
    def __init__(self, layer_idx: int):
        self.layer_idx = layer_idx
        self.layers = [_LinearLayerState() for _ in range(layer_idx + 1)]

    def has_previous_state(self, layer_idx: int) -> bool:
        layer = self.layers[layer_idx]
        return layer.conv_states is not None and layer.recurrent_states is not None

    def update_conv_state(self, new_conv_state: torch.Tensor, layer_idx: int) -> None:
        self.layers[layer_idx].conv_states = new_conv_state.detach().clone()

    def update_recurrent_state(self, recurrent_state: torch.Tensor, layer_idx: int) -> None:
        self.layers[layer_idx].recurrent_states = recurrent_state.detach().clone()


def _target_text_config():
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True).text_config
    cfg._attn_implementation = "eager"
    return cfg


def _randn(shape, seed: int, scale: float = 0.02):
    gen = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=gen, dtype=torch.float32) * scale).to(torch.bfloat16)


def _zero(shape):
    return torch.zeros(shape, dtype=torch.bfloat16)


def _synthetic_layer_state(cfg, layer_idx: int):
    state = {
        "input_layernorm.weight": _zero((cfg.hidden_size,)),
        "post_attention_layernorm.weight": _zero((cfg.hidden_size,)),
        "mlp.gate.weight": _zero((cfg.num_experts, cfg.hidden_size)),
        "mlp.experts.gate_up_proj": _zero((cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size)),
        "mlp.experts.down_proj": _zero((cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)),
        "mlp.shared_expert.gate_proj.weight": _zero((cfg.shared_expert_intermediate_size, cfg.hidden_size)),
        "mlp.shared_expert.up_proj.weight": _zero((cfg.shared_expert_intermediate_size, cfg.hidden_size)),
        "mlp.shared_expert.down_proj.weight": _zero((cfg.hidden_size, cfg.shared_expert_intermediate_size)),
        "mlp.shared_expert_gate.weight": _zero((1, cfg.hidden_size)),
    }
    if cfg.layer_types[layer_idx] == "full_attention":
        q_width = cfg.num_attention_heads * cfg.head_dim
        state.update(
            {
                "self_attn.q_proj.weight": _randn((2 * q_width, cfg.hidden_size), seed=10 + layer_idx),
                "self_attn.k_proj.weight": _randn(
                    (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size), seed=20 + layer_idx
                ),
                "self_attn.v_proj.weight": _randn(
                    (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size), seed=30 + layer_idx
                ),
                "self_attn.o_proj.weight": _randn((cfg.hidden_size, q_width), seed=40 + layer_idx),
                "self_attn.q_norm.weight": _zero((cfg.head_dim,)),
                "self_attn.k_norm.weight": _zero((cfg.head_dim,)),
            }
        )
    else:
        key_dim = cfg.linear_key_head_dim * cfg.linear_num_key_heads
        value_dim = cfg.linear_value_head_dim * cfg.linear_num_value_heads
        conv_dim = key_dim * 2 + value_dim
        state.update(
            {
                "linear_attn.A_log": torch.linspace(0.1, 1.0, cfg.linear_num_value_heads).log().to(torch.bfloat16),
                "linear_attn.dt_bias": torch.ones((cfg.linear_num_value_heads,), dtype=torch.bfloat16),
                "linear_attn.conv1d.weight": _randn((conv_dim, 1, cfg.linear_conv_kernel_dim), seed=50 + layer_idx),
                "linear_attn.in_proj_qkv.weight": _randn((conv_dim, cfg.hidden_size), seed=60 + layer_idx),
                "linear_attn.in_proj_z.weight": _randn((value_dim, cfg.hidden_size), seed=70 + layer_idx),
                "linear_attn.in_proj_a.weight": _randn(
                    (cfg.linear_num_value_heads, cfg.hidden_size), seed=80 + layer_idx
                ),
                "linear_attn.in_proj_b.weight": _randn(
                    (cfg.linear_num_value_heads, cfg.hidden_size), seed=90 + layer_idx
                ),
                "linear_attn.norm.weight": torch.ones((cfg.linear_value_head_dim,), dtype=torch.bfloat16),
                "linear_attn.out_proj.weight": _randn((cfg.hidden_size, value_dim), seed=100 + layer_idx),
            }
        )
    return state


def _load_real_layer_state(layer_idx: int):
    index_path = hf_hub_download(MODEL_ID, "model.safetensors.index.json", local_files_only=True)
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)

    prefix = f"model.language_model.layers.{layer_idx}."
    keys = [key for key in index["weight_map"] if key.startswith(prefix)]
    if not keys:
        raise RuntimeError(f"no checkpoint tensors found for {prefix}")

    snapshot_dir = Path(index_path).parent
    state = {}
    for shard_name in sorted({index["weight_map"][key] for key in keys}):
        with safe_open(snapshot_dir / shard_name, framework="pt", device="cpu") as shard:
            for key in keys:
                if index["weight_map"][key] == shard_name:
                    state[key.removeprefix(prefix)] = shard.get_tensor(key)
    return state


def _torch_layer(cfg, layer_idx: int, state: dict[str, torch.Tensor]):
    layer = Qwen3_5MoeDecoderLayer(cfg, layer_idx=layer_idx).eval()
    missing, unexpected = layer.load_state_dict(state, strict=False)
    assert not missing
    assert not unexpected
    return layer.to(dtype=torch.bfloat16)


def _rotary(cfg, hidden: torch.Tensor, position_ids: torch.Tensor):
    rotary = Qwen3_5MoeTextRotaryEmbedding(cfg)
    return rotary(hidden, position_ids)


def _causal_mask(batch: int, seq_len: int):
    mask = torch.full((batch, 1, seq_len, seq_len), torch.finfo(torch.float32).min, dtype=torch.float32)
    return torch.triu(mask, diagonal=1).to(torch.bfloat16)


def _tt_bf16(tensor: torch.Tensor, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=device,
        dtype=ttnn.bfloat16,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _tt_int(tensor: torch.Tensor, device):
    return ttnn.Tensor(tensor.to(torch.int32), ttnn.int32).to(device)


def _to_torch(tensor: ttnn.Tensor):
    return tensor.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch().float()


def _page_table(batch: int, max_seq_len: int, block_size: int = BLOCK_SIZE):
    blocks_per_user = math.ceil(max_seq_len / block_size)
    table = torch.arange(batch * blocks_per_user, dtype=torch.int32).reshape(batch, blocks_per_user)
    return torch.flip(table, dims=[1])


def _assert_pcc(label: str, expected: torch.Tensor, actual: torch.Tensor, pcc: float = PCC_BAR):
    passed, message = comp_pcc(expected.float(), actual.float(), pcc)
    assert passed, f"{label}: {message}"
    return message


def _signpost(name: str) -> None:
    try:
        from tracy import signpost
    except ImportError:
        return
    signpost(name)


def _state_for_perf(cfg, layer_idx: int):
    if os.environ.get("RUN_QWEN36_PERF_REAL_WEIGHTS", "1") == "1":
        return _load_real_layer_state(layer_idx)
    return _synthetic_layer_state(cfg, layer_idx)


def _full_prefill_inputs(device, cfg, seq_len: int):
    batch = 1
    max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=600 + seq_len, scale=0.01)
    position_ids = torch.arange(seq_len, dtype=torch.long).reshape(batch, seq_len)
    position_embeddings = _rotary(cfg, hidden, position_ids)
    page_table = _page_table(batch, max_seq_len)
    return {
        "hidden": _tt_bf16(hidden.unsqueeze(0), device),
        "position_embeddings": (
            _tt_bf16(position_embeddings[0].unsqueeze(1), device),
            _tt_bf16(position_embeddings[1].unsqueeze(1), device),
        ),
        "page_table": _tt_int(page_table, device),
        "max_seq_len": max_seq_len,
    }


def _linear_prefill_inputs(device, cfg, seq_len: int):
    hidden = _randn((1, seq_len, cfg.hidden_size), seed=700 + seq_len, scale=0.01)
    return {"hidden": _tt_bf16(hidden.unsqueeze(0), device)}


def _prepare_decode_after_prefill(device, cfg, tt_layer: FunctionalDecoder, layer_idx: int, seq_len: int):
    batch = 1
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=800 + layer_idx + seq_len, scale=0.01)
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    current_pos = _tt_int(torch.tensor([seq_len], dtype=torch.int32), device)

    if cfg.layer_types[layer_idx] == "full_attention":
        prefill = _full_prefill_inputs(device, cfg, seq_len)
        kv_cache = FunctionalDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=prefill["max_seq_len"],
            block_size=BLOCK_SIZE,
        )
        tt_layer.prefill_forward(
            prefill["hidden"],
            position_embeddings=prefill["position_embeddings"],
            page_table=prefill["page_table"],
            kv_cache=kv_cache,
        )
        decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[seq_len]], dtype=torch.long))
        kwargs = {
            "current_pos": current_pos,
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": prefill["page_table"],
            "kv_cache": kv_cache,
        }
    else:
        state = FunctionalDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=batch)
        prefill = _linear_prefill_inputs(device, cfg, seq_len)
        tt_prefill = tt_layer.prefill_forward(prefill["hidden"], linear_state=state)
        kwargs = {"current_pos": current_pos, "linear_state": tt_prefill.linear_state}
    return decode_input, kwargs


def _run_signposted_prefill(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)

    if cfg.layer_types[layer_idx] == "full_attention":
        inputs = _full_prefill_inputs(device, cfg, seq_len)
        warm_cache = FunctionalDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=1,
            max_seq_len=inputs["max_seq_len"],
            block_size=BLOCK_SIZE,
        )
        tt_layer.prefill_forward(
            inputs["hidden"],
            position_embeddings=inputs["position_embeddings"],
            page_table=inputs["page_table"],
            kv_cache=warm_cache,
        )
        measure_cache = FunctionalDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=1,
            max_seq_len=inputs["max_seq_len"],
            block_size=BLOCK_SIZE,
        )
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(
            inputs["hidden"],
            position_embeddings=inputs["position_embeddings"],
            page_table=inputs["page_table"],
            kv_cache=measure_cache,
        ).hidden_states
    else:
        inputs = _linear_prefill_inputs(device, cfg, seq_len)
        warm_state = FunctionalDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
        tt_layer.prefill_forward(inputs["hidden"], linear_state=warm_state)
        measure_state = FunctionalDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=device, batch_size=1
        )
        ttnn.synchronize_device(device)
        _signpost(signpost_name)
        start = time.perf_counter()
        out = tt_layer.prefill_forward(inputs["hidden"], linear_state=measure_state).hidden_states

    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} wall_ms={elapsed_ms:.3f} output_shape={tuple(out.shape)}")


def _run_signposted_traced_decode(device, *, layer_idx: int, seq_len: int, signpost_name: str):
    cfg = _target_text_config()
    state = _state_for_perf(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    decode_input, decode_kwargs = _prepare_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)

    _signpost(signpost_name)
    start = time.perf_counter()
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _signpost(f"{signpost_name}_END")
    print(f"{signpost_name} traced_wall_ms={elapsed_ms:.3f} output_shape={tuple(traced.shape)}")
    ttnn.release_trace(device, trace_id)


def _run_traced_decode(device, tt_layer: FunctionalDecoder, decode_input: ttnn.Tensor, decode_kwargs: dict):
    tt_layer.decode_forward(decode_input, **decode_kwargs)
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    ttnn.synchronize_device(device)
    return traced


def _run_prefill_decode_parity(
    device,
    *,
    cfg,
    layer_idx: int,
    state: dict[str, torch.Tensor],
    seq_len: int,
    batch: int = 1,
    trace_decode: bool = False,
):
    torch.manual_seed(1234 + layer_idx + seq_len)
    max_seq_len = max(96, math.ceil((seq_len + 1) / BLOCK_SIZE) * BLOCK_SIZE)
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=200 + batch + layer_idx + seq_len, scale=0.01)
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=300 + batch + layer_idx + seq_len, scale=0.01)

    layer = _torch_layer(cfg, layer_idx, state)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)

    if cfg.layer_types[layer_idx] == "full_attention":
        cache = DynamicCache()
        pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
        position_embeddings = _rotary(cfg, hidden, pos)
        with torch.no_grad():
            ref_prefill = layer(
                hidden,
                position_embeddings=position_embeddings,
                attention_mask=_causal_mask(batch, seq_len),
                past_key_values=cache,
            )
        page_table = _page_table(batch, max_seq_len)
        kv_cache = FunctionalDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=max_seq_len,
            block_size=BLOCK_SIZE,
        )
        tt_prefill = tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        ).hidden_states
        tt_prefill_host = _to_torch(tt_prefill).squeeze(0)
        prefill_msg = _assert_pcc("full_attention prefill", ref_prefill, tt_prefill_host)

        current_pos = torch.full((batch,), seq_len, dtype=torch.int32)
        decode_pos = current_pos.to(torch.long).reshape(batch, 1)
        decode_position_embeddings = _rotary(cfg, decode_hidden, decode_pos)
        with torch.no_grad():
            ref_decode = layer(
                decode_hidden,
                position_embeddings=decode_position_embeddings,
                attention_mask=None,
                past_key_values=cache,
            )
        decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
        decode_kwargs = {
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
            "current_pos": _tt_int(current_pos, device),
        }
        if trace_decode:
            tt_decode = _run_traced_decode(device, tt_layer, decode_input, decode_kwargs)
        else:
            tt_decode = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
        tt_decode_host = _to_torch(tt_decode).squeeze(0).transpose(0, 1)
        decode_msg = _assert_pcc("full_attention decode", ref_decode, tt_decode_host)
        return prefill_msg, decode_msg

    linear_cache = _LinearCache(layer_idx)
    dummy_pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len).expand(batch, seq_len)
    dummy_position_embeddings = _rotary(cfg, hidden, dummy_pos)
    with torch.no_grad():
        ref_prefill = layer(
            hidden,
            position_embeddings=dummy_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    linear_state = FunctionalDecoder.allocate_linear_attention_state(
        hf_config=cfg, mesh_device=device, batch_size=batch
    )
    tt_prefill_result = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=linear_state)
    tt_prefill_host = _to_torch(tt_prefill_result.hidden_states).squeeze(0)
    prefill_msg = _assert_pcc("linear_attention prefill", ref_prefill, tt_prefill_host)

    decode_pos = torch.tensor([[seq_len]], dtype=torch.long)
    decode_position_embeddings = _rotary(cfg, decode_hidden, decode_pos)
    with torch.no_grad():
        ref_decode = layer(
            decode_hidden,
            position_embeddings=decode_position_embeddings,
            attention_mask=None,
            past_key_values=linear_cache,
        )
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    decode_kwargs = {
        "current_pos": _tt_int(torch.full((batch,), seq_len, dtype=torch.int32), device),
        "linear_state": tt_prefill_result.linear_state,
    }
    if trace_decode:
        tt_decode = _run_traced_decode(device, tt_layer, decode_input, decode_kwargs)
    else:
        tt_decode = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    tt_decode_host = _to_torch(tt_decode).squeeze(0).transpose(0, 1)
    decode_msg = _assert_pcc("linear_attention decode", ref_decode, tt_decode_host)
    return prefill_msg, decode_msg


def test_target_decoder_contract():
    cfg = _target_text_config()
    assert cfg.max_position_embeddings == 262144
    assert cfg.hidden_size == 2048
    assert cfg.layer_types.count("linear_attention") == 30
    assert cfg.layer_types.count("full_attention") == 10
    assert cfg.layer_types[:4] == ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 8_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_seq5", "full_seq33"])
def test_synthetic_functional_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"prefill {prefill_msg}")
    print(f"traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_batch2", "full_batch2"])
def test_synthetic_functional_decoder_batch2_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    prefill_msg, decode_msg = _run_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        batch=2,
        trace_decode=True,
    )
    print(f"batch2 prefill {prefill_msg}")
    print(f"batch2 traced decode {decode_msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 1), (3, 1)], ids=["linear_trace", "full_trace"])
def test_synthetic_functional_decoder_traced_decode(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    batch = 1
    hidden = _randn((batch, seq_len, cfg.hidden_size), seed=401 + layer_idx, scale=0.01)
    decode_hidden = _randn((batch, 1, cfg.hidden_size), seed=501 + layer_idx, scale=0.01)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)

    if cfg.layer_types[layer_idx] == "full_attention":
        page_table = _page_table(batch, 96)
        kv_cache = FunctionalDecoder.allocate_full_attention_cache(
            hf_config=cfg,
            mesh_device=device,
            max_batch_size=batch,
            max_seq_len=96,
            block_size=BLOCK_SIZE,
        )
        pos = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
        position_embeddings = _rotary(cfg, hidden, pos)
        tt_layer.prefill_forward(
            _tt_bf16(hidden.unsqueeze(0), device),
            position_embeddings=(
                _tt_bf16(position_embeddings[0].unsqueeze(1), device),
                _tt_bf16(position_embeddings[1].unsqueeze(1), device),
            ),
            page_table=_tt_int(page_table, device),
            kv_cache=kv_cache,
        )
        decode_pos = torch.tensor([seq_len], dtype=torch.int32)
        decode_position_embeddings = _rotary(cfg, decode_hidden, decode_pos.to(torch.long).reshape(1, 1))
        decode_kwargs = {
            "current_pos": _tt_int(decode_pos, device),
            "position_embeddings": (
                _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
                _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
            ),
            "page_table": _tt_int(page_table, device),
            "kv_cache": kv_cache,
        }
    else:
        linear_state = FunctionalDecoder.allocate_linear_attention_state(
            hf_config=cfg, mesh_device=device, batch_size=batch
        )
        tt_prefill = tt_layer.prefill_forward(_tt_bf16(hidden.unsqueeze(0), device), linear_state=linear_state)
        decode_kwargs = {
            "current_pos": _tt_int(torch.tensor([seq_len], dtype=torch.int32), device),
            "linear_state": tt_prefill.linear_state,
        }

    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    eager = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    traced = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.release_trace(device, trace_id)
    ttnn.synchronize_device(device)

    eager_host = _to_torch(eager)
    traced_host = _to_torch(traced)
    msg = _assert_pcc(f"{cfg.layer_types[layer_idx]} traced decode", eager_host, traced_host, pcc=0.999)
    print(f"traced decode {msg}")


@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(("layer_idx", "seq_len"), [(0, 5), (3, 33)], ids=["linear_repeat", "full_repeat"])
def test_synthetic_functional_decoder_repeated_input_determinism(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    decode_input, decode_kwargs = _prepare_decode_after_prefill(device, cfg, tt_layer, layer_idx, seq_len)

    first = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    second = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    msg = _assert_pcc(f"{cfg.layer_types[layer_idx]} repeated decode", _to_torch(first), _to_torch(second), pcc=0.9999)
    print(f"repeated decode {msg}")


@pytest.mark.real_weights
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_REAL_WEIGHTS") != "1", reason="set RUN_QWEN36_REAL_WEIGHTS=1 to load checkpoint weights"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 16_000_000}], indirect=True)
@pytest.mark.parametrize(
    ("layer_idx", "seq_len"),
    [(0, 1), (3, 1), (0, 5), (3, 5)],
    ids=["real_linear_layer0_seq1", "real_full_layer3_seq1", "real_linear_layer0_seq5", "real_full_layer3_seq5"],
)
def test_real_weight_functional_decoder_prefill_decode_against_hf(device, layer_idx, seq_len):
    cfg = _target_text_config()
    state = _load_real_layer_state(layer_idx)
    prefill_msg, decode_msg = _run_prefill_decode_parity(
        device,
        cfg=cfg,
        layer_idx=layer_idx,
        state=state,
        seq_len=seq_len,
        trace_decode=True,
    )
    print(f"real prefill {prefill_msg}")
    print(f"real traced decode {decode_msg}")


def test_runtime_fallback_audit_source():
    runtime_functions = {
        "_slice": functional_decoder_module._slice,
        "_slice_last": functional_decoder_module._slice_last,
        "_concat_dim2_bounded": functional_decoder_module._concat_dim2_bounded,
        "_silu_mul": functional_decoder_module._silu_mul,
        "_rms_norm": functional_decoder_module._rms_norm,
        "_l2_norm_last_dim": functional_decoder_module._l2_norm_last_dim,
        "_rotate_half": functional_decoder_module._rotate_half,
        "_apply_partial_rope": functional_decoder_module._apply_partial_rope,
        "_sparse_matmul_program_config": functional_decoder_module._sparse_matmul_program_config,
        "_QwenFullAttention._project_qkgv": functional_decoder_module._QwenFullAttention._project_qkgv,
        "_QwenFullAttention._reshape_prefill_heads": functional_decoder_module._QwenFullAttention._reshape_prefill_heads,
        "_QwenFullAttention._reshape_decode_heads": functional_decoder_module._QwenFullAttention._reshape_decode_heads,
        "_QwenFullAttention._norm_and_rope": functional_decoder_module._QwenFullAttention._norm_and_rope,
        "_QwenFullAttention._decode_update_mem_config": functional_decoder_module._QwenFullAttention._decode_update_mem_config,
        "_QwenFullAttention._cache_update_tensor": functional_decoder_module._QwenFullAttention._cache_update_tensor,
        "_QwenFullAttention.prefill_forward": functional_decoder_module._QwenFullAttention.prefill_forward,
        "_QwenFullAttention.decode_forward": functional_decoder_module._QwenFullAttention.decode_forward,
        "_QwenLinearAttention._conv_step": functional_decoder_module._QwenLinearAttention._conv_step,
        "_QwenLinearAttention._step": functional_decoder_module._QwenLinearAttention._step,
        "_QwenLinearAttention._conv_prefill": functional_decoder_module._QwenLinearAttention._conv_prefill,
        "_QwenLinearAttention._reshape_prefill_heads": functional_decoder_module._QwenLinearAttention._reshape_prefill_heads,
        "_QwenLinearAttention._fold_prefill_heads": functional_decoder_module._QwenLinearAttention._fold_prefill_heads,
        "_QwenLinearAttention._pad_linear_chunk": functional_decoder_module._QwenLinearAttention._pad_linear_chunk,
        "_QwenLinearAttention._solve_chunk_attn": functional_decoder_module._QwenLinearAttention._solve_chunk_attn,
        "_QwenLinearAttention._chunk_gated_delta_rule": functional_decoder_module._QwenLinearAttention._chunk_gated_delta_rule,
        "_QwenLinearAttention._finish_prefill_chunk": functional_decoder_module._QwenLinearAttention._finish_prefill_chunk,
        "_QwenLinearAttention.prefill_forward": functional_decoder_module._QwenLinearAttention.prefill_forward,
        "_QwenLinearAttention.decode_forward": functional_decoder_module._QwenLinearAttention.decode_forward,
        "_QwenMoe._router_dense": functional_decoder_module._QwenMoe._router_dense,
        "_QwenMoe._shared": functional_decoder_module._QwenMoe._shared,
        "_QwenMoe._routed_decode": functional_decoder_module._QwenMoe._routed_decode,
        "_QwenMoe._routed_prefill_chunk": functional_decoder_module._QwenMoe._routed_prefill_chunk,
        "_QwenMoe._routed_chunk": functional_decoder_module._QwenMoe._routed_chunk,
        "_QwenMoe._forward_chunk": functional_decoder_module._QwenMoe._forward_chunk,
        "_QwenMoe.forward": functional_decoder_module._QwenMoe.forward,
        "FunctionalDecoder.prefill_forward": functional_decoder_module.FunctionalDecoder.prefill_forward,
        "FunctionalDecoder.decode_forward": functional_decoder_module.FunctionalDecoder.decode_forward,
    }
    forbidden_names = {"torch"}
    forbidden_attrs = {"from_torch", "to_torch", "get_fallback_function"}
    violations = []
    for name, func in runtime_functions.items():
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                violations.append(f"{name}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in forbidden_attrs:
                violations.append(f"{name}: attribute {node.attr}")
    assert not violations


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_PERF") != "1", reason="set RUN_QWEN36_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_linear_prefill(device):
    _run_signposted_prefill(device, layer_idx=0, seq_len=5, signpost_name="PERF_LINEAR_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_PERF") != "1", reason="set RUN_QWEN36_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_full_prefill(device):
    _run_signposted_prefill(device, layer_idx=3, seq_len=33, signpost_name="PERF_FULL_PREFILL")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_PERF") != "1", reason="set RUN_QWEN36_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_linear_decode(device):
    _run_signposted_traced_decode(device, layer_idx=0, seq_len=5, signpost_name="PERF_LINEAR_DECODE")


@pytest.mark.perf
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_PERF") != "1", reason="set RUN_QWEN36_PERF=1 for Tracy performance evidence"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_perf_qwen36_full_decode(device):
    _run_signposted_traced_decode(device, layer_idx=3, seq_len=33, signpost_name="PERF_FULL_DECODE")


@pytest.mark.context
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_CONTEXT_PROBE") != "1", reason="set RUN_QWEN36_CONTEXT_PROBE=1 for larger context probes"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_context_probe_full_attention_decode_advertised_context(device):
    cfg = _target_text_config()
    layer_idx = 3
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    max_context = cfg.max_position_embeddings
    kv_cache = FunctionalDecoder.allocate_full_attention_cache(
        hf_config=cfg,
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=max_context,
        block_size=BLOCK_SIZE,
    )
    page_table = _page_table(1, max_context)
    decode_hidden = _randn((1, 1, cfg.hidden_size), seed=900, scale=0.01)
    decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[max_context - 1]], dtype=torch.long))
    out = tt_layer.decode_forward(
        _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device),
        current_pos=_tt_int(torch.tensor([max_context - 1], dtype=torch.int32), device),
        position_embeddings=(
            _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
            _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
        ),
        page_table=_tt_int(page_table, device),
        kv_cache=kv_cache,
    ).hidden_states
    ttnn.synchronize_device(device)
    assert tuple(out.shape) == (1, 1, 1, cfg.hidden_size)
    print(f"full_attention advertised decode context={max_context} current_pos={max_context - 1}")


@pytest.mark.context
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_CONTEXT_PROBE") != "1", reason="set RUN_QWEN36_CONTEXT_PROBE=1 for larger context probes"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_context_probe_full_attention_decode_advertised_context_traced_control(device):
    cfg = _target_text_config()
    layer_idx = 3
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    max_context = cfg.max_position_embeddings
    kv_cache = FunctionalDecoder.allocate_full_attention_cache(
        hf_config=cfg,
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=max_context,
        block_size=BLOCK_SIZE,
    )

    page_table = _page_table(1, max_context)
    assert int(page_table[0, 0]) != 0
    assert int(page_table[0, -1]) == 0

    prefix_len = 33
    prefix_hidden = _randn((1, prefix_len, cfg.hidden_size), seed=975, scale=0.01)
    prefix_pos = torch.arange(prefix_len, dtype=torch.long).reshape(1, prefix_len)
    prefix_position_embeddings = _rotary(cfg, prefix_hidden, prefix_pos)
    tt_layer.prefill_forward(
        _tt_bf16(prefix_hidden.unsqueeze(0), device),
        position_embeddings=(
            _tt_bf16(prefix_position_embeddings[0].unsqueeze(1), device),
            _tt_bf16(prefix_position_embeddings[1].unsqueeze(1), device),
        ),
        page_table=_tt_int(page_table, device),
        kv_cache=kv_cache,
    )

    decode_hidden = _randn((1, 1, cfg.hidden_size), seed=976, scale=0.01)
    decode_position_embeddings = _rotary(cfg, decode_hidden, torch.tensor([[max_context - 1]], dtype=torch.long))
    decode_input = _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device)
    decode_kwargs = {
        "current_pos": _tt_int(torch.tensor([max_context - 1], dtype=torch.int32), device),
        "position_embeddings": (
            _tt_bf16(decode_position_embeddings[0].unsqueeze(0), device),
            _tt_bf16(decode_position_embeddings[1].unsqueeze(0), device),
        ),
        "page_table": _tt_int(page_table, device),
        "kv_cache": kv_cache,
    }

    eager = tt_layer.decode_forward(decode_input, **decode_kwargs).hidden_states
    ttnn.synchronize_device(device)
    traced = _run_traced_decode(device, tt_layer, decode_input, decode_kwargs)
    msg = _assert_pcc(
        "full_attention traced advertised-context control", _to_torch(eager), _to_torch(traced), pcc=0.999
    )
    assert tuple(traced.shape) == (1, 1, 1, cfg.hidden_size)
    print(f"full_attention traced advertised decode context={max_context} current_pos={max_context - 1} control {msg}")


@pytest.mark.context
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_CONTEXT_PROBE") != "1", reason="set RUN_QWEN36_CONTEXT_PROBE=1 for larger context probes"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_context_probe_full_attention_prefill_non_aligned(device):
    cfg = _target_text_config()
    seq_len = int(os.environ.get("QWEN36_CONTEXT_PREFILL_SEQ", "1025"))
    layer_idx = 3
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    inputs = _full_prefill_inputs(device, cfg, seq_len)
    kv_cache = FunctionalDecoder.allocate_full_attention_cache(
        hf_config=cfg,
        mesh_device=device,
        max_batch_size=1,
        max_seq_len=inputs["max_seq_len"],
        block_size=BLOCK_SIZE,
    )
    out = tt_layer.prefill_forward(
        inputs["hidden"],
        position_embeddings=inputs["position_embeddings"],
        page_table=inputs["page_table"],
        kv_cache=kv_cache,
    ).hidden_states
    ttnn.synchronize_device(device)
    assert tuple(out.shape) == (1, 1, seq_len, cfg.hidden_size)
    print(f"full_attention non_aligned prefill seq_len={seq_len}")


@pytest.mark.context
@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_CONTEXT_PROBE") != "1", reason="set RUN_QWEN36_CONTEXT_PROBE=1 for larger context probes"
)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_context_probe_linear_attention_prefill_decode_non_aligned(device):
    cfg = _target_text_config()
    seq_len = int(os.environ.get("QWEN36_CONTEXT_LINEAR_SEQ", "65"))
    layer_idx = 0
    state = _synthetic_layer_state(cfg, layer_idx)
    tt_layer = FunctionalDecoder.from_state_dict(state, hf_config=cfg, layer_idx=layer_idx, mesh_device=device)
    linear_state = FunctionalDecoder.allocate_linear_attention_state(hf_config=cfg, mesh_device=device, batch_size=1)
    inputs = _linear_prefill_inputs(device, cfg, seq_len)
    prefill = tt_layer.prefill_forward(inputs["hidden"], linear_state=linear_state)
    decode_hidden = _randn((1, 1, cfg.hidden_size), seed=950, scale=0.01)
    decode = tt_layer.decode_forward(
        _tt_bf16(decode_hidden.transpose(0, 1).unsqueeze(0), device),
        current_pos=_tt_int(torch.tensor([seq_len], dtype=torch.int32), device),
        linear_state=prefill.linear_state,
    ).hidden_states
    ttnn.synchronize_device(device)
    assert tuple(prefill.hidden_states.shape) == (1, 1, seq_len, cfg.hidden_size)
    assert tuple(decode.shape) == (1, 1, 1, cfg.hidden_size)
    print(f"linear_attention non_aligned prefill/decode seq_len={seq_len}")
