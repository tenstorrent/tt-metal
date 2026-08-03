"""Phi-3.5 adapter for advisor-challenger's capture template, decode batch 32."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from types import MethodType, SimpleNamespace

import torch

import ttnn
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizationPolicy, OptimizedDecoder

BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "32"))
LAYER_IDX = int(os.environ.get("CHALLENGER_LAYER_IDX", "0"))
OUT_DIR = os.environ.get("CHALLENGER_OUT_DIR", ".")
INCUMBENT = os.environ.get("CHALLENGER_INCUMBENT_JSON", "models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/incumbent.json")


def _policy(record):
    values = dict(record)
    dtypes = {"bfloat4_b": ttnn.bfloat4_b, "bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}
    fidelities = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
    for key in ("attention_weight_dtype", "gate_up_weight_dtype", "down_weight_dtype", "kv_cache_dtype"):
        values[key] = dtypes[values[key]]
    for key in ("attention_math_fidelity", "gate_up_math_fidelity", "down_math_fidelity"):
        values[key] = fidelities[values[key]]
    return OptimizationPolicy(**values)


_DECODER = None
_KWARGS = None


def _config():
    path = "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77/config.json"
    with open(path) as handle:
        return SimpleNamespace(**json.load(handle))


def _synthetic_state(config):
    generator = torch.Generator().manual_seed(20260728)
    prefix = f"model.layers.{LAYER_IDX}."
    def sample(shape):
        return (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)
    return {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.qkv_proj.weight": sample((3 * config.hidden_size, config.hidden_size)),
        prefix + "self_attn.o_proj.weight": sample((config.hidden_size, config.hidden_size)),
        prefix + "mlp.gate_up_proj.weight": sample((2 * config.intermediate_size, config.hidden_size)),
        prefix + "mlp.down_proj.weight": sample((config.hidden_size, config.intermediate_size)),
    }


def _to_tt_decode(hidden, device):
    return ttnn.from_torch(hidden.transpose(0, 1).unsqueeze(0), device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device), dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _page_table(batch, max_context, device):
    blocks = math.ceil(max_context / 32)
    value = torch.arange(batch * blocks, dtype=torch.int32).reshape(batch, blocks).flip(-1)
    return ttnn.from_torch(value, device=device, mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _positions(values, device):
    return ttnn.from_torch(torch.tensor(values, dtype=torch.int32), device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device), dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def make_inputs(device):
    global _DECODER, _KWARGS
    with open(INCUMBENT) as handle:
        incumbent = json.load(handle)
    config = _config()
    _DECODER = OptimizedDecoder.from_state_dict(
        _synthetic_state(config), hf_config=config, layer_idx=LAYER_IDX, mesh_device=device,
        batch=BATCH, max_context=128, optimization_policy=_policy(incumbent["shipped_policy"]),
    )
    # Capture-template rule: the tracer cannot query a symbolic tensor's
    # memory_config. The real decode path restores the create-heads output
    # config; spell that phase-specific config explicitly for capture.
    def capture_decode_rope(self, query, key, current_positions, *, use_long_rope):
        cos_table = self.long_cos if use_long_rope else self.short_cos
        sin_table = self.long_sin if use_long_rope else self.short_sin
        rope_positions = ttnn.typecast(current_positions, ttnn.uint32)
        cos = ttnn.reshape(ttnn.embedding(rope_positions, cos_table, layout=ttnn.TILE_LAYOUT),
                           [1, 1, self.batch, self.head_dim])
        sin = ttnn.reshape(ttnn.embedding(rope_positions, sin_table, layout=ttnn.TILE_LAYOUT),
                           [1, 1, self.batch, self.head_dim])
        query = self._apply_rope(ttnn.to_memory_config(query, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        key = self._apply_rope(ttnn.to_memory_config(key, ttnn.DRAM_MEMORY_CONFIG), cos, sin)
        return (
            ttnn.to_memory_config(query, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
            ttnn.to_memory_config(key, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
        )
    _DECODER._decode_rope = MethodType(capture_decode_rope, _DECODER)
    hidden = torch.randn(BATCH, 1, config.hidden_size, generator=torch.Generator().manual_seed(232)).to(torch.bfloat16)
    key_cache, value_cache = _DECODER.create_paged_kv_cache()
    _KWARGS = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": _page_table(BATCH, 128, device),
        "current_positions": _positions([33] * BATCH, device),
        "use_long_rope": False,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "traced_dtypes.json"), "w") as handle:
        json.dump({
            "layer_kind": "dense", "layer_idx": LAYER_IDX, "batch": BATCH,
            "traced_weight_dtypes": incumbent["shipped_weight_dtypes"],
            "shipped_weight_dtypes": incumbent["shipped_weight_dtypes"],
            "policy_source": incumbent["shipped_policy_source"],
            "advisor_commit": "618cd4e75d", "advisor_pin_expected": "618cd4e75d",
            "advisor_home": os.environ.get("TTMLIR_ADVISOR_HOME"),
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "decode_batch": BATCH, "requested_decode_batch": BATCH, "capture_batch": BATCH,
        }, handle, indent=2)
    return (_to_tt_decode(hidden, device),)


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)
