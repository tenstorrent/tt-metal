#!/usr/bin/env python3
"""Phi-3.5 hooks for the mandated advisor-challenger capture template."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[5]
TEMPLATE = ROOT / ".agents/skills/advisor-challenger/scripts/capture_template.py"
spec = importlib.util.spec_from_file_location("advisor_challenger_capture_template", TEMPLATE)
template = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(template)


def _config():
    return SimpleNamespace(
        hidden_size=3072, intermediate_size=8192, num_attention_heads=32, num_key_value_heads=32,
        head_dim=96, num_hidden_layers=32, max_position_embeddings=64,
        original_max_position_embeddings=4096, rope_theta=10000.0, rms_norm_eps=1e-5,
        attention_bias=False, rope_scaling={"short_factor": [1.0] * 48, "long_factor": [1.0] * 48},
    )


def _synthetic_state_dict(_cfg):
    return {
        "self_attn.qkv_proj.weight": torch.zeros(9216, 3072, dtype=torch.bfloat16),
        "self_attn.o_proj.weight": torch.zeros(3072, 3072, dtype=torch.bfloat16),
        "mlp.gate_up_proj.weight": torch.zeros(16384, 3072, dtype=torch.bfloat16),
        "mlp.down_proj.weight": torch.zeros(3072, 8192, dtype=torch.bfloat16),
        "input_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
        "post_attention_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
    }


def _build(device):
    import ttnn

    cfg = _config()
    decoder = template.OptimizedDecoder.from_state_dict(
        _synthetic_state_dict(cfg), hf_config=cfg, layer_idx=template.LAYER_IDX,
        mesh_device=device, max_position_embeddings=64, batch=template.BATCH, **template.SHIPPED_POLICY,
    )
    assert decoder.qkv_weight.dtype == ttnn.bfloat8_b
    assert decoder.o_weight.dtype == ttnn.bfloat8_b
    assert decoder.gate_up_weight.dtype == ttnn.bfloat4_b
    assert decoder.down_weight.dtype == ttnn.bfloat4_b
    kv_cache = decoder.allocate_paged_kv_cache(
        hf_config=cfg, mesh_device=device, max_batch_size=template.BATCH, max_seq_len=64, block_size=32
    )
    current_host = torch.full((template.BATCH,), 32, dtype=torch.int32)
    kwargs = {
        "current_pos": ttnn.Tensor(current_host, ttnn.int32).to(device),
        "position_ids": ttnn.Tensor(current_host.to(torch.uint32), ttnn.uint32).to(device),
        "page_table": ttnn.Tensor(
            torch.arange(template.BATCH * 2, dtype=torch.int32).reshape(template.BATCH, 2), ttnn.int32
        ).to(device),
        "kv_cache": kv_cache,
        "rope_sequence_length": 33,
    }
    template._DECODER, template._CONFIG, template._WEIGHTS = decoder, cfg, decoder
    globals()["_KWARGS"] = kwargs
    return decoder


def decode(hidden):
    if template._DECODER is None:
        raise RuntimeError("capture wrapper must be built by ttnn-advise before decode")
    return template._DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    import ttnn

    _build(device)
    hidden = ttnn.Tensor(torch.zeros(1, 1, template.BATCH, 3072), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return (hidden,)


template._config = _config
template._synthetic_state_dict = _synthetic_state_dict
template._build = _build
template.decode = decode


if __name__ == "__main__":
    import ttnn

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        _build(mesh)
        template._record_traced_dtypes(os.environ.get("CHALLENGER_OUT_DIR", "."))
        print(f"capture target builds: kind={template.LAYER_KIND} idx={template.LAYER_IDX} batch={template.BATCH}")
    finally:
        ttnn.close_mesh_device(mesh)
