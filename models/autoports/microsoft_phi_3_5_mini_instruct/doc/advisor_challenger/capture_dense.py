"""Batch-32 shipped-precision capture target for the dense Phi-3.5 decoder."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import ttnn


ROOT = Path(os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder


BATCH = 32
CONTEXT = 32
INCUMBENT = ROOT / "models/autoports/microsoft_phi_3_5_mini_instruct/doc/advisor_challenger/incumbent.json"
with INCUMBENT.open() as fh:
    INCUMBENT_RECORD = json.load(fh)
if INCUMBENT_RECORD["decode_batch"] != BATCH:
    raise RuntimeError("capture batch must match the frozen incumbent")

_DECODER = None
_KWARGS = None


def _state_dict():
    return {
        "self_attn.qkv_proj.weight": torch.zeros(9216, 3072, dtype=torch.bfloat16),
        "self_attn.o_proj.weight": torch.zeros(3072, 3072, dtype=torch.bfloat16),
        "mlp.gate_up_proj.weight": torch.zeros(16384, 3072, dtype=torch.bfloat16),
        "mlp.down_proj.weight": torch.zeros(3072, 8192, dtype=torch.bfloat16),
        "input_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
        "post_attention_layernorm.weight": torch.ones(3072, dtype=torch.bfloat16),
    }


def _build(device):
    cfg = SimpleNamespace(
        hidden_size=3072,
        intermediate_size=8192,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=96,
        max_position_embeddings=64,
        original_max_position_embeddings=4096,
        rope_theta=10000.0,
        rms_norm_eps=1e-5,
        attention_bias=False,
        rope_scaling={
            "short_factor": [1.0] * 48,
            "long_factor": [1.0] * 48,
        },
    )
    decoder = OptimizedDecoder.from_state_dict(
        _state_dict(),
        hf_config=cfg,
        layer_idx=0,
        mesh_device=device,
        max_position_embeddings=64,
        **INCUMBENT_RECORD["shipped_policy"],
    )
    assert decoder.qkv_weight.dtype == ttnn.bfloat8_b
    assert decoder.o_weight.dtype == ttnn.bfloat8_b
    assert decoder.gate_up_weight.dtype == ttnn.bfloat4_b
    assert decoder.down_weight.dtype == ttnn.bfloat4_b
    kv_cache = OptimizedDecoder.allocate_paged_kv_cache(
        hf_config=cfg,
        mesh_device=device,
        max_batch_size=BATCH,
        max_seq_len=64,
        block_size=32,
    )
    hidden = ttnn.Tensor(torch.zeros(1, 1, BATCH, cfg.hidden_size), ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    current_host = torch.full((BATCH,), CONTEXT, dtype=torch.int32)
    current_pos = ttnn.Tensor(current_host, ttnn.int32).to(device)
    position_ids = ttnn.Tensor(current_host.to(torch.uint32), ttnn.uint32).to(device)
    page_table = ttnn.Tensor(
        torch.arange(BATCH * 2, dtype=torch.int32).reshape(BATCH, 2), ttnn.int32
    ).to(device)
    kwargs = {
        "current_pos": current_pos,
        "position_ids": position_ids,
        "page_table": page_table,
        "kv_cache": kv_cache,
        "rope_sequence_length": CONTEXT + 1,
    }
    return decoder, kwargs, hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
