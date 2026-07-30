"""Shard-advisor capture target for one Phi-3.5 optimized decode block."""

from __future__ import annotations

import json
import math
import os
import sys
from types import SimpleNamespace

import torch

import ttnn

TT_METAL_ROOT = os.environ.get("TT_METAL_ROOT", "/home/mvasiljevic/tt-metal")
# Append: prepending tt-metal would let its source ``ttnn/`` shadow the
# advisor environment's installed package.
if TT_METAL_ROOT not in sys.path:
    sys.path.append(TT_METAL_ROOT)

BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
MAX_CONTEXT = 128

_DECODER = None
_KWARGS = None


def _build(device):
    from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import OptimizedDecoder

    config_path = (
        "/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/"
        "snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77/config.json"
    )
    with open(config_path) as handle:
        config = SimpleNamespace(**json.load(handle))
    layer_idx = 0
    prefix = f"model.layers.{layer_idx}."
    generator = torch.Generator().manual_seed(20260728)

    def sample(shape):
        return (torch.randn(*shape, generator=generator) * 0.02).to(torch.bfloat16)

    state = {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "post_attention_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.qkv_proj.weight": sample((3 * config.hidden_size, config.hidden_size)),
        prefix + "self_attn.o_proj.weight": sample((config.hidden_size, config.hidden_size)),
        prefix + "mlp.gate_up_proj.weight": sample((2 * config.intermediate_size, config.hidden_size)),
        prefix + "mlp.down_proj.weight": sample((config.hidden_size, config.intermediate_size)),
    }
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        batch=BATCH,
        max_context=MAX_CONTEXT,
    )
    hidden = torch.randn(
        BATCH,
        1,
        config.hidden_size,
        generator=torch.Generator().manual_seed(9876 + BATCH),
    ).to(torch.bfloat16)
    key_cache, value_cache = decoder.create_paged_kv_cache()
    blocks = math.ceil(MAX_CONTEXT / 32)
    page_table = ttnn.from_torch(
        torch.arange(BATCH * blocks, dtype=torch.int32).reshape(BATCH, blocks).flip(-1),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    positions = ttnn.from_torch(
        torch.zeros(BATCH, dtype=torch.int32),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_hidden = ttnn.from_torch(
        hidden.transpose(0, 1).unsqueeze(0),
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    kwargs = {
        "key_cache": key_cache,
        "value_cache": value_cache,
        "page_table": page_table,
        "current_positions": positions,
        "use_long_rope": False,
    }
    return decoder, kwargs, tt_hidden


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
