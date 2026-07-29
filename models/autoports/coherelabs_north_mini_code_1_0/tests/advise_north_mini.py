# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shard-advisor capture target for North-Mini's rewritten dense decode block."""

import os
import sys
from types import SimpleNamespace

import torch

import ttnn

MODEL_DIR = os.environ.get("SHARD_ADVISE_MODEL_DIR", "/home/mvasiljevic/tt-metal")
BATCH = int(os.environ.get("SHARD_ADVISE_BATCH", "1"))
_DECODER = None
_KWARGS = None


def _build(device):
    if MODEL_DIR not in sys.path:
        sys.path.append(MODEL_DIR)
    from models.autoports.coherelabs_north_mini_code_1_0.tt.optimized_decoder import OptimizedDecoder

    config = SimpleNamespace(
        num_hidden_layers=40,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        max_position_embeddings=500000,
        rms_norm_eps=1.0e-5,
        layer_types=["full_attention"] + ["sliding_attention"] * 39,
        mlp_layer_types=["dense"] + ["sparse"] * 39,
        sliding_window=4096,
        prefix_dense_sliding_window_pattern=1,
        prefix_dense_intermediate_size=3072,
        intermediate_size=1024,
        num_experts=64,
        num_experts_per_tok=4,
        rope_parameters={"rope_theta": 1000000.0},
    )
    generator = torch.Generator().manual_seed(20260729)
    prefix = "model.layers.0."
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.bfloat16) * 0.01
    state = {
        prefix + "input_layernorm.weight": torch.ones(config.hidden_size, dtype=torch.bfloat16),
        prefix + "self_attn.q_proj.weight": randn(4096, 2048),
        prefix + "self_attn.k_proj.weight": randn(512, 2048),
        prefix + "self_attn.v_proj.weight": randn(512, 2048),
        prefix + "self_attn.o_proj.weight": randn(2048, 4096),
        prefix + "mlp.gate_proj.weight": randn(3072, 2048),
        prefix + "mlp.up_proj.weight": randn(3072, 2048),
        prefix + "mlp.down_proj.weight": randn(2048, 3072),
    }
    decoder = OptimizedDecoder.from_state_dict(
        state,
        hf_config=config,
        layer_idx=0,
        mesh_device=device,
        batch=BATCH,
        max_cache_len=32,
        candidate="bfp8_hifi2",
    )
    hidden = torch.randn(BATCH, 1, config.hidden_size, dtype=torch.bfloat16).unsqueeze(0)
    hidden_tt = ttnn.from_torch(
        hidden,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    key_cache, value_cache = decoder.create_paged_kv_cache()
    page_table = ttnn.from_torch(
        torch.arange(BATCH, dtype=torch.int32).reshape(BATCH, 1),
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
    cos_host, sin_host = decoder.build_rope_rows(torch.zeros(BATCH, dtype=torch.long), hf_config=config, decode=True)
    rope_kwargs = dict(
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    cos, sin = ttnn.from_torch(cos_host, **rope_kwargs), ttnn.from_torch(sin_host, **rope_kwargs)
    kwargs = dict(
        key_cache=key_cache,
        value_cache=value_cache,
        page_table=page_table,
        current_positions=positions,
        position_cos=cos,
        position_sin=sin,
    )
    return decoder, kwargs, hidden_tt


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    _DECODER, _KWARGS, hidden = _build(device)
    return (hidden,)
