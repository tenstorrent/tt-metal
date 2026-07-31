from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path

import torch
import ttnn

from models.demos.gemma4.config import MeshConfig, ModeConfig


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "tests" / "test_functional_decoder.py"
BLOCK_SIZE = 64
CONTEXT = 64


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


functional = _load(TEST, "gemma4_12b_advisor_functional_helpers")


def dtype(name):
    return {
        "bf16": ttnn.bfloat16,
        "bfp8": ttnn.bfloat8_b,
        "bfp4": ttnn.bfloat4_b,
    }[name]


def ctor_policy(policy: dict, layer_kind: str) -> dict:
    def select(key):
        value = policy[key]
        if isinstance(value, dict):
            value = value[layer_kind]
        return dtype(value) if isinstance(value, str) and value in {"bf16", "bfp8", "bfp4"} else value

    return {
        "dtype": select("activation_dtype"),
        "attention_dtype": select("attention_dtype"),
        "attention_qkv_dtype": select("attention_qkv_dtype"),
        "attention_o_dtype": select("attention_o_dtype"),
        "shared_mlp_dtype": select("shared_mlp_dtype"),
        "shared_mlp_down_dtype": select("shared_mlp_down_dtype"),
        "shared_mlp_decode_dtype": select("shared_mlp_decode_dtype"),
        "shared_mlp_decode_down_dtype": select("shared_mlp_decode_down_dtype"),
        "kv_cache_dtype": select("kv_cache_dtype"),
        "fuse_mlp_gelu": select("fuse_mlp_gelu"),
        "decode_norm_sharded": select("decode_norm_sharded"),
        "attention_decode_o_interleaved": select("attention_decode_o_interleaved"),
    }


def to_tt(tensor, device, *, layout=ttnn.TILE_LAYOUT, tt_dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        tensor,
        device=device,
        layout=layout,
        dtype=tt_dtype,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def build_state(device, policy: dict, layer_kind: str, batch: int = 32):
    from models.autoports.google_gemma_4_12b.tt.optimized_decoder import OptimizedDecoder

    config = functional._hf_text_config()
    layer_idx = functional._find_layer_idx(config, layer_kind)
    hf_layer = functional._synthetic_hf_layer(config, layer_idx)
    decoder = OptimizedDecoder.from_state_dict(
        hf_layer.state_dict(),
        hf_config=config,
        layer_idx=layer_idx,
        mesh_device=device,
        mesh_config=MeshConfig(device.shape, decode=ModeConfig(tp=1)),
        **ctor_policy(policy, layer_kind),
    )

    blocks_per_user = math.ceil((CONTEXT + 1) / BLOCK_SIZE)
    max_num_blocks = batch * blocks_per_user
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(batch, blocks_per_user)
    page_table_tt = to_tt(page_table, device, layout=ttnn.ROW_MAJOR_LAYOUT, tt_dtype=ttnn.int32)
    kv_cache = decoder.create_paged_kv_cache(block_size=BLOCK_SIZE, max_num_blocks=max_num_blocks)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    max_seq_len = blocks_per_user * BLOCK_SIZE
    rope = Gemma4TextRotaryEmbedding(config)
    dummy = torch.zeros(1, max_seq_len, config.hidden_size)
    positions = torch.arange(max_seq_len).unsqueeze(0)
    cos, sin = rope(dummy, positions, layer_type=config.layer_types[layer_idx])
    # Match the executed full-model create_rope_caches policy.  The older
    # decoder test helper still creates these decode caches tiled.
    rope2 = (
        to_tt(cos[0].to(torch.bfloat16), device, layout=ttnn.ROW_MAJOR_LAYOUT),
        to_tt(sin[0].to(torch.bfloat16), device, layout=ttnn.ROW_MAJOR_LAYOUT),
    )

    torch.manual_seed(12000 + layer_idx)
    hidden = to_tt(torch.randn(1, 1, batch, config.hidden_size, dtype=torch.bfloat16), device)
    pos_u32 = to_tt(
        torch.full((1, batch), CONTEXT, dtype=torch.uint32),
        device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        tt_dtype=ttnn.uint32,
    )
    pos_i32 = to_tt(
        torch.full((batch,), CONTEXT, dtype=torch.int32),
        device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        tt_dtype=ttnn.int32,
    )
    return {
        "decoder": decoder,
        "hidden": hidden,
        "rope": rope2,
        "page_table": page_table_tt,
        "kv_cache": kv_cache,
        "position": pos_u32,
        "position_cache": pos_i32,
        "hf_layer": hf_layer,
        "layer_idx": layer_idx,
        "config": config,
    }


def decode(state):
    return state["decoder"].decode_forward(
        state["hidden"],
        rope_mats=state["rope"],
        page_table=state["page_table"],
        kv_cache=state["kv_cache"],
        position_idx=state["position"],
        position_idx_cache=state["position_cache"],
    )
