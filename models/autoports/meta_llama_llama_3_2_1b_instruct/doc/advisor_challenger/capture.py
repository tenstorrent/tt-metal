"""Batch-32 dense-layer capture target based on capture_template.py."""
import json
import os
import subprocess
from types import SimpleNamespace

import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.optimized_decoder import (
    OptimizedDecoder,
    precision_policy_from_config,
)

_DECODER = None
_KWARGS = None


def decode(hidden):
    return _DECODER.decode_forward(hidden, **_KWARGS)


def make_inputs(device):
    global _DECODER, _KWARGS
    policy = json.load(open(os.environ["CHALLENGER_INCUMBENT_JSON"]))["shipped_policy"]
    cfg = SimpleNamespace(
        hidden_size=2048, intermediate_size=8192, num_attention_heads=32,
        num_key_value_heads=8, head_dim=64, max_position_embeddings=131072,
        rms_norm_eps=1e-5, attention_bias=False, mlp_bias=False,
    )
    gen = torch.Generator().manual_seed(20260731)
    rand = lambda shape: torch.randn(shape, generator=gen, dtype=torch.bfloat16) * 0.02
    state = {
        "model.layers.0.self_attn.q_proj.weight": rand((2048, 2048)),
        "model.layers.0.self_attn.k_proj.weight": rand((512, 2048)),
        "model.layers.0.self_attn.v_proj.weight": rand((512, 2048)),
        "model.layers.0.self_attn.o_proj.weight": rand((2048, 2048)),
        "model.layers.0.mlp.gate_proj.weight": rand((8192, 2048)),
        "model.layers.0.mlp.up_proj.weight": rand((8192, 2048)),
        "model.layers.0.mlp.down_proj.weight": rand((2048, 8192)),
        "model.layers.0.input_layernorm.weight": torch.ones(2048, dtype=torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(2048, dtype=torch.bfloat16),
    }
    _DECODER = OptimizedDecoder.from_state_dict(
        state, hf_config=cfg, layer_idx=0, mesh_device=device, page_block_size=64,
        max_seq_len=64, max_batch_size=32, precision_policy=precision_policy_from_config(policy),
    )
    _DECODER.mlp._advisor_capture = True
    hidden = ttnn.from_torch(
        rand((1, 1, 32, 2048)), device=device, dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    hidden = ttnn.to_memory_config(hidden, _DECODER.decode_input_memcfg)
    positions = torch.zeros(32, dtype=torch.int32)
    current_pos = ttnn.from_torch(
        positions, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rot_mem = ttnn.create_sharded_memory_config(
        shape=(32, 64), core_grid=ttnn.num_cores_to_corerangeset(
            32, device.compute_with_storage_grid_size(), row_wise=True
        ), strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    cos = ttnn.from_torch(torch.ones(1, 32, 1, 64, dtype=torch.bfloat16), device=device,
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=rot_mem)
    sin = ttnn.from_torch(torch.zeros(1, 32, 1, 64, dtype=torch.bfloat16), device=device,
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=rot_mem)
    page_table = ttnn.from_torch(
        torch.arange(32, dtype=torch.int32).reshape(32, 1), device=device, dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    rot_mats = (cos, sin)
    _KWARGS = {"current_pos": current_pos, "rot_mats": rot_mats, "page_table": page_table}
    return (hidden,)


def record_provenance(out_dir):
    incumbent = json.load(open(os.environ["CHALLENGER_INCUMBENT_JSON"]))
    commit = subprocess.check_output(
        ["git", "-C", os.environ["TTMLIR_ADVISOR_HOME"], "rev-parse", "HEAD"], text=True
    ).strip()
    with open(os.path.join(out_dir, "traced_dtypes.json"), "w") as fh:
        json.dump({
            "layer_kind": "dense", "layer_idx": 0, "batch": 32,
            "traced_weight_dtypes": incumbent["shipped_weight_dtypes"],
            "shipped_weight_dtypes": incumbent["shipped_weight_dtypes"],
            "policy_source": incumbent["shipped_policy_source"],
            "advisor_commit": commit, "advisor_pin_expected": "618cd4e75d",
            "advisor_home": os.environ["TTMLIR_ADVISOR_HOME"],
        }, fh, indent=2)
