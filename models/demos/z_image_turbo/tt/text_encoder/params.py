# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Weight loading for TextEncoder (Qwen3) TTNN model."""

import torch

import ttnn

NUM_LAYERS = 36
HEAD_DIM = 128
ROPE_THETA = 1_000_000.0


def load_weights(mesh_device, pt_model):
    """Load all weights from a PyTorch Qwen3Model and convert to TTNN.

    Column-parallel (Q/K/V/gate/up): ShardTensorToMesh(dim=0)
    Row-parallel (O/down) + all-reduce: ShardTensorToMesh(dim=1)
    Replicated: embed_tokens, all RMSNorm weights
    """
    sd = {k: v.bfloat16() for k, v in pt_model.state_dict().items()}

    col_par = ttnn.ShardTensorToMesh(mesh_device, dim=0)
    row_par = ttnn.ShardTensorToMesh(mesh_device, dim=1)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    mem = ttnn.DRAM_MEMORY_CONFIG

    def _tile(t, mapper):
        return ttnn.from_torch(
            t,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
            device=mesh_device,
            memory_config=mem,
            mesh_mapper=mapper,
        )

    weights = {}

    weights["embed_tokens.weight"] = ttnn.from_torch(
        sd["embed_tokens.weight"],
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=mem,
        mesh_mapper=replicate,
    )

    weights["norm.weight"] = _tile(sd["norm.weight"], replicate)

    weights["inv_freq"] = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))

    _COL_PAR = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ]
    _ROW_PAR = [
        "self_attn.o_proj.weight",
        "mlp.down_proj.weight",
    ]
    _REPLICATED = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ]

    for i in range(NUM_LAYERS):
        for suffix in _COL_PAR:
            key = f"layers.{i}.{suffix}"
            weights[key] = _tile(sd[key], col_par)
        for suffix in _ROW_PAR:
            key = f"layers.{i}.{suffix}"
            weights[key] = _tile(sd[key], row_par)
        for suffix in _REPLICATED:
            key = f"layers.{i}.{suffix}"
            weights[key] = _tile(sd[key], replicate)

    return weights
