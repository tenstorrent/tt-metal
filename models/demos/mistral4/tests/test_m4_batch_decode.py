# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user batched decode: B users decode together; each user's logits match the single-user golden.

Prerequisite for batched serving (and the all-to-all sparse-dispatch MoE, which shards the batch
across devices). Tiles the 2-layer golden prompt across B identical users, decodes token-by-token
with batch=B through the per-user KV cache + batch-agnostic MoE, and PCCs each user's logit stream
against the single-user golden (small batched-SDPA reduction differences expected).
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
BATCH = 8


def _pos(positions, mesh):
    return ttnn.from_torch(
        torch.tensor(positions, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_batch_decode(mesh_device, reset_seeds):
    pcc_required = 0.98
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, N_LAYERS, 0, 32)
    ids, cos, sin = g["input_ids"], g["rope_cos"], g["rope_sin"]  # [1,S,*]
    S, rope = ids.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    kv = tt.init_kv_caches(BATCH, max_seq=64)
    emb = tsd["model.embed_tokens.weight"][ids]  # [1,S,H]

    outs = []
    for i in range(S):
        x_i = _repl(emb[:, i : i + 1, :].repeat(BATCH, 1, 1), mesh_device)  # [B,1,H] (B identical users)
        c_i = _repl(cos[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(BATCH, 1, 1, 1), mesh_device)
        s_i = _repl(sin[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(BATCH, 1, 1, 1), mesh_device)
        o = tt.forward_decode(x_i, _pos([i] * BATCH, mesh_device), c_i, s_i, kv)  # [B,1,vocab]
        outs.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:BATCH])
    out = torch.cat(outs, dim=1)  # [B, S, vocab]

    for u in range(BATCH):
        passing, msg = comp_pcc(g["logits"], out[u : u + 1], pcc_required)
        logger.info(f"user {u}: {msg}")
        assert passing, f"user {u} decode PCC below {pcc_required}: {msg}"
    logger.info(f"Mistral-Small-4 batched decode (B={BATCH}): all users >= {pcc_required}")
