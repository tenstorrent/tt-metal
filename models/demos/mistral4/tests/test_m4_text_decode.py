# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-level decode PCC: incremental token-by-token text-model decode vs the prefill logit golden.

Drives TtMistral4TextModel.forward_decode (per-layer KV caches; MLA forward_decode + MoE per token)
over the reference sequence and matches the prefill golden logits — i.e. the full decode path
(KV cache across all layers + LM head) reproduces the prefill result.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2


def _repl(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_text_decode(mesh_device, reset_seeds):
    pcc_required = 0.98
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, N_LAYERS, 0, 32)
    ids, cos, sin = g["input_ids"], g["rope_cos"], g["rope_sin"]
    B, S, rope = ids.shape[0], ids.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    kv = tt.init_kv_caches(B, max_seq=64)
    embed = tsd["model.embed_tokens.weight"][ids]  # [B,S,hidden]

    outs = []
    for i in range(S):
        h_i = _repl(embed[:, i : i + 1, :], mesh_device)
        c_i = _repl(cos[:, i : i + 1, :].reshape(B, 1, 1, rope), mesh_device)
        s_i = _repl(sin[:, i : i + 1, :].reshape(B, 1, 1, rope), mesh_device)
        o = tt.forward_decode(h_i, i, c_i, s_i, kv)
        outs.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B])
    out = torch.cat(outs, dim=1)  # [B, S, vocab]
    passing, msg = comp_pcc(g["logits"], out, pcc_required)
    logger.info(f"Mistral-Small-4 {N_LAYERS}-layer decode logit PCC: {msg}")
    assert passing, f"decode logit PCC below {pcc_required}: {msg}"
