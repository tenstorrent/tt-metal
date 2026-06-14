# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-layer logit PCC for the Mistral-Small-4 text core (embed -> N layers -> norm -> lm_head).

Validates the full forward assembly + LM head through the model-local module classes, on a small
layer count (memory: replicated dense experts cost ~6.4 GB/layer; full 36-layer depth needs the
expert-sharding optimization). Embedding is a host row-gather (trivial); RoPE cos/sin fed from the
reference (position-only, shared across layers).
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tests.multimodal.mistral_24b.m4_text_reference import get_cached_golden, load_m4_weights
from models.tt_transformers.tt.multimodal.mistral_24b.mistral4_text import TtMistral4TextModel

N_LAYERS = int(os.environ.get("M4_N_LAYERS", "2"))
SHARD = os.environ.get("M4_SHARD", "0") == "1"
EXPERT_DTYPE = {"bf16": "bfloat16", "bfp8": "bfloat8_b"}[os.environ.get("M4_EXPERT_DTYPE", "bf16")]
SEED, SEQ = 0, 32


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_text_model_logits(mesh_device, reset_seeds):
    pcc_required = 0.98  # full forward stacks bf16 error across layers + on-device routing
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config

    # golden: cached HF reference (built once); weights: loaded directly from checkpoint (fast,
    # no HF model build/forward) — decoupled so TT iterations never pay the ~40-min reference.
    g = get_cached_golden(ckpt, N_LAYERS, SEED, SEQ)
    sd = load_m4_weights(ckpt, N_LAYERS)
    ids = g["input_ids"]
    B, S, rope = ids.shape[0], ids.shape[1], cfg.qk_rope_head_dim

    edtype = getattr(ttnn, EXPERT_DTYPE)
    tt_model = TtMistral4TextModel(
        mesh_device, sd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=SHARD, expert_dtype=edtype
    )
    logger.info(f"text model: N_LAYERS={N_LAYERS} shard_experts={SHARD} expert_dtype={edtype}")

    # embedding: host row-gather (trivial lookup) -> device
    embed = sd["model.embed_tokens.weight"].detach()[ids]  # [B,S,hidden]

    def _from(t, shape=None):
        return ttnn.from_torch(
            (t if shape is None else t.view(shape)).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    hidden = _from(embed)
    cos = _from(g["rope_cos"], (B, 1, S, rope))
    sin = _from(g["rope_sin"], (B, 1, S, rope))

    logits = tt_model(hidden, cos, sin)
    logits_t = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]
    passing, msg = comp_pcc(g["logits"], logits_t, pcc_required)
    logger.info(f"Mistral-Small-4 {N_LAYERS}-layer logit PCC: {msg}")
    assert passing, f"logit PCC below {pcc_required}: {msg}"
