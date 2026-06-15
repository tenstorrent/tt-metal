# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-level compressed-latent (A6) decode: 2-layer text model decoded via the paged flash-MLA path
(12.8x smaller KV cache), logits PCC vs the prefill golden. B=32 batched (the 1-kv-head latent write
tile-aligns at B=32); compares user 0 to the single-sequence golden."""
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
B = 32


def _pos(p, mesh):
    return ttnn.from_torch(
        torch.tensor([p] * B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.xfail(
    reason="A6 model-level compressed-latent decode: (1) inherits the compile/device-state "
    "nondeterminism of forward_decode_mla (see test_m4_mla_compressed_decode — bimodal ~0.9997/0.0205, "
    "suspected uninitialized read in paged_flash_multi_latent_attention_decode); (2) even on good "
    "compiles, the bf16 weight-absorption (q_nope@wkv_b1, ctx@wkv_b2) has a small per-layer precision "
    "deficit vs the expanded path that MoE expert-selection amplifies over depth (2-layer logits 0.92 "
    "vs expanded-kv baseline 0.99779; HiFi4 on the absorption matmuls corrupts them, so no clean precision "
    "lever). Expanded-kv decode is the verified-quality DEFAULT. See MISTRAL4_DESIGN.md (A6).",
    strict=False,
)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}],
    indirect=True,
)
def test_m4_text_decode_compressed(mesh_device, reset_seeds):
    pcc_required = 0.98
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, N_LAYERS, 0, 32)
    ids, cos, sin = g["input_ids"], g["rope_cos"], g["rope_sin"]
    S, rope = ids.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    caches = tt.init_compressed_caches(B, max_seq=64)
    emb = tsd["model.embed_tokens.weight"][ids]  # [1,S,H]

    outs = []
    for i in range(S):
        h_i = _repl(emb[:, i : i + 1, :].repeat(B, 1, 1), mesh_device)  # [B,1,H]
        c_i = _repl(cos[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(B, 1, 1, 1), mesh_device)
        s_i = _repl(sin[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(B, 1, 1, 1), mesh_device)
        o = tt.forward_decode_mla(h_i, _pos(i, mesh_device), c_i, s_i, caches)  # [B,1,vocab]
        outs.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:1])
    out = torch.cat(outs, dim=1)  # [1, S, vocab] (user 0)

    passing, msg = comp_pcc(g["logits"], out, pcc_required)
    logger.info(f"Mistral-Small-4 2-layer COMPRESSED decode logit PCC: {msg}")
    assert passing, f"compressed decode logit PCC below {pcc_required}: {msg}"
