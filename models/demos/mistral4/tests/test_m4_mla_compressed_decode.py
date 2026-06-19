# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A6 compressed-latent paged flash-MLA decode PCC vs the golden (expanded-kv) MLA output.

Incrementally decodes the reference sequence through TtMistral4MLA.forward_decode_mla (compressed
latent paged cache + weight absorption + paged_flash_multi_latent_attention_decode) for B=32 users
(tile-aligned batched decode) and matches the golden mla_out. The KV cache is 12.8x smaller than the
expanded path (kv_lora+rope=320 vs n_heads*qk=4096).
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4MLA

B = 32


def _repl(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


def _pos(p, mesh):
    return ttnn.from_torch(
        torch.tensor([p] * B, dtype=torch.int32), device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
    )


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}],
    indirect=True,
)
def test_m4_mla_compressed_decode(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, 1, 0, 32)
    mla_in, mla_out, cos, sin = g["mla_in"], g["mla_out"], g["rope_cos"], g["rope_sin"]
    S, rope = mla_in.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)
    cache, page_table = mla.init_compressed_cache(B, max_seq=64)

    outs = []
    for i in range(S):
        x_i = _repl(mla_in[:, i : i + 1, :].repeat(B, 1, 1), mesh_device)  # [B,1,hidden]
        c_i = _repl(cos[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(B, 1, 1, 1), mesh_device)
        s_i = _repl(sin[:, i : i + 1, :].reshape(1, 1, 1, rope).repeat(B, 1, 1, 1), mesh_device)
        o = mla.forward_decode_mla(x_i, _pos(i, mesh_device), c_i, s_i, cache, page_table)  # [B,1,hidden]
        outs.append(ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B])
    out = torch.cat(outs, dim=1)  # [B, S, hidden]

    # localization: step-0 (single token, attend-to-self) isolates per-step correctness from accumulation
    s0 = comp_pcc(mla_out[:, 0:1].repeat(B, 1, 1), out[:, 0:1], 0.99)
    logger.info(f"compressed MLA decode STEP-0 PCC: {s0[1]}")
    passing, msg = comp_pcc(mla_out.repeat(B, 1, 1), out, 0.99)
    logger.info(f"Mistral-Small-4 compressed-latent MLA decode (B={B}) PCC: {msg}")
    assert passing, f"compressed MLA decode PCC below 0.99: {msg}"
