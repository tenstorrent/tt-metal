# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill (criteria A6/C1/C6) vs single-shot prefill: TtMistral4MLA.forward_prefill_chunked
(paged k/v + per-chunk chunked_scaled_dot_product_attention) must match forward_prefill (single-shot
SDPA) — same causal math. Self-consistency on random input at S=256 (2x128 chunks). Unlocks ISL >> the
single-shot ~4K L1 cap. B=1."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference, load_m4_weights
from models.demos.mistral4.tt.mistral4_text import TtMistral4MLA

B = 1
S = 256
CHUNK = 128


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_chunked_prefill(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    rope = cfg.qk_rope_head_dim
    ref, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    ref = ref.float()
    torch.manual_seed(0)
    x = torch.randn(B, S, cfg.hidden_size) * 0.02
    cos1, sin1 = ref.model.rotary_emb(x, torch.arange(S)[None])  # [1,S,rope]
    cosT, sinT = cos1[0].float(), sin1[0].float()

    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)
    rep = lambda t: ttnn.from_torch(
        t.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    xd = rep(x)
    cosd = rep(cosT.reshape(1, 1, S, rope))
    sind = rep(sinT.reshape(1, 1, S, rope))

    # single-shot reference (the current default prefill)
    kv = mla.init_kv_cache(B, S)
    ref_o = mla.forward_prefill(xd, cosd, sind, kv)
    ref_t = ttnn.to_torch(ref_o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]

    # chunked prefill over a paged cache
    paged, pt = mla.init_paged_kv_cache(B, S, block_size=CHUNK)
    chk_o = mla.forward_prefill_chunked(xd, cosd, sind, paged, pt, chunk=CHUNK, block_size=CHUNK)
    chk_t = ttnn.to_torch(chk_o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]

    passing, msg = comp_pcc(ref_t, chk_t, 0.99)
    logger.info(f"Mistral-Small-4 chunked-vs-singleshot prefill (S={S}, chunk={CHUNK}) PCC: {msg}")
    assert passing, f"chunked prefill PCC below 0.99: {msg}"
