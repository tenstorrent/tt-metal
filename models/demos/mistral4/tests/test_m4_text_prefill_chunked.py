# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-level chunked prefill (criteria A6/C1/C6) vs single-shot: TtMistral4TextModel.forward_prefill_chunked
(paged k/v + per-chunk chunked SDPA) must match forward_prefill logits. 2 layers, B=1, S=256 (2x128 chunks),
self-consistency on random hidden. Proves the chunked path threads correctly through the decoder stack."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import load_m4_text_reference, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4TextModel

N_LAYERS = 2
B = 1
S = 256
CHUNK = 128


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_text_prefill_chunked(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    rope = cfg.qk_rope_head_dim
    ref, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    ref = ref.float()
    torch.manual_seed(0)
    h = torch.randn(B, S, cfg.hidden_size) * 0.02
    cos1, sin1 = ref.model.rotary_emb(h, torch.arange(S)[None])
    cosT, sinT = cos1[0].float(), sin1[0].float()

    tsd = load_m4_weights(ckpt, N_LAYERS)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, N_LAYERS, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )
    hd = _repl(h, mesh_device)
    cosd = _repl(cosT.reshape(1, 1, S, rope), mesh_device)
    sind = _repl(sinT.reshape(1, 1, S, rope), mesh_device)

    kv = tt.init_kv_caches(B, S)
    ref_o = tt.forward_prefill(hd, cosd, sind, kv)
    ref_t = ttnn.to_torch(ref_o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]

    caches = tt.init_paged_kv_caches(B, S, block_size=CHUNK)
    chk_o = tt.forward_prefill_chunked(hd, cosd, sind, caches, chunk=CHUNK, block_size=CHUNK)
    chk_t = ttnn.to_torch(chk_o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]

    passing, msg = comp_pcc(ref_t, chk_t, 0.98)
    logger.info(f"Mistral-Small-4 2-layer chunked-vs-singleshot prefill logits (S={S}) PCC: {msg}")
    assert passing, f"chunked prefill logit PCC below 0.98: {msg}"
