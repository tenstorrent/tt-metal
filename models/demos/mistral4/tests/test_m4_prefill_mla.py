# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""A6 compressed-latent PREFILL: forward_prefill_mla (flash_mla_prefill over the kvl+rope latent, absorbed
q) vs the golden expanded-MLA output. Same compression as the compressed decode, over all S query
positions in one shot — should match the expanded MLA (no per-head k/v materialized)."""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import get_cached_golden, load_m4_weights
from models.demos.mistral4.tt.mistral4_generator import _repl
from models.demos.mistral4.tt.mistral4_text import TtMistral4MLA, TtMistral4TextModel

B = 1


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_prefill_mla(mesh_device, reset_seeds):
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, 1, 0, 32)
    mla_in, mla_out, cos, sin = g["mla_in"], g["mla_out"], g["rope_cos"], g["rope_sin"]
    S, rope = mla_in.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)

    x = _repl(mla_in, mesh_device)  # [1,S,hidden]
    cos_t = _repl(cos.reshape(1, 1, S, rope), mesh_device)
    sin_t = _repl(sin.reshape(1, 1, S, rope), mesh_device)
    out = ttnn.to_torch(
        mla.forward_prefill_mla(x, cos_t, sin_t), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    ).float()[:B]

    passing, msg = comp_pcc(mla_out, out, 0.99)
    logger.info(f"Mistral-Small-4 compressed-latent MLA PREFILL (S={S}) PCC: {msg}")
    assert passing, f"compressed MLA prefill PCC below 0.99: {msg}"


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_prefill_mla_chunked(mesh_device, reset_seeds):
    """Chunked compressed prefill (long-context path) over a paged latent cache, multi-chunk (S=256, 2x128).
    Validated vs both the golden expanded MLA and the single-shot compressed prefill."""
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    g = get_cached_golden(ckpt, 1, 0, 256)
    mla_in, mla_out, cos, sin = g["mla_in"], g["mla_out"], g["rope_cos"], g["rope_sin"]
    S, rope = mla_in.shape[1], cfg.qk_rope_head_dim

    tsd = load_m4_weights(ckpt, 1)
    layer_sd = {k[len("model.layers.0.") :]: v for k, v in tsd.items() if k.startswith("model.layers.0.")}
    mla = TtMistral4MLA(mesh_device, layer_sd, cfg, cfg.rms_norm_eps)

    x = _repl(mla_in, mesh_device)
    cos_t = _repl(cos.reshape(1, 1, S, rope), mesh_device)
    sin_t = _repl(sin.reshape(1, 1, S, rope), mesh_device)

    cache, pt = mla.init_paged_compressed_cache(B, max_seq=S, block_size=128)
    chunked = ttnn.to_torch(
        mla.forward_prefill_mla_chunked(x, cos_t, sin_t, cache, pt, chunk=128, block_size=128),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[:B]
    single = ttnn.to_torch(
        mla.forward_prefill_mla(x, cos_t, sin_t), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    ).float()[:B]

    pg, mg = comp_pcc(mla_out, chunked, 0.99)
    ps, ms = comp_pcc(single, chunked, 0.99)
    logger.info(
        f"Mistral-Small-4 CHUNKED compressed MLA prefill (S={S}, 128-chunks) vs golden: {mg}; vs single-shot: {ms}"
    )
    assert pg and ps, f"chunked prefill PCC below 0.99: golden {mg}, single-shot {ms}"


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000}], indirect=True
)
def test_m4_text_prefill_mla_chunked(mesh_device, reset_seeds):
    """Model-level long-context chunked prefill (chunk loop OUTSIDE the layer stack, MoE per-chunk) must
    produce the same last-token logits as the single-shot compressed prefill (both correct prefill, same
    weights). TT-vs-TT, 2 layers, S=256 (2x128 chunks) — validates the cross-chunk KV accumulation + the
    per-chunk MoE restructure. No golden needed."""
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    S, chunk, rope, hidden = 256, 128, cfg.qk_rope_head_dim, cfg.hidden_size
    tsd = load_m4_weights(ckpt, 2)
    tt = TtMistral4TextModel(
        mesh_device, tsd, cfg, 2, cfg.rms_norm_eps, shard_experts=True, expert_dtype=ttnn.bfloat8_b
    )

    torch.manual_seed(0)
    emb = _repl(torch.randn(1, S, hidden) * 0.02, mesh_device)
    cos_t = _repl(torch.randn(1, 1, S, rope), mesh_device)
    sin_t = _repl(torch.randn(1, 1, S, rope), mesh_device)

    single = ttnn.to_torch(
        tt.forward_prefill_mla(emb, cos_t, sin_t), mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)
    ).float()[
        :1
    ]  # [1,S,vocab]
    caches = tt.init_paged_compressed_caches(1, max_seq=S, block_size=128)
    chunked = ttnn.to_torch(
        tt.forward_prefill_mla_chunked(emb, cos_t, sin_t, caches, chunk=chunk, block_size=128),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    ).float()[
        :1
    ]  # [1,1,vocab] (final-position next-token logits)

    passing, msg = comp_pcc(single[:, S - 1 : S, :], chunked, 0.99)
    logger.info(f"Mistral-Small-4 2-layer chunk-outside prefill vs single-shot (final token, S={S}): {msg}")
    assert passing, f"model-level chunked prefill PCC below 0.99: {msg}"
