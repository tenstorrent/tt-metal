# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DSA / MLA layer (dense prefill <= 2048, paged latent cache, absorbed decode) vs HF
Glm5NextTextAttention with real layer weights.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_dsa.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import (
    cast_like_from_pretrained,
    convert_to_hf_module_keys,
    load_config,
)
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache

LAYER = int(os.environ.get("GLM53_DSA_LAYER", "3"))


def _hf_attention(text_config, raw_attn_prefixed):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention

    sd = {k[len("self_attn.") :]: v for k, v in convert_to_hf_module_keys(raw_attn_prefixed).items()}
    with torch.device("meta"):
        hf = Glm5NextTextAttention(text_config, LAYER)
    hf = hf.to_empty(device="cpu")
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    cast_like_from_pretrained(hf)
    return hf


@pytest.fixture(scope="module")
def weights():
    snapshot = default_snapshot_dir()
    _, text = load_config(snapshot)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot), name_filter=lambda n: n.startswith("self_attn."))
    return snapshot, text, raw


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("seq_len", [96, 1000])
def test_dsa_prefill_then_decode(mesh_device, weights, seq_len):
    from transformers import DynamicCache

    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_attention(text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    tt = TtDSA(mesh_device, cfg, attn, layer_idx=LAYER)

    torch.manual_seed(0)
    n_decode = 3
    T = seq_len
    x = (torch.randn(1, T + n_decode, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf, _, _ = hf(x[:, :T], attention_mask=torch.ones(1, T, dtype=torch.bool), past_key_values=cache)

    block = 32
    T_phys = -(-T // block) * block
    tt_cache = TtLatentCache(mesh_device, cfg, max_batch=1, max_seq_len=T_phys + 64, block_size=block)
    x_pad = torch.zeros(1, 1, T_phys, cfg.hidden_size, dtype=torch.bfloat16)
    x_pad[:, :, :T] = x[:, :T]
    x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    y_tt = tt.prefill_forward(x_tt, tt_cache, user=0, seq_len=T)
    y = ttnn.to_torch(y_tt).reshape(1, T_phys, -1)[:, :T]
    p = pcc(y_hf, y)
    logger.info(f"DSA layer {LAYER} prefill T={T}: pcc {p:.6f}")
    # latent cache content vs HF (keys = latent [1,1,T,512])
    lat_hf = cache.layers[LAYER].keys[0, 0]  # [T, 512]
    lat_tt = ttnn.to_torch(tt_cache.cache).reshape(-1, cfg.kv_lora_rank)[:T]
    p_lat = pcc(lat_hf, lat_tt)
    logger.info(f"  latent cache pcc {p_lat:.6f}")
    assert p > 0.995 and p_lat > 0.995

    for step in range(n_decode):
        xt = x[:, T + step : T + step + 1]
        with torch.no_grad():
            y_hf_d, _, _ = hf(xt, attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache)
        xt_tt = ttnn.from_torch(
            xt.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        pos = ttnn.from_torch(
            torch.tensor([T + step], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
        )
        y_tt_d = tt.decode_forward(xt_tt, tt_cache, pos)
        y_d = ttnn.to_torch(y_tt_d).reshape(1, 1, -1)
        p_d = pcc(y_hf_d, y_d)
        logger.info(f"  decode step {step} (pos {T + step}): pcc {p_d:.6f}")
        assert p_d > 0.995
