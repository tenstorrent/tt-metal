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
from models.demos.glm53_flash.tt.dsa import TtDSA, TtLatentCache, decode_position_tensors

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
@pytest.mark.parametrize("seq_len", [1, 2, 3, 31, 33, 96, 1000])
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
        pos, pool_ids, ring_slots = decode_position_tensors(mesh_device, [T + step], cfg.index_kpool)
        y_tt_d = tt.decode_forward(xt_tt, tt_cache, pos, pool_ids, ring_slots)
        y_d = ttnn.to_torch(y_tt_d).reshape(1, 1, -1)
        p_d = pcc(y_hf_d, y_d)
        logger.info(f"  decode step {step} (pos {T + step}): pcc {p_d:.6f}")
        assert p_d > 0.995


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [4, 32])
def test_dsa_fused_vs_composite(mesh_device, weights, batch):
    """The 7-op fused decode block vs the composite decode on real layer weights: ``batch`` users, 10 decode steps on
    the same evolving caches (latent, index rings, pools), output PCC per step and the cache / ring / pool PCCs at
    the end; the HF reference for user 0 as the absolute anchor. The caches are bit-exact between the paths. The
    output gate is 0.9999 with ``GLM53_DSA_FOLD=none`` (the state program alone: 0.99990 measured) and 0.9998
    with the folded absorb weights (default: 0.99983 measured; the bfp8 products Wqb.Wkb1 / Wkb2.Wo round
    differently from the composite's two bfp8 matmuls with a bf16 intermediate, while vs HF the folded block is at
    or above the composite)."""
    from transformers import DynamicCache

    from models.demos.glm53_flash.tt.dsa import DSA_FOLD, TtLatentCache, decode_position_tensors

    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_attention(text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    comp = TtDSA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="composite")
    fused = TtDSA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="fused")
    B = batch
    T0 = 96  # user 0's prefill length; the other users start at a staggered position (same prefill, replayed)
    torch.manual_seed(2)
    x_pre = (torch.randn(1, T0, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    cache_hf = DynamicCache(config=text)
    with torch.no_grad():
        hf(x_pre, attention_mask=torch.ones(1, T0, dtype=torch.bool), past_key_values=cache_hf)
    caches = []
    for tt in (comp, fused):
        c = TtLatentCache(mesh_device, cfg, max_batch=B, max_seq_len=T0 + 64, block_size=32)
        x_tt = ttnn.from_torch(
            x_pre.reshape(1, 1, T0, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        for u in range(B):
            tt.prefill_forward(x_tt, c, user=u, seq_len=T0)
        caches.append(c)
    n_steps = 10
    worst, worst_hf = 1.0, 1.0
    for step in range(n_steps):
        xt = (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16)
        x_dev = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        pos = [T0 + step] * B
        ys = []
        for tt, c in zip((comp, fused), caches):
            p, pool_ids, ring_slots = decode_position_tensors(mesh_device, pos, cfg.index_kpool)
            ys.append(ttnn.to_torch(tt.decode_forward(x_dev, c, p, pool_ids, ring_slots)).float()[0, 0])
        pc = pcc(ys[0], ys[1])
        with torch.no_grad():
            y_hf, _, _ = hf(xt[:, :, 0], attention_mask=torch.ones(1, 1, dtype=torch.bool), past_key_values=cache_hf)
        p_hf_f, p_hf_c = pcc(y_hf[0], ys[1][:1]), pcc(y_hf[0], ys[0][:1])
        worst, worst_hf = min(worst, pc), min(worst_hf, p_hf_f)
        logger.info(
            f"  step {step}: fused vs composite pcc {pc:.6f}; vs HF (user 0): fused {p_hf_f:.6f} composite {p_hf_c:.6f}"
        )
    c0, c1 = caches
    p_lat = pcc(ttnn.to_torch(c0.cache).float(), ttnn.to_torch(c1.cache).float())
    p_pool = pcc(ttnn.to_torch(c0.index.pool_cache).float(), ttnn.to_torch(c1.index.pool_cache).float())
    rk0, rg0 = c0.index.rings()
    rk1, rg1 = c1.index.rings()
    p_rk = pcc(ttnn.to_torch(rk0).float(), ttnn.to_torch(rk1).float())
    p_rg = pcc(ttnn.to_torch(rg0).float(), ttnn.to_torch(rg1).float())
    logger.info(
        f"DSA layer {LAYER} B={B} after {n_steps} steps: latent cache pcc fused vs composite {p_lat:.6f}, pools {p_pool:.6f}, "
        f"ring k {p_rk:.6f} g {p_rg:.6f}; worst output pcc {worst:.6f} (vs HF {worst_hf:.6f})"
    )
    bar = 0.9999 if DSA_FOLD == "none" else 0.9998
    assert worst > bar and p_lat > 0.9999 and p_pool > 0.9999 and p_rk > 0.9999 and p_rg > 0.9999
    assert worst_hf > 0.995
