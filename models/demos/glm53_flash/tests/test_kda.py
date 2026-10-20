# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA layer (prefill via ttKDA, decode composite) vs HF Glm5NextTextLinearAttention, real weights.

Run on one Blackhole chip of the Galaxy:
    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_kda.py -s
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
from models.demos.glm53_flash.tt.kda import TtKDA, device_scalar_u32

LAYER = int(os.environ.get("GLM53_KDA_LAYER", "0"))


def _hf_linear_attention(snapshot, text_config, raw_attn_prefixed):
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention

    sd = {k[len("self_attn.") :]: v for k, v in convert_to_hf_module_keys(raw_attn_prefixed).items()}
    with torch.device("meta"):
        hf = Glm5NextTextLinearAttention(text_config, LAYER)
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
def test_kda_prefill_then_decode(mesh_device, weights, seq_len):
    from transformers import DynamicCache

    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    hf = _hf_linear_attention(snapshot, text, raw)
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    tt = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER)

    torch.manual_seed(0)
    n_decode = 3
    T = seq_len
    x = (torch.randn(1, T + n_decode, cfg.hidden_size) * 0.8).to(torch.bfloat16)
    cache = DynamicCache(config=text)
    with torch.no_grad():
        y_hf = hf(x[:, :T], cache_params=cache, attention_mask=torch.ones(1, T, dtype=torch.bool))
    hf_layer = cache.layers[LAYER]

    # ---- prefill on device: pad to 32 rows, actual_end marks the logical length
    T_phys = -(-T // 32) * 32
    x_pad = torch.zeros(1, T_phys, cfg.hidden_size, dtype=torch.bfloat16)
    x_pad[:, :T] = x[:, :T]
    x_tt = ttnn.from_torch(x_pad, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    state = tt.allocate_state()
    y_tt, state = tt.prefill_forward(
        x_tt, state, device_scalar_u32(mesh_device, 0), device_scalar_u32(mesh_device, T) if T != T_phys else None
    )
    y = ttnn.to_torch(y_tt)[:, :T]
    p = pcc(y_hf, y)
    logger.info(f"KDA layer {LAYER} prefill T={T}: pcc {p:.6f}")
    rec = ttnn.to_torch(state.recurrent)
    p_rec = pcc(hf_layer.recurrent_states[0], rec)
    conv_hf = hf_layer.conv_states[0][:, :, -3:].transpose(1, 2)  # [1,3,W]
    p_conv = pcc(conv_hf, ttnn.to_torch(state.convolution))
    logger.info(f"  recurrent state pcc {p_rec:.6f}   conv history pcc {p_conv:.6f}")
    assert p > 0.995 and p_rec > 0.995 and p_conv > 0.995

    # ---- decode steps continuing from the device state
    for step in range(n_decode):
        xt = x[:, T + step : T + step + 1]
        with torch.no_grad():
            y_hf_d = hf(xt, cache_params=cache, attention_mask=torch.ones(1, 1, dtype=torch.bool))
        xt_tt = ttnn.from_torch(
            xt.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
        )
        y_tt_d, state = tt.decode_forward(xt_tt, state)
        y_d = ttnn.to_torch(y_tt_d).reshape(1, 1, -1)
        p_d = pcc(y_hf_d, y_d)
        p_rec = pcc(hf_layer.recurrent_states[0], ttnn.to_torch(state.recurrent))
        logger.info(f"  decode step {step} (pos {T + step}): pcc {p_d:.6f}  state pcc {p_rec:.6f}")
        assert p_d > 0.995 and p_rec > 0.995


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("batch", [4, 32])
def test_kda_block_vs_composite(mesh_device, weights, batch):
    """The 5-op block decode (``kda_decode``) vs the composite decode on real weights: ``batch`` users, 10
    decode steps on the same evolving recurrent state (from a random prefill-like carry), output PCC per
    step and the final state PCC; the HF reference for user 0 as the absolute anchor."""
    snapshot, text, raw = weights
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    attn = {k[len("self_attn.") :]: v for k, v in raw.items()}
    comp = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="composite")
    blk = TtKDA(mesh_device, cfg, attn, layer_idx=LAYER, fused_decode="block")
    hf = _hf_linear_attention(snapshot, text, raw)
    torch.manual_seed(1)
    B, H, K, V = batch, comp.H, comp.K, comp.V
    S0 = 0.05 * torch.randn(B, H, K, V)
    conv0 = (0.5 * torch.randn(B, 3, comp.conv_width)).to(torch.bfloat16)
    states = []
    for tt in (comp, blk):
        st = tt.allocate_state(B)
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(S0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), st.recurrent)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(conv0, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT), st.convolution
        )
        states.append(st)
    # the HF module tracks user 0 through its cache
    from transformers import DynamicCache

    cache = DynamicCache(config=text)
    with torch.no_grad():  # seed the HF cache with the same carry (user 0)
        x_seed = torch.zeros(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        hf(x_seed, cache_params=cache, attention_mask=torch.ones(1, 4, dtype=torch.bool))
    layer_cache = cache.layers[LAYER]
    layer_cache.recurrent_states[0] = S0[:1].clone()
    layer_cache.conv_states[0][:, :, -3:] = conv0[:1].float().transpose(1, 2)
    n_steps = 10
    worst_o, worst_hf = 1.0, 1.0
    for step in range(n_steps):
        xt = (torch.randn(1, 1, B, cfg.hidden_size) * 0.8).to(torch.bfloat16)
        x_dev = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
        y_c, states[0] = comp.decode_forward(x_dev, states[0])
        y_b, states[1] = blk.decode_forward(x_dev, states[1])
        yc, yb = ttnn.to_torch(y_c).float()[0, 0], ttnn.to_torch(y_b).float()[0, 0]
        p = pcc(yc, yb)
        with torch.no_grad():
            y_hf = hf(xt[:, :, 0], cache_params=cache, attention_mask=torch.ones(1, 1, dtype=torch.bool))
        p_hf_b = pcc(y_hf[0], yb[:1])
        p_hf_c = pcc(y_hf[0], yc[:1])
        worst_o, worst_hf = min(worst_o, p), min(worst_hf, p_hf_b)
        logger.info(
            f"  step {step}: block vs composite pcc {p:.6f}; vs HF (user 0): block {p_hf_b:.6f} composite {p_hf_c:.6f}"
        )
    s_c, s_b = ttnn.to_torch(states[0].recurrent), ttnn.to_torch(states[1].recurrent)
    p_state = pcc(s_c, s_b)
    p_state_hf = pcc(layer_cache.recurrent_states[0], s_b[:1])
    p_conv = pcc(ttnn.to_torch(states[0].convolution).float(), ttnn.to_torch(states[1].convolution).float())
    logger.info(
        f"KDA layer {LAYER} B={B} after {n_steps} steps: state pcc block vs composite {p_state:.6f} (vs HF user 0 "
        f"{p_state_hf:.6f}), conv history pcc {p_conv:.6f}; worst output pcc {worst_o:.6f} (vs HF {worst_hf:.6f})"
    )
    assert worst_o > 0.9999 and p_state > 0.9999 and p_conv > 0.9999
    assert worst_hf > 0.999 and p_state_hf > 0.999
