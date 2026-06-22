# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# I2I multi-step denoise loop with CFG — multi-span mask + base_embeds + gen timestep.
#
# Run:
#   HY_NUM_LAYERS=2 HY_CFG=2 HY_STEPS=2 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_i2i_denoise_loop.py -v -s --timeout=7200

from __future__ import annotations

import gc

import pytest
import torch

from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler

STEPS = int(__import__("os").environ.get("HY_STEPS", "2"))
GUIDANCE = float(__import__("os").environ.get("HY_GUIDANCE", "5.0"))
PCC_THR = 0.80 if h.NUM_LAYERS > 4 else 0.90


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_i2i_denoise_loop_cfg_pcc(device):
    host = h.build_host_bundle(cfg_factor=2)
    assert host["uncond"] is not None

    c = h.model_cfg()
    down_sd = h.load_prefix("patch_embed")
    up_sd = h.load_prefix("final_layer")
    latent_ch, hid, hsz = h.pe_dims(down_sd)
    h_dim = c["H"]
    seq_len = host["seq_len"]
    img_slice = host["img_slice"]
    grid = host["grid_hw"]

    torch.manual_seed(1)
    init_latent = torch.randn(1, latent_ch, grid[0], grid[1])
    timestep_emb = h.ref_timestep_emb(h_dim)

    sched_ref = HunyuanTtScheduler(device)
    sched_ref.set_timesteps(STEPS)
    sigmas = sched_ref.sigmas
    timesteps = sched_ref.timesteps

    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        pred = h.ref_i2i_step(
            c,
            lat,
            float(t),
            host["cond"],
            down_sd,
            up_sd,
            host["image_infos"],
            host["mask_add"],
            img_slice,
            timestep_emb,
        )
        pred_u = h.ref_i2i_step(
            c,
            lat,
            float(t),
            host["uncond"],
            down_sd,
            up_sd,
            host["image_infos"],
            host["uncond"]["attention_mask"],
            img_slice,
            timestep_emb,
        )
        pred_cfg = pred_u + GUIDANCE * (pred - pred_u)
        lat = lat + float(sigmas[i + 1] - sigmas[i]) * pred_cfg
    ref_final = lat
    gc.collect()

    patch_embed = HunyuanTtUNetDown(
        device,
        {f"patch_embed.{k}": v for k, v in down_sd.items()},
        in_channels=latent_ch,
        hidden_channels=hid,
        out_channels=hsz,
    )
    final_layer = HunyuanTtUNetUp(
        device,
        {f"final_layer.{k}": v for k, v in up_sd.items()},
        in_channels=hsz,
        hidden_channels=hid,
        out_channels=latent_ch,
    )
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    backbone = HunyuanTtModel(
        device,
        num_layers=h.NUM_LAYERS,
        hidden_size=h_dim,
        num_heads=c["HEADS"],
        num_kv_heads=c["KV_HEADS"],
        head_dim=c["HEAD_DIM"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        use_qk_norm=c["USE_QK_NORM"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=True,
        layer_loader=layer_loader,
        apply_final_norm=False,
    )
    time_embed = HunyuanTtTimestepEmbedder(
        device, h_dim, {f"time_embed.{k}": v for k, v in h.load_prefix("time_embed").items()}, "time_embed"
    )
    time_embed_2 = HunyuanTtTimestepEmbedder(
        device, h_dim, {f"time_embed_2.{k}": v for k, v in h.load_prefix("time_embed_2").items()}, "time_embed_2"
    )
    step = HunyuanTtDenoiseStep(
        device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=img_slice,
        grid_hw=grid,
        seq_len=seq_len,
    )

    cond_tt = h.upload_loop_cond(device, host["cond"])
    uncond_tt = h.upload_loop_cond(device, host["uncond"])
    sched_tt = HunyuanTtScheduler(device)
    sched_tt.set_timesteps(STEPS)
    tt_final = denoise_loop(
        step,
        sched_tt,
        init_latent.clone(),
        time_embed=time_embed,
        time_embed_2=time_embed_2,
        cond=cond_tt,
        uncond=uncond_tt,
        guidance_scale=GUIDANCE,
        timestep_emb=timestep_emb,
    )

    score = h.pcc(ref_final, tt_final)
    print(f"I2I denoise loop+CFG PCC={score:.6f} steps={STEPS} seq={seq_len}")
    assert score >= PCC_THR, f"I2I loop PCC {score:.6f} < {PCC_THR}"
    gc.collect()
