# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# I2I on-device single denoise step — multi-span mask + base_embeds + gen timestep.
#
# Run:
#   HY_NUM_LAYERS=2 HY_VIT_LAYERS=1 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_i2i_denoise_step.py -v -s --timeout=3600

from __future__ import annotations

import gc

import pytest
import torch

from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep

PCC_THR = 0.85 if h.NUM_LAYERS > 4 else 0.95


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_i2i_denoise_step_pcc(device):
    import ttnn

    host = h.build_host_bundle(cfg_factor=1)
    c = h.model_cfg()
    down_sd = h.load_prefix("patch_embed")
    up_sd = h.load_prefix("final_layer")
    latent_ch, hid, hsz = h.pe_dims(down_sd)
    h_dim = c["H"]
    seq_len = host["seq_len"]
    img_slice = host["img_slice"]
    grid = host["grid_hw"]
    n_img = grid[0] * grid[1]

    torch.manual_seed(0)
    latent = torch.randn(1, latent_ch, grid[0], grid[1])
    t_scalar = 0.42
    timestep_emb = h.ref_timestep_emb(h_dim)

    ref_pred = h.ref_i2i_step(
        c,
        latent,
        t_scalar,
        host["cond"],
        down_sd,
        up_sd,
        host["image_infos"],
        host["mask_add"],
        img_slice,
        timestep_emb,
    )

    t_emb1 = h.ref_time_embed("time_embed", h_dim, torch.tensor([t_scalar]))
    t_emb2 = h.ref_time_embed("time_embed_2", h_dim, torch.tensor([t_scalar]))
    t1_tt = ttnn.from_torch(t_emb1.reshape(1, 1, 1, h_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    t2_tt = ttnn.from_torch(t_emb2.reshape(1, 1, 1, h_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

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
    step = HunyuanTtDenoiseStep(
        device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=img_slice,
        grid_hw=grid,
        seq_len=seq_len,
    )

    step_embeds = h.prepare_step_host_embeds(host["cond"], t_scalar, timestep_emb)
    base_tt = h.upload_base_embeds(device, step_embeds)
    mask_tt = h.upload_mask(device, host["mask_add"])

    pred_tt = step(
        latent,
        base_embeds=base_tt,
        t_emb1=t1_tt,
        t_emb2=t2_tt,
        image_infos=host["image_infos"],
        attention_mask=mask_tt,
        batch=1,
    )
    pred = ttnn.to_torch(pred_tt).reshape(1, grid[0], grid[1], latent_ch).permute(0, 3, 1, 2)

    score = h.pcc(ref_pred, pred)
    print(f"I2I denoise step PCC={score:.6f} seq={seq_len} layers={h.NUM_LAYERS}")
    assert score >= PCC_THR, f"I2I step PCC {score:.6f} < {PCC_THR}"

    ttnn.deallocate(base_tt)
    ttnn.deallocate(mask_tt)
    ttnn.deallocate(pred_tt)
    ttnn.deallocate(t1_tt)
    ttnn.deallocate(t2_tt)
    gc.collect()
