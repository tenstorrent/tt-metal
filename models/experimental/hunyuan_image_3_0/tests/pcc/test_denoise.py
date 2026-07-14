# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated denoise PCC tests:
#   - Host-routed single step (patch_embed + backbone + final_layer)
#   - Multi-step denoise loop (single device + mesh resident)
#   - I2I denoise step + loop with CFG (multi-span mask, base_embeds)
#
# Lean layout: GRID=8 S=128 (fast); GRID=64 S=4160 production (slow).
#
# Run (fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_denoise.py -m "not slow" -v

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.experimental.hunyuan_image_3_0.ref.weights import load_prefixed_state_dict, resolve_base_model_dir
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as i2i
from models.experimental.hunyuan_image_3_0.tt.image_gen.patch_embed import HunyuanTtUNetDown, HunyuanTtUNetUp
from models.experimental.hunyuan_image_3_0.tt.image_gen.timestep_embedder import HunyuanTtTimestepEmbedder
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.pipeline import HunyuanTtDenoiseStep, denoise_loop
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler
from denoise_helpers import (
    denoise_steps,
    host_step_pcc_threshold,
    loop_pcc_threshold,
    num_layers_loop,
    num_layers_step,
    reference_host_step,
    reference_loop,
    resident_loop_pcc_threshold,
    run_denoise_loop_tt,
    run_host_routed_step_tt,
)
from pcc_common import PIPELINE_LAYOUT_FAST, PIPELINE_LAYOUT_PROD, pcc_metrics, transformer_cfg
from pipeline_helpers import patch_embed_dims, reference_time_embed

BATCH = 1
LAYOUT_FAST = [("fast", PIPELINE_LAYOUT_FAST)]
LAYOUT_SLOW = [("prod", PIPELINE_LAYOUT_PROD)]

I2I_STEPS = int(os.environ.get("HY_STEPS", "2"))
I2I_GUIDANCE = float(os.environ.get("HY_GUIDANCE", "5.0"))


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


@pytest.fixture(scope="function")
def device_params():
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}


def _host_step_run(device, layout: dict):
    c = transformer_cfg()
    num_layers = num_layers_step()
    down_sd = load_prefixed_state_dict(resolve_base_model_dir(), "patch_embed.")
    up_sd = load_prefixed_state_dict(resolve_base_model_dir(), "final_layer.")
    h = c["H"]
    grid = layout["grid"]
    s = layout["seq_len"]
    latent_ch, _, _ = patch_embed_dims(down_sd)
    thr = host_step_pcc_threshold(num_layers)

    torch.manual_seed(0)
    latent = torch.randn(BATCH, latent_ch, grid, grid)
    text_embeds = torch.randn(BATCH, s, h) * 0.02
    timesteps = torch.rand(BATCH)
    t_emb1 = reference_time_embed("time_embed", h, timesteps)
    t_emb2 = reference_time_embed("time_embed_2", h, timesteps)

    ref_pred, _ = reference_host_step(c, layout, num_layers, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd, BATCH)
    pred = run_host_routed_step_tt(
        device, c, layout, num_layers, latent, t_emb1, t_emb2, text_embeds, down_sd, up_sd, BATCH
    )
    p, d = pcc_metrics(ref_pred, pred, thr)
    return p, d, thr, num_layers


def _loop_run(device, layout: dict, mesh: bool = False):
    c = transformer_cfg()
    num_layers = num_layers_loop()
    steps = denoise_steps()
    down_sd = load_prefixed_state_dict(resolve_base_model_dir(), "patch_embed.")
    up_sd = load_prefixed_state_dict(resolve_base_model_dir(), "final_layer.")
    h = c["H"]
    grid = layout["grid"]
    s = layout["seq_len"]
    latent_ch, _, _ = patch_embed_dims(down_sd)
    thr = resident_loop_pcc_threshold() if mesh else loop_pcc_threshold()

    torch.manual_seed(0)
    init_latent = torch.randn(BATCH, latent_ch, grid, grid)
    text_embeds = torch.randn(BATCH, s, h) * 0.02

    ref_final = reference_loop(c, layout, num_layers, init_latent, text_embeds, down_sd, up_sd, steps, BATCH)
    gc.collect()
    tt_final = run_denoise_loop_tt(
        device, layout, num_layers, init_latent, text_embeds, steps, c, down_sd, up_sd, mesh=mesh
    )
    p, d = pcc_metrics(ref_final, tt_final, thr)
    return p, d, thr, steps, num_layers


# ---------------------------------------------------------------------------
# Host-routed single step (test_denoise_step.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag,layout", LAYOUT_FAST)
def test_host_routed_denoise_step(device, tag, layout):
    p, d, thr, layers = _host_step_run(device, layout)
    print(
        f"host-routed step [{tag}] GRID={layout['grid']} S={layout['seq_len']} "
        f"layers={layers}: PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}"
    )
    assert p >= thr


@pytest.mark.slow
@pytest.mark.parametrize("tag,layout", LAYOUT_SLOW)
def test_host_routed_denoise_step_production(device, tag, layout):
    p, d, thr, _ = _host_step_run(device, layout)
    assert p >= thr


# ---------------------------------------------------------------------------
# Multi-step loop (test_denoise_loop.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tag,layout", LAYOUT_FAST)
def test_denoise_loop_pcc(device, tag, layout):
    p, d, thr, steps, layers = _loop_run(device, layout)
    print(f"denoise loop [{tag}] steps={steps} layers={layers}: PCC={p:.8f}  max|diff|={d:.6f}  thr={thr}")
    assert p >= thr


# ---------------------------------------------------------------------------
# Mesh resident loop (test_denoise_loop_resident.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize("tag,layout", LAYOUT_FAST)
def test_denoise_loop_resident_mesh(mesh_device, tag, layout):
    mesh_device.enable_program_cache()
    p, d, thr, steps, layers = _loop_run(mesh_device, layout, mesh=True)
    logger.info(f"resident denoise loop [{tag}] steps={steps} layers={layers} PCC={p:.6f}  thr={thr}")
    assert p >= thr


# ---------------------------------------------------------------------------
# I2I denoise step (test_i2i_denoise_step.py)
# ---------------------------------------------------------------------------
def _i2i_denoise_step_run(device):
    host = i2i.build_host_bundle(cfg_factor=1)
    c = i2i.model_cfg()
    down_sd = i2i.load_prefix("patch_embed")
    up_sd = i2i.load_prefix("final_layer")
    latent_ch, hid, hsz = i2i.pe_dims(down_sd)
    h_dim = c["H"]
    thr = 0.85 if i2i.NUM_LAYERS > 4 else 0.95

    torch.manual_seed(0)
    latent = torch.randn(1, latent_ch, host["grid_hw"][0], host["grid_hw"][1])
    t_scalar = 0.42
    timestep_emb = i2i.ref_timestep_emb(h_dim)
    ref_pred = i2i.ref_i2i_step(
        c,
        latent,
        t_scalar,
        host["cond"],
        down_sd,
        up_sd,
        host["image_infos"],
        host["mask_add"],
        host["img_slice"],
        timestep_emb,
    )

    t_emb1 = i2i.ref_time_embed("time_embed", h_dim, torch.tensor([t_scalar]))
    t_emb2 = i2i.ref_time_embed("time_embed_2", h_dim, torch.tensor([t_scalar]))
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
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in i2i.load_prefix(f"model.layers.{i}").items()}
    backbone = HunyuanTtModel(
        device,
        num_layers=i2i.NUM_LAYERS,
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
        img_slice=host["img_slice"],
        grid_hw=host["grid_hw"],
        seq_len=host["seq_len"],
    )
    step_embeds = i2i.prepare_step_host_embeds(host["cond"], t_scalar, timestep_emb)
    base_tt = i2i.upload_base_embeds(device, step_embeds)
    mask_tt = i2i.upload_mask(device, host["mask_add"])
    pred_tt = step(
        latent,
        base_embeds=base_tt,
        t_emb1=t1_tt,
        t_emb2=t2_tt,
        image_infos=host["image_infos"],
        attention_mask=mask_tt,
        batch=1,
    )
    grid = host["grid_hw"]
    pred = ttnn.to_torch(pred_tt).reshape(1, grid[0], grid[1], latent_ch).permute(0, 3, 1, 2)
    p, d = pcc_metrics(ref_pred, pred, thr)
    for t in (base_tt, mask_tt, pred_tt, t1_tt, t2_tt):
        t.deallocate(True)
    gc.collect()
    return p, d, thr, host["seq_len"], i2i.NUM_LAYERS


@pytest.mark.skipif(not i2i.has_weights(), reason="Hunyuan I2I checkpoint not available")
@pytest.mark.slow
def test_i2i_denoise_step_pcc(device):
    p, d, thr, seq_len, layers = _i2i_denoise_step_run(device)
    print(f"I2I denoise step seq={seq_len} layers={layers}: PCC={p:.8f}")
    assert p >= thr


@pytest.mark.skipif(not i2i.has_weights(), reason="Hunyuan I2I checkpoint not available")
@pytest.mark.skipif(int(os.environ.get("HY_NUM_LAYERS", "2")) != 32, reason="requires HY_NUM_LAYERS=32")
@pytest.mark.slow
def test_i2i_denoise_step_production_32l_pcc(device):
    """I2I denoise step at full depth (32L) with real instruct bundle @ 1024²."""
    p, d, thr, seq_len, layers = _i2i_denoise_step_run(device)
    assert layers == 32
    print(f"I2I denoise step production seq={seq_len} layers={layers}: PCC={p:.8f}  thr={thr}")
    assert p >= thr


# ---------------------------------------------------------------------------
# I2I denoise loop + CFG (test_i2i_denoise_loop.py)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not i2i.has_weights(), reason="Hunyuan I2I checkpoint not available")
@pytest.mark.slow
def test_i2i_denoise_loop_cfg_pcc(device):
    host = i2i.build_host_bundle(cfg_factor=2)
    assert host["uncond"] is not None
    c = i2i.model_cfg()
    down_sd = i2i.load_prefix("patch_embed")
    up_sd = i2i.load_prefix("final_layer")
    latent_ch, hid, hsz = i2i.pe_dims(down_sd)
    h_dim = c["H"]
    thr = 0.80 if i2i.NUM_LAYERS > 4 else 0.90

    torch.manual_seed(1)
    grid = host["grid_hw"]
    init_latent = torch.randn(1, latent_ch, grid[0], grid[1])
    timestep_emb = i2i.ref_timestep_emb(h_dim)

    sched_ref = HunyuanTtScheduler(device)
    sched_ref.set_timesteps(I2I_STEPS)
    sigmas, timesteps = sched_ref.sigmas, sched_ref.timesteps
    lat = init_latent.clone()
    for i, t in enumerate(timesteps):
        pred = i2i.ref_i2i_step(
            c,
            lat,
            float(t),
            host["cond"],
            down_sd,
            up_sd,
            host["image_infos"],
            host["mask_add"],
            host["img_slice"],
            timestep_emb,
        )
        pred_u = i2i.ref_i2i_step(
            c,
            lat,
            float(t),
            host["uncond"],
            down_sd,
            up_sd,
            host["image_infos"],
            host["uncond"]["attention_mask"],
            host["img_slice"],
            timestep_emb,
        )
        lat = lat + float(sigmas[i + 1] - sigmas[i]) * (pred_u + I2I_GUIDANCE * (pred - pred_u))
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
    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in i2i.load_prefix(f"model.layers.{i}").items()}
    backbone = HunyuanTtModel(
        device,
        num_layers=i2i.NUM_LAYERS,
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
        device, h_dim, {f"time_embed.{k}": v for k, v in i2i.load_prefix("time_embed").items()}, "time_embed"
    )
    time_embed_2 = HunyuanTtTimestepEmbedder(
        device, h_dim, {f"time_embed_2.{k}": v for k, v in i2i.load_prefix("time_embed_2").items()}, "time_embed_2"
    )
    step = HunyuanTtDenoiseStep(
        device,
        patch_embed=patch_embed,
        backbone=backbone,
        final_layer=final_layer,
        img_slice=host["img_slice"],
        grid_hw=host["grid_hw"],
        seq_len=host["seq_len"],
    )
    cond_tt = i2i.upload_loop_cond(device, host["cond"])
    uncond_tt = i2i.upload_loop_cond(device, host["uncond"])
    sched_tt = HunyuanTtScheduler(device)
    sched_tt.set_timesteps(I2I_STEPS)
    tt_final = denoise_loop(
        step,
        sched_tt,
        init_latent.clone(),
        time_embed=time_embed,
        time_embed_2=time_embed_2,
        cond=cond_tt,
        uncond=uncond_tt,
        guidance_scale=I2I_GUIDANCE,
        timestep_emb=timestep_emb,
    )
    p, d = pcc_metrics(ref_final, tt_final, thr)
    print(f"I2I denoise loop+CFG steps={I2I_STEPS} seq={host['seq_len']}: PCC={p:.8f}")
    assert p >= thr
    gc.collect()
