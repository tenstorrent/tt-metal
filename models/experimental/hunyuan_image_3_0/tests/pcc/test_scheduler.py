# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN FlowMatchDiscreteScheduler vs PyTorch reference.
# Smoke uses a small latent; slow cases use production 64×64 (1024² image) latents.

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import ttnn

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

from models.experimental.hunyuan_image_3_0.ref.scheduler import FlowMatchDiscreteScheduler, classifier_free_guidance
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS
from models.experimental.hunyuan_image_3_0.tt.scheduler import HunyuanTtScheduler, classifier_free_guidance_tt
from pcc_common import PCC_BLOCK, pcc_metrics

PCC_THR = PCC_BLOCK
NUM_STEPS = 50
FLOW_SHIFT = 3.0
GUIDANCE = 5.0
# Smoke: small spatial. Full: production diffusion latent (64×64 for 1024² decode).
B = 1
C = Z_CHANNELS
H_SMOKE, W_SMOKE = 8, 8
H_FULL, W_FULL = 64, 64


def _to_tt(device, x):
    return ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _from_tt(x):
    return ttnn.to_torch(x).float()


def _make_ref():
    s = FlowMatchDiscreteScheduler(shift=FLOW_SHIFT, reverse=True, solver="euler")
    s.set_timesteps(NUM_STEPS)
    return s


def _make_tt(device):
    s = HunyuanTtScheduler(device, shift=FLOW_SHIFT, reverse=True, solver="euler")
    s.set_timesteps(NUM_STEPS)
    return s


def _denoising_loop_pcc(device, h: int, w: int) -> float:
    torch.manual_seed(0)
    ref = _make_ref()
    tt = _make_tt(device)
    latent_ref = torch.randn(B, C, h, w, dtype=torch.float32)
    latent_tt = _to_tt(device, latent_ref.clone())
    worst = 1.0
    for i in range(len(ref.timesteps)):
        model_out = torch.randn(B, C, h, w, dtype=torch.float32)
        mo_tt = _to_tt(device, model_out)
        latent_ref = ref.step(model_out, ref.timesteps[i], latent_ref, return_dict=False)[0]
        new_tt = tt.step(mo_tt, tt.timesteps[i], latent_tt)
        mo_tt.deallocate(True)
        latent_tt.deallocate(True)
        latent_tt = new_tt
        p, _ = pcc_metrics(latent_ref, _from_tt(latent_tt), PCC_THR)
        worst = min(worst, p)
    latent_tt.deallocate(True)
    return worst


def _cfg_combine_pcc(device, h: int, w: int) -> float:
    torch.manual_seed(1)
    pred_cond = torch.randn(B, C, h, w, dtype=torch.float32)
    pred_uncond = torch.randn(B, C, h, w, dtype=torch.float32)
    ref = classifier_free_guidance(pred_cond, pred_uncond, GUIDANCE)
    cond_tt = _to_tt(device, pred_cond)
    uncond_tt = _to_tt(device, pred_uncond)
    out_tt = classifier_free_guidance_tt(cond_tt, uncond_tt, GUIDANCE)
    out = _from_tt(out_tt)
    cond_tt.deallocate(True)
    uncond_tt.deallocate(True)
    out_tt.deallocate(True)
    p, _ = pcc_metrics(ref, out, PCC_THR)
    return p


def test_schedule_matches_reference(device):
    ref = _make_ref()
    tt = _make_tt(device)
    ds = float(abs(tt.sigmas - ref.sigmas.numpy()).max())
    dt = float(abs(tt.timesteps - ref.timesteps.numpy()).max())
    df = float(abs(tt.timesteps_full - ref.timesteps_full.numpy()).max())
    assert ds < 1e-6 and dt < 1e-3 and df < 1e-3, f"schedule differs: dsig={ds} dts={dt} dfull={df}"


def test_denoising_loop_pcc(device):
    worst = _denoising_loop_pcc(device, H_SMOKE, W_SMOKE)
    assert worst >= PCC_THR


def test_cfg_combine_pcc(device):
    p = _cfg_combine_pcc(device, H_SMOKE, W_SMOKE)
    assert p >= PCC_THR


@pytest.mark.slow
def test_denoising_loop_full_latent_pcc(device):
    """Scheduler Euler update at production latent spatial size (64×64 / 1024² image)."""
    worst = _denoising_loop_pcc(device, H_FULL, W_FULL)
    print(f"scheduler denoise loop full latent [{B},{C},{H_FULL},{W_FULL}] worst PCC={worst:.8f}")
    assert worst >= PCC_THR


@pytest.mark.slow
def test_cfg_combine_full_latent_pcc(device):
    """CFG combine at production latent spatial size (64×64)."""
    p = _cfg_combine_pcc(device, H_FULL, W_FULL)
    print(f"scheduler CFG full latent [{B},{C},{H_FULL},{W_FULL}] PCC={p:.8f}")
    assert p >= PCC_THR
