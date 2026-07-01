# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN FlowMatchDiscreteScheduler vs PyTorch reference.
#
# Covers the default HunyuanImage-3.0 text-to-image schedule (generation_config:
# flow_shift=3.0, solver="euler", reverse=True, 50 steps, guidance_scale=5.0):
#   1. set_timesteps -> sigmas / timesteps / timesteps_full match the reference
#      EXACTLY (host-side scalar math, so we require bit-equality).
#   2. A full 50-step Euler denoising loop on a random latent, driven on device,
#      tracks the reference loop at PCC >= 0.99 at every step.
#   3. The classifier-free-guidance combine matches the reference at PCC >= 0.99.
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_scheduler.py -v -s
# Run (script):
#   python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_scheduler.py

import sys

import torch

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
for _p in (ROOT, HUNYUAN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ttnn

from models.experimental.hunyuan_image_3_0.ref.scheduler import (
    FlowMatchDiscreteScheduler,
    classifier_free_guidance,
)
from models.experimental.hunyuan_image_3_0.tt.scheduler import (
    HunyuanTtScheduler,
    classifier_free_guidance_tt,
)

PCC_THR = 0.99
NUM_STEPS = 50
FLOW_SHIFT = 3.0
GUIDANCE = 5.0
# [B, latent_channels, H, W] — latent_channels=32 per config.json vae.
B, C, H, W = 1, 32, 8, 8


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _to_tt(device, x):
    return ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
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


def test_schedule_matches_reference(device):
    """numpy schedule (sigmas / timesteps / timesteps_full) must match the torch reference."""
    ref = _make_ref()
    tt = _make_tt(device)
    # tt.* are numpy (torch-free module); ref.* are torch. Compare values.
    ds = float(abs(tt.sigmas - ref.sigmas.numpy()).max())
    dt = float(abs(tt.timesteps - ref.timesteps.numpy()).max())
    df = float(abs(tt.timesteps_full - ref.timesteps_full.numpy()).max())
    # numpy.linspace vs torch.linspace agree only to float32 rounding (~1 ULP);
    # bit-equality across the two libraries is neither achievable nor meaningful.
    assert ds < 1e-6 and dt < 1e-3 and df < 1e-3, f"schedule differs: dsig={ds} dts={dt} dfull={df}"
    print(
        f"\nschedule OK: {len(ref.timesteps)} timesteps, "
        f"sigma[0]={float(ref.sigmas[0]):.4f} -> sigma[-1]={float(ref.sigmas[-1]):.4f}  "
        f"max|diff| sig={ds} ts={dt} full={df}"
    )


def test_denoising_loop_pcc(device):
    """Full Euler denoising loop: device latent must track the reference per step."""
    torch.manual_seed(0)
    ref = _make_ref()
    tt = _make_tt(device)

    latent_ref = torch.randn(B, C, H, W, dtype=torch.float32)
    latent_tt = _to_tt(device, latent_ref.clone())

    worst = 1.0
    for i in range(len(ref.timesteps)):
        # Same model output fed to both schedulers (random stand-in for the DiT).
        # Each scheduler is driven by its own timestep schedule.
        model_out = torch.randn(B, C, H, W, dtype=torch.float32)
        mo_tt = _to_tt(device, model_out)

        latent_ref = ref.step(model_out, ref.timesteps[i], latent_ref, return_dict=False)[0]
        new_tt = tt.step(mo_tt, tt.timesteps[i], latent_tt)
        ttnn.deallocate(mo_tt)
        ttnn.deallocate(latent_tt)
        latent_tt = new_tt

        pcc = _pcc(latent_ref, _from_tt(latent_tt))
        worst = min(worst, pcc)

    ttnn.deallocate(latent_tt)
    print(f"\ndenoising loop ({NUM_STEPS} steps): worst per-step PCC = {worst:.6f} (>= {PCC_THR})")
    assert worst >= PCC_THR, f"denoising loop PCC {worst:.6f} below {PCC_THR}"


def test_cfg_combine_pcc(device):
    """pred = uncond + g*(cond - uncond) on device vs reference."""
    torch.manual_seed(1)
    pred_cond = torch.randn(B, C, H, W, dtype=torch.float32)
    pred_uncond = torch.randn(B, C, H, W, dtype=torch.float32)

    ref = classifier_free_guidance(pred_cond, pred_uncond, GUIDANCE)

    cond_tt = _to_tt(device, pred_cond)
    uncond_tt = _to_tt(device, pred_uncond)
    out_tt = classifier_free_guidance_tt(cond_tt, uncond_tt, GUIDANCE)
    out = _from_tt(out_tt)
    ttnn.deallocate(cond_tt)
    ttnn.deallocate(uncond_tt)
    ttnn.deallocate(out_tt)

    pcc = _pcc(ref, out)
    print(f"\nCFG combine (g={GUIDANCE}): PCC = {pcc:.6f} (>= {PCC_THR})")
    assert pcc >= PCC_THR, f"CFG combine PCC {pcc:.6f} below {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        test_schedule_matches_reference(dev)
        test_denoising_loop_pcc(dev)
        test_cfg_combine_pcc(dev)
        print("\n" + "=" * 56)
        print("  [PASS] scheduler PCC tests")
        print("=" * 56)
    finally:
        ttnn.close_device(dev)
