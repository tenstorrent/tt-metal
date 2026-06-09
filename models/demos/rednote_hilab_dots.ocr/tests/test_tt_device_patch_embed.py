# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the dots.ocr device patch_embed (host Conv2d -> ttnn matmul+rms_norm).

The patch_embed Conv2d(3,1536,k=14,s=14) over non-overlapping patches is exactly a
matmul (flatten each patch to 588=3*14*14, C,H,W order). This migrates it from the
host (torch CPU, ~240 ms at 19,520 patches) to the device. The golden is the EXISTING
verified host_patch_embed reference; the device path must match it to PCC > 0.99
(target >= 0.999) at the production num_patches (19,520).

Run: pytest models/demos/rednote_hilab_dots.ocr/tests/test_tt_device_patch_embed.py -v -m device
"""
import importlib.util
import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

_TT_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "tt"))
_spec = importlib.util.spec_from_file_location("dots_tt_vision_tower", os.path.join(_TT_DIR, "vision_tower.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
TtVisionTower = _mod.TtVisionTower
host_patch_embed = _mod.host_patch_embed

EMBED_DIM = 1536
NUM_CHANNELS = 3
TEMPORAL_PATCH_SIZE = 1
PATCH_SIZE = 14
RMS_EPS = 1e-5


def _build_state_dict():
    torch.manual_seed(0)
    return {
        "patch_embed.proj.weight": torch.randn(EMBED_DIM, NUM_CHANNELS, PATCH_SIZE, PATCH_SIZE),
        "patch_embed.proj.bias": torch.randn(EMBED_DIM),
        "patch_embed.norm.weight": torch.randn(EMBED_DIM),
    }


def _make_tower(device, state_dict):
    # grid_thw must factor num_patches; only patch_embed weights are exercised, so
    # build a tower with 0 transformer layers and no post-norm/merger dependence.
    # We only call device_patch_embed / host_patch_embed, never forward().
    return TtVisionTower(
        device=device,
        state_dict={
            **state_dict,
            # post_trunk_norm + merger weights are required by __init__ even when
            # num_layers=0; provide dummies (never used by patch_embed paths).
            "post_trunk_norm.weight": torch.ones(EMBED_DIM),
            "merger.ln_q.weight": torch.ones(EMBED_DIM),
            "merger.ln_q.bias": torch.zeros(EMBED_DIM),
            "merger.mlp.0.weight": torch.randn(EMBED_DIM * 4, EMBED_DIM * 4),
            "merger.mlp.0.bias": torch.zeros(EMBED_DIM * 4),
            "merger.mlp.2.weight": torch.randn(EMBED_DIM * 4, EMBED_DIM * 4),
            "merger.mlp.2.bias": torch.zeros(EMBED_DIM * 4),
        },
        grid_thw=torch.tensor([[1, 2, 2]]),  # arbitrary; patch_embed ignores grid
        num_layers=0,
        embed_dim=EMBED_DIM,
        num_channels=NUM_CHANNELS,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        rms_norm_eps=RMS_EPS,
        post_norm=False,
    )


def _run_for_num_patches(device, num_patches: int) -> float:
    state_dict = _build_state_dict()
    tower = _make_tower(device, state_dict)

    torch.manual_seed(1)
    pixel_values = torch.randn(num_patches, NUM_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)

    golden = host_patch_embed(
        pixel_values,
        state_dict["patch_embed.proj.weight"].to(torch.float32),
        state_dict["patch_embed.proj.bias"].to(torch.float32),
        state_dict["patch_embed.norm.weight"].to(torch.float32),
        num_channels=NUM_CHANNELS,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        eps=RMS_EPS,
    )

    tt_out = tower.device_patch_embed(pixel_values)
    ttnn.synchronize_device(device)
    tt_out_torch = ttnn.to_torch(tt_out).to(torch.float32)[:num_patches].reshape(golden.shape)

    passing, pcc_message = comp_pcc(golden, tt_out_torch, 0.99)
    print(f"device_patch_embed PCC @ {num_patches} patches: passing={passing}, {pcc_message}")
    msg = str(pcc_message)
    pcc = float(msg.split("PCC:")[-1].strip()) if "PCC:" in msg else float(pcc_message)
    assert passing, f"device_patch_embed @ {num_patches} below PCC 0.99: {pcc_message}"
    return pcc


@pytest.mark.device
def test_device_patch_embed_19520(device):
    pcc = _run_for_num_patches(device, 19520)
    print(f"device_patch_embed PCC (19520) = {pcc}")


@pytest.mark.device
def test_device_patch_embed_4096(device):
    pcc = _run_for_num_patches(device, 4096)
    print(f"device_patch_embed PCC (4096) = {pcc}")


@pytest.mark.device
def test_device_vs_host_timing(device):
    """Informational: time device_patch_embed vs host_patch_embed at 19,520 patches."""
    state_dict = _build_state_dict()
    tower = _make_tower(device, state_dict)
    num_patches = 19520
    torch.manual_seed(1)
    pixel_values = torch.randn(num_patches, NUM_CHANNELS * TEMPORAL_PATCH_SIZE * PATCH_SIZE * PATCH_SIZE)

    # warm up device path (alloc/compile)
    _ = tower.device_patch_embed(pixel_values)
    ttnn.synchronize_device(device)

    t0 = time.perf_counter()
    _ = host_patch_embed(
        pixel_values,
        state_dict["patch_embed.proj.weight"].to(torch.float32),
        state_dict["patch_embed.proj.bias"].to(torch.float32),
        state_dict["patch_embed.norm.weight"].to(torch.float32),
        num_channels=NUM_CHANNELS,
        temporal_patch_size=TEMPORAL_PATCH_SIZE,
        patch_size=PATCH_SIZE,
        embed_dim=EMBED_DIM,
        eps=RMS_EPS,
    )
    host_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    out = tower.device_patch_embed(pixel_values)
    ttnn.synchronize_device(device)
    dev_ms = (time.perf_counter() - t0) * 1e3

    print(f"patch_embed @ {num_patches}: host={host_ms:.1f} ms, device={dev_ms:.1f} ms")


if __name__ == "__main__":
    device = ttnn.open_device(device_id=0, l1_small_size=32768)
    try:
        _run_for_num_patches(device, 19520)
        _run_for_num_patches(device, 4096)
    finally:
        ttnn.close_device(device)
