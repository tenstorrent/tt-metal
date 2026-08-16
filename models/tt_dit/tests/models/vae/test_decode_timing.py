# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TEMP end-to-end DiffVAE decode timing (uncommitted): full replicated decode on shipped weights,
timing the whole pipeline per NA3D backend so we can see where the fused kernel places us e2e."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.diffvae_ltx import DiffVAEDecoder, decoder_config

CHECKPOINT = Path(
    os.environ.get(
        "DIFFVAE_CHECKPOINT",
        os.path.expanduser("~/.cache/ltx-checkpoints/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors"),
    )
)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("backend", ["gather", "fused"], ids=["gather", "fused"])
@pytest.mark.parametrize("latent_hw", [(16, 16), (34, 60)], ids=["s16", "s34x60"])
def test_decode_timing(*, mesh_device, backend, latent_hw):
    if not CHECKPOINT.exists():
        pytest.skip(f"missing {CHECKPOINT}")
    config = decoder_config(CHECKPOINT)
    lh, lw = latent_hw
    torch.manual_seed(0)
    latent = torch.randn(1, config["in_channels"], 4, lh, lw)

    dec = DiffVAEDecoder(config, mesh_device=mesh_device, stage5_na3d_backend=backend, stages_na3d_backend=backend)
    dec.load_checkpoint(CHECKPOINT)

    px = dec.decode(latent, seed=0)  # warmup (also builds fused mask cache)
    ttnn.synchronize_device(mesh_device)
    px_shape = tuple(px.shape)

    t0 = time.perf_counter()
    px = dec.decode(latent, seed=0)
    ttnn.synchronize_device(mesh_device)
    dt = time.perf_counter() - t0
    print(f"\n[decode {backend}] latent(1,{config['in_channels']},4,{lh},{lw}) -> {px_shape}: {dt * 1000:8.0f} ms\n")
