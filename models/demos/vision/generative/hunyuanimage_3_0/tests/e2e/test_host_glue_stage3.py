# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Checks a fully on-device text->image step against the host reference (PCC), with the patch-embed,
transformer, and velocity head all run on the mesh and the hidden state kept resident across the step."""
from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt import host_glue_stage3 as hg3
from models.demos.vision.generative.hunyuanimage_3_0.tt import pipeline as ttpipe

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

PROMPT = (
    "a red panda floating in space with a samurai sword and laser eyes, ultra realistic, galaxy lots of galactic colors"
)
PCC_TARGET = float(os.environ.get("HUNYUAN_HG3_PCC", "0.99"))
N_LAYERS = int(os.environ.get("HUNYUAN_GENIMG_NUM_LAYERS", "2"))
IMAGE_SIZE = (1024, 1024)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "l1_small_size": 24576,
            "fabric_config": (
                ttnn.FabricConfig.FABRIC_1D_RING
                if os.environ.get("HUNYUAN_SP_RING") == "1"
                else ttnn.FabricConfig.FABRIC_1D
            ),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_stage3_velocity_pcc(device_params, mesh_device):
    torch.manual_seed(0)
    model = gi.load_model(num_layers=N_LAYERS)
    kwargs, attn_mask = gi.prepare_gen_image_inputs(model, PROMPT, list(IMAGE_SIZE), seed=0)
    cfg = int(kwargs["input_ids"].shape[0])
    h = kwargs["batch_gen_image_info"][0].token_height
    w = kwargs["batch_gen_image_info"][0].token_width

    sched = model.pipeline.scheduler
    sched.set_timesteps(50, device="cpu")
    t = sched.timesteps[len(sched.timesteps) // 2]
    g = torch.Generator("cpu").manual_seed(0)
    latents = torch.randn(1, int(model.config.vae["latent_channels"]), h, w, generator=g, dtype=torch.float32)

    tt_pipe = ttpipe.HunyuanImage3Pipeline(mesh_device, model, num_layers=N_LAYERS)
    uninstall = gi.install_tt_layer_stack(model, tt_pipe)
    try:
        vel_ref = gi.run_velocity_once(model, kwargs, attn_mask, latents, t, cfg)  # existing TT path (host convs)
        print(
            f"[hg3] ref velocity={tuple(vel_ref.shape)} cfg={cfg} token_hw=({h},{w}) S={int(kwargs['input_ids'].shape[1])}",
            flush=True,
        )
        ctx = hg3.setup_ondevice_headglue(model, tt_pipe, kwargs, attn_mask, token_h=h, token_w=w)
        print(
            f"[hg3] image block: " + ", ".join(f"row{i} [{r['start']}:{r['end']}]" for i, r in enumerate(ctx["rows"])),
            flush=True,
        )
        vel_tt = hg3.run_velocity_once_ondevice(model, ctx, tt_pipe, latents, t, cfg)  # stage-3 fully on-device
        print(f"[hg3] ondevice velocity={tuple(vel_tt.shape)}", flush=True)
    finally:
        uninstall()

    ok, pcc = comp_pcc(vel_ref, vel_tt, PCC_TARGET)
    print(f"HOST_GLUE_STAGE3_PCC={pcc} target={PCC_TARGET} ok={ok}", flush=True)
    assert ok, f"stage-3 on-device velocity PCC {pcc} < {PCC_TARGET} (vs existing TT path)"
