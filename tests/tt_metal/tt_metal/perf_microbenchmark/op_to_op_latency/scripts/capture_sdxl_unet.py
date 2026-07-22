#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Capture ONE SDXL-Base UNet forward (one denoising step) under ttnn graph capture, dump JSON for
raw_hazard_analyzer.py. Single Wormhole chip.

Weights are RANDOM (UNet2DConditionModel.from_config -> from_config, no pretrained download): the
RAW-hazard analysis depends only on the op DEPENDENCY graph (which op reads which buffer), which is
weight-VALUE independent -- same trick used for the ResNet capture. Only the tiny config.json is
fetched from HF. Structure/prep mirror models/demos/stable_diffusion_xl_base/tests/pcc/
test_module_tt_unet.py (prepare_ttnn_tensors + run_unet_model).
"""
import json
import sys

import torch
import ttnn
from diffusers import UNet2DConditionModel

from models.demos.stable_diffusion_xl_base.tt.model_configs import load_model_optimisations
from models.demos.stable_diffusion_xl_base.tt.tt_unet import TtUNet2DConditionModel

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
RES = (512, 512)
INPUT_SHAPE = (1, 4, 64, 64)  # B,C,H,W for 512x512 (op structure is resolution-independent)
OUT = "/tmp/sdxl_unet_capture.json"


def prepare_ttnn_tensors(device, x, ts, temb, enc, time_ids):
    torch.manual_seed(2025)
    tt_ts = ttnn.from_torch(
        ts, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_enc = ttnn.from_torch(
        enc, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_temb = ttnn.from_torch(
        temb, dtype=ttnn.bfloat16, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_tids = ttnn.from_torch(
        time_ids, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    B, C, H, W = list(tt_x.shape)
    tt_x = ttnn.permute(tt_x, (0, 2, 3, 1))
    tt_x = ttnn.reshape(tt_x, (B, 1, H * W, C))
    return tt_x, [B, C, H, W], tt_ts, tt_enc, {"text_embeds": tt_temb, "time_ids": tt_tids}


def main():
    # random-init UNet from config (tiny download), correctly-shaped state_dict, no weight fetch
    cfg = UNet2DConditionModel.load_config(MODEL, subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg).eval()
    state_dict = unet.state_dict()
    print(f"UNet2DConditionModel from_config: {len(state_dict)} tensors (random weights)")

    device = ttnn.open_device(device_id=0, l1_small_size=32000)  # SDXL_L1_SMALL_SIZE (conv2d/halo)
    try:
        model_config = load_model_optimisations(RES)
        tt_unet = TtUNet2DConditionModel(device, state_dict, "unet", model_config=model_config, debug_mode=False)

        x = torch.rand(*INPUT_SHAPE, dtype=torch.float32) * 0.2 - 0.1
        ts = torch.tensor([500.0], dtype=torch.float32)
        temb = torch.rand(1, 1280, dtype=torch.float32) * 0.2 - 0.1
        enc = torch.rand(1, 77, 2048, dtype=torch.float32) * 0.2 - 0.1
        time_ids = torch.tensor([512, 512, 0, 0, 512, 512])

        tt_x, bchw, tt_ts, tt_enc, added = prepare_ttnn_tensors(device, x, ts, temb, enc, time_ids)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        out, out_shape = tt_unet.forward(
            tt_x,
            bchw,
            timestep=tt_ts,
            encoder_hidden_states=tt_enc,
            time_ids=added["time_ids"],
            text_embeds=added["text_embeds"],
        )
        ttnn.synchronize_device(device)
        captured = ttnn.graph.end_graph_capture()

        json.dump(captured, open(OUT, "w"))
        print(f"captured {len(captured)} nodes -> {OUT}  (UNet out_shape={out_shape})")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    sys.exit(main())
