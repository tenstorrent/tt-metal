# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage-3 host-glue PERF: full-depth (32-layer) 50-step e2e render with the
fully-on-device head-glue (hidden stays on device). Emits
ONDEVICE_E2E_TOTAL_LATENCY_S=<s> to compare vs the hybrid baseline (406.8s:
loop 370.0 @7401ms/step + vae 36.3s). Same banked wins (vae_bf16 + ccl_links2)
+ red-panda prompt for an apples-to-apples number isolating the host-glue win.

Run:  HUNYUAN_VAE_AUTOCAST=bf16 HUNYUAN_CCL_LINKS=2 ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_host_glue_stage3_perf.py -s
Env:  HUNYUAN_STAGE3_STEPS (default 50), HUNYUAN_STAGE3_NUM_LAYERS (default 32),
      HUNYUAN_STAGE3_OUT, HUNYUAN_STAGE3_PROMPT
"""
from __future__ import annotations

import os

import pytest

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi
from models.demos.vision.generative.hunyuanimage_3_0.tt import host_glue_stage3 as hg3

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

PROMPT = os.environ.get(
    "HUNYUAN_STAGE3_PROMPT",
    "a red panda floating in space with a samurai sword and laser eyes, ultra realistic, galaxy lots of galactic colors",
)
STEPS = int(os.environ.get("HUNYUAN_STAGE3_STEPS", "50"))
NUM_LAYERS = int(os.environ.get("HUNYUAN_STAGE3_NUM_LAYERS", "32"))
OUT = os.environ.get("HUNYUAN_STAGE3_OUT", "hunyuan_t2i_redpanda_stage3.png")


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
def test_stage3_perf(device_params, mesh_device):
    model, tt_pipe, uninstall = gi.build_tt_backed_model(mesh_device, num_layers=NUM_LAYERS, use_trace=False)
    try:
        img, timing = hg3.generate_image_ondevice(model, tt_pipe, PROMPT, num_inference_steps=STEPS, out_path=OUT)
    finally:
        uninstall()
    assert img is not None
    print(
        f"STAGE3 PERF: total={timing['total_s']:.1f}s loop={timing['loop_s']:.1f}s "
        f"@ {timing['ms_per_step']:.0f} ms/step x{timing['steps']} vae={timing['vae_s']:.1f}s "
        f"num_layers={NUM_LAYERS} (baseline hybrid=406.8s)",
        flush=True,
    )
