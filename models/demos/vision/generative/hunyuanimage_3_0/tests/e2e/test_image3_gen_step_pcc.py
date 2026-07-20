# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage-2 correctness gate for the HunyuanImage-3.0 hybrid text->image path.

HunyuanImage-3.0 generates images via a diffusion-in-transformer `gen_image`
forward (2D image RoPE + text-causal/image-bidirectional block mask +
timestep-conditioned velocity head). This test drives ONE denoising step and
asserts the TT per-step VELOCITY (`diffusion_prediction`) matches the pure-host
(first-N HF layers) velocity at reduced depth:

    per-step velocity PCC(TT, HF-host) >= 0.95

Reduced depth (N layers, default 2) keeps it cheap while exercising the exact
gen_image forward that a full render uses. This is the gate that must pass before
trusting a full-depth rendered image.

Run:  ./python_env/bin/python -m pytest \
        models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_image3_gen_step_pcc.py -s
Env:  HUNYUAN_GENIMG_NUM_LAYERS (default 2), HUNYUAN_GENIMG_SIZE (default 1024,1024)
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt import gen_image as gi

try:
    _MESH = tuple(int(x) for x in ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
except Exception:
    _MESH = (1, 8)

PROMPT = "A serene mountain lake at sunrise, photorealistic, ultra detailed."
PCC_TARGET = 0.95
N_LAYERS = int(os.environ.get("HUNYUAN_GENIMG_NUM_LAYERS", "2"))
_SZ = os.environ.get("HUNYUAN_GENIMG_SIZE", "1024,1024")
IMAGE_SIZE = tuple(int(x) for x in _SZ.replace("x", ",").split(","))


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_gen_image_step_pcc(device_params, mesh_device):
    torch.manual_seed(0)
    res = gi.step_pcc(
        mesh_device,
        prompt=PROMPT,
        image_size=IMAGE_SIZE,
        num_layers=N_LAYERS,
        seed=0,
        pcc_target=PCC_TARGET,
    )
    # ALWAYS print achieved PCC (pass or fail), on its own line, before the assert.
    print(
        f"\ngen_image per-step velocity PCC={res['pcc']} "
        f"(N={res['num_layers']} layers, token_hw={res['token_hw']}, seq_len={res['seq_len']})"
    )
    assert (
        res["pcc_ok"] and res["pcc"] >= PCC_TARGET
    ), f"Stage-2 FAIL — gen_image per-step velocity PCC {res['pcc']} < {PCC_TARGET}"
