# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end PCC: prompt->image (diffusion loop + scheduler + VAE), host vs TT decoder layers."""

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
N_STEPS = int(os.environ.get("HUNYUAN_GENIMG_STEPS", "8"))
_SZ = os.environ.get("HUNYUAN_GENIMG_SIZE", "1024,1024")
IMAGE_SIZE = tuple(int(x) for x in _SZ.replace("x", ",").split(","))
USE_TRACE = os.environ.get("HUNYUAN_GENIMG_TRACE", "0") != "0"  # run the TT trajectory host-free (traced)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [_MESH], indirect=True)
def test_t2i_e2e_image_pcc(device_params, mesh_device):
    torch.manual_seed(0)
    res = gi.e2e_image_pcc(
        mesh_device,
        prompt=PROMPT,
        image_size=IMAGE_SIZE,
        num_layers=N_LAYERS,
        num_steps=N_STEPS,
        seed=0,
        pcc_target=PCC_TARGET,
        decode=True,
        use_trace_tt=USE_TRACE,
    )
    # ALWAYS print achieved PCCs (pass or fail) before the asserts.
    print(
        f"\nt2i e2e PCC: final_latent={res['pcc_final_latent']} image={res.get('pcc_image')} "
        f"velocity_step0={res['pcc_velocity_step0']} "
        f"(N={res['num_layers']} layers, steps={res['num_steps']}, token_hw={res['token_hw']}, "
        f"image_finite={res.get('image_finite')}, image_std={res.get('image_std')})"
    )
    assert (
        res["pcc_final_latent"] >= PCC_TARGET
    ), f"t2i e2e FAIL — final-latent PCC {res['pcc_final_latent']} < {PCC_TARGET}"
    assert (
        res.get("pcc_image", 1.0) >= PCC_TARGET
    ), f"t2i e2e FAIL — decoded-image PCC {res.get('pcc_image')} < {PCC_TARGET}"
    assert res.get("image_finite", True), "t2i e2e FAIL — TT image has non-finite values"
    assert res.get("image_std", 1.0) > 0.0, "t2i e2e FAIL — TT image is degenerate (std=0)"
