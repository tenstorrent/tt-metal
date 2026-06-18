# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full VAE decode, H/W-spatial-parallel on a 2x2 mesh, vs the torch reference.

This is the end-to-end Phase V5 gate: the real decoder runs sharded (H on axis 0,
W on axis 1) — convs keep a neighbor-pad halo, GroupNorm/attention gather to full
spatial — and the 1024x1024 output is gathered with ConcatMesh2dToTensor. Spatial
sharding also shrinks the conv im2col 4x (the full-res OOM driver), so this should
fit where the replicated full-res decode struggles.

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vae/test_decode_latent_spatial.py -v -s --timeout=1200
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS, load_decoder
from models.experimental.hunyuan_image_3_0.tt.pipeline import decode_latent

SCALING_FACTOR = 0.562679178327931


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decode_latent_spatial_vs_reference(mesh_device):
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch.manual_seed(42)
    latent_bchw = torch.randn(1, Z_CHANNELS, 64, 64, dtype=torch.float32)

    # Torch fp32 reference (CPU).
    z_bcthw = (latent_bchw / SCALING_FACTOR).unsqueeze(2)
    with torch.no_grad():
        pt_out = load_decoder()(z_bcthw)
    pt_img = (pt_out[:, :, 0] / 2 + 0.5).clamp(0, 1)  # [1, 3, 1024, 1024]

    # Spatial-parallel TTNN decode: H -> axis 0, W -> axis 1.
    tt_img = decode_latent(
        mesh_device,
        latent_bchw,
        scaling_factor=SCALING_FACTOR,
        ccl_manager=ccl,
        h_mesh_axis=0,
        w_mesh_axis=1,
    )

    assert tuple(tt_img.shape) == tuple(pt_img.shape), f"shape {tt_img.shape} vs {pt_img.shape}"
    passing, pcc = comp_pcc(pt_img, tt_img, 0.99)
    logger.info(f"spatial VAE decode vs reference PCC: {pcc:.6f}  shape={tuple(tt_img.shape)}")
    assert passing, f"PCC {pcc:.6f} < 0.99 — spatial decode wrong"
