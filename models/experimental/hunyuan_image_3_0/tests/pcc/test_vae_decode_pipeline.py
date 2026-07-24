# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated VAE decode pipeline tests:
#   decode_latent glue (identity decoder), full-res OOM documentation, spatial-parallel decode.
#
# Run (glue only, fast):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_vae_decode_pipeline.py -k glue -v -s
#
# Full spatial decode (slow, needs 2×2 mesh):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_vae_decode_pipeline.py \
#     -k spatial -v -s --timeout=1200

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_dit.parallel.manager import CCLManager
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS, load_decoder, vae_decode_output_to_rgb
from models.experimental.hunyuan_image_3_0.ref.model_config import VAE_SCALING_FACTOR
from models.experimental.hunyuan_image_3_0.tt.pipeline import decode_latent

PCC_THRESHOLD = 0.99
SCALING_FACTOR = VAE_SCALING_FACTOR


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


class _IdentityDecoder:
    """Returns the first 3 channels of a BTHWC tensor — stand-in for the VAE decoder."""

    def __call__(self, x_bthwc):
        b, t, h, w, _c = x_bthwc.shape
        return ttnn.slice(x_bthwc, [0, 0, 0, 0, 0], [b, t, h, w, 3])


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decode_latent_glue(mesh_device):
    """decode_latent scaling / temporal-dim / denorm wiring, identity decoder."""
    mesh_device.enable_program_cache()

    torch.manual_seed(0)
    latent_bchw = torch.randn(1, 3, 16, 16, dtype=torch.float32)

    pt_img = ((latent_bchw / SCALING_FACTOR) / 2 + 0.5).clamp(0, 1)

    tt_img = decode_latent(mesh_device, latent_bchw, scaling_factor=SCALING_FACTOR, decoder=_IdentityDecoder())

    assert pt_img.shape == tt_img.shape, f"shape {pt_img.shape} vs {tt_img.shape}"
    assert tt_img.min() >= 0.0 and tt_img.max() <= 1.0, "image not in [0,1]"
    passing, pcc = comp_pcc(pt_img, tt_img, PCC_THRESHOLD)
    logger.info(f"decode_latent glue PCC: {pcc:.6f}  shape={tuple(tt_img.shape)}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD}"


@pytest.mark.xfail(
    reason="Full-res VAE decode OOMs on a single Blackhole device (pre-existing); ~16GB intermediate", strict=False
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_full_res_decode_oom(mesh_device):
    """Documents the real-decoder full-res OOM until chunked/sharded upsample lands."""
    mesh_device.enable_program_cache()
    torch.manual_seed(42)
    latent_bchw = torch.randn(1, Z_CHANNELS, 64, 64, dtype=torch.float32)

    z_bcthw = (latent_bchw / SCALING_FACTOR).unsqueeze(2)
    with torch.no_grad():
        pt_out = load_decoder()(z_bcthw)
    pt_img = vae_decode_output_to_rgb(pt_out)

    tt_img = decode_latent(mesh_device, latent_bchw, scaling_factor=SCALING_FACTOR)
    assert_pcc = comp_pcc(pt_img, tt_img, PCC_THRESHOLD)[0]
    assert assert_pcc


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decode_latent_spatial_vs_reference(mesh_device):
    """Full VAE decode with H/W spatial sharding on 2×2 mesh vs torch reference."""
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)

    torch.manual_seed(42)
    latent_bchw = torch.randn(1, Z_CHANNELS, 64, 64, dtype=torch.float32)

    z_bcthw = (latent_bchw / SCALING_FACTOR).unsqueeze(2)
    with torch.no_grad():
        pt_out = load_decoder()(z_bcthw)
    pt_img = vae_decode_output_to_rgb(pt_out)

    tt_img = decode_latent(
        mesh_device,
        latent_bchw,
        scaling_factor=SCALING_FACTOR,
        ccl_manager=ccl,
        h_mesh_axis=0,
        w_mesh_axis=1,
    )

    assert tuple(tt_img.shape) == tuple(pt_img.shape), f"shape {tt_img.shape} vs {pt_img.shape}"
    passing, pcc = comp_pcc(pt_img, tt_img, PCC_THRESHOLD)
    logger.info(f"spatial VAE decode vs reference PCC: {pcc:.6f}  shape={tuple(tt_img.shape)}")
    assert passing, f"PCC {pcc:.6f} < {PCC_THRESHOLD} — spatial decode wrong"
