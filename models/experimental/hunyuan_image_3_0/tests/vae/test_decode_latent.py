# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for tt/pipeline.decode_latent — diffusion latent -> RGB image.

decode_latent wraps the (already block-gated) TTNN VAE decoder with the glue a
diffusion pipeline needs: scaling-factor division, temporal-dim insertion
([B,C,h,w] -> BCTHW), host upload/download around the mesh decoder, and the
(image/2 + 0.5) denormalization.

Two things are validated separately:
  * test_decode_latent_glue — the wiring math, with an INJECTED identity decoder
    (no real VAE weights), so it runs at any size and isolates the glue.
  * test_full_res_decode_oom — documents that the REAL decoder OOMs at the only
    resolution it supports (64x64 latent -> 1024x1024). This is a PRE-EXISTING
    VAE limitation (upstream test_full_decoder_vs_pytorch OOMs identically): the
    decoder's GroupNorm3D bakes in input_nhw=4096 at construction, so it cannot
    be shrunk to fit, and the DCAE upsample needs a ~16 GB DRAM intermediate.
    Tracked in the README as VAE-decode memory work (chunked/sharded upsample).

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vae/test_decode_latent.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.hunyuan_image_3_0.ref.vae.decoder import Z_CHANNELS, load_decoder
from models.experimental.hunyuan_image_3_0.tt.pipeline import decode_latent

PCC_THRESHOLD = 0.99
SCALING_FACTOR = 0.562679178327931


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


class _IdentityDecoder:
    """Returns the first 3 channels of a BTHWC tensor — a stand-in for the VAE
    decoder that lets us exercise decode_latent's glue without the real weights
    (and without the full-res OOM)."""

    def __call__(self, x_bthwc):
        b, t, h, w, _c = x_bthwc.shape
        return ttnn.slice(x_bthwc, [0, 0, 0, 0, 0], [b, t, h, w, 3])


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_decode_latent_glue(mesh_device):
    """decode_latent's scaling / temporal-dim / denorm wiring, identity decoder."""
    mesh_device.enable_program_cache()

    # 3-channel latent so the identity decoder is a true passthrough.
    torch.manual_seed(0)
    latent_bchw = torch.randn(1, 3, 16, 16, dtype=torch.float32)

    # Reference glue (host): scale -> (identity) -> denormalize.
    pt_img = ((latent_bchw / SCALING_FACTOR) / 2 + 0.5).clamp(0, 1)  # [1, 3, 16, 16]

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
    """Documents the real-decoder full-res OOM. Expected to xfail until the VAE
    decode gets chunked/sharded upsample memory work."""
    mesh_device.enable_program_cache()
    torch.manual_seed(42)
    latent_bchw = torch.randn(1, Z_CHANNELS, 64, 64, dtype=torch.float32)

    z_bcthw = (latent_bchw / SCALING_FACTOR).unsqueeze(2)
    with torch.no_grad():
        pt_out = load_decoder()(z_bcthw)
    pt_img = (pt_out[:, :, 0] / 2 + 0.5).clamp(0, 1)

    tt_img = decode_latent(mesh_device, latent_bchw, scaling_factor=SCALING_FACTOR)
    assert_pcc = comp_pcc(pt_img, tt_img, PCC_THRESHOLD)[0]
    assert assert_pcc
