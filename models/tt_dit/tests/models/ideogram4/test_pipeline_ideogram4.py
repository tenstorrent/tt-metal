# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 pipeline glue. The denoise loop (CFG blend + Euler) is covered by
# the sampler + transformer tests and the text encode by the encoder test; this
# verifies the pipeline-specific NEW device computation:
#   * the decode tail: per-channel latent denorm -> 2x2 unpatch -> VAE decode,
#     end-to-end on device vs a torch reference (latent_norm + diffusers
#     AutoencoderKL.decode);
#   * the asymmetric-CFG velocity blend v = gw*v_cond + (1-gw)*v_uncond.
# =============================================================================

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger

import ttnn

from ....models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....pipelines.ideogram4.pipeline import Ideogram4DecodeStage, cfg_blend, unpatchify_latent
from ....reference.ideogram4.latent_norm import get_latent_norm
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

Z_CHANNELS = 32
PATCH = 2


def _build_akl(torch_dtype):
    return (
        AutoencoderKL(
            in_channels=3,
            out_channels=3,
            latent_channels=Z_CHANNELS,
            down_block_types=("DownEncoderBlock2D",) * 4,
            up_block_types=("UpDecoderBlock2D",) * 4,
            block_out_channels=(128, 256, 512, 512),
            layers_per_block=2,
            norm_num_groups=32,
        )
        .to(torch_dtype)
        .eval()
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(("grid_h", "grid_w"), [(16, 16)], ids=["256px"])
def test_pipeline_decode(*, mesh_device: ttnn.MeshDevice, grid_h: int, grid_w: int) -> None:
    """Decode tail end-to-end (latent denorm + 2x2 unpatch + VAE) vs torch reference."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    tokens = grid_h * grid_w

    akl = _build_akl(torch_dtype)
    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae = Ideogram4VAEDecoder.from_torch(
        akl, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
    )
    stage = Ideogram4DecodeStage(vae, mesh_device=mesh_device, patch=PATCH)

    z = torch.randn(1, tokens, PATCH * PATCH * Z_CHANNELS, dtype=torch.float32)  # patchified latent

    # torch reference: denorm -> unpatch -> diffusers decode
    shift, scale = get_latent_norm()
    z_dn = (z * scale + shift).to(torch_dtype)
    z_nchw = unpatchify_latent(z_dn, grid_h=grid_h, grid_w=grid_w, patch=PATCH)
    with torch.no_grad():
        ref = akl.decode(z_nchw).sample

    out = stage.decode(bf16_tensor(z, device=mesh_device), grid_h=grid_h, grid_w=grid_w)

    logger.info(f"ideogram4 pipeline decode: {tokens} tokens -> image {grid_h*PATCH*8}x{grid_w*PATCH*8}")
    assert_quality(ref.float(), out.float(), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_pipeline_cfg_blend(*, mesh_device: ttnn.MeshDevice) -> None:
    """Asymmetric CFG: v = gw*v_cond + (1-gw)*v_uncond on device vs torch."""
    torch.manual_seed(0)
    gw = 7.0
    v_cond = torch.randn(1, 4096, 128, dtype=torch.float32)
    v_uncond = torch.randn(1, 4096, 128, dtype=torch.float32)
    ref = gw * v_cond + (1.0 - gw) * v_uncond

    out = cfg_blend(bf16_tensor(v_cond, device=mesh_device), bf16_tensor(v_uncond, device=mesh_device), gw)
    out_torch = tensor.to_torch(out, mesh_axes=[None, None, None])

    logger.info("ideogram4 pipeline CFG blend")
    assert_quality(ref, out_torch, pcc=0.999)
