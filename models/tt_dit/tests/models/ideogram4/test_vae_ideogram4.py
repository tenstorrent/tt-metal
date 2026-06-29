# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 "Flux2 KL autoencoder" decode path. The reference autoencoder.py
# ships a diffusers->Flux state-dict converter, so the model IS a diffusers
# AutoencoderKL (block_out [128,256,512,512], latent_channels 32, mid attention).
# We reuse vae_sd35.VAEDecoder + a post_quant_conv and verify the decode against a
# diffusers AutoencoderKL with Ideogram's config (random init; weights are gated).
# =============================================================================

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger

import ttnn

from ....models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor

Z_CHANNELS = 32


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(("latent_h", "latent_w"), [(32, 32), (32, 48)], ids=["256sq", "256x384"])
def test_vae_decoder(*, mesh_device: ttnn.MeshDevice, latent_h: int, latent_w: int) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    akl = (
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

    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt_decoder = Ideogram4VAEDecoder.from_torch(
        akl, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl_manager
    )

    z = torch.randn(1, Z_CHANNELS, latent_h, latent_w, dtype=torch_dtype)
    with torch.no_grad():
        ref = akl.decode(z).sample  # post_quant_conv + decoder

    tt_z = ttnn.from_torch(
        z.permute(0, 2, 3, 1),  # NCHW -> NHWC
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )
    tt_out = tt_decoder(tt_z)
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)  # NHWC -> NCHW

    logger.info(f"ideogram4 VAE decode: latent {latent_h}x{latent_w} -> image {latent_h*8}x{latent_w*8}")
    assert_quality(ref.float(), tt_out_torch.float(), pcc=0.99)


@pytest.mark.parametrize(
    ("mesh_device", "submesh_shape", "tp_axis"),
    [pytest.param((2, 4), (1, 4), 1, id="tp4")],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768}], indirect=True
)
@pytest.mark.parametrize(("latent_h", "latent_w"), [(64, 64), (128, 128)], ids=["512px", "1024px"])
def test_vae_decoder_tp4(*, mesh_device: ttnn.MeshDevice, submesh_shape, tp_axis, latent_h: int, latent_w: int) -> None:
    """VAE decode channel-parallel at TP=4 across the mesh (GroupNorm shards channels)."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    akl = (
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

    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis))
    ccl_manager = CCLManager(submesh, num_links=1, topology=ttnn.Topology.Linear)
    tt_decoder = Ideogram4VAEDecoder.from_torch(
        akl, mesh_device=submesh, parallel_config=parallel_config, ccl_manager=ccl_manager
    )

    z = torch.randn(1, Z_CHANNELS, latent_h, latent_w, dtype=torch_dtype)
    with torch.no_grad():
        ref = akl.decode(z).sample

    # Feed replicated: post_quant_conv is replicated; the decoder's conv_in (out_mesh_axis=tp)
    # shards channels from there.
    tt_z = bf16_tensor(z.permute(0, 2, 3, 1), device=submesh)  # NCHW -> NHWC, replicated
    tt_out = tt_decoder(tt_z)
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)  # replicated output

    logger.info(f"ideogram4 VAE decode TP={tp_factor}: latent {latent_h}x{latent_w}")
    assert_quality(ref.float(), tt_out_torch.float(), pcc=0.99)
