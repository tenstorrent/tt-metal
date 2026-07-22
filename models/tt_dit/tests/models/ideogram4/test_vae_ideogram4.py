# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Ideogram 4.0 "Flux2 KL autoencoder" decode path. The reference autoencoder.py
# ships a diffusers->Flux state-dict converter, so the model IS a diffusers
# AutoencoderKL (block_out [128,256,512,512], latent_channels 32, mid attention).
# We reuse vae_sd35.VAEDecoder + a post_quant_conv and verify the decode against a
# diffusers AutoencoderKL with Ideogram's config. weights="random" is the wiring /
# channel-parallel (GroupNorm shards channels) check across single device + TP;
# weights="real" loads the shipped VAE checkpoint for a fidelity check.
# =============================================================================

import os

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from ....utils.test import line_params, ring_params

Z_CHANNELS = 32
FP8 = os.environ.get("IDEOGRAM4_WEIGHTS")
# Real-weight cases need the gated fp8 checkpoint; skip (don't error) when it isn't configured.
_NEEDS_WEIGHTS = pytest.mark.skipif(not FP8, reason="IDEOGRAM4_WEIGHTS not set (gated fp8 checkpoint)")
_L1 = {"l1_small_size": 32768}


def _make_akl(weights: str, torch_dtype) -> AutoencoderKL:
    """diffusers AutoencoderKL with Ideogram's config; random init or the shipped VAE weights."""
    akl = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=Z_CHANNELS,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
    )
    if weights == "real":
        vae_sd = load_file(f"{FP8}/vae/diffusion_pytorch_model.safetensors")  # diffusers layout, not fp8
        # bn.* are checkpoint-only batchnorm running stats absent from diffusers AutoencoderKL.
        incompat = akl.load_state_dict({k: v for k, v in vae_sd.items() if not k.startswith("bn.")}, strict=False)
        # Prove the shipped weights actually landed; otherwise random init passes PCC without
        # testing the real checkpoint. (Verified empty on the real load.)
        assert not incompat.missing_keys and not incompat.unexpected_keys, (
            f"real VAE load key mismatch: missing={incompat.missing_keys[:5]} "
            f"unexpected={incompat.unexpected_keys[:5]}"
        )
    return akl.to(torch_dtype).eval()


@pytest.mark.parametrize(
    (
        "mesh_device",
        "submesh_shape",
        "tp_axis",
        "num_links",
        "device_params",
        "topology",
        "latent_h",
        "latent_w",
        "weights",
    ),
    [
        # single device (TP=1): small latents, random + real fidelity.
        pytest.param((1, 1), (1, 1), 1, 1, _L1, ttnn.Topology.Linear, 32, 32, "random", id="tp1_256sq"),
        pytest.param((1, 1), (1, 1), 1, 1, _L1, ttnn.Topology.Linear, 32, 48, "random", id="tp1_256x384"),
        pytest.param(
            (1, 1), (1, 1), 1, 1, _L1, ttnn.Topology.Linear, 32, 32, "real", id="tp1_256sq_real", marks=_NEEDS_WEIGHTS
        ),
        # TP=4 on the 2x4 loudbox (channel-parallel GroupNorm), 512px + 1024px.
        pytest.param(
            (2, 4), (1, 4), 1, 1, {**line_params, **_L1}, ttnn.Topology.Linear, 64, 64, "random", id="tp4_512px"
        ),
        pytest.param(
            (2, 4), (1, 4), 1, 1, {**line_params, **_L1}, ttnn.Topology.Linear, 128, 128, "random", id="tp4_1024px"
        ),
        # SP4xTP2 denoiser: VAE TP=2, replicated on the size-4 axis.
        pytest.param(
            (4, 2), (4, 2), 1, 1, {**line_params, **_L1}, ttnn.Topology.Linear, 64, 64, "random", id="tp2_512px"
        ),
        pytest.param(
            (4, 2), (4, 2), 1, 1, {**line_params, **_L1}, ttnn.Topology.Linear, 128, 128, "random", id="tp2_1024px"
        ),
        # BH Galaxy 4x8, 2D torus Ring: VAE TP=4 (axis 0), replicated on the size-8 axis, 2 links/neighbor.
        pytest.param(
            (4, 8), (4, 8), 0, 2, {**ring_params, **_L1}, ttnn.Topology.Ring, 64, 64, "random", id="bh_galaxy_tp4_512px"
        ),
        pytest.param(
            (4, 8),
            (4, 8),
            0,
            2,
            {**ring_params, **_L1},
            ttnn.Topology.Ring,
            128,
            128,
            "random",
            id="bh_galaxy_tp4_1024px",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_vae_decoder(
    *,
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    latent_h: int,
    latent_w: int,
    weights: str,
) -> None:
    """VAE decode, channel-parallel at TP across the mesh (GroupNorm shards channels)."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))
    tp_factor = tuple(submesh.shape)[tp_axis]

    akl = _make_akl(weights, torch_dtype)

    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis))
    ccl_manager = CCLManager(submesh, num_links=num_links, topology=topology)
    tt_decoder = Ideogram4VAEDecoder.from_torch(
        akl, mesh_device=submesh, parallel_config=parallel_config, ccl_manager=ccl_manager
    )

    z = torch.randn(1, Z_CHANNELS, latent_h, latent_w, dtype=torch_dtype)
    with torch.no_grad():
        ref = akl.decode(z).sample  # post_quant_conv + decoder

    # Feed replicated NHWC: post_quant_conv is replicated; the decoder's conv_in
    # (out_mesh_axis=tp) shards channels from there. Output is replicated -> take device 0.
    tt_z = bf16_tensor(z.permute(0, 2, 3, 1), device=submesh)  # NCHW -> NHWC
    tt_out = tt_decoder(tt_z)
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)  # NHWC -> NCHW

    logger.info(
        f"ideogram4 VAE decode [{weights}] TP={tp_factor}: latent {latent_h}x{latent_w} "
        f"-> image {latent_h*8}x{latent_w*8}"
    )
    assert_quality(ref.float(), tt_out_torch.float(), pcc=0.99, relative_rmse=0.15)
