# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import time

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 8), id="1x8"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [
        (1, 1024, 1024),
    ],
)
def test_vae_flux2_decoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis = 1

    torch_model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(torch_model, reference.AutoencoderKLFlux2)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = (
        VAEParallelConfig(tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis))
        if tp_axis is not None
        else None
    )

    z_channels = torch_model.config.latent_channels
    patch_size = 2
    vae_scale_factor = 8

    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=torch_model.config.block_out_channels,
        layers_per_block=torch_model.config.layers_per_block,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    f = vae_scale_factor * patch_size
    inp = torch.randn(batch_size, z_channels * patch_size**2, height // f, width // f)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1).flatten(1, 2), device=mesh_device)

    with torch.no_grad():
        # https://github.com/huggingface/diffusers/blob/1b91856d0eee7b6fb58340e9b54ea2c3d5424311/src/diffusers/pipelines/flux2/pipeline_flux2.py#L866
        m = torch_model.bn.running_mean.view(1, -1, 1, 1)
        s = torch.sqrt(torch_model.bn.running_var.view(1, -1, 1, 1) + torch_model.config.batch_norm_eps)
        latents = inp * s + m
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // (2 * 2), 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(b, c // (2 * 2), h * 2, w * 2)
        torch_output = torch_model.decode(latents).sample

    tt_inp = tt_model.preprocess_and_unpatchify(
        tt_inp, height=height // vae_scale_factor, width=width // vae_scale_factor
    )
    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.9978, relative_rmse=0.034)

    start = time()
    tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"VAE time taken: {time() - start}")
