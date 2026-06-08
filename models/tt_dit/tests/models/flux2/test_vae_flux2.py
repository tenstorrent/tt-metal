# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch

import ttnn

from ....models.vae.vae_flux2_opt import Flux2VaeDecoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor


def prep_data(
    vae_parallel_config: Flux2VaeParallelConfig,
    inp: torch.Tensor,
    mesh_device: ttnn.Device,
    sp_axis: int,
    tt_model: Flux2VaeDecoder,
) -> ttnn.Tensor:
    # inp: [B, C*p^2, H_t, W_t] patchified latent in torch.
    # Convert to patchified token format [B, H_t*W_t, C*p^2].
    inp_flat = inp.permute(0, 2, 3, 1).flatten(1, 2)

    # Both pipeline branches (h_parallel.mesh_axis == sp_axis, or all-gather + re-shard)
    # result in the same target layout when starting from the full tensor:
    # shard on dim=1 (token/H dimension) with h_parallel.mesh_axis, or replicate if None.
    if vae_parallel_config.h_parallel is not None:
        tt_latents = tensor.from_torch(
            inp_flat,
            device=mesh_device,
            mesh_axes=(None, vae_parallel_config.h_parallel.mesh_axis, None),
        )
    else:
        tt_latents = tensor.from_torch(inp_flat, device=mesh_device)

    # Derive unpatchify dimensions from inp.shape; avoids needing explicit height/width args.
    # inp.shape[2] = height // (vae_scale_factor * p),  so height // vae_scale_factor = shape[2] * p.
    p = Flux2VaeDecoder._PATCH_SIZE
    height_for_unp = inp.shape[2] * p
    width_for_unp = inp.shape[3] * p

    tt_latents = tt_model.preprocess_and_unpatchify(
        tt_latents,
        height=height_for_unp,
        width=width_for_unp,
    )

    # W-sharding can only be applied after unpatchify because H and W are interleaved
    # in the patchified token dimension.
    if vae_parallel_config.w_parallel is not None:
        tt_latents = ttnn.mesh_partition(
            tt_latents,
            dim=2,
            cluster_axis=vae_parallel_config.w_parallel.mesh_axis,
            memory_config=tt_latents.memory_config(),
        )

    return tt_latents


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 8), id="1x8"),
        pytest.param((4, 8), id="4x8"),
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
        (1, 2048, 2048),
        (1, 4096, 4096),
    ],
    ids=["1024", "2048", "4096"],
)
def test_vae_flux2_decoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis = 0
    h_axis = 1 - tp_axis
    w_axis = None

    torch_model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(torch_model, reference.AutoencoderKLFlux2)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    z_channels = torch_model.config.latent_channels
    patch_size = 2
    vae_scale_factor = 8

    # tt_model = Flux2VaeDecoder(
    #     out_channels=torch_model.config.out_channels,
    #     block_out_channels=torch_model.config.block_out_channels,
    #     layers_per_block=torch_model.config.layers_per_block,
    #     z_channels=z_channels,
    #     device=mesh_device,
    #     parallel_config=vae_parallel_config,
    #     ccl_manager=ccl_manager,
    # )

    tt_model = Flux2VaeDecoder(
        out_channels=3,
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        z_channels=32,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    f = vae_scale_factor * patch_size
    inp = torch.randn(batch_size, z_channels * patch_size**2, height // f, width // f)

    # sp_axis is the DiT sequence-parallel axis; in this test it matches tp_axis.
    tt_inp = prep_data(vae_parallel_config, inp, mesh_device, sp_axis=tp_axis, tt_model=tt_model)

    # tt_out = tt_model.forward(tt_inp)

    # tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    # assert_quality(torch_output, tt_out_torch, pcc=0.9978, relative_rmse=0.034)

    # start = time()
    tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)
    # logger.info(f"VAE time taken: {time() - start}")
