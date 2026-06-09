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
from ....utils.check import assert_quality

_LAYERS_PER_BLOCK = 1  # pruned from pretrained (2) to keep host run fast


def _load_pruned_torch_model() -> reference.AutoencoderKLFlux2:
    """Load the pretrained VAE and truncate each up_block to _LAYERS_PER_BLOCK resnets.

    The mid_block is left intact — it already uses num_layers=1 (2 resnets, 1 attention)
    matching TT's VaeMidBlock default, regardless of layers_per_block.
    """
    model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(model, reference.AutoencoderKLFlux2)
    model.eval()
    keep = _LAYERS_PER_BLOCK + 1
    for up_block in model.decoder.up_blocks:
        up_block.resnets = torch.nn.ModuleList(list(up_block.resnets)[:keep])
    return model


def _torch_decode_reference(torch_model: reference.AutoencoderKLFlux2, inp: torch.Tensor) -> torch.Tensor:
    """Apply BN inv-normalize, unpatchify, then run the torch decoder.

    inp: [B, C*p^2, H_t, W_t] patchified latent.
    Returns [B, 3, H, W] (NCHW) for comparison with the TT output permuted to NCHW.
    """
    p = 2  # _PATCH_SIZE
    z_channels = torch_model.config.latent_channels
    bn_eps = torch_model.bn.eps
    s = (torch_model.bn.running_var + bn_eps).sqrt()  # [C*p^2]
    m = torch_model.bn.running_mean  # [C*p^2]
    # Matches TT _inv_normalize: z * sqrt(var + eps) + mean
    inp_norm = inp * s.view(1, -1, 1, 1) + m.view(1, -1, 1, 1)

    # Unpatchify: [B, C*p^2, H_t, W_t] -> [B, C, H_lat, W_lat]
    B, _, h_t, w_t = inp_norm.shape
    z = inp_norm.reshape(B, z_channels, p, p, h_t, w_t)
    z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
    z = z.reshape(B, z_channels, h_t * p, w_t * p)

    with torch.no_grad():
        return torch_model.decode(z).sample  # [B, 3, H, W]


def prep_data(
    vae_parallel_config: Flux2VaeParallelConfig,
    inp: torch.Tensor,
    mesh_device: ttnn.Device,
    tt_model: Flux2VaeDecoder,
) -> ttnn.Tensor:
    # inp: [B, C*p^2, H_t, W_t] patchified latent in torch.
    # Convert to patchified token format [B, H_t*W_t, C*p^2].
    inp_flat = inp.permute(0, 2, 3, 1).flatten(1, 2)

    if vae_parallel_config.h_parallel is not None:
        tt_latents = tensor.from_torch(
            inp_flat,
            device=mesh_device,
            mesh_axes=(None, vae_parallel_config.h_parallel.mesh_axis, None),
        )
    else:
        tt_latents = tensor.from_torch(inp_flat, device=mesh_device)

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
    [(1, 1024, 1024)],
    ids=["1024"],
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

    torch_model = _load_pruned_torch_model()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    z_channels = torch_model.config.latent_channels
    patch_size = 2
    vae_scale_factor = 8

    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    f = vae_scale_factor * patch_size
    inp = torch.randn(batch_size, z_channels * patch_size**2, height // f, width // f)

    torch_output = _torch_decode_reference(torch_model, inp)  # [B, 3, H, W]

    tt_inp = prep_data(vae_parallel_config, inp, mesh_device, tt_model=tt_model)
    tt_out = tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)  # [B, 3, H, W]
    assert_quality(torch_output, tt_out_torch, pcc=0.9978, relative_rmse=0.034)
