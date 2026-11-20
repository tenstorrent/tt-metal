# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch

import ttnn

from ....models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from .device_params import line_params_flux2, single_device_params_flux2

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
    split_height: bool = False,
) -> ttnn.Tensor:
    # inp: [B, C*p^2, H_t, W_t] patchified latent in torch.
    # Convert to patchified token format [B, H_t*W_t, C*p^2].
    inp_flat = inp.permute(0, 2, 3, 1).flatten(1, 2)

    # With split_height the tokens arrive replicated, as the pipeline passes them when the VAE rows
    # are not on the SP axis, and preprocess_and_unpatchify splits them.
    if vae_parallel_config.h_parallel is not None and not split_height:
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
        split_height=split_height,
    )

    return tt_latents


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(1, 1), single_device_params_flux2],
        [(2, 2), line_params_flux2],
        [(1, 8), line_params_flux2],
        [(2, 4), line_params_flux2],
        [(4, 8), line_params_flux2],
    ],
    ids=["1x1", "2x2", "1x8", "2x4", "4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    # 1088x1184 gives 34 x 74 latent pixels per device on 2x4, not whole tiles for the group norm.
    [(1, 1024, 1024), (1, 1024, 1184), (1, 1056, 1024), (1, 1088, 1184)],
    ids=["1024", "1024x1184", "1056x1024", "1088x1184"],
)
# Rows on mesh axis 1, and channels (tp_h) or columns (h_w) on axis 0.
@pytest.mark.parametrize("split", ["tp_h", "h_w"])
# Tokens H-sharded on the host, or replicated and split by the decoder.
@pytest.mark.parametrize("split_height", [False, True], ids=["host_rows", "device_rows"])
def test_vae_flux2_decoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
    split: str,
    split_height: bool,
) -> None:
    torch.manual_seed(0)

    tp_axis = 0 if split == "tp_h" else None
    h_axis = 1
    w_axis = 0 if split == "h_w" else None

    torch_model = _load_pruned_torch_model()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    # Host-split tokens need whole token rows (2 latent rows) per device, the decoder's split only
    # whole latent rows. 1056x1024 has 132 latent rows: 66 token rows split over 2 devices but not
    # over 4, and 132 latent rows split over 4 but not over 8.
    h_factor = vae_parallel_config.h_parallel.factor if vae_parallel_config.h_parallel is not None else 1
    rows_per_device_unit = 1 if split_height else 2
    if (height // 8) % (rows_per_device_unit * h_factor) != 0:
        pytest.skip(f"{height // 8} latent rows do not split into whole {'latent' if split_height else 'token'} rows")

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

    tt_inp = prep_data(vae_parallel_config, inp, mesh_device, tt_model=tt_model, split_height=split_height)
    tt_out = tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)

    # The output stays split over the H and W axes.
    h_mesh_axis = vae_parallel_config.h_parallel.mesh_axis if vae_parallel_config.h_parallel is not None else None
    w_mesh_axis = vae_parallel_config.w_parallel.mesh_axis if vae_parallel_config.w_parallel is not None else None
    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[None, h_mesh_axis, w_mesh_axis, None]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.9978, relative_rmse=0.034)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        [(2, 2), line_params_flux2],
        [(2, 4), line_params_flux2],
        [(1, 8), line_params_flux2],
        [(4, 8), line_params_flux2],
    ],
    ids=["2x2", "2x4", "1x8", "4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024)], ids=["1024"])
@pytest.mark.parametrize(
    ("tp_axis", "h_axis", "w_axis"),
    [(0, 1, None), (None, 1, 0)],
    ids=["tp_h", "h_w"],
)
def test_vae_flux2_encoder(
    *,
    mesh_device: ttnn.MeshDevice,
    height: int,
    width: int,
    tp_axis: int | None,
    h_axis: int,
    w_axis: int | None,
) -> None:
    """The encoder with the channels and rows split, and with the rows and columns split as the
    pipeline's presets do. The output is read as the pipeline reads it."""
    torch.manual_seed(0)

    torch_model = reference.AutoencoderKLFlux2.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae")
    assert isinstance(torch_model, reference.AutoencoderKLFlux2)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=w_axis)

    tt_model = Flux2VaeEncoder(
        in_channels=torch_model.config.in_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=torch_model.config.layers_per_block,
        z_channels=torch_model.config.latent_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    image = torch.rand(1, 3, height, width) * 2 - 1

    with torch.no_grad():
        torch_output = torch_model.encode(image).latent_dist.mode()  # [B, C, H / 8, W / 8]

    h_mesh_axis = vae_parallel_config.h_parallel.mesh_axis if vae_parallel_config.h_parallel is not None else None
    w_mesh_axis = vae_parallel_config.w_parallel.mesh_axis if vae_parallel_config.w_parallel is not None else None
    tt_image = tensor.from_torch(
        image.permute(0, 2, 3, 1),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_axes=[None, h_mesh_axis, w_mesh_axis, None],
    )
    tt_out = tt_model.forward(tt_image)
    ttnn.synchronize_device(mesh_device)

    tt_out_torch = tensor.fast_device_to_host(
        tt_out, mesh_device, tt_model.output_concat_dims(), ccl_manager=ccl_manager
    ).permute(
        0, 3, 1, 2
    )  # [B, C, H / 8, W / 8]
    assert_quality(torch_output, tt_out_torch, pcc=0.999, relative_rmse=0.03)
