# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import time

import pytest
import torch
import ttnn
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger

from ....models.vae import vae_sd35
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality


def skip_invalid_submesh_shape(mesh_device: ttnn.Device, submesh_shape: tuple[int, int]):
    mesh_device_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > mesh_device_shape[0] or submesh_shape[1] > mesh_device_shape[1]:
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")


@pytest.mark.parametrize("mesh_device", [(2, 4), (4, 8)], ids=["t3k", "tg"], indirect=True)
@pytest.mark.parametrize("submesh_shape", [(1, 4)])
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    (
        "batch",
        "in_channels",
        "out_channels",
        "layers_per_block",
        "height",
        "width",
        "norm_num_groups",
        "block_out_channels",
    ),
    [
        (1, 16, 3, 2, 128, 128, 32, (128, 256, 512, 512)),
    ],
)
def test_sd35_vae_vae_decoder(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: tuple[int, int],
    batch: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    height: int,
    width: int,
    norm_num_groups: int,
    block_out_channels: list[int] | tuple[int, ...],
):
    torch.manual_seed(0)

    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = AutoencoderKL(
        in_channels=out_channels,
        out_channels=out_channels,
        up_block_types=["UpDecoderBlock2D"] * len(block_out_channels),
        block_out_channels=block_out_channels,
        layers_per_block=layers_per_block,
        latent_channels=in_channels,
        norm_num_groups=norm_num_groups,
        use_quant_conv=False,
        use_post_quant_conv=False,
    )
    torch_model.eval()

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=4, mesh_axis=1))

    tt_model = vae_sd35.VAEDecoder.from_torch(
        torch_model.decoder,
        mesh_device=submesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_model.load_torch_state_dict(torch_model.decoder.state_dict())

    torch_input = torch.randn(batch, in_channels, height, width)

    tt_input_tensor = ttnn.from_torch(
        torch_input.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=submesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh_device),
    )

    with torch.no_grad():
        torch_output = torch_model.decode(torch_input).sample

    tt_out = tt_model(tt_input_tensor)

    tt_final_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_final_out_torch, pcc=0.99_000)

    start = time()
    tt_out = tt_model(tt_input_tensor)
    ttnn.synchronize_device(submesh_device)
    logger.info(f"VAE Time taken: {time() - start}")
