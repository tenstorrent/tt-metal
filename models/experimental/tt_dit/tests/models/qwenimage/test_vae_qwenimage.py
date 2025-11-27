# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import time

import diffusers.models.autoencoders.autoencoder_kl_qwenimage as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....models.vae.vae_qwenimage import QwenImageVaeDecoder
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((1, 1), id="1x1"),
        pytest.param((1, 4), id="1x4"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width"),
    [
        (1, 128, 128),
    ],
)
def test_vae_qwenimage_decoder(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis = 1

    torch_model = reference.AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
    assert isinstance(torch_model, reference.AutoencoderKLQwenImage)
    torch_model.eval()

    in_channels = torch_model.config["z_dim"]

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = (
        VAEParallelConfig(tensor_parallel=ParallelFactor(factor=mesh_device.shape[tp_axis], mesh_axis=tp_axis))
        if tp_axis is not None
        else None
    )

    tt_model = QwenImageVaeDecoder(
        base_dim=torch_model.config["base_dim"],
        z_dim=torch_model.config["z_dim"],
        dim_mult=torch_model.config["dim_mult"],
        num_res_blocks=torch_model.config["num_res_blocks"],
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    inp = torch.randn(batch_size, in_channels, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model.decode(inp.unsqueeze(2)).sample.squeeze(2)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.9997, relative_rmse=0.023)

    start = time()
    tt_model(tt_inp)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"VAE time taken: {time() - start}")
