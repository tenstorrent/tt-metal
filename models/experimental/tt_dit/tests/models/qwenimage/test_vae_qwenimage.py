# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from time import time

import diffusers.models.autoencoders.autoencoder_kl_qwenimage as reference
import pytest
import torch
import ttnn
from loguru import logger

from ....models.vae.vae_qwenimage import (
    QwenImageAttentionBlock,
    QwenImageRmsNorm,
    QwenImageVaeContext,
    QwenImageVaeDecoder,
)
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="t3k"),
        pytest.param((4, 8), id="tg"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("submesh_shape", "tp_axis"),
    [
        ((1, 4), None),
        ((1, 4), 1),
        ((1, 8), 1),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width", "channels"),
    [
        (2, 128, 128, 384),
    ],
)
def test_vae_qwenimage_norm(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: tuple[int, int],
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    tp_axis: int | None,
) -> None:
    torch.manual_seed(0)

    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = reference.QwenImageRMS_norm(channels)
    torch_model.eval()

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)

    ctx = QwenImageVaeContext(tp_axis=tp_axis, device=submesh_device, ccl_manager=ccl_manager)
    tt_model = QwenImageRmsNorm(channels, ctx=ctx)

    tt_model.load_torch_state_dict(torch_model.state_dict())

    inp = torch.randn(batch_size, channels, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), mesh_axes=[..., tp_axis], device=submesh_device)

    with torch.no_grad():
        torch_output = torch_model.forward(inp)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[..., tp_axis]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.999992, relative_rmse=0.007)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="t3k"),
        pytest.param((4, 8), id="tg"),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    ("submesh_shape", "tp_axis"),
    [
        ((1, 4), None),
        ((1, 4), 1),
        ((1, 8), 1),
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width", "dim"),
    [
        (2, 128, 128, 384),
    ],
)
def test_vae_qwenimage_attention(
    *,
    mesh_device: ttnn.Device,
    submesh_shape: tuple[int, int],
    batch_size: int,
    height: int,
    width: int,
    dim: int,
    tp_axis: int | None,
) -> None:
    torch.manual_seed(0)

    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = reference.QwenImageAttentionBlock(dim)
    torch_model.eval()

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)

    ctx = QwenImageVaeContext(tp_axis=tp_axis, device=submesh_device, ccl_manager=ccl_manager)
    tt_model = QwenImageAttentionBlock(dim=dim, ctx=ctx)

    tt_model.load_torch_state_dict(torch_model.state_dict())

    inp = torch.randn(batch_size, dim, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), mesh_axes=[..., tp_axis], device=submesh_device)

    with torch.no_grad():
        torch_output = torch_model.forward(inp.unsqueeze(2)).squeeze(2)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[..., tp_axis]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.99996, relative_rmse=0.009)


@pytest.mark.parametrize(
    "mesh_device",
    [
        pytest.param((2, 4), id="t3k"),
        pytest.param((4, 8), id="tg"),
    ],
    indirect=True,
)
@pytest.mark.parametrize("submesh_shape", [(1, 4), (1, 8)])
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
    submesh_shape: tuple[int, int],
    batch_size: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    tp_axis = 1

    skip_invalid_submesh_shape(mesh_device, submesh_shape)
    submesh_device = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    torch_model = reference.AutoencoderKLQwenImage.from_pretrained("Qwen/Qwen-Image", subfolder="vae")
    assert isinstance(torch_model, reference.AutoencoderKLQwenImage)
    torch_model.eval()

    in_channels = torch_model.config["z_dim"]

    ccl_manager = CCLManager(submesh_device, topology=ttnn.Topology.Linear)
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
        temperal_downsample=torch_model.config["temperal_downsample"],
        device=submesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )

    tt_model.load_torch_state_dict(torch_model.state_dict())

    inp = torch.randn(batch_size, in_channels, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), device=submesh_device)

    with torch.no_grad():
        torch_output = torch_model.decode(inp.unsqueeze(2)).sample.squeeze(2)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.9997, relative_rmse=0.023)

    start = time()
    tt_model(tt_inp)
    ttnn.synchronize_device(submesh_device)
    logger.info(f"VAE time taken: {time() - start}")


def skip_invalid_submesh_shape(mesh_device: ttnn.Device, submesh_shape: tuple[int, int]) -> None:
    mesh_device_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > mesh_device_shape[0] or submesh_shape[1] > mesh_device_shape[1]:
        pytest.skip("submesh shape is larger than parent mesh shape, skipping")
