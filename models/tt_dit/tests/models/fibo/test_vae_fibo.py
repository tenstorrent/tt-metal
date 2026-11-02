# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter, WanVaeEncoder2DAdapter
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality

_CHECKPOINT = "briaai/FIBO"

_MESHES = pytest.mark.parametrize(
    ("mesh_device", "h_axis", "w_axis"),
    [
        pytest.param((1, 4), 1, 0, id="1x4_h1_w0"),
        pytest.param((2, 2), 0, 1, id="2x2_h0_w1"),
        pytest.param((2, 4), 0, 1, id="2x4_h0_w1"),
    ],
    indirect=["mesh_device"],
)
_DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "l1_small_size": 32768,
            "trace_region_size": 8000000,
        }
    ],
    indirect=True,
)
_SIZES = pytest.mark.parametrize(("height", "width"), [(1024, 1024)])


@_MESHES
@_DEVICE_PARAMS
@_SIZES
def test_vae(
    *,
    mesh_device: ttnn.MeshDevice,
    h_axis: int,
    w_axis: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig.from_axes(mesh_device, h_axis=h_axis, w_axis=w_axis)

    logger.info("constructing tt VAE...")
    tt_vae = WanVaeDecoder2DAdapter(
        checkpoint_name=_CHECKPOINT,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_torch=False,
    )

    logger.info("constructing torch VAE...")
    torch_vae = WanVaeDecoder2DAdapter(
        checkpoint_name=_CHECKPOINT,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_torch=True,
    )

    # Latents are laid out (B, H, W, C) for the adapter's decode signature. Shape derived from
    # the loaded VAE config so we don't carry stale values when checkpoints change.
    batch_size = 2
    latents_h = height // tt_vae.spatial_compression_ratio
    latents_w = width // tt_vae.spatial_compression_ratio
    latents = torch.randn(batch_size, latents_h, latents_w, tt_vae.z_dim, dtype=torch.float32)

    logger.info("running torch VAE decode...")
    with torch.no_grad():
        torch_out = torch_vae.decode(latents, traced=True)

    logger.info("running tt VAE decode...")
    tt_out = tt_vae.decode(latents, traced=True)

    assert_quality(torch_out, tt_out, pcc=0.99, relative_rmse=0.1)


@_MESHES
@_DEVICE_PARAMS
@_SIZES
def test_vae_encoder(
    *,
    mesh_device: ttnn.MeshDevice,
    h_axis: int,
    w_axis: int,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = VaeHWParallelConfig.from_axes(mesh_device, h_axis=h_axis, w_axis=w_axis)

    logger.info("constructing tt VAE encoder...")
    tt_vae = WanVaeEncoder2DAdapter(
        checkpoint_name=_CHECKPOINT,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_torch=False,
    )

    logger.info("constructing torch VAE encoder...")
    torch_vae = WanVaeEncoder2DAdapter(
        checkpoint_name=_CHECKPOINT,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_torch=True,
    )

    batch_size = 2
    images = torch.rand(batch_size, 3, height, width) * 2 - 1

    logger.info("running torch VAE encode...")
    torch_out = torch_vae.encode(images, traced=True)

    logger.info("running tt VAE encode...")
    tt_out = tt_vae.encode(images, traced=True)

    assert_quality(torch_out, tt_out, pcc=0.99, relative_rmse=0.1)
