# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_wan_2d import WanVaeDecoder2DAdapter, WanVaeEncoder2DAdapter
from models.tt_dit.parallel.config import VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.cfg import submesh_shape
from models.tt_dit.pipelines.fibo.pipeline_fibo import FiboPipelineConfig
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.test import line_params_req_exact_devices

_CHECKPOINT = "briaai/FIBO"

_MESHES = pytest.mark.parametrize("mesh_device", [(2, 2), (2, 4), (4, 8)], ids=["2x2", "2x4", "4x8"], indirect=True)
_DEVICE_PARAMS = pytest.mark.parametrize(
    "device_params",
    # The encoder's avg_pool2d needs L1_SMALL.
    [{**line_params_req_exact_devices, "l1_small_size": 32768, "trace_region_size": 8000000}],
    indirect=True,
)
_SIZES = pytest.mark.parametrize(("height", "width"), [(1024, 1024)])


@_MESHES
@_DEVICE_PARAMS
@_SIZES
def test_vae(
    *,
    mesh_device: ttnn.MeshDevice,
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    ccl_manager, parallel_config = _pipeline_vae_setup(mesh_device)

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
    height: int,
    width: int,
) -> None:
    torch.manual_seed(0)

    ccl_manager, parallel_config = _pipeline_vae_setup(mesh_device)

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


def _pipeline_vae_setup(mesh_device: ttnn.MeshDevice) -> tuple[CCLManager, VaeHWParallelConfig]:
    """Returns the CCL manager and parallel config the pipeline runs the VAE with, on its submesh."""
    config = FiboPipelineConfig.default(mesh_shape=mesh_device.shape, edit=False)
    submesh_device = mesh_device.create_submesh(submesh_shape(config.dit_parallel_config))
    ccl_manager = CCLManager(submesh_device, num_links=config.num_links, topology=config.topology)
    return ccl_manager, config.vae_parallel_config
