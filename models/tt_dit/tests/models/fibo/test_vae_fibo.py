# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.models.vae.vae_fibo import FiboVAEDecoderAdapter
from models.tt_dit.parallel.config import Flux2VaeParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality

_CHECKPOINT = "briaai/FIBO"


@pytest.mark.parametrize(
    ("mesh_device", "h_axis", "w_axis"),
    [
        pytest.param((1, 4), 1, None, id="1x4_h4"),
        pytest.param((2, 2), 0, 1, id="2x2_h0_w1"),
        pytest.param((2, 4), 0, 1, id="2x4_h0_w1"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 32768, "trace_region_size": 8000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("height", "width"),
    [(1024, 1024)],
)
@pytest.mark.parametrize(
    "traced",
    [
        pytest.param(True, id="traced"),
        pytest.param(False, id="not_traced"),
    ],
)
def test_vae(
    *,
    mesh_device: ttnn.MeshDevice,
    h_axis: int | None,
    w_axis: int | None,
    height: int,
    width: int,
    traced: bool,
) -> None:
    torch.manual_seed(0)

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, h_axis=h_axis, w_axis=w_axis)

    logger.info("constructing tt VAE...")
    tt_vae = FiboVAEDecoderAdapter(
        checkpoint_name=_CHECKPOINT,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_torch=False,
    )
    tt_vae.reload_weights()

    logger.info("constructing torch VAE...")
    torch_vae = FiboVAEDecoderAdapter(
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
        torch_out = torch_vae.decode(latents, traced=False)

    logger.info("running tt VAE decode...")
    tt_out = tt_vae.decode(latents, traced=traced)

    assert_quality(torch_out, tt_out, pcc=0.99, relative_rmse=0.1)
