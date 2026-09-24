# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The neighbour stitch pinned to the gather stitch through a whole 4x8 decode: same latents, weights and geometry,
only the exchange differs. Real weights: a per-device stub would hide the tile placement this checks."""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig, split_tiles
from ....parallel.config import ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MODEL_NAME
from ....utils import cache
from ....utils.check import assert_quality
from ....utils.conv3d import conv3d_blocking_hash
from .common import MESH_4X8_RING, weights_subdir

MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)
# The served 1344x768 canvas: a 4x7 tile grid over a 48x84 latent chunk.
LATENT_HW = (48, 84)
HEIGHT, WIDTH = 768, 1344


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_neighbor_stitch_matches_gather(mesh_device, reset_seeds):
    weights = weights_subdir("vae")
    if weights is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    config = MiniMaxH3VaeConfig.from_pretrained(weights)
    torch.manual_seed(5)

    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(mesh_axis=0, factor=1))

    def load_cached(module, subfolder, state):
        blocking = conv3d_blocking_hash(module)
        cache.load_model(
            module,
            model_name=MODEL_NAME,
            subfolder=f"{subfolder}_{blocking}" if blocking else subfolder,
            parallel_config=parallel_config,
            mesh_shape=tuple(mesh_device.shape),
            mesh_device=mesh_device,
            dtype="fp32",
            get_torch_state_dict=lambda: (_ for _ in ()).throw(
                RuntimeError(f"cache miss for {subfolder}; run a served decode first to populate it")
            ),
        )

    vae = MiniMaxH3Vae(
        config,
        task="t2va",
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring),
        device_stitch=True,
        weight_loader=load_cached,
        pixel_denorm=(MINIMAX_H3_PIXEL_MEAN, MINIMAX_H3_PIXEL_STD),
    )
    load_cached(vae.decoder, vae._decoder_subfolder(), {})

    num_frames = vae.decoder.latent_shape[0]
    latent_h, latent_w = LATENT_HW
    chunk = torch.randn(1, config.latent_channels, num_frames, latent_h, latent_w)

    vae._profile = vae._empty_profile()
    gathered = vae._decode_clips_device_stitched([chunk], "float")[0]
    vae._profile = vae._empty_profile()
    neighboured = vae._decode_clips_neighbor_stitched([chunk], "float")[0]

    assert (
        gathered.shape == neighboured.shape
    ), f"gather gave {tuple(gathered.shape)}, neighbour {tuple(neighboured.shape)}"
    assert gathered.shape[-2:] == (HEIGHT, WIDTH), f"canvas is {tuple(gathered.shape[-2:])}, not {(HEIGHT, WIDTH)}"
    logger.info(f"canvas {tuple(gathered.shape)}; comparing the two exchanges")
    assert_quality(gathered, neighboured, pcc=0.9999, relative_rmse=0.02)

    # The seams on their own: a whole-canvas metric averages a boundary defect away, and the
    # boundaries are the only place the two exchanges do different arithmetic.
    ratio = config.spatial_compression_ratio
    _, _, height_overlaps = split_tiles(HEIGHT, 256, 64, ratio)
    _, _, width_overlaps = split_tiles(WIDTH, 256, 64, ratio)
    y = 0
    for index, overlap in enumerate(height_overlaps):
        y += 256 - overlap
        logger.info(f"horizontal seam {index} at y={y}, extent {overlap}")
        assert_quality(gathered[..., y : y + overlap, :], neighboured[..., y : y + overlap, :], pcc=0.9999)
    x = 0
    for index, overlap in enumerate(width_overlaps):
        x += 256 - overlap
        logger.info(f"vertical seam {index} at x={x}, extent {overlap}")
        assert_quality(gathered[..., x : x + overlap], neighboured[..., x : x + overlap], pcc=0.9999)
