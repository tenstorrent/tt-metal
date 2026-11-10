# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3-Image VAE decode timing on the 4x8: a 2K and a 4K canvas, cold then warm, with the stage breakdown.

Env: MINIMAX_H3_IMAGE_MODEL_PATH -> converted snapshot (only vae/ is read).
"""

import os
import time

import pytest
import torch
from loguru import logger

from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig, stitch_tiles
from .common import MESH_4X8_RING


def _vae(mesh_device, waves_per_device=2):
    path = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not path:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    from safetensors.torch import load_file

    config = MiniMaxH3VaeConfig.from_pretrained(os.path.join(path, "vae"))
    vae = MiniMaxH3Vae(config, task="image", mesh_device=mesh_device, waves_per_device=waves_per_device)
    vae.load_state(load_file(os.path.join(path, "vae", "diffusion_pytorch_model.safetensors")))
    return vae, config


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("waves_per_device", [2, 8])
def test_image_decode_timing(mesh_device, waves_per_device):
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // 2))  # as the pipeline does; see its __call__
    vae, config = _vae(mesh_device, waves_per_device)
    for pixels in (2048, 4096):
        z = torch.randn(1, config.latent_channels, 1, pixels // 16, pixels // 16)
        for run in ("cold", "warm1", "warm2"):
            vae._profile = vae._empty_profile()
            mark = time.perf_counter()
            vae.decode_clip(z)
            seconds = time.perf_counter() - mark
            profile = {k: round(v, 3) for k, v in vae._profile.items() if isinstance(v, float) and v > 0.005}
            logger.info(f"RESULT decode {pixels} waves{waves_per_device} {run}: {seconds:.2f}s profile={profile}")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_stitch_slowness(mesh_device):
    """Why is the host stitch 100x slower on decoded tiles than on random ones? Same tiles, three treatments."""
    vae, config = _vae(mesh_device)
    z = torch.randn(1, config.latent_channels, 1, 128, 128)
    vae._ensure_loaded(vae.decoder, vae._decoder_subfolder())
    units = vae._latent_tiles(z)
    decoded = list(vae._stream_decoder_units(units))
    (_, y_len, y_ov), (_, x_len, x_ov) = vae._decode_tile_grid(128, 128)
    cols = len(x_len)
    t0 = decoded[0]
    sub = sum(int(((t != 0) & (t.abs() < torch.finfo(torch.float32).tiny)).sum()) for t in decoded)
    logger.info(
        f"RESULT tiles: n={len(decoded)} dtype={t0.dtype} shape={tuple(t0.shape)} contiguous={t0.is_contiguous()} "
        f"storage_elems={t0.untyped_storage().nbytes() // t0.element_size()} vs numel={t0.numel()} subnormals={sub}"
    )

    def run(tag, tiles):
        rows = [tiles[i * cols : (i + 1) * cols] for i in range(len(y_len))]
        mark = time.perf_counter()
        stitch_tiles(rows, y_ov, x_ov)
        logger.info(f"RESULT stitch {tag}: {time.perf_counter() - mark:.3f}s")

    affinity = len(os.sched_getaffinity(0))
    threads = torch.get_num_threads()
    logger.info(
        f"RESULT with mesh open: affinity={affinity} cores, torch threads={threads}, cpu_count={os.cpu_count()}"
    )
    run("as_decoded", decoded)
    for n in sorted({1, affinity, max(1, affinity // 2)}):
        torch.set_num_threads(n)
        run(f"threads={n}", decoded)
    torch.set_num_threads(threads)
