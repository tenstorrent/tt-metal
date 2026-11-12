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


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_decode_breakdown_4k(mesh_device):
    """Where the pipeline's 4K decode stage goes: VAE decode_clip (host stitch, profiled) and the host post-processing
    the pipeline applies after it, then the device-stitch variants (time, and max abs diff vs the host stitch)."""
    from ....pipelines.minimax_h3 import image_host_minimax_h3 as host

    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // 2))
    path = os.environ.get("MINIMAX_H3_IMAGE_MODEL_PATH")
    if not path:
        pytest.skip("set MINIMAX_H3_IMAGE_MODEL_PATH")
    from safetensors.torch import load_file

    config = MiniMaxH3VaeConfig.from_pretrained(os.path.join(path, "vae"))
    state = load_file(os.path.join(path, "vae", "diffusion_pytorch_model.safetensors"))
    z = torch.randn(1, config.latent_channels, 1, 256, 256, generator=torch.Generator().manual_seed(0))
    mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(config.latents_std).view(1, -1, 1, 1, 1)
    z = z * std + mean
    pm, ps = torch.tensor(host.PIXEL_MEAN).view(3, 1, 1), torch.tensor(host.PIXEL_STD).view(3, 1, 1)

    def post(decoded):
        mark = time.perf_counter()
        pixels01 = (decoded[0, :, 0].float() * ps + pm).clamp(0, 1)
        t_norm = time.perf_counter() - mark
        mark = time.perf_counter()
        image = host.finish_image(pixels01, (4096, 4096))
        return image, t_norm, time.perf_counter() - mark

    vae = MiniMaxH3Vae(
        config, task="image", mesh_device=mesh_device, waves_per_device=2, pixel_norm=(host.PIXEL_MEAN, host.PIXEL_STD)
    )
    vae.load_state(state)
    reference = None
    for run in ("cold", "warm1", "warm2"):
        vae._profile = vae._empty_profile()
        mark = time.perf_counter()
        decoded = vae.decode_clip(z)
        t_decode = time.perf_counter() - mark
        image, t_norm, t_finish = post(decoded)
        profile = {k: round(v, 3) for k, v in vae._profile.items() if isinstance(v, float) and v > 0.003}
        logger.info(
            f"RESULT host-stitch {run}: decode_clip {t_decode:.3f}s + denorm/clamp {t_norm:.3f}s + uint8 {t_finish:.3f}s "
            f"| out {tuple(decoded.shape)} {decoded.dtype} | profile={profile}"
        )
        reference = decoded

    for exchange in ("gather", "strips", "neighbor"):
        vae.device_stitch, vae.stitch_exchange = True, exchange
        try:
            for run in ("cold", "warm"):
                vae._profile = vae._empty_profile()
                mark = time.perf_counter()
                decoded = vae.decode_clip(z)
                t_decode = time.perf_counter() - mark
                _, t_norm, t_finish = post(decoded)
                diff = (
                    (decoded.float() - reference.float()).abs().max().item()
                    if decoded.shape == reference.shape
                    else None
                )
                logger.info(
                    f"RESULT device-stitch {exchange} {run}: decode_clip {t_decode:.3f}s + post {t_norm + t_finish:.3f}s "
                    f"| out {tuple(decoded.shape)} {decoded.dtype} | max abs diff vs host stitch {diff}"
                )
        except Exception as err:  # report and move on: each variant is a separate question
            logger.info(f"RESULT device-stitch {exchange}: failed: {type(err).__name__}: {str(err)[:300]}")
    vae.device_stitch = False


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_image_decode_attribution_4k(mesh_device):
    """4K decode with the VAE's `profile` attribution on (it serializes the waves, so the total is not the served
    number): how the time splits between host prep, upload, device decode and readback, and how much crosses PCIe."""
    torch.set_num_threads(max(1, len(os.sched_getaffinity(0)) // 2))
    vae, config = _vae(mesh_device, 2)
    vae.profile = True
    z = torch.randn(1, config.latent_channels, 1, 4096 // 16, 4096 // 16)
    for run in ("cold", "warm1", "warm2"):
        vae._profile = vae._empty_profile()
        mark = time.perf_counter()
        vae.decode_clip(z)
        seconds = time.perf_counter() - mark
        p = vae._profile
        keys = ("host_prep", "upload", "device", "readback", "readback_mb", "tiling", "stitch", "waves", "units")
        logger.info(f"RESULT decode_attr 4096 {run}: {seconds:.2f}s " + " ".join(f"{k}={p.get(k)}" for k in keys))
