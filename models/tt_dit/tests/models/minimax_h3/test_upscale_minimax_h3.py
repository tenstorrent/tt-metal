# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device bicubic 768p -> 1080p (`CanvasUpscaler`, `MiniMaxH3Vae(upscale_to=...)`) against host bicubic.

Both arms end in the same device YUV 4:2:0 conversion, so a difference is the resize alone. The host
arm is `F.interpolate(..., mode="bicubic", align_corners=False)` on float RGB, as the 1080p export does.

    # CPU only: the per-device block math equals F.interpolate.
    pytest models/tt_dit/tests/models/minimax_h3/test_upscale_minimax_h3.py -k blocks
    # 4x8: real frames (one mp4 per seed), and a 10 s VAE decode with the resize inside the readback.
    MINIMAX_H3_UPSCALE_MP4S=s0.mp4,s1.mp4 MINIMAX_H3_UPSCALE_OUT=out/ \\
    MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3 pytest ... -k "frames or decode" -s
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import ttnn

from ....models.vae.minimax_h3.upscale_minimax_h3 import CanvasUpscaler, bicubic_matrix, upscale_blocks_reference
from ....utils.yuv_d2h import fast_device_to_host_yuv

IN_HW = (768, 1344)
OUT_HW = (1080, 1920)
CHUNK_FRAMES = 20

MESH_4X8_RING = [
    pytest.param(
        (4, 8),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
            "require_exact_physical_num_devices": True,
            "l1_small_size": 65536,
        },
        id="4x8ring",
    )
]


def host_bicubic(video_BCTHW: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """The host reference: float bicubic in chunks of 16 frames, clamped to [-1, 1]."""
    _, c, t, h, w = video_BCTHW.shape
    frames = video_BCTHW[0].permute(1, 0, 2, 3)
    out = torch.empty(t, c, *out_hw)
    for start in range(0, t, 16):
        out[start : start + 16] = F.interpolate(
            frames[start : start + 16], size=out_hw, mode="bicubic", align_corners=False
        ).clamp_(-1.0, 1.0)
    return out.permute(1, 0, 2, 3).unsqueeze(0)


def psnr_uint8(reference: np.ndarray, test: np.ndarray) -> float:
    mse = float(np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


def boundary_report(reference: np.ndarray, test: np.ndarray, mesh_shape: tuple[int, int]) -> dict[str, float]:
    """Max |dY| on the luma rows and columns either side of a device patch boundary, and elsewhere."""
    height, width = OUT_HW
    diff = np.abs(reference[:, :height].astype(np.int16) - test[:, :height].astype(np.int16))
    rows = sorted({r + d for k in range(1, mesh_shape[0]) for d in (-1, 0) for r in [k * height // mesh_shape[0]]})
    cols = sorted({c + d for k in range(1, mesh_shape[1]) for d in (-1, 0) for c in [k * width // mesh_shape[1]]})
    inner = np.ones((height, width), dtype=bool)
    inner[rows, :] = False
    inner[:, cols] = False
    return {
        "boundary_rows_max": float(diff[:, rows, :].max()),
        "boundary_cols_max": float(diff[:, :, cols].max()),
        "interior_max": float(diff[:, inner].max()),
        "boundary_rows_mean": float(diff[:, rows, :].mean()),
        "interior_mean": float(diff[:, inner].mean()),
    }


def yuv420_to_rgb(planar: np.ndarray, frame: int) -> np.ndarray:
    """One frame of limited-range BT.601 planar 4:2:0 to RGB uint8, for stills."""
    height = planar.shape[1] * 2 // 3
    width = planar.shape[2]
    y = planar[frame, :height].astype(np.float32)
    chroma = planar[frame, height:].reshape(2, height // 2, width // 2).astype(np.float32)
    cb, cr = (np.repeat(np.repeat(p, 2, axis=0), 2, axis=1) - 128.0 for p in chroma)
    y = (y - 16.0) * (255.0 / 219.0)
    rgb = np.stack([y + 1.596 * cr, y - 0.392 * cb - 0.813 * cr, y + 2.017 * cb], axis=-1)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def save_still(planar: np.ndarray, frame: int, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(yuv420_to_rgb(planar, frame)).save(path)


def read_mp4(path: str) -> torch.Tensor:
    """`(1, 3, T, H, W)` float in [-1, 1]."""
    import av

    with av.open(path) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    video = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)
    return video.float().div_(127.5).sub_(1.0)


def patches_to_yuv(patch_BCTHW: ttnn.Tensor, mesh_device, out_hw) -> np.ndarray:
    planar = fast_device_to_host_yuv(patch_BCTHW, mesh_device, use_persistent_buffer=False)
    return np.asarray(planar).reshape(-1, out_hw[0] * 3 // 2, out_hw[1])


def host_to_yuv(video_BCTHW: torch.Tensor, mesh_device) -> np.ndarray:
    """Host pixels through the same device YUV conversion the device arm ends in."""
    mesh_shape = tuple(mesh_device.shape)
    parts = []
    for start in range(0, video_BCTHW.shape[2], CHUNK_FRAMES):
        chunk = video_BCTHW[:, :, start : start + CHUNK_FRAMES]
        patch = ttnn.from_torch(
            chunk,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(3, 4)),
        )
        parts.append(patches_to_yuv(patch, mesh_device, video_BCTHW.shape[-2:]))
        ttnn.deallocate(patch)
    return np.concatenate(parts)


@pytest.mark.parametrize(("in_hw", "out_hw", "mesh_shape"), [(IN_HW, OUT_HW, (4, 8)), ((64, 96), (90, 160), (2, 4))])
def test_upscale_blocks_match_interpolate(in_hw, out_hw, mesh_shape):
    """Each device's patch, assembled, is exactly the host resize: no seam at a device boundary."""
    torch.manual_seed(0)
    canvas = torch.rand(1, 3, 2, *in_hw, dtype=torch.float64) * 2 - 1
    expected = F.interpolate(canvas[0].transpose(0, 1), size=out_hw, mode="bicubic", align_corners=False)
    expected = expected.transpose(0, 1).unsqueeze(0)
    got = upscale_blocks_reference(canvas, out_hw, mesh_shape)
    assert got.shape == expected.shape
    assert (got - expected).abs().max().item() < 1e-12

    matrix = bicubic_matrix(in_hw[1], out_hw[1])
    assert torch.allclose(matrix.sum(dim=1), torch.ones(out_hw[1], dtype=torch.float64))


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8_RING, indirect=["mesh_device", "device_params"])
def test_upscale_frames(mesh_device):
    """Real 768p frames per seed: device resize vs host bicubic, PSNR and device-boundary error."""
    from loguru import logger

    paths = [p for p in os.environ.get("MINIMAX_H3_UPSCALE_MP4S", "").split(",") if p]
    if not paths:
        pytest.skip("set MINIMAX_H3_UPSCALE_MP4S to 768p mp4s, one per seed")
    out_dir = Path(os.environ.get("MINIMAX_H3_UPSCALE_OUT", "upscale_out"))
    mesh_shape = tuple(mesh_device.shape)
    upscaler = CanvasUpscaler(mesh_device, IN_HW, OUT_HW)

    def device_arm(video):
        parts = []
        for start in range(0, video.shape[2], CHUNK_FRAMES):
            rows = ttnn.from_torch(
                video[:, :, start : start + CHUNK_FRAMES],
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(3, None)),
            )
            parts.append(patches_to_yuv(upscaler(rows), mesh_device, OUT_HW))
            ttnn.deallocate(rows)
        return np.concatenate(parts)

    device_arm(read_mp4(paths[0])[:, :, :CHUNK_FRAMES])
    worst = math.inf
    for path in paths:
        name = Path(path).stem
        video = read_mp4(path)
        assert tuple(video.shape[-2:]) == IN_HW, f"{path} is {tuple(video.shape[-2:])}, want {IN_HW}"

        started = time.perf_counter()
        host = host_bicubic(video, OUT_HW)
        host_s = time.perf_counter() - started
        reference = host_to_yuv(host, mesh_device)
        del host

        started = time.perf_counter()
        test = device_arm(video)
        device_s = time.perf_counter() - started

        assert test.shape == reference.shape
        score = psnr_uint8(reference, test)
        luma = psnr_uint8(reference[:, : OUT_HW[0]], test[:, : OUT_HW[0]])
        seams = boundary_report(reference, test, mesh_shape)
        worst = min(worst, score)
        logger.info(
            f"UPSCALE_FRAMES {name} frames={video.shape[2]} psnr_yuv={score:.2f} psnr_y={luma:.2f} "
            f"host_bicubic={host_s:.3f}s device_upscale_yuv={device_s:.3f}s "
            + " ".join(f"{k}={v:.2f}" for k, v in seams.items())
        )
        middle = video.shape[2] // 2
        save_still(test, middle, out_dir / f"{name}_device_f{middle}.png")
        save_still(reference, middle, out_dir / f"{name}_host_f{middle}.png")
    assert worst >= 40.0, f"worst PSNR {worst:.2f} dB < 40"


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8_RING, indirect=["mesh_device", "device_params"])
def test_decode_upscale(mesh_device):
    """10 s VAE decode: yuv420 768p, yuv420 resized on device, and float + host bicubic, same latents."""
    from loguru import logger

    from ....models.vae.minimax_h3.vae_minimax_h3 import MiniMaxH3Vae, MiniMaxH3VaeConfig
    from ....parallel.manager import CCLManager
    from .common import weights_subdir
    from .test_performance_vae_minimax_h3 import DECODE_STAGE_MODES, DECODE_STAGE_WARM_FRAMES, _decode_stage_state

    weights_dir = weights_subdir("vae")
    if weights_dir is None:
        pytest.skip("MiniMax-H3 vae not found; set MINIMAX_H3_MODEL_PATH")
    out_dir = Path(os.environ.get("MINIMAX_H3_UPSCALE_OUT", "upscale_out"))
    # 10 s is 243 frames on the 17n + 5 grid, which decodes from 72 latent frames.
    latent_frames = int(os.environ.get("MINIMAX_H3_UPSCALE_LATENT_FRAMES", "72"))
    seed = int(os.environ.get("MINIMAX_H3_UPSCALE_SEED", "0"))

    config = MiniMaxH3VaeConfig.from_pretrained(weights_dir)
    ccl_manager = CCLManager(mesh_device, num_links=2, topology=ttnn.Topology.Ring)
    vae = MiniMaxH3Vae(
        config, task="t2va", mesh_device=mesh_device, ccl_manager=ccl_manager, **DECODE_STAGE_MODES["yuv420"]
    )
    vae.load_state(_decode_stage_state(weights_dir))

    torch.manual_seed(seed)
    warm = torch.randn(1, config.latent_channels, DECODE_STAGE_WARM_FRAMES, 48, 84)
    latents = torch.randn(1, config.latent_channels, latent_frames, 48, 84)

    def decode(upscale_to, output_type, z):
        vae.upscale_to = upscale_to
        started = time.perf_counter()
        video = vae.decode(z, output_type=output_type)
        return video, time.perf_counter() - started

    for upscale_to, output_type in ((None, "yuv420"), (OUT_HW, "yuv420"), (None, "float")):
        decode(upscale_to, output_type, warm)

    plain, plain_s = decode(None, "yuv420", latents)
    test, device_s = decode(OUT_HW, "yuv420", latents)
    pixels, float_s = decode(None, "float", latents)
    started = time.perf_counter()
    host = host_bicubic(pixels.float().clamp(-1.0, 1.0), OUT_HW)
    host_s = time.perf_counter() - started
    reference = host_to_yuv(host, mesh_device)

    test = np.asarray(test)
    assert test.shape == reference.shape, f"{test.shape} vs {reference.shape}"
    score = psnr_uint8(reference, test)
    seams = boundary_report(reference, test, tuple(mesh_device.shape))
    logger.info(
        f"UPSCALE_DECODE frames={test.shape[0]} plain_768p={np.asarray(plain).shape} psnr_yuv={score:.2f} "
        f"psnr_y={psnr_uint8(reference[:, :OUT_HW[0]], test[:, :OUT_HW[0]]):.2f} "
        f"decode_yuv768={plain_s:.3f}s decode_yuv1080_device={device_s:.3f}s "
        f"device_upscale_stage={device_s - plain_s:.3f}s decode_float={float_s:.3f}s host_bicubic={host_s:.3f}s "
        + " ".join(f"{k}={v:.2f}" for k, v in seams.items())
    )
    middle = test.shape[0] // 2
    save_still(test, middle, out_dir / f"decode_seed{seed}_device_f{middle}.png")
    save_still(reference, middle, out_dir / f"decode_seed{seed}_host_f{middle}.png")
    assert score >= 40.0, f"PSNR {score:.2f} dB < 40"
