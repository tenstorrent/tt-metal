# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Output-quality checks shared by the TT-DiT video tests: CLIP prompt alignment, reading frames
back from a written clip, and the I2V first-frame check. Each test keeps its own thresholds."""

from __future__ import annotations

import functools
import shutil
import subprocess

import numpy as np
import torch
from loguru import logger
from PIL import Image


@functools.lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    """The ffmpeg used to read test outputs back: the system binary if it is on PATH, else the static
    build bundled with imageio-ffmpeg. `utils.video.export_to_video` encodes with imageio-ffmpeg, so any
    host that wrote a clip can also read it. Neither available is an error, not a skip."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        raise RuntimeError("no ffmpeg: put one on PATH or install imageio-ffmpeg") from None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _decode(path, pix_fmt: str, *filters: str) -> bytes:
    return subprocess.run(
        [
            ffmpeg_exe(),
            "-v",
            "error",
            # Index what the file STORES. Muxed AAC writes an mp4 edit list, and a decoder honouring it
            # drops frames at the edges.
            "-ignore_editlist",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            *filters,
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            pix_fmt,
            "-",
        ],
        check=True,
        capture_output=True,
    ).stdout


def decoded_frames(path, count: int, *, height: int, width: int, luma: bool = True) -> np.ndarray:
    """``count`` frames sampled evenly by index from the clip at ``path`` (fewer if it is shorter), as
    uint8 ``(n, H, W)`` luma or ``(n, H, W, 3)`` RGB.

    ``height``/``width`` are the clip's encoded size; imageio-ffmpeg ships no ffprobe to read it from.
    """
    total = len(_decode(path, "gray", "-vf", "scale=8:8")) // 64
    assert total, f"no frames decoded from {path}"
    indices = np.linspace(0, total - 1, min(count, total), dtype=int)
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    channels = 1 if luma else 3
    raw = _decode(path, "gray" if luma else "rgb24", "-vf", f"select='{select}'")
    frame_size = height * width * channels
    assert len(raw) == len(indices) * frame_size, (
        f"decoded {len(raw)} bytes from {path}, expected {len(indices)} frames of {width}x{height}x{channels}; "
        "wrong height/width?"
    )
    shape = (len(indices), height, width) if luma else (len(indices), height, width, 3)
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape)


def clip_alignment(frames, prompt: str, num_frames: int = 8) -> dict:
    """CLIP similarity (x100) between ``prompt`` and ``num_frames`` evenly spaced RGB frames of
    ``frames`` ((T, H, W, 3), uint8). Returns the per-frame ``scores`` and their mean/min/max."""
    from models.tt_dit.tests.dataset_eval.clip_encoder import CLIPEncoder

    indices = np.linspace(0, len(frames) - 1, min(num_frames, len(frames)), dtype=int)
    encoder = CLIPEncoder()
    scores = []
    for i in indices:
        frame = frames[i]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        scores.append(
            encoder.get_clip_score(prompt, Image.fromarray(np.asarray(frame).astype(np.uint8))).item() * 100.0
        )
    return {"scores": scores, "mean": sum(scores) / len(scores), "min": min(scores), "max": max(scores)}


def assert_clip_alignment(frames, prompt: str, threshold: float, num_frames: int = 8) -> dict:
    """Fails if the mean CLIP alignment of ``frames`` with ``prompt`` is below ``threshold``."""
    alignment = clip_alignment(frames, prompt, num_frames)
    logger.info(f"CLIP scores: min={alignment['min']:.2f}, max={alignment['max']:.2f}, mean={alignment['mean']:.2f}")
    assert alignment["mean"] >= threshold, (
        f"Mean CLIP score {alignment['mean']:.2f} is below threshold {threshold:.2f}. "
        f"Per-frame scores: {[f'{s:.2f}' for s in alignment['scores']]}"
    )
    return alignment


def first_frame_pcc(frame, image: Image.Image, *, luma: bool = False) -> float:
    """Pearson correlation between a decoded first frame ((H, W, 3) RGB or (H, W) luma, uint8) and the
    image it was conditioned on.

    The image is center-cropped to the frame's aspect, as the pipelines do before encoding it (same
    arithmetic as Wan's `center_crop_to_aspect`), then resized to the frame; both are no-ops for an image
    already at the output size.
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    mode = "L" if luma else "RGB"
    frame = Image.fromarray(np.asarray(frame).astype(np.uint8)).convert(mode)
    image = image.convert(mode)

    fw, fh = frame.size
    iw, ih = image.size
    cw, ch = min(iw, round(ih * fw / fh)), min(ih, round(iw * fh / fw))
    left, top = (iw - cw) // 2, (ih - ch) // 2
    image = image.crop((left, top, left + cw, top + ch))
    if image.size != frame.size:
        image = image.resize(frame.size)

    a = np.asarray(frame, dtype=np.float64).ravel()
    b = np.asarray(image, dtype=np.float64).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def assert_first_frame_matches(frame, image: Image.Image, pcc_floor: float, *, luma: bool = False) -> float:
    """Fails if a decoded I2V first frame barely correlates with its conditioning image.

    The pipeline encodes the image into the first latent frame, so a working conditioning path gives a
    strong correlation; the VAE round trip and denoising keep it from being exact.
    """
    pcc = first_frame_pcc(frame, image, luma=luma)
    logger.info(f"I2V frame-0 vs conditioning image correlation (PCC) = {pcc:.4f}")
    assert pcc > pcc_floor, (
        f"Decoded frame 0 barely correlates with the conditioning image (PCC={pcc:.4f} <= {pcc_floor}); "
        "the I2V image-conditioning path is likely broken"
    )
    return pcc
