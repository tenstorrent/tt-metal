#!/usr/bin/env python3
"""Synthetic baseline-shaped run dirs for exercising the eval scripts without a device run.

  make_fixture.py OUT [--image IMG] [--seeds 0,1] [--width 1920 --height 1088 --frames 241] [--noise 2.0]

Writes OUT/ref and OUT/cand, each with seedN.mp4, seedN_frames_u8_every16.npy, seedN_latents.pt,
seedN.wav and seedN_timings.json in the layout test_fasth3_baseline_minimax_h3.py produces. The clip is a
slow zoom-pan over IMG; cand adds Gaussian pixel noise of --noise (uint8 sigma) and a matching latent
perturbation, so the expected PSNR is known (~20*log10(255/noise)).
"""

import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ffmpeg  # noqa: E402

DEFAULT_IMAGE = Path(__file__).resolve().parents[2] / "models/sample_data/house_in_field_1080p.jpg"
FPS = 24


def clip(image: np.ndarray, width: int, height: int, frames: int, seed: int):
    src = cv2.resize(image, (int(width * 1.3), int(height * 1.3)), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed)
    dx, dy = rng.uniform(-1, 1, 2)
    for t in range(frames):
        a = t / max(frames - 1, 1)
        zoom = 1.0 + 0.2 * a
        w, h = int(src.shape[1] / 1.3 / zoom * 1.2), int(src.shape[0] / 1.3 / zoom * 1.2)
        cx = src.shape[1] / 2 + dx * a * (src.shape[1] - w) / 2
        cy = src.shape[0] / 2 + dy * a * (src.shape[0] - h) / 2
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        yield cv2.resize(src[y0 : y0 + h, x0 : x0 + w], (width, height), interpolation=cv2.INTER_LINEAR)


def write_run(out: Path, stem: str, frames: np.ndarray, latents: dict, audio: np.ndarray, wall: float, prompt: str):
    out.mkdir(parents=True, exist_ok=True)
    num, height, width, _ = frames.shape
    subprocess.run(
        [
            ffmpeg(),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(out / f"{stem}.mp4"),
        ],
        input=frames.tobytes(),
        check=True,
    )
    np.save(out / f"{stem}_frames_u8_every16.npy", frames[::16])
    torch.save(latents, out / f"{stem}_latents.pt")
    with wave.open(str(out / f"{stem}.wav"), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
    (out / f"{stem}_timings.json").write_text(
        json.dumps(
            dict(seed=int(stem[4:]), call_wall_s=wall, prompt=prompt, num_frames=num, height=height, width=width)
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1088)
    parser.add_argument("--frames", type=int, default=241)
    parser.add_argument("--noise", type=float, default=2.0)
    args = parser.parse_args()

    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    prompt = "A slow cinematic camera pan across a house in a field."
    for seed in (int(s) for s in args.seeds.split(",")):
        rng = np.random.default_rng(1000 + seed)
        ref = np.stack(list(clip(image, args.width, args.height, args.frames, seed)))
        cand = np.clip(ref + rng.normal(0, args.noise, ref.shape), 0, 255).astype(np.uint8)
        lat = {
            "video_rows": torch.randn(4096, 128, generator=torch.Generator().manual_seed(seed)),
            "audio_rows": torch.randn(256, 64, generator=torch.Generator().manual_seed(seed + 1)),
        }
        lat_c = {k: v + 0.05 * torch.randn_like(v) for k, v in lat.items()}
        t = np.arange(16000 * args.frames // FPS) / 16000
        audio = np.stack([0.3 * np.sin(2 * np.pi * 220 * t), 0.3 * np.sin(2 * np.pi * 330 * t)], axis=1)
        audio_c = audio + rng.normal(0, 0.003, audio.shape)
        stem = f"seed{seed}"
        write_run(Path(args.out) / "ref", stem, ref, lat, audio, 30.0, prompt)
        write_run(Path(args.out) / "cand", stem, cand, lat_c, audio_c, 20.0, prompt)
        print(f"wrote {stem} ref/cand {ref.shape}")


if __name__ == "__main__":
    main()
