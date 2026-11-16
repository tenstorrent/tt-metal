# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side I/O for Wan2.2-Animate: the official preprocessing outputs in, an mp4 out. No ttnn."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from PIL import Image


def read_video(path: str, max_frames: int | None = None) -> tuple[list[Image.Image], float]:
    """Frames as RGB PIL images, and the video's fps (OpenCV, already a tt-metal dependency)."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        msg = f"cannot open video {path}"
        raise FileNotFoundError(msg)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        msg = f"no frames decoded from {path}"
        raise ValueError(msg)
    return frames, float(fps)


def write_mp4(frames: np.ndarray, path: str, fps: float) -> str:
    """uint8 [T, H, W, 3] -> H.264 mp4 (imageio-ffmpeg via tt_dit's exporter, else OpenCV mp4v)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    try:
        from models.tt_dit.utils.video import export_to_video

        return export_to_video(frames, path, fps=round(fps), preset="medium")
    except ImportError:
        import cv2

        t, h, w, _ = frames.shape
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return path


@dataclass
class AnimateInputs:
    image: Image.Image
    pose: list[Image.Image]
    face: list[Image.Image]
    background: list[Image.Image] | None
    mask: list[Image.Image] | None
    fps: float

    @property
    def size(self) -> tuple[int, int]:
        return self.pose[0].size


def load_preprocessed(src_root: str, *, mode: str, max_frames: int | None = None) -> AnimateInputs:
    """Load the directory written by the official ``preprocess_data.py``:
    src_ref.png, src_pose.mp4, src_face.mp4 (+ src_bg.mp4, src_mask.mp4 with --replace_flag)."""
    pose, fps = read_video(os.path.join(src_root, "src_pose.mp4"), max_frames)
    face, _ = read_video(os.path.join(src_root, "src_face.mp4"), max_frames)
    image = Image.open(os.path.join(src_root, "src_ref.png")).convert("RGB")
    bg = mask = None
    if mode == "replace":
        bg, _ = read_video(os.path.join(src_root, "src_bg.mp4"), max_frames)
        mask, _ = read_video(os.path.join(src_root, "src_mask.mp4"), max_frames)
    n = min(len(pose), len(face), *(len(x) for x in (bg, mask) if x is not None))
    return AnimateInputs(
        image=image,
        pose=pose[:n],
        face=face[:n],
        background=bg[:n] if bg else None,
        mask=mask[:n] if mask else None,
        fps=fps,
    )


def run_official_preprocess(
    *,
    wan_repo: str,
    ckpt_path: str,
    video: str,
    image: str,
    save_path: str,
    mode: str,
    width: int,
    height: int,
    fps: int = 30,
    retarget: bool = False,
    python: str = sys.executable,
) -> None:
    """Run ``wan/modules/animate/preprocess/preprocess_data.py`` from a Wan2.2 checkout (CPU is fine for
    animate mode; replace mode needs SAM2 - see the runbook for its CPU patch)."""
    script_dir = os.path.join(wan_repo, "wan", "modules", "animate", "preprocess")
    cmd = [
        python,
        "preprocess_data.py",
        "--ckpt_path", ckpt_path,
        "--video_path", os.path.abspath(video),
        "--refer_path", os.path.abspath(image),
        "--save_path", os.path.abspath(save_path),
        "--resolution_area", str(width), str(height),
        "--fps", str(fps),
    ]  # fmt: skip
    if mode == "replace":
        cmd += ["--replace_flag", "--iterations", "3", "--k", "7", "--w_len", "1", "--h_len", "1"]
    elif retarget:
        cmd += ["--retarget_flag"]
    print("+ (cd", script_dir, "&&", shlex.join(cmd), ")", flush=True)
    subprocess.run(cmd, cwd=script_dir, check=True)
