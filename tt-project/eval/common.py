"""Shared loaders for the FastH3 eval scripts (CPU only, no ttnn)."""

import glob
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np

# Lives outside git so every worktree shares one CPU env; setup_env.sh creates it.
EVAL_VENV = Path("/home/smarton/fasth3/tt-metal/.venv-eval")


def seed_stems(run_dir: Path) -> list[str]:
    """`seedN` stems a baseline run dir holds, in seed order."""
    stems = set()
    for path in run_dir.glob("seed*"):
        match = re.match(r"(seed\d+)", path.name)
        if match:
            stems.add(match.group(1))
    return sorted(stems, key=lambda s: int(s[4:]))


def video_path(run_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.mp4", f"{stem}_silent.mp4"):
        if (run_dir / name).is_file():
            return run_dir / name
    return None


def iter_mp4(path: Path, every: int = 1):
    """Yields (index, HxWx3 uint8 RGB) without holding the clip in memory; a 10 s 1080p clip is ~1.5 GB."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % every == 0:
            yield index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    cap.release()


def mp4_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = dict(
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def read_frames(path: Path, indices: list[int]) -> list[np.ndarray]:
    wanted = set(indices)
    got = {i: f for i, f in iter_mp4(path) if i in wanted}
    return [got[i] for i in indices if i in got]


def load_timings(run_dir: Path, stem: str) -> dict | None:
    path = run_dir / f"{stem}_timings.json"
    return json.loads(path.read_text()) if path.is_file() else None


def expand_videos(inputs: list[str]) -> list[Path]:
    """Accepts mp4 files, run dirs (seedN.mp4 inside) or globs."""
    out = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            for stem in seed_stems(path):
                video = video_path(path, stem)
                if video is not None:
                    out.append(video)
            if not seed_stems(path):
                out += sorted(p for p in path.glob("*.mp4") if not p.name.endswith("_silent.mp4"))
        elif path.is_file():
            out.append(path)
        else:
            out += [Path(p) for p in sorted(glob.glob(item))]
    return out


def ffmpeg() -> str:
    for exe in (os.environ.get("FFMPEG"), "/usr/bin/ffmpeg"):
        if exe and os.path.isfile(exe):
            return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()
