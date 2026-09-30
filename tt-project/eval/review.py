#!/usr/bin/env python3
"""Visual review over seeds: contact-sheet PNG plus a side-by-side grid mp4.

  review.py RUN_DIR [RUN_DIR...] --out OUT_PREFIX [--seeds 0,1,2,3,4] [--cols 6] [--tile-width 320]

The first RUN_DIR is the reference; later dirs are labelled with their per-seed PSNR against it on the
sampled frames. Contact sheet: one row per (seed, run), --cols evenly spaced frames. Grid mp4: runs side
by side, one row per seed, at --video-width per tile. Writes OUT_PREFIX_sheet.png and OUT_PREFIX_grid.mp4.
"""

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ffmpeg, iter_mp4, mp4_info, read_frames, seed_stems, video_path  # noqa: E402

LABEL_H = 28


def label(tile: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((LABEL_H, tile.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, LABEL_H - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def resize(frame: np.ndarray, width: int) -> np.ndarray:
    height = int(round(frame.shape[0] * width / frame.shape[1] / 2)) * 2
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255**2 / mse)


def contact_sheet(runs: list[Path], names: list[str], stems: list[str], cols: int, width: int, out: Path):
    rows = []
    for stem in stems:
        ref_frames = None
        for run, name in zip(runs, names):
            video = video_path(run, stem)
            if video is None:
                continue
            count = mp4_info(video)["frames"]
            indices = [round(i * (count - 1) / max(cols - 1, 1)) for i in range(cols)]
            frames = read_frames(video, indices)
            text = f"{name} {stem}"
            if ref_frames is None:
                ref_frames = frames
            elif len(frames) == len(ref_frames) and frames[0].shape == ref_frames[0].shape:
                text += f"  PSNR {np.mean([psnr(a, b) for a, b in zip(ref_frames, frames)]):.2f} dB vs {names[0]}"
            tiles = [resize(f, width) for f in frames]
            tiles = [label(t, f"f{i}") for t, i in zip(tiles, indices)]
            rows.append(label(np.hstack(tiles), text))
        rows.append(np.full((6, rows[-1].shape[1], 3), 64, np.uint8))
    sheet = np.vstack(rows[:-1])
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def grid_video(runs: list[Path], names: list[str], stems: list[str], width: int, out: Path):
    cells = [[video_path(run, stem) for run in runs] for stem in stems]
    info = mp4_info(next(v for row in cells for v in row if v))
    tile_h = int(round(info["height"] * width / info["width"] / 2)) * 2
    blank = np.zeros((tile_h, width, 3), np.uint8)
    iters = [[iter_mp4(v) if v else None for v in row] for row in cells]
    grid_w = width * len(runs)
    grid_h = (tile_h + LABEL_H) * len(stems)
    proc = subprocess.Popen(
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
            f"{grid_w}x{grid_h}",
            "-r",
            str(info["fps"] or 24),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        stdin=subprocess.PIPE,
    )
    for _ in range(info["frames"]):
        grid_rows = []
        live = False
        for stem, row, row_iters in zip(stems, cells, iters):
            tiles = []
            for name, it in zip(names, row_iters):
                item = next(it, None) if it else None
                frame = item[1] if item else None
                live |= frame is not None
                tiles.append(label(resize(frame, width) if frame is not None else blank, f"{name} {stem}"))
            grid_rows.append(np.hstack(tiles))
        if not live:
            break
        proc.stdin.write(np.vstack(grid_rows).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed writing the grid mp4")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True, help="output prefix")
    parser.add_argument("--names", help="comma list of labels (default = dir names)")
    parser.add_argument("--seeds", help="comma list (default = seeds in the first run)")
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    runs = [Path(r) for r in args.runs]
    names = args.names.split(",") if args.names else [r.name for r in runs]
    stems = [f"seed{s}" for s in args.seeds.split(",")] if args.seeds else seed_stems(runs[0])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    sheet = out.with_name(out.name + "_sheet.png")
    contact_sheet(runs, names, stems, args.cols, args.tile_width, sheet)
    print(f"sheet {sheet} ({time.time() - t0:.1f}s)")
    if not args.no_video:
        t1 = time.time()
        grid = out.with_name(out.name + "_grid.mp4")
        grid_video(runs, names, stems, args.video_width, grid)
        print(f"grid  {grid} ({time.time() - t1:.1f}s)")


if __name__ == "__main__":
    main()
