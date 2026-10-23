# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Find flat mid-grey patches (decoded zeros) in a Wan output video and map them to VAE shards.

    python -I handoff/wan-bin6/find_grey_patches.py video.mp4 [--mesh 4x8] [--save-dir DIR]

A 32x32 pixel tile counts as a patch when its gray-level std is < 2 and its mean is in (100, 200):
an all-zero latent region decodes to about 124. Natural flat regions (dark backdrops, blown-out
light) are excluded by the mean range; an occasional single tile in a smooth background is noise,
so look for runs of 3+ tiles. The BH 4x8 preset splits VAE height across mesh axis 0 and width
across mesh axis 1 (pipeline_wan.py: height=(tp_factor, tp_axis), width=(sp_factor, sp_axis)), so
the shard of pixel (y, x) is mesh coordinate (y // (H / rows), x // (W / cols)).
"""

import argparse
import collections
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--mesh", default="4x8", help="rows x cols of the VAE H x W split")
ap.add_argument("--min-tiles", type=int, default=3, help="ignore frames with fewer flat tiles")
ap.add_argument("--save-dir", help="write the worst frame here as a JPEG")
args = ap.parse_args()
rows, cols = map(int, args.mesh.lower().split("x"))

cap = cv2.VideoCapture(args.video)
per_frame = {}
shards = collections.Counter()
frame_shape = None
for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = frame_shape = gray.shape
    tiles = [
        (y, x)
        for y in range(0, h - 31, 32)
        for x in range(0, w - 31, 32)
        if gray[y : y + 32, x : x + 32].std() < 2 and 100 < gray[y : y + 32, x : x + 32].mean() < 200
    ]
    if len(tiles) >= args.min_tiles:
        per_frame[i] = tiles
        for y, x in tiles:
            shards[((y + 16) * rows // h, (x + 16) * cols // w)] += 1

if not per_frame:
    print(f"{args.video}: no grey patches ({args.min_tiles}+ flat tiles in a frame)")
    raise SystemExit

frames = sorted(per_frame)
ys = [y for t in per_frame.values() for y, _ in t]
xs = [x for t in per_frame.values() for _, x in t]
print(f"{args.video}: grey patches in {len(frames)} frames: {frames[0]}..{frames[-1]} -> {frames}")
print(
    f"  pixel bbox: x {min(xs)}..{max(xs) + 32}, y {min(ys)}..{max(ys) + 32} (frame {frame_shape[1]}x{frame_shape[0]})"
)
print(f"  VAE shard (mesh row, col) -> flat tiles: {dict(shards.most_common())}")
if args.save_dir:
    worst = max(per_frame, key=lambda k: len(per_frame[k]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, worst)
    _, frame = cap.read()
    out = os.path.join(args.save_dir, f"{os.path.splitext(os.path.basename(args.video))[0]}_f{worst}.jpg")
    cv2.imwrite(out, frame)
    print(f"  worst frame {worst} ({len(per_frame[worst])} tiles) -> {out}")
