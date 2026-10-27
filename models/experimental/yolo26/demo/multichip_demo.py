# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLO26 multi-stream people counting on all chips of a QB2 (data parallel, one mesh).

Same pipeline as multistream_demo.py (run_pipeline), on a 1 x --chips mesh: the traced model runs --batch-size images
per chip, so one mesh pass carries batch_size x chips images. Each chip serves its own --streams-per-chip streams
(default 20, i.e. 80 streams on 4 chips); the same clips are reused on every chip, each chip starting at a different
frame. Only the device setup and the mosaic layout differ from the 1-chip demo.

    python models/experimental/yolo26/demo/multichip_demo.py --videos <dir> --record out.mp4
"""

import argparse

import cv2
import numpy as np
import ttnn

from models.experimental.yolo26.common import YOLO26_L1_SMALL_SIZE
from models.experimental.yolo26.demo.multistream_demo import add_common_args, load_clips, run_pipeline, stream_table

PANEL_W = 420
TILE_COLS = 10  # 2 rows of 10 tiles per chip


class Display:
    """80 small tiles (one band of 2 rows per chip) and a side panel of per-chip camera counts."""

    def __init__(self, args, num_streams, chips):
        self.n, self.chips, self.args = num_streams, chips, args
        self.tile_w, self.tile_h = map(int, args.tile.split("x"))
        self.per_chip = num_streams // chips
        clips = load_clips(args.videos, self.per_chip, args.max_frames, rgba=False)
        self.table = stream_table(clips, num_streams, chips)
        src_h, src_w = clips[0]["frames"][0].shape[:2]
        self.sx, self.sy = self.tile_w / src_w, self.tile_h / src_h
        # Tiles are drawn from frames resized once at load (tile size, ~2 GB for the CAVIAR set)
        self.small = [
            [cv2.resize(f, (self.tile_w, self.tile_h), interpolation=cv2.INTER_AREA) for f in c["frames"]]
            for c in clips
        ]
        rows = (num_streams + TILE_COLS - 1) // TILE_COLS
        self.canvas = np.zeros((rows * self.tile_h, TILE_COLS * self.tile_w + PANEL_W, 3), np.uint8)

    def tile(self, s, frame_idx, people_src, fps):
        r, c = divmod(s, TILE_COLS)
        view = self.canvas[r * self.tile_h : (r + 1) * self.tile_h, c * self.tile_w : (c + 1) * self.tile_w]
        view[:] = self.small[self.table[s][0]][frame_idx]
        sx, sy = self.sx, self.sy
        for x1, y1, x2, y2, _ in people_src:
            cv2.rectangle(view, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (60, 220, 60), 1)
        cv2.rectangle(view, (0, 0), (self.tile_w, 15), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(view, f"CAM {s + 1:02d}: {len(people_src)}", (4, 11), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        text = f"{fps:4.1f} FPS"
        (w, _), _ = cv2.getTextSize(text, font, 0.38, 1)
        cv2.putText(view, text, (self.tile_w - w - 4, 11), font, 0.38, (90, 230, 90), 1, cv2.LINE_AA)
        cv2.rectangle(view, (0, 0), (self.tile_w - 1, self.tile_h - 1), (40, 40, 40), 1)

    def panel(self, stats, counts):
        x0 = self.canvas.shape[1] - PANEL_W
        p = self.canvas[:, x0:]
        p[:] = (24, 24, 24)
        font = cv2.FONT_HERSHEY_SIMPLEX
        put = lambda text, x, y, scale=0.55, color=(230, 230, 230), thick=1: cv2.putText(
            p, text, (x, y), font, scale, color, thick, cv2.LINE_AA
        )
        put(f"YOLO26{self.args.variant[-1]} people counting", 16, 32, 0.7, (255, 255, 255), 2)
        put(f"{self.chips} Tenstorrent Blackhole chips, batch {self.args.batch_size} each", 16, 58)
        put(f"{self.n} streams x {stats['stream_fps']:.1f} FPS", 16, 96, 0.75, (90, 230, 90), 2)
        put(f"= {stats['total_fps']:.0f} images/s", 16, 124)
        put(f"latency {stats['latency_ms']:.0f} ms (median)   display {stats['display_fps']:.0f} FPS", 16, 148)
        put(f"People now: {sum(counts)}", 16, 186, 0.75, (255, 255, 255), 2)

        top, cols = 204, 5
        block_h = (p.shape[0] - top - 6) // self.chips
        cell_w = (PANEL_W - 32) // cols
        rows = (self.per_chip + cols - 1) // cols
        cell_h = (block_h - 30) // rows
        scale = max(4, max(counts))
        for g in range(self.chips):
            y0 = top + g * block_h
            first, last = g * self.per_chip, (g + 1) * self.per_chip
            chip_counts = counts[first:last]
            put(f"Chip {g + 1}   CAM {first + 1:02d}-{last:02d}", 16, y0 + 18, 0.5, (210, 210, 210))
            text = f"people {sum(chip_counts)}"
            (w, _), _ = cv2.getTextSize(text, font, 0.5, 1)
            put(text, PANEL_W - 16 - w, y0 + 18, 0.5, (255, 255, 255))
            for k, n in enumerate(chip_counts):
                r, c = divmod(k, cols)
                x, y = 16 + c * cell_w, y0 + 26 + r * cell_h
                cv2.rectangle(p, (x, y), (x + cell_w - 4, y + cell_h - 4), (45, 45, 45), -1)
                bar = int((cell_h - 4) * n / scale)
                cv2.rectangle(p, (x, y + cell_h - 4 - bar), (x + cell_w - 4, y + cell_h - 4), (60, 140, 60), -1)
                put(f"{first + k + 1:02d}", x + 4, y + 13, 0.36, (170, 170, 170))
                put(f"{n}", x + cell_w - 22, y + cell_h - 12, 0.6, (255, 255, 255), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--chips", type=int, default=4)
    ap.add_argument("--streams-per-chip", type=int, default=20)
    ap.add_argument("--tile", default="192x144")
    args = ap.parse_args()
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, args.chips),
        l1_small_size=YOLO26_L1_SMALL_SIZE,
        trace_region_size=8000000,
        num_command_queues=2,
    )
    try:
        run_pipeline(
            device,
            args,
            args.chips,
            args.streams_per_chip,
            Display,
            ttnn.ShardTensorToMesh(device, dim=0),
            ttnn.ConcatMeshToTensor(device, dim=0),
        )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
