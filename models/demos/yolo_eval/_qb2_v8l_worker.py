#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""YOLOv8L SAHI worker for the qb2 camera demo: 4 chips, 2x2 tile of 640x640.

The supervisor (`unified_camera_demo.py`) writes letterboxed 1280x1280 JPEG
frames to ``--frame-input-file`` and we poll for new mtimes. Each frame is
sliced into a 2x2 grid of 640x640 tiles, batched across a 2x2 sub-mesh
(4 devices), run through ``YOLOv8lPerformantRunner``, post-processed via the
existing ``_tile_nms`` + ``torch_merge_tiles`` pair, and the merged
detections are written as a single-line JSON to ``--dets-out-file``
(consumed by the supervisor and forwarded to the browser via the WebRTC
DataChannel).

Why a separate worker process: the runner constructor opens TT devices and
captures a trace; once captured, the mesh is held for the lifetime of the
process. Mode switches in the supervisor kill this process and respawn the
side-by-side workers; tt-metal needs the kill so device locks release.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import cv2
import numpy as np
import torch

# Cross-tile merge + seam-adjacent merge + overlap grid builder, all from the
# production 640 SAHI script. The overlap grid is the architectural fix for
# duplicate boxes on tile seams: when adjacent tiles share an overlap band,
# an object straddling the seam is FULLY visible in both tiles, so cross-tile
# WBF gets a high IoU/IoS match and dedup actually works.
from models.demos.yolo_eval.yolov8l_sahi_640_pipelined import (
    _cross_tile_merge,
    _merge_seam_adjacent,
    build_overlap_grid,
)
from models.demos.yolo_eval.yolov8l_sahi_pipelined import _tile_nms, slice_and_preprocess

# Letterbox to 1216 (not 1280) so that with 640×640 tiles, build_overlap_grid
# shifts the last row/col inward to start=576, giving a 64-px overlap zone
# on both axes. Same 4-chip cost; sustained dedup quality.
_LETTERBOX_RES = 1216
_TILE = 640
# Seams are computed from the grid below at startup, so they always reflect
# the actual tile geometry (no manual coupling to constants).
_SEAMS_X: tuple = ()
_SEAMS_Y: tuple = ()
# bfloat8 sigmoid noise floor — class logits hover near zero at 640×640 on
# the YOLOv8L mesh path, producing sigmoid ≈ 0.50. Conf below this lets the
# floor noise through; conf at or above it correctly suppresses the noise
# AND admits real-signal class scores. Mirrors _CONF_FLOOR_640 in the
# production script.
_V8L_CONF_FLOOR = 0.50


def _read_frame_file(path: str, last_mtime: float, timeout_s: float = 600.0) -> tuple[np.ndarray, float] | None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            mt = os.stat(path).st_mtime
        except (FileNotFoundError, OSError):
            time.sleep(0.01)
            continue
        if mt == last_mtime:
            time.sleep(0.005)
            continue
        for _ in range(3):
            img = cv2.imread(path)
            if img is not None and img.size > 0:
                return img, mt
            time.sleep(0.005)
        last_mtime = mt
    return None


def _write_dets(path: str, payload: dict) -> None:
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(json.dumps(payload, separators=(",", ":")))
        os.replace(tmp, path)
    except OSError:
        pass


def parse_args() -> argparse.Namespace:
    # Defaults mirror the supervisor demo's camera transport (demo_supervisor.py):
    # conf=0.7, iou=0.45, merge=wbf+ios@0.6, seam-merge with tol=100.
    p = argparse.ArgumentParser(description="YOLOv8L SAHI worker — 4 chips × 640×640.")
    p.add_argument("--frame-input-file", required=True)
    p.add_argument("--dets-out-file", required=True)
    p.add_argument("--conf", type=float, default=0.7)
    p.add_argument("--iou", type=float, default=0.45, help="per-tile NMS IoU")
    p.add_argument("--merge-mode", choices=["nms", "greedy-nmm", "nmm", "wbf"], default="wbf")
    p.add_argument("--merge-match", choices=["iou", "ios"], default="ios")
    p.add_argument(
        "--merge-threshold", type=float, default=0.5, help="cross-tile WBF threshold (lower → more aggressive merge)"
    )
    p.add_argument(
        "--class-agnostic", action="store_true", help="merge across classes (off by default — camera-mode default)"
    )
    p.add_argument(
        "--seam-merge", action="store_true", help="extra pass merging same-class boxes split across a tile seam"
    )
    p.add_argument("--seam-tol", type=int, default=160, help="pixel tolerance for seam-adjacent matching")
    p.add_argument(
        "--seam-perp-overlap-frac",
        type=float,
        default=0.05,
        help="required perpendicular-axis overlap fraction for seam-adjacent merge "
        "(lower → more aggressive merge across the seam)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.conf < _V8L_CONF_FLOOR:
        print(
            f"[qb2-v8l] conf {args.conf:.2f} below floor {_V8L_CONF_FLOOR:.2f} — raising "
            f"(bfloat8 sigmoid noise would otherwise dominate)",
            flush=True,
        )
        args.conf = _V8L_CONF_FLOOR

    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers
    from models.demos.yolo_eval.yolov8l_native_vs_sahi_demo import _coco_names_dict, _load_coco_names
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    TAG = "[qb2-v8l]"

    # 1216x1216 letterboxed → 2x2 grid with 64px overlap on each axis → 4 chips.
    # build_overlap_grid shifts the last row/col inward when it would extend
    # past the frame: x_starts=[0,576], y_starts=[0,576] for 1216 / 640.
    grid = build_overlap_grid(_LETTERBOX_RES, _LETTERBOX_RES, _TILE, _TILE, add_whole_frame=False)
    n_tiles = grid.n_tiles
    if n_tiles != 4:
        print(f"{TAG} unexpected n_tiles={n_tiles} (need 4 for 2x2)", file=sys.stderr, flush=True)
        return 2

    # Derive seams from the grid (same idiom as the production supervisor
    # demo). For our case: col_starts={0,576}, col_ends={640,1216}; the
    # interior seam pair is (640, 576) on each axis.
    _col_starts = sorted({ts.col_start for ts in grid.tiles})
    _col_ends = sorted({ts.col_start + ts.src_w for ts in grid.tiles})
    _row_starts = sorted({ts.row_start for ts in grid.tiles})
    _row_ends = sorted({ts.row_start + ts.src_h for ts in grid.tiles})
    seams_x = tuple(
        zip(
            [e for e in _col_ends if e < _LETTERBOX_RES],
            [s for s in _col_starts if s > 0],
        )
    )
    seams_y = tuple(
        zip(
            [e for e in _row_ends if e < _LETTERBOX_RES],
            [s for s in _row_starts if s > 0],
        )
    )
    print(
        f"{TAG} OverlapGrid {_LETTERBOX_RES}x{_LETTERBOX_RES} "
        f"-> {grid.n_cols}x{grid.n_rows} = {n_tiles} tiles of {_TILE}x{_TILE}; "
        f"seams_x={seams_x} seams_y={seams_y}",
        flush=True,
    )
    for i, ts in enumerate(grid.tiles):
        print(
            f"{TAG}   tile {i}: start=({ts.col_start},{ts.row_start}) " f"src={ts.src_w}x{ts.src_h}",
            flush=True,
        )

    # --- Open 2x2 sub-mesh ---
    mesh_shape = ttnn.MeshShape(2, 2)
    print(f"{TAG} Opening 2x2 mesh (4 devices)...", flush=True)
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        l1_small_size=yolov8l_l1_small_size_for_res(_TILE, _TILE),
        trace_region_size=35_000_000,
        num_command_queues=2,
    )
    mesh_device.enable_program_cache()

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    print(f"{TAG} Building YOLOv8lPerformantRunner ({_TILE}x{_TILE}, batch={n_tiles})...", flush=True)
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=n_tiles,
        inp_h=_TILE,
        inp_w=_TILE,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    # Reverse map cat_name -> id, since torch_merge_tiles only gives us cat_name back.
    name_to_id = {v: k for k, v in names_dict.items()}
    print(f"{TAG} ready.", flush=True)

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    last_mtime = 0.0
    frame_id = 0
    ema_fps = 0.0
    EMA_ALPHA = 0.15
    first = True

    try:
        while not stop:
            got = _read_frame_file(args.frame_input_file, last_mtime)
            if got is None:
                print(f"{TAG} frame source idle — exiting.", flush=True)
                break
            bgr, last_mtime = got

            # The supervisor letterboxes to exactly 1280x1280; a guard helps on
            # the (rare) first frame where the file is the wrong size.
            h, w = bgr.shape[:2]
            if (h, w) != (_LETTERBOX_RES, _LETTERBOX_RES):
                bgr = cv2.resize(bgr, (_LETTERBOX_RES, _LETTERBOX_RES), interpolation=cv2.INTER_LINEAR)

            t0 = time.perf_counter()
            tensor, shifts, _timings = slice_and_preprocess(bgr, grid, n_tiles)
            preds = runner.run(torch_input_tensor=tensor)
            ttnn.synchronize_device(mesh_device)
            dt = time.perf_counter() - t0

            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(preds, dtype=torch.float32, mesh_composer=output_mesh_composer)

            tile_results = _tile_nms(preds_torch, args.conf, args.iou)

            # --- Concat all tiles' boxes into full-image (1280×1280) coords ---
            all_b, all_s, all_c = [], [], []
            for j, r in enumerate(tile_results):
                b = r["boxes"]["xyxy"]
                if b.numel() == 0:
                    continue
                sx, sy = shifts[j]
                shifted = b.clone()
                shifted[:, 0] += sx
                shifted[:, 2] += sx
                shifted[:, 1] += sy
                shifted[:, 3] += sy
                all_b.append(shifted)
                all_s.append(r["boxes"]["conf"])
                all_c.append(r["boxes"]["cls"])

            if all_b:
                boxes = torch.cat(all_b, dim=0)
                scores = torch.cat(all_s, dim=0)
                cls_ids = torch.cat(all_c, dim=0).int()

                # --- Cross-tile merge (WBF + IoS by default) ---
                boxes, scores, cls_ids = _cross_tile_merge(
                    mode=args.merge_mode,
                    boxes=boxes,
                    scores=scores,
                    cls_ids=cls_ids,
                    threshold=args.merge_threshold,
                    match_metric=args.merge_match,
                    class_agnostic=args.class_agnostic,
                    max_det=300,
                )

                # --- Seam-adjacent merge (catches same-class boxes split
                # across the x=640 / y=640 tile boundaries that IoS can't
                # see because they barely overlap). ---
                if args.seam_merge and boxes.shape[0] > 0:
                    boxes, scores, cls_ids = _merge_seam_adjacent(
                        boxes,
                        scores,
                        cls_ids,
                        seams_x=seams_x,
                        seams_y=seams_y,
                        tol=args.seam_tol,
                        perp_overlap_frac=args.seam_perp_overlap_frac,
                    )
            else:
                boxes = torch.empty((0, 4))
                scores = torch.empty((0,))
                cls_ids = torch.empty((0,), dtype=torch.int32)

            dets = []
            for i in range(boxes.shape[0]):
                if scores[i] < args.conf:
                    continue
                x1, y1, x2, y2 = boxes[i].tolist()
                dets.append([x1, y1, x2, y2, float(scores[i]), int(cls_ids[i])])

            if first:
                ema_fps = 1.0 / max(dt, 1e-9)
                first = False
                print(f"{TAG} First inference: {dt:.4f}s ({ema_fps:.0f} FPS)", flush=True)
            else:
                ema_fps = EMA_ALPHA * (1.0 / max(dt, 1e-9)) + (1 - EMA_ALPHA) * ema_fps

            _write_dets(
                args.dets_out_file,
                {
                    "t": time.time(),
                    "frame_id": frame_id,
                    "model": "YOLOv8L-SAHI(4ch)",
                    "input_res": _LETTERBOX_RES,
                    "fps": round(ema_fps, 1),
                    "dets": dets,
                },
            )
            frame_id += 1
    finally:
        print(f"{TAG} shutting down...", flush=True)
        try:
            runner.release()
        except Exception:
            pass
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception:
            pass
        print(f"{TAG} done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
