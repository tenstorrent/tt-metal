#!/usr/bin/env python3
"""
YOLOv8L SAHI-sliced worker subprocess.

Reads a 1280x1280 video, slices each frame into 4x 640x640 tiles (no overlap),
runs parallel inference on 4 Tenstorrent devices via TTYoloBackend, merges
detections, draws results on the 1280x1280 canvas, and writes the annotated
JPEG to a shared frame file.

Launched by `unified_video_demo.py --unified` when the user selects
"Large Model Mode" in the browser UI.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time

import cv2
import numpy as np
import torch

_EMA_ALPHA = 0.15


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv8L 4-device SAHI worker.")
    p.add_argument("--video", required=True, help="Path to 1280x1280 letterboxed video.")
    p.add_argument("--frame-file", required=True, help="Shared JPEG output file.")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--jpeg-quality", type=int, default=75)
    p.add_argument(
        "--device-ids", type=str, default="0,1,2,3", help="Comma-separated TT device IDs (default: 0,1,2,3)."
    )
    return p.parse_args()


def _write_frame(path: str, img: np.ndarray, quality: int = 75):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, path)
    except OSError:
        pass


def _draw_hud(img: np.ndarray, fps: float) -> np.ndarray:
    text = f"YOLOv8L (4-device SAHI)  |  FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def _draw_detections(img: np.ndarray, object_predictions: list) -> np.ndarray:
    for pred in object_predictions:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        score = pred.score.value
        cat = pred.category.name
        label = f"{cat} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    device_ids = [int(x) for x in args.device_ids.split(",")]
    n_devices = len(device_ids)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}", file=sys.stderr, flush=True)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[sahi-worker] Video: {args.video}  fps={fps_src:.1f}  frames={total}", flush=True)

    # Build a fake argparse.Namespace that TTYoloBackend expects
    from types import SimpleNamespace

    backend_args = SimpleNamespace(
        tt_model="yolov8l",
        backend="tt",
        tt_device_id=0,
        tt_l1_small_size=24576,
        tt_trace_region_size=6434816,
        tt_mesh_shape=None,
        tt_force_single_device=False,
        tt_eth_dispatch=False,
        tt_slice_parallel_devices=n_devices,
        tt_slice_parallel_mesh_shape=None,
        confidence_threshold=args.conf,
    )

    # Apply yolov8l device defaults
    from models.demos.yolov8l.common import YOLOV8L_L1_SMALL_SIZE, YOLOV8L_TRACE_REGION_SIZE_E2E

    backend_args.tt_l1_small_size = YOLOV8L_L1_SMALL_SIZE
    backend_args.tt_trace_region_size = YOLOV8L_TRACE_REGION_SIZE_E2E

    import ttnn
    from models.demos.utils.common_demo_utils import postprocess as tt_postprocess
    from models.demos.utils.common_demo_utils import preprocess

    print(f"[sahi-worker] Initializing TTYoloBackend with {n_devices} devices...", flush=True)
    from models.demos.yolo_eval.sahi_ultralytics_eval import (
        TTYoloBackend,
        build_postprocess,
        result_to_object_predictions,
    )

    backend = TTYoloBackend(backend_args)
    print("[sahi-worker] TTYoloBackend ready.", flush=True)

    postprocess_fn = build_postprocess(
        postprocess_type="NMS",
        match_metric="IOU",
        match_threshold=0.5,
        class_agnostic=False,
    )

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("ready.", flush=True)

    ema_fps = 0.0
    first = True
    in_res = backend._TT_INPUT_RES

    try:
        while not stop:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break

            h, w = frame.shape[:2]

            tile_h, tile_w = h // 2, w // 2
            tiles_bgr = [
                frame[0:tile_h, 0:tile_w],
                frame[0:tile_h, tile_w:w],
                frame[tile_h:h, 0:tile_w],
                frame[tile_h:h, tile_w:w],
            ]
            shifts = [
                (0, 0),
                (tile_w, 0),
                (0, tile_h),
                (tile_w, tile_h),
            ]

            batch_bgr = tiles_bgr[:n_devices]
            while len(batch_bgr) < n_devices:
                batch_bgr.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))

            # CPU preprocessing (outside timer)
            parts = [preprocess([im], res=in_res) for im in batch_bgr]
            im_tensor = torch.cat(parts, dim=0)

            # Device inference only (timed) — matches YOLOv8s/v11s measurement
            t_dev = time.perf_counter()
            preds = backend.runner.run(im_tensor)
            ttnn.synchronize_device(backend.device)
            dt_dev = time.perf_counter() - t_dev

            # D2H + postprocess (outside timer)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(
                preds,
                dtype=torch.float32,
                mesh_composer=backend.output_mesh_composer,
            )
            paths = ([f"frame#slot{i}" for i in range(n_devices)],)
            results = tt_postprocess(preds_torch, im_tensor, batch_bgr, paths, backend.names)

            all_preds = []
            full_shape = [h, w]
            for j in range(min(n_devices, len(tiles_bgr))):
                preds_list = result_to_object_predictions(
                    results[j],
                    shift_xy=shifts[j],
                    full_shape=full_shape,
                    confidence_threshold=args.conf,
                )
                all_preds.extend(preds_list)

            merged = [p.get_shifted_object_prediction() for p in all_preds]
            if len(merged) > 1:
                merged = postprocess_fn(merged)

            canvas = frame.copy()
            canvas = _draw_detections(canvas, merged)

            if first:
                ema_fps = 1.0 / max(dt_dev, 1e-9)
                first = False
                print(f"[sahi-worker] First inference: {dt_dev:.4f}s " f"({ema_fps:.0f} device FPS)", flush=True)
            else:
                ema_fps = _EMA_ALPHA * (1.0 / max(dt_dev, 1e-9)) + (1 - _EMA_ALPHA) * ema_fps

            canvas = _draw_hud(canvas, ema_fps)
            _write_frame(args.frame_file, canvas, args.jpeg_quality)

    finally:
        print("[sahi-worker] Shutting down...", flush=True)
        cap.release()
        try:
            backend.close()
        except Exception:
            pass
        print("[sahi-worker] Done.", flush=True)


if __name__ == "__main__":
    main()
