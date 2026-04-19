#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8L comparison demo: Native 1280×1280 vs. 4K SAHI slicing.

Runs two YOLOv8L inference modes side-by-side from a single 4K video:
  Left  – Native 1280: letterbox 4K → 1280×1280, run on 1 device.
  Right – SAHI 4K: slice into tiles using sahi.slicing.slice_image(),
          run on a multi-device mesh, merge detections via SAHI.

The SAHI worker mirrors the proven pipeline from sahi_ultralytics_eval.py:
  1. sahi.slicing.slice_image()       – tile the frame
  2. common_demo_utils.preprocess()   – letterbox + normalise each tile
  3. YOLOv8lPerformantRunner.run()    – device inference (batch across mesh)
  4. common_demo_utils.postprocess()  – NMS + scale_boxes (de-letterbox)
  5. result_to_object_predictions()   – create SAHI ObjectPrediction with shift
  6. SAHI build_postprocess()         – merge cross-tile detections

Usage:
    python models/demos/yolo_eval/yolov8l_native_vs_sahi_demo.py \\
        --input path/to/4k_video.mp4 \\
        --serve --port 9090

    python models/demos/yolo_eval/yolov8l_native_vs_sahi_demo.py \\
        --input path/to/image.jpg --sahi-only \\
        --serve --port 9090

    SSH tunnel:  ssh -L 9090:localhost:9090 user@server
    Then open:   http://localhost:9090/

Requirements:
    • 6+ Tenstorrent devices (e.g. Galaxy with 8×4 mesh)
    • 4K video or image (3840×2160 recommended)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_TILE_SIZE = 1280
_EMA_ALPHA = 0.15
_CONF_FLOOR_1280 = 0.50
_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "_mjpeg_server.py")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in _IMAGE_EXTS


class FrameSource:
    """Unified frame source for video files and single images."""

    def __init__(self, path: str):
        self.path = path
        self._is_image = _is_image_path(path)
        if self._is_image:
            self._frame = cv2.imread(path)
            if self._frame is None:
                raise RuntimeError(f"Cannot read image: {path}")
            self._served = False
        else:
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video: {path}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._is_image:
            if not self._served:
                self._served = True
                return True, self._frame.copy()
            return False, None
        return self._cap.read()

    def reset(self):
        if self._is_image:
            self._served = False
        else:
            # Re-open instead of seek — cv2 seek can leave the decoder in a
            # degraded state that holds the GIL longer during subsequent reads.
            self._cap.release()
            self._cap = cv2.VideoCapture(self.path)

    def peek(self) -> np.ndarray:
        """Read one frame without advancing (for setup / size detection)."""
        if self._is_image:
            return self._frame.copy()
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read first frame from: {self.path}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return frame

    @property
    def is_image(self) -> bool:
        return self._is_image

    def release(self):
        if not self._is_image:
            self._cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8L comparison: native 1280 vs. 4K SAHI (multi-chip).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", help="Path to a 4K video file or a single image (jpg/png/bmp/tiff).")
    p.add_argument("--video", dest="input", help="Alias for --input (backwards compatibility).")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save the annotated result to this path (image mode only). "
        "When set with --serve, the image is also saved before streaming.",
    )
    p.add_argument(
        "--conf", type=float, default=0.25, help="NMS confidence threshold (auto-raised to 0.50 for bfloat8)."
    )
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--serve", action="store_true", help="Stream MJPEG over HTTP.")
    p.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address.")
    p.add_argument("--port", type=int, default=9090, help="HTTP port.")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality (1-100).")
    p.add_argument(
        "--native-device-id",
        type=int,
        default=0,
        help="TT device id for native 1280 worker (default: 0).",
    )
    p.add_argument(
        "--sahi-device-ids",
        type=str,
        default="0,1,2,3,4,5",
        help="(Legacy, ignored) Device IDs were used with TT_VISIBLE_DEVICES; "
        "now the worker opens a sub-mesh automatically.",
    )
    p.add_argument(
        "--display-width",
        type=int,
        default=0,
        help="Width to scale each output panel. 0 = no scaling (native resolution). "
        "Example: 960 for dual-panel 1920 total, 1920 for full-HD single panel.",
    )
    p.add_argument(
        "--sahi-only",
        action="store_true",
        help="Run SAHI 4K slicing only (full screen, no native comparison).",
    )
    p.add_argument(
        "--native-only",
        action="store_true",
        help="Run native 1280×1280 only (full screen, no SAHI slicing).",
    )
    p.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Input is already 1280×1280 letterboxed BGR. Skip resize/pad. "
        "Use with a pre-processed video created by ffmpeg.",
    )
    p.add_argument(
        "--pipeline",
        action="store_true",
        help="Use 3-stage threaded pipeline (preprocess | device | postprocess) "
        "to overlap host and device work. Improves throughput for video.",
    )
    p.add_argument(
        "--sahi-merge-type",
        type=str,
        default="GREEDYNMM",
        choices=["NMS", "NMM", "GREEDYNMM", "LSNMS"],
        help="SAHI postprocess merge strategy (default: GREEDYNMM).",
    )
    p.add_argument(
        "--sahi-merge-metric",
        type=str,
        default="IOS",
        choices=["IOU", "IOS"],
        help="SAHI merge match metric (default: IOS).",
    )
    p.add_argument(
        "--sahi-merge-threshold",
        type=float,
        default=0.4,
        help="SAHI merge match threshold (default: 0.4).",
    )
    p.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.0,
        help="SAHI vertical overlap ratio between slices (default: 0.0).",
    )
    p.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.0,
        help="SAHI horizontal overlap ratio between slices (default: 0.0).",
    )
    p.add_argument(
        "--sahi-class-agnostic",
        action="store_true",
        help="Merge boxes across classes during SAHI postprocess. "
        "Useful when the same object gets different class labels from different tiles.",
    )
    p.add_argument(
        "--perform-standard-pred",
        action="store_true",
        help="Also run a full-image prediction (letterboxed to 1280×1280) and merge with "
        "sliced detections. Catches large objects spanning multiple tiles.",
    )
    p.add_argument(
        "--mesh-shape",
        type=str,
        default="auto",
        help="Mesh shape for SAHI sub-mesh: 'auto' matches tile count (e.g. 2x3 for 4K), "
        "or explicit like '3x2'. Total devices must equal total tiles.",
    )
    # Internal: worker mode flags
    p.add_argument("--_native-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_sahi-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_frame-file", type=str, default=None, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.input is None:
        p.error("one of --input or --video is required")
    return args


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_coco_names() -> list[str]:
    names_file = Path(__file__).resolve().parents[2] / "demos" / "utils" / "coco.names"
    if names_file.exists():
        return [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
    return [str(i) for i in range(80)]


def _coco_names_dict(names_list: list[str]) -> dict[int, str]:
    return {i: n for i, n in enumerate(names_list)}


def draw_hud(img: np.ndarray, title: str, fps: float, color=(0, 255, 0)) -> np.ndarray:
    text = f"{title}  |  FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return img


def write_frame(path: str, img: np.ndarray, quality: int = 75):
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


def scale_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    if target_w <= 0:
        return img
    h, w = img.shape[:2]
    if w == target_w:
        return img
    target_h = int(round(h * target_w / w))
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _draw_sahi_detections(img: np.ndarray, preds: list) -> np.ndarray:
    """Draw SAHI ObjectPrediction list onto img (BGR)."""
    for pred in preds:
        bbox = pred.bbox
        x1, y1, x2, y2 = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        score = pred.score.value
        cat = pred.category.name
        label = f"{cat} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# Native 1280 worker (kept as-is; uses custom letterbox + NMS for single frame)
# ---------------------------------------------------------------------------


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    xy, wh = x[..., :2], x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def _nms_single(
    preds: torch.Tensor,
    conf_thres: float,
    iou_thres: float,
) -> torch.Tensor:
    """Run NMS on [1, 84, N] predictions. Returns [M, 6]: (x1,y1,x2,y2,conf,cls)."""
    import torchvision

    nc = preds.shape[1] - 4
    xc = preds[:, 4 : 4 + nc].amax(1) > conf_thres
    preds = preds.transpose(-1, -2)
    preds[..., :4] = _xywh2xyxy(preds[..., :4])

    results = []
    for xi in range(preds.shape[0]):
        x = preds[xi][xc[xi]]
        if x.shape[0] == 0:
            continue
        box, cls_scores = x[:, :4], x[:, 4:]
        conf, j = cls_scores.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if x.shape[0] == 0:
            continue
        c = x[:, 5:6] * 7680
        i = torchvision.ops.nms(x[:, :4] + c, x[:, 4], iou_thres)
        results.append(x[i[:300]])

    if results:
        return torch.cat(results, 0)
    return torch.zeros((0, 6))


def _letterbox(img: np.ndarray, target: tuple[int, int] = (1280, 1280)) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(target[0] / h, target[1] / w)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    dw = (target[1] - new_w) / 2
    dh = (target[0] - new_h) / 2
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _bgr_to_tensor(bgr: np.ndarray) -> torch.Tensor:
    """BGR HWC uint8 → [1,3,H,W] float32 NCHW tensor normalised to [0,1]."""
    rgb = bgr[:, :, ::-1].transpose(2, 0, 1)
    return torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0


def _draw_detections(
    img: np.ndarray,
    dets: torch.Tensor,
    class_names: list[str],
    scale_from: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw [M,6] detections on ``img``, optionally un-letterboxing from ``scale_from``."""
    if dets.shape[0] == 0:
        return img

    oh, ow = img.shape[:2]
    if scale_from is not None:
        mh, mw = scale_from
        gain = min(mh / oh, mw / ow)
        pad_x = round((mw - ow * gain) / 2 - 0.1)
        pad_y = round((mh - oh * gain) / 2 - 0.1)
        dets = dets.clone()
        dets[:, [0, 2]] -= pad_x
        dets[:, [1, 3]] -= pad_y
        dets[:, :4] /= gain

    dets[:, 0].clamp_(0, ow)
    dets[:, 1].clamp_(0, oh)
    dets[:, 2].clamp_(0, ow)
    dets[:, 3].clamp_(0, oh)

    for det in dets:
        x1, y1, x2, y2 = map(int, det[:4])
        score = float(det[4])
        cls_id = int(det[5])
        label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id} {score:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img


def _run_native_worker(args):
    """Single-device YOLOv8L at 1280×1280: letterbox 4K → 1280, infer, draw on 4K."""
    import ttnn
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    skip_pp = args.skip_preprocess

    print("[native] Opening device 0...", flush=True)
    device = ttnn.CreateDevice(
        0,
        l1_small_size=yolov8l_l1_small_size_for_res(1280, 1280),
        trace_region_size=35_000_000,
        num_command_queues=2,
    )
    device.enable_program_cache()

    print("[native] Building YOLOv8lPerformantRunner (1280×1280)...", flush=True)
    runner = YOLOv8lPerformantRunner(device, device_batch_size=1, inp_h=1280, inp_w=1280)
    print("[native] ready.", flush=True)

    coco_names = _load_coco_names()
    conf = max(args.conf, _CONF_FLOOR_1280)

    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"[native] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    if skip_pp:
        print("[native] --skip-preprocess: input assumed 1280×1280 BGR, skipping letterbox", flush=True)

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ema_fps = 0.0
    first = True
    frame_count = 0
    t_read_sum = t_letterbox_sum = t_tensor_sum = 0.0
    t_device_sum = t_d2h_sum = t_nms_sum = 0.0
    t_draw_sum = t_encode_sum = t_total_sum = 0.0
    t_host_prep_sum = t_h2d_trace_sum = t_sync_sum = 0.0
    LOG_INTERVAL = 30

    print("[native] Running...", flush=True)

    try:
        while not stop:
            t_frame_start = time.perf_counter()

            t0 = time.perf_counter()
            ok, frame = src.read()
            if not ok:
                if src.is_image:
                    break
                src.reset()
                ok, frame = src.read()
                if not ok:
                    break
            t_read = time.perf_counter() - t0

            t0 = time.perf_counter()
            if skip_pp:
                lb = frame
            else:
                lb = _letterbox(frame, (1280, 1280))
            t_letterbox = time.perf_counter() - t0

            t0 = time.perf_counter()
            inp = _bgr_to_tensor(lb)
            t_tensor = time.perf_counter() - t0

            t0 = time.perf_counter()
            preds = runner.run(torch_input_tensor=inp)
            t_pre_sync = time.perf_counter() - t0
            t0_sync = time.perf_counter()
            ttnn.synchronize_device(device)
            t_sync = time.perf_counter() - t0_sync
            t_device = t_pre_sync + t_sync
            t_host_prep = runner.last_timing["host_prep_ms"]
            t_h2d_trace = runner.last_timing["h2d_and_trace_ms"]

            t0 = time.perf_counter()
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(preds, dtype=torch.float32)
            t_d2h = time.perf_counter() - t0

            t0 = time.perf_counter()
            dets = _nms_single(preds_torch, conf, args.iou)
            t_nms = time.perf_counter() - t0

            t0 = time.perf_counter()
            if skip_pp:
                canvas = frame.copy()
                canvas = _draw_detections(canvas, dets, coco_names, scale_from=None)
            else:
                canvas = frame.copy()
                canvas = _draw_detections(canvas, dets, coco_names, scale_from=(1280, 1280))
            t_draw = time.perf_counter() - t0

            dt_total = time.perf_counter() - t_frame_start

            if first:
                ema_fps = 1.0 / max(dt_total, 1e-9)
                first = False
                print(
                    f"[native] First frame: {dt_total*1000:.1f}ms "
                    f"(read={t_read*1000:.1f} letterbox={t_letterbox*1000:.1f} "
                    f"tensor={t_tensor*1000:.1f} device={t_device*1000:.1f}"
                    f"[host_prep={t_host_prep:.1f} h2d+trace={t_h2d_trace:.1f} sync={t_sync*1000:.1f}] "
                    f"d2h={t_d2h*1000:.1f} nms={t_nms*1000:.1f} draw={t_draw*1000:.1f})",
                    flush=True,
                )
            else:
                ema_fps = _EMA_ALPHA * (1.0 / max(dt_total, 1e-9)) + (1 - _EMA_ALPHA) * ema_fps

            t0 = time.perf_counter()
            canvas = draw_hud(canvas, "YOLOv8L Native 1280", ema_fps)
            canvas = scale_to_width(canvas, args.display_width)
            write_frame(args._frame_file, canvas, args.jpeg_quality)
            t_encode = time.perf_counter() - t0

            frame_count += 1
            t_read_sum += t_read
            t_letterbox_sum += t_letterbox
            t_tensor_sum += t_tensor
            t_device_sum += t_device
            t_host_prep_sum += t_host_prep
            t_h2d_trace_sum += t_h2d_trace
            t_sync_sum += t_sync
            t_d2h_sum += t_d2h
            t_nms_sum += t_nms
            t_draw_sum += t_draw
            t_encode_sum += t_encode
            t_total_sum += dt_total + t_encode

            if frame_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                print(
                    f"[native] Avg over last {n} frames: "
                    f"total={t_total_sum/n*1000:.1f}ms "
                    f"({1.0/(t_total_sum/n):.1f} FPS)  |  "
                    f"read={t_read_sum/n*1000:.1f}  "
                    f"letterbox={t_letterbox_sum/n*1000:.1f}  "
                    f"tensor={t_tensor_sum/n*1000:.1f}  "
                    f"device={t_device_sum/n*1000:.1f}"
                    f"[host_prep={t_host_prep_sum/n:.1f} "
                    f"h2d+trace={t_h2d_trace_sum/n:.1f} "
                    f"sync={t_sync_sum/n*1000:.1f}]  "
                    f"d2h={t_d2h_sum/n*1000:.1f}  "
                    f"nms={t_nms_sum/n*1000:.1f}  "
                    f"draw={t_draw_sum/n*1000:.1f}  "
                    f"hud+encode+write={t_encode_sum/n*1000:.1f}  (ms)",
                    flush=True,
                )
                t_read_sum = t_letterbox_sum = t_tensor_sum = 0.0
                t_device_sum = t_d2h_sum = t_nms_sum = 0.0
                t_draw_sum = t_encode_sum = t_total_sum = 0.0
                t_host_prep_sum = t_h2d_trace_sum = t_sync_sum = 0.0

            if src.is_image:
                if args.output:
                    cv2.imwrite(args.output, canvas)
                    print(f"[native] Saved annotated image to {args.output}", flush=True)
                print("[native] Image inference complete. Waiting for shutdown...", flush=True)
                while not stop:
                    time.sleep(1)
                break
    finally:
        print("[native] Shutting down...", flush=True)
        src.release()
        try:
            runner.release()
        except Exception:
            pass
        try:
            ttnn.synchronize_device(device)
        except Exception:
            pass
        try:
            ttnn.CloseDevice(device)
        except Exception:
            pass
        print("[native] Done.", flush=True)


# ---------------------------------------------------------------------------
# Native 1280 worker — 2-stage pipeline (main=preprocess+device, bg=postprocess)
# ---------------------------------------------------------------------------


def _run_native_worker_pipelined(args):
    """2-stage pipelined YOLOv8L: main thread does preprocess+device,
    background thread does NMS+draw+encode. The overlap happens during
    synchronize_device (C-level wait releases GIL, letting bg thread run)."""
    import queue
    import threading

    import ttnn
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    skip_pp = args.skip_preprocess

    print("[native-pipe] Opening device 0...", flush=True)
    device = ttnn.CreateDevice(
        0,
        l1_small_size=yolov8l_l1_small_size_for_res(1280, 1280),
        trace_region_size=35_000_000,
        num_command_queues=2,
    )
    device.enable_program_cache()

    print("[native-pipe] Building YOLOv8lPerformantRunner (1280×1280)...", flush=True)
    runner = YOLOv8lPerformantRunner(device, device_batch_size=1, inp_h=1280, inp_w=1280)
    print("[native-pipe] ready.", flush=True)

    coco_names = _load_coco_names()
    conf = max(args.conf, _CONF_FLOOR_1280)

    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"[native-pipe] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    if skip_pp:
        print("[native-pipe] --skip-preprocess: skipping letterbox", flush=True)

    _SENTINEL = None
    q_post = queue.Queue(maxsize=2)
    stop = False
    bg_error = [None]

    # --- Background thread: postprocess (NMS + draw + encode + write) ---
    ema_fps_holder = [0.0]
    frame_count_holder = [0]
    t_post_sum_holder = [0.0]
    t_wall_start_holder = [0.0]

    def _postprocess_bg():
        try:
            while True:
                item = q_post.get()
                if item is _SENTINEL:
                    return
                frame, preds_torch = item

                t0 = time.perf_counter()
                fc = frame_count_holder[0]
                if fc == 0:
                    t_wall_start_holder[0] = t0

                dets = _nms_single(preds_torch, conf, args.iou)

                if skip_pp:
                    canvas = frame.copy()
                    canvas = _draw_detections(canvas, dets, coco_names, scale_from=None)
                else:
                    canvas = frame.copy()
                    canvas = _draw_detections(canvas, dets, coco_names, scale_from=(1280, 1280))

                canvas = draw_hud(canvas, "YOLOv8L Native 1280 [pipe]", ema_fps_holder[0])
                canvas = scale_to_width(canvas, args.display_width)
                write_frame(args._frame_file, canvas, args.jpeg_quality)

                dt_post = time.perf_counter() - t0
                fc += 1
                frame_count_holder[0] = fc
                t_post_sum_holder[0] += dt_post

                wall_elapsed = time.perf_counter() - t_wall_start_holder[0]
                if fc > 1:
                    wall_fps = (fc - 1) / wall_elapsed
                    ema_fps_holder[0] = _EMA_ALPHA * wall_fps + (1 - _EMA_ALPHA) * ema_fps_holder[0]
                else:
                    ema_fps_holder[0] = 1.0 / max(dt_post, 1e-9)
                    print(f"[native-pipe] First post-process: {dt_post*1000:.1f}ms", flush=True)

                if fc % 30 == 0:
                    avg_post = t_post_sum_holder[0] / 30 * 1000
                    throughput = fc / wall_elapsed
                    print(
                        f"[native-pipe] Frames: {fc}  |  "
                        f"Throughput: {throughput:.1f} FPS  |  "
                        f"Avg post: {avg_post:.1f}ms  "
                        f"(q={q_post.qsize()})",
                        flush=True,
                    )
                    t_post_sum_holder[0] = 0.0

                if src.is_image:
                    if args.output:
                        cv2.imwrite(args.output, canvas)
                        print(f"[native-pipe] Saved to {args.output}", flush=True)
                    print("[native-pipe] Image done. Waiting...", flush=True)
        except Exception as exc:
            bg_error[0] = exc

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    bg = threading.Thread(target=_postprocess_bg, name="postprocess", daemon=True)
    bg.start()
    print("[native-pipe] 2-stage pipeline running...", flush=True)

    # --- Main thread: preprocess + device (all ttnn in one thread) ---
    try:
        while not stop:
            ok, frame = src.read()
            if not ok:
                if src.is_image:
                    break
                src.reset()
                ok, frame = src.read()
                if not ok:
                    break

            if skip_pp:
                lb = frame
            else:
                lb = _letterbox(frame, (1280, 1280))
            inp = _bgr_to_tensor(lb)

            preds = runner.run(torch_input_tensor=inp)
            ttnn.synchronize_device(device)

            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(preds, dtype=torch.float32)

            q_post.put((frame, preds_torch))

            if src.is_image:
                while not stop and bg.is_alive():
                    time.sleep(1)
                break
    finally:
        q_post.put(_SENTINEL)
        bg.join(timeout=10)

        if bg_error[0]:
            print(f"[native-pipe] BG error: {bg_error[0]}", file=sys.stderr, flush=True)

        print("[native-pipe] Shutting down...", flush=True)
        src.release()
        try:
            runner.release()
        except Exception:
            pass
        try:
            ttnn.synchronize_device(device)
        except Exception:
            pass
        try:
            ttnn.CloseDevice(device)
        except Exception:
            pass
        print("[native-pipe] Done.", flush=True)


# ---------------------------------------------------------------------------
# SAHI 4K worker — uses sahi.slicing + common_demo_utils pipeline
# ---------------------------------------------------------------------------


def _run_sahi_worker(args):
    """Multi-device YOLOv8L SAHI: slice with sahi.slicing.slice_image, infer on mesh,
    merge with SAHI postprocess.  Mirrors sahi_ultralytics_eval.run_tt_sliced_prediction."""
    from PIL import Image
    from sahi.slicing import slice_image

    import ttnn
    from models.demos.utils.common_demo_utils import get_mesh_mappers, postprocess, preprocess
    from models.demos.yolo_eval.sahi_ultralytics_eval import (
        build_postprocess,
        parallel_slice_chunk_bounds,
        result_to_object_predictions,
    )
    from models.demos.yolov8l.common import yolov8l_l1_small_size_for_res
    from models.demos.yolov8l.runner.performant_runner import YOLOv8lPerformantRunner

    l1_small = yolov8l_l1_small_size_for_res(1280, 1280)
    trace_region = 35_000_000
    in_res = (_TILE_SIZE, _TILE_SIZE)

    try:
        src = FrameSource(args.input)
    except RuntimeError as e:
        print(f"[sahi] ERROR: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    sample = src.peek()
    frame_h, frame_w = sample.shape[:2]

    # Use SAHI to determine the tile count from a sample frame
    sample_pil = Image.fromarray(cv2.cvtColor(sample, cv2.COLOR_BGR2RGB))
    sample_slice = slice_image(
        image=sample_pil,
        slice_height=_TILE_SIZE,
        slice_width=_TILE_SIZE,
        overlap_height_ratio=args.overlap_height_ratio,
        overlap_width_ratio=args.overlap_width_ratio,
        auto_slice_resolution=False,
    )
    total_tiles = len(sample_slice.images)

    if args.mesh_shape == "auto":
        n_tile_rows = max(1, (frame_h + _TILE_SIZE - 1) // _TILE_SIZE)
        n_tile_cols = max(1, (frame_w + _TILE_SIZE - 1) // _TILE_SIZE)
        if n_tile_rows * n_tile_cols != total_tiles:
            n_tile_rows, n_tile_cols = 1, total_tiles
        mesh_rows, mesh_cols = n_tile_rows, n_tile_cols
    else:
        mesh_rows, mesh_cols = (int(x) for x in args.mesh_shape.split("x"))
        assert mesh_rows * mesh_cols == total_tiles, (
            f"--mesh-shape {args.mesh_shape} = {mesh_rows*mesh_cols} devices " f"but SAHI produced {total_tiles} tiles"
        )

    n_devices = mesh_rows * mesh_cols

    sys_shape = tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape())
    mesh_shape = ttnn.MeshShape(mesh_rows, mesh_cols)
    print(
        f"[sahi] System mesh: {sys_shape[0]}×{sys_shape[1]} = {sys_shape[0]*sys_shape[1]} devices",
        flush=True,
    )
    print(
        f"[sahi] SAHI slicing: {total_tiles} tiles of {_TILE_SIZE}×{_TILE_SIZE} "
        f"(overlap h={args.overlap_height_ratio}, w={args.overlap_width_ratio})",
        flush=True,
    )
    for idx, (sx, sy) in enumerate(sample_slice.starting_pixels):
        sh, sw = sample_slice.images[idx].shape[:2]
        print(f"[sahi]   tile {idx}: start=({sx},{sy}) size={sw}×{sh}", flush=True)
    print(
        f"[sahi] Opening sub-mesh {mesh_rows}×{mesh_cols} = {n_devices} devices",
        flush=True,
    )

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        l1_small_size=l1_small,
        trace_region_size=trace_region,
        num_command_queues=2,
    )
    mesh_device.enable_program_cache()

    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)

    print(f"[sahi] Building YOLOv8lPerformantRunner (1280×1280, batch={n_devices})...", flush=True)
    runner = YOLOv8lPerformantRunner(
        mesh_device,
        device_batch_size=n_devices,
        inp_h=1280,
        inp_w=1280,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
    )
    print("[sahi] ready.", flush=True)

    coco_names = _load_coco_names()
    names_dict = _coco_names_dict(coco_names)
    conf = max(args.conf, _CONF_FLOOR_1280)

    sahi_merge = build_postprocess(
        postprocess_type=args.sahi_merge_type,
        match_metric=args.sahi_merge_metric,
        match_threshold=args.sahi_merge_threshold,
        class_agnostic=args.sahi_class_agnostic,
    )
    do_full_pred = args.perform_standard_pred
    print(
        f"[sahi] Merge: {args.sahi_merge_type} metric={args.sahi_merge_metric} "
        f"threshold={args.sahi_merge_threshold}  conf={conf}  "
        f"perform_standard_pred={do_full_pred}",
        flush=True,
    )

    black_bgr = np.zeros((_TILE_SIZE, _TILE_SIZE, 3), dtype=np.uint8)

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    ema_fps = 0.0
    first = True
    frame_count = 0
    LOG_INTERVAL = 10
    t_read_sum = t_sahi_slice_sum = 0.0
    t_colorconv_sum = t_letterbox_sum = t_tensor_sum = 0.0
    t_device_sum = t_host_prep_sum = t_h2d_trace_sum = t_sync_sum = 0.0
    t_d2h_sum = t_nms_sum = t_to_objpred_sum = 0.0
    t_sahi_shift_sum = t_sahi_merge_sum = 0.0
    t_draw_sum = t_encode_sum = t_total_sum = 0.0
    print("[sahi] Running...", flush=True)

    try:
        while not stop:
            t_frame_start = time.perf_counter()

            t0 = time.perf_counter()
            ok, frame_4k = src.read()
            if not ok:
                if src.is_image:
                    break
                src.reset()
                ok, frame_4k = src.read()
                if not ok:
                    break
            t_read = time.perf_counter() - t0

            # Step 1: SAHI slices the frame
            t0 = time.perf_counter()
            frame_pil = Image.fromarray(cv2.cvtColor(frame_4k, cv2.COLOR_BGR2RGB))
            slice_result = slice_image(
                image=frame_pil,
                slice_height=_TILE_SIZE,
                slice_width=_TILE_SIZE,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
                auto_slice_resolution=False,
            )
            full_shape = [slice_result.original_image_height, slice_result.original_image_width]
            slice_entries = list(zip(slice_result.starting_pixels, slice_result.images))
            t_sahi_slice = time.perf_counter() - t0

            all_obj_preds = []
            t_colorconv_frame = 0.0
            t_letterbox_frame = 0.0
            t_tensor_frame = 0.0
            t_device_frame = 0.0
            t_host_prep_frame = 0.0
            t_h2d_trace_frame = 0.0
            t_sync_frame = 0.0
            t_d2h_frame = 0.0
            t_nms_frame = 0.0
            t_to_objpred_frame = 0.0

            # Step 2–5: Batch tiles across mesh, infer, postprocess, convert to SAHI objects
            for chunk_start, n_valid in parallel_slice_chunk_bounds(len(slice_entries), n_devices):
                chunk = slice_entries[chunk_start : chunk_start + n_valid]
                shifts = []
                batch_bgr = []

                t0 = time.perf_counter()
                for (start_x, start_y), slice_img in chunk:
                    shifts.append((start_x, start_y))
                    batch_bgr.append(cv2.cvtColor(slice_img, cv2.COLOR_RGB2BGR))

                for _ in range(n_devices - len(chunk)):
                    shifts.append((0, 0))
                    batch_bgr.append(black_bgr.copy())
                t_colorconv_frame += time.perf_counter() - t0

                t0 = time.perf_counter()
                parts = [preprocess([im], res=in_res) for im in batch_bgr]
                t_letterbox_frame += time.perf_counter() - t0

                t0 = time.perf_counter()
                im_tensor = torch.cat(parts, dim=0)
                t_tensor_frame += time.perf_counter() - t0

                # Device inference
                t0 = time.perf_counter()
                preds = runner.run(torch_input_tensor=im_tensor)
                t_pre_sync = time.perf_counter() - t0
                t0_sync = time.perf_counter()
                ttnn.synchronize_device(mesh_device)
                t_sync_chunk = time.perf_counter() - t0_sync
                t_device_frame += t_pre_sync + t_sync_chunk
                t_host_prep_frame += runner.last_timing["host_prep_ms"]
                t_h2d_trace_frame += runner.last_timing["h2d_and_trace_ms"]
                t_sync_frame += t_sync_chunk

                t0 = time.perf_counter()
                if isinstance(preds, (list, tuple)):
                    preds = preds[0]
                preds_torch = ttnn.to_torch(
                    preds,
                    dtype=torch.float32,
                    mesh_composer=output_mesh_composer,
                )
                t_d2h_frame += time.perf_counter() - t0

                # postprocess: NMS + scale_boxes to undo letterbox
                t0 = time.perf_counter()
                paths = ([f"tile{chunk_start + i}" for i in range(n_devices)],)
                results = postprocess(preds_torch, im_tensor, batch_bgr, paths, names_dict, conf=conf, iou=args.iou)
                t_nms_frame += time.perf_counter() - t0

                # Convert each valid tile's results to SAHI ObjectPredictions
                t0 = time.perf_counter()
                for j in range(n_valid):
                    all_obj_preds.extend(
                        result_to_object_predictions(
                            results[j],
                            shift_xy=shifts[j],
                            full_shape=full_shape,
                            confidence_threshold=conf,
                        )
                    )
                t_to_objpred_frame += time.perf_counter() - t0

            # Optional: full-image prediction (letterboxed 4K→1280) for large objects
            if do_full_pred:
                full_im_tensor = preprocess([frame_4k], res=in_res)
                full_im_tensor = full_im_tensor.repeat(n_devices, 1, 1, 1)
                full_preds = runner.run(torch_input_tensor=full_im_tensor)
                ttnn.synchronize_device(mesh_device)
                if isinstance(full_preds, (list, tuple)):
                    full_preds = full_preds[0]
                full_preds_torch = ttnn.to_torch(
                    full_preds,
                    dtype=torch.float32,
                    mesh_composer=output_mesh_composer,
                )
                full_orig = [frame_4k] * n_devices
                full_paths = (["full_image"] * n_devices,)
                full_results = postprocess(
                    full_preds_torch,
                    full_im_tensor,
                    full_orig,
                    full_paths,
                    names_dict,
                    conf=conf,
                    iou=args.iou,
                )
                full_obj_preds = result_to_object_predictions(
                    full_results[0],
                    shift_xy=(0, 0),
                    full_shape=full_shape,
                    confidence_threshold=conf,
                )
                all_obj_preds.extend(full_obj_preds)

            # Step 6: SAHI shifts to full-image coords and merges
            t0 = time.perf_counter()
            shifted = [p.get_shifted_object_prediction() for p in all_obj_preds]
            t_sahi_shift = time.perf_counter() - t0

            t0 = time.perf_counter()
            if len(shifted) > 1:
                merged_preds = sahi_merge(shifted)
            else:
                merged_preds = shifted
            t_sahi_merge = time.perf_counter() - t0

            t0 = time.perf_counter()
            canvas = frame_4k.copy()
            canvas = _draw_sahi_detections(canvas, merged_preds)
            t_draw = time.perf_counter() - t0

            dt_total = time.perf_counter() - t_frame_start

            if first:
                ema_fps = 1.0 / max(dt_total, 1e-9)
                first = False
                print(
                    f"[sahi] First frame: {dt_total*1000:.1f}ms "
                    f"(read={t_read*1000:.1f} sahi_slice={t_sahi_slice*1000:.1f} "
                    f"colorconv={t_colorconv_frame*1000:.1f} "
                    f"letterbox={t_letterbox_frame*1000:.1f} "
                    f"tensor={t_tensor_frame*1000:.1f} "
                    f"device={t_device_frame*1000:.1f}"
                    f"[host_prep={t_host_prep_frame:.1f} "
                    f"h2d+trace={t_h2d_trace_frame:.1f} "
                    f"sync={t_sync_frame*1000:.1f}] "
                    f"d2h={t_d2h_frame*1000:.1f} nms={t_nms_frame*1000:.1f} "
                    f"to_objpred={t_to_objpred_frame*1000:.1f} "
                    f"sahi_shift={t_sahi_shift*1000:.1f} sahi_merge={t_sahi_merge*1000:.1f} "
                    f"draw={t_draw*1000:.1f})",
                    flush=True,
                )
            else:
                ema_fps = _EMA_ALPHA * (1.0 / max(dt_total, 1e-9)) + (1 - _EMA_ALPHA) * ema_fps

            t0 = time.perf_counter()
            canvas = draw_hud(
                canvas,
                f"YOLOv8L SAHI 4K ({mesh_rows}x{mesh_cols}={n_devices} chips)",
                ema_fps,
            )
            canvas = scale_to_width(canvas, args.display_width)
            write_frame(args._frame_file, canvas, args.jpeg_quality)
            t_encode = time.perf_counter() - t0

            frame_count += 1
            t_read_sum += t_read
            t_sahi_slice_sum += t_sahi_slice
            t_colorconv_sum += t_colorconv_frame
            t_letterbox_sum += t_letterbox_frame
            t_tensor_sum += t_tensor_frame
            t_device_sum += t_device_frame
            t_host_prep_sum += t_host_prep_frame
            t_h2d_trace_sum += t_h2d_trace_frame
            t_sync_sum += t_sync_frame
            t_d2h_sum += t_d2h_frame
            t_nms_sum += t_nms_frame
            t_to_objpred_sum += t_to_objpred_frame
            t_sahi_shift_sum += t_sahi_shift
            t_sahi_merge_sum += t_sahi_merge
            t_draw_sum += t_draw
            t_encode_sum += t_encode
            t_total_sum += dt_total + t_encode

            if frame_count % LOG_INTERVAL == 0:
                n = LOG_INTERVAL
                print(
                    f"[sahi] Avg over {n} frames: "
                    f"total={t_total_sum/n*1000:.1f}ms "
                    f"({1.0/(t_total_sum/n):.1f} FPS)  |  "
                    f"read={t_read_sum/n*1000:.1f}  "
                    f"sahi_slice={t_sahi_slice_sum/n*1000:.1f}  "
                    f"colorconv={t_colorconv_sum/n*1000:.1f}  "
                    f"letterbox={t_letterbox_sum/n*1000:.1f}  "
                    f"tensor={t_tensor_sum/n*1000:.1f}  "
                    f"device={t_device_sum/n*1000:.1f}"
                    f"[host_prep={t_host_prep_sum/n:.1f} "
                    f"h2d+trace={t_h2d_trace_sum/n:.1f} "
                    f"sync={t_sync_sum/n*1000:.1f}]  "
                    f"d2h={t_d2h_sum/n*1000:.1f}  "
                    f"nms={t_nms_sum/n*1000:.1f}  "
                    f"to_objpred={t_to_objpred_sum/n*1000:.1f}  "
                    f"sahi_shift={t_sahi_shift_sum/n*1000:.1f}  "
                    f"sahi_merge={t_sahi_merge_sum/n*1000:.1f}  "
                    f"draw={t_draw_sum/n*1000:.1f}  "
                    f"hud+encode+write={t_encode_sum/n*1000:.1f}  (ms)",
                    flush=True,
                )
                t_read_sum = t_sahi_slice_sum = 0.0
                t_colorconv_sum = t_letterbox_sum = t_tensor_sum = 0.0
                t_device_sum = t_host_prep_sum = t_h2d_trace_sum = t_sync_sum = 0.0
                t_d2h_sum = t_nms_sum = t_to_objpred_sum = 0.0
                t_sahi_shift_sum = t_sahi_merge_sum = 0.0
                t_draw_sum = t_encode_sum = t_total_sum = 0.0

            if src.is_image:
                if args.output:
                    cv2.imwrite(args.output, canvas)
                    print(f"[sahi] Saved annotated image to {args.output}", flush=True)
                print("[sahi] Image inference complete. Waiting for shutdown...", flush=True)
                while not stop:
                    time.sleep(1)
                break
    finally:
        print("[sahi] Shutting down...", flush=True)
        src.release()
        try:
            runner.release()
        except Exception:
            pass
        try:
            ttnn.synchronize_device(mesh_device)
        except Exception:
            pass
        try:
            ttnn.close_mesh_device(mesh_device)
        except Exception:
            pass
        print("[sahi] Done.", flush=True)


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _wait_for_ready(proc: subprocess.Popen, name: str, timeout: int = 600):
    import select

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print(f"[main] {name} exited unexpectedly (code {proc.returncode})", flush=True)
            return False
        if proc.stdout and select.select([proc.stdout], [], [], 0.5)[0]:
            line = proc.stdout.readline()
            if line:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                if b"ready." in line:
                    return True
        else:
            time.sleep(0.5)
    print(f"[main] WARNING: {name} did not become ready within {timeout}s", flush=True)
    return True


def _pipe_output(proc: subprocess.Popen, name: str):
    import threading

    def _run():
        try:
            for line in iter(proc.stdout.readline, b""):
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
        except (ValueError, OSError):
            pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    if args._native_worker:
        if args.pipeline:
            _run_native_worker_pipelined(args)
        else:
            _run_native_worker(args)
        return
    if args._sahi_worker:
        if args.pipeline:
            from models.demos.yolo_eval.yolov8l_sahi_pipelined import run_sahi_worker_pipelined

            run_sahi_worker_pipelined(args)
        else:
            _run_sahi_worker(args)
        return

    # --- Orchestrator: launch workers + MJPEG server ---
    tmpdir = tempfile.gettempdir()
    left_file = os.path.join(tmpdir, "yolov8l_native_frame.jpg")
    right_file = os.path.join(tmpdir, "yolov8l_sahi_frame.jpg")

    this_script = os.path.abspath(__file__)
    common_args = (
        [
            "--input",
            args.input,
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--jpeg-quality",
            str(args.jpeg_quality),
            "--display-width",
            str(args.display_width),
            "--sahi-merge-type",
            args.sahi_merge_type,
            "--sahi-merge-metric",
            args.sahi_merge_metric,
            "--sahi-merge-threshold",
            str(args.sahi_merge_threshold),
            "--overlap-height-ratio",
            str(args.overlap_height_ratio),
            "--overlap-width-ratio",
            str(args.overlap_width_ratio),
            "--mesh-shape",
            args.mesh_shape,
        ]
        + (["--sahi-class-agnostic"] if args.sahi_class_agnostic else [])
        + (["--perform-standard-pred"] if args.perform_standard_pred else [])
        + (["--skip-preprocess"] if args.skip_preprocess else [])
        + (["--pipeline"] if args.pipeline else [])
        + (["--output", args.output] if args.output else [])
    )

    sahi_only = args.sahi_only
    native_only = args.native_only
    if sahi_only and native_only:
        print("ERROR: --sahi-only and --native-only are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    single_panel = sahi_only or native_only

    env_native = {**os.environ, "TT_VISIBLE_DEVICES": str(args.native_device_id)}
    env_sahi = {k: v for k, v in os.environ.items() if k != "TT_VISIBLE_DEVICES"}

    print("=" * 70, flush=True)
    if sahi_only:
        print("YOLOv8L 4K SAHI Slicing Demo (full screen)", flush=True)
        print("  SAHI mesh:      auto (sub-mesh sized to tile grid)", flush=True)
    elif native_only:
        print("YOLOv8L Native 1280 Demo (full screen)", flush=True)
        print(f"  Native device:  {args.native_device_id}", flush=True)
    else:
        print("YOLOv8L Native 1280 vs. 4K SAHI Comparison Demo", flush=True)
        print(f"  Native device:  {args.native_device_id}", flush=True)
        print("  SAHI mesh:      auto (sub-mesh sized to tile grid)", flush=True)
    input_type = "Image" if _is_image_path(args.input) else "Video"
    print(f"  {input_type}:          {args.input}", flush=True)
    print(f"  Display width:  {args.display_width}", flush=True)
    print("=" * 70, flush=True)

    proc_native = None
    if not sahi_only:
        print("[main] Launching native 1280 worker...", flush=True)
        native_frame_file = left_file
        proc_native = subprocess.Popen(
            [
                sys.executable,
                "-u",
                this_script,
                "--_native-worker",
                "--_frame-file",
                native_frame_file,
                "--native-device-id",
                "0",
            ]
            + common_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_native,
        )
        _wait_for_ready(proc_native, "Native")
        _pipe_output(proc_native, "Native")

    proc_sahi = None
    if not native_only:
        print("[main] Launching SAHI 4K worker...", flush=True)
        sahi_frame_file = right_file if not sahi_only else left_file
        proc_sahi = subprocess.Popen(
            [
                sys.executable,
                "-u",
                this_script,
                "--_sahi-worker",
                "--_frame-file",
                sahi_frame_file,
            ]
            + common_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_sahi,
        )
        _wait_for_ready(proc_sahi, "SAHI")
        _pipe_output(proc_sahi, "SAHI")

    # Launch MJPEG server
    http_proc = None
    if args.serve:
        print(f"[main] Starting MJPEG server on http://{args.host}:{args.port}/", flush=True)
        if single_panel:
            http_proc = subprocess.Popen(
                [
                    sys.executable,
                    _SERVER_SCRIPT,
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--frame-file",
                    left_file,
                ]
            )
        else:
            http_proc = subprocess.Popen(
                [
                    sys.executable,
                    _SERVER_SCRIPT,
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--left-file",
                    left_file,
                    "--right-file",
                    right_file,
                ]
            )
        time.sleep(1)

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    workers = []
    if proc_native is not None:
        workers.append((proc_native, "Native"))
    if proc_sahi is not None:
        workers.append((proc_sahi, "SAHI"))

    print("[main] Demo running. Press Ctrl+C to stop.", flush=True)
    try:
        while not stop:
            for p, name in workers:
                if p.poll() is not None:
                    print(f"[main] {name} exited (code {p.returncode})", flush=True)
                    stop = True
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[main] Shutting down...", flush=True)
        for p, name in workers:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                print(f"[main] {name} terminated.", flush=True)
        if http_proc and http_proc.poll() is None:
            http_proc.terminate()
            try:
                http_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                http_proc.kill()
        for f in (left_file, right_file):
            for p in (f, f + ".tmp"):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        print("[main] Done.", flush=True)


if __name__ == "__main__":
    main()
