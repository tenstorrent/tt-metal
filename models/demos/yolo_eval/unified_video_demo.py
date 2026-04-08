#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
YOLO video/image demo on Tenstorrent device(s).

Supports three modes:
  Single model  – YOLOv8s or YOLOv11s on one device (default).
  Dual model    – YOLOv8s + YOLOv11s side-by-side on two devices (--dual).
  Unified demo  – Side-by-side + large-model (YOLOv8L SAHI 4-device) with
                  browser-based mode switching (--unified).

In dual/unified mode each model runs in its own **subprocess** (separate UMD
instance) to avoid the cluster-wide device lock that causes hangs when two
devices are opened in the same process.

Usage (single model, YOLOv8s):
    python models/demos/yolo_eval/unified_video_demo.py \\
        --video models/demos/yolo_eval/sample_images/14025986_640x640_29fps.mp4 \\
        --device-id 0 --serve --port 9090 --pre-letterboxed

Usage (single model, YOLOv11s):
    python models/demos/yolo_eval/unified_video_demo.py \\
        --model yolov11s \\
        --video models/demos/yolo_eval/sample_images/14025986_640x640_29fps.mp4 \\
        --device-id 0 --serve --port 9090 --pre-letterboxed

Usage (dual model, side-by-side):
    python models/demos/yolo_eval/unified_video_demo.py \\
        --video models/demos/yolo_eval/sample_images/14025986_640x640_29fps.mp4 \\
        --dual --device-id 0 --device-id-v11s 1 --serve --port 9090 --pre-letterboxed

Usage (unified demo with mode toggle):
    python models/demos/yolo_eval/unified_video_demo.py \\
        --video models/demos/yolo_eval/sample_images/14025986_640x640_29fps.mp4 \\
        --unified --device-id 0 --device-id-v11s 1 --serve --port 9090 \\
        --pre-letterboxed \\
        --video-1280 models/demos/yolo_eval/sample_images/14025986_1280x1280_29fps.mp4

Browser access:
    SSH tunnel:  ssh -L 9090:localhost:9090 user@server
    Then open:   http://localhost:9090/
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

_INPUT_RES = (640, 640)
_EMA_ALPHA = 0.15


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO single/dual-device video demo.")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--video", type=str, default=None, help="Path to video file.")
    src.add_argument("--image", type=str, default=None, help="Path to a static image (loops).")
    src.add_argument("--camera-index", type=int, default=0, help="Webcam index (default 0).")
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s",
        choices=["yolov8s", "yolov11s"],
        help="Which model to run in single-model mode (default: yolov8s).",
    )
    p.add_argument(
        "--unified", action="store_true", help="Unified demo: side-by-side + large-model mode toggle via browser UI."
    )
    p.add_argument(
        "--mode-file",
        type=str,
        default=None,
        help="Shared mode file for unified demo (auto-created if omitted with --unified).",
    )
    p.add_argument(
        "--video-1280", type=str, default=None, help="Path to 1280x1280 letterboxed video for large-model mode."
    )
    p.add_argument("--device-id", type=int, default=0, help="TT device id.")
    p.add_argument("--dual", action="store_true", help="Run YOLOv8s + YOLOv11s side-by-side.")
    p.add_argument("--device-id-v11s", type=int, default=1, help="TT device id for YOLOv11s (dual mode).")
    p.add_argument("--conf", type=float, default=0.25, help="NMS confidence threshold.")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    p.add_argument("--serve", action="store_true", help="Stream MJPEG over HTTP.")
    p.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind address.")
    p.add_argument("--port", type=int, default=9090, help="HTTP port.")
    p.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality (1-100).")
    p.add_argument(
        "--pre-letterboxed",
        action="store_true",
        help="Input is already 640x640 letterboxed — skip resize, draw on 640x640.",
    )
    # Internal: used by --dual to launch a headless worker subprocess
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_frame-file", type=str, default=None, help=argparse.SUPPRESS)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def letterbox(img: np.ndarray, target: tuple[int, int] = _INPUT_RES) -> np.ndarray:
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


def frame_to_tensor(bgr: np.ndarray, skip_letterbox: bool = False) -> torch.Tensor:
    """BGR frame -> [1,3,640,640] float32 NCHW tensor."""
    lb = bgr if skip_letterbox else letterbox(bgr, _INPUT_RES)
    rgb = lb[:, :, ::-1].transpose(2, 0, 1)  # HWC BGR -> CHW RGB
    return torch.from_numpy(np.ascontiguousarray(rgb)).float().unsqueeze(0) / 255.0


# ---------------------------------------------------------------------------
# Post-processing: NMS + draw
# ---------------------------------------------------------------------------


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    xy, wh = x[..., :2], x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def _load_coco_names() -> list[str]:
    names_file = Path(__file__).resolve().parents[2] / "demos" / "utils" / "coco.names"
    if names_file.exists():
        return [l.strip() for l in names_file.read_text().splitlines() if l.strip()]
    return [str(i) for i in range(80)]


def nms_and_draw(
    preds: torch.Tensor,
    orig_bgr: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    class_names: list[str],
) -> np.ndarray:
    import torchvision

    if isinstance(preds, (list, tuple)):
        preds = preds[0]

    nc = preds.shape[1] - 4
    xc = preds[:, 4 : 4 + nc].amax(1) > conf_thres
    preds = preds.transpose(-1, -2)
    preds[..., :4] = _xywh2xyxy(preds[..., :4])

    img = orig_bgr.copy()
    oh, ow = img.shape[:2]
    gain = min(640 / oh, 640 / ow)
    pad_x = round((640 - ow * gain) / 2 - 0.1)
    pad_y = round((640 - oh * gain) / 2 - 0.1)

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
        x = x[i[:300]]

        x[:, [0, 2]] -= pad_x
        x[:, [1, 3]] -= pad_y
        x[:, :4] /= gain
        x[:, 0].clamp_(0, ow)
        x[:, 1].clamp_(0, oh)
        x[:, 2].clamp_(0, ow)
        x[:, 3].clamp_(0, oh)

        for det in x:
            x1, y1, x2, y2 = map(int, det[:4])
            score = float(det[4])
            cls_id = int(det[5])
            label = f"{class_names[cls_id] if cls_id < len(class_names) else cls_id} {score:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------


def draw_hud(img: np.ndarray, model_name: str, fps: float, color: tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    text = f"{model_name}  |  FPS: {fps:.1f}"
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    return img


# ---------------------------------------------------------------------------
# MJPEG server helpers
# ---------------------------------------------------------------------------

_SERVER_SCRIPT = str(Path(__file__).resolve().parent / "_mjpeg_server.py")
_SAHI_WORKER_SCRIPT = str(Path(__file__).resolve().parent / "_yolov8l_sahi_worker.py")


def start_server_single(host: str, port: int, frame_file: str, mode_file: str | None = None) -> subprocess.Popen:
    cmd = [sys.executable, _SERVER_SCRIPT, "--host", host, "--port", str(port), "--frame-file", frame_file]
    if mode_file:
        cmd += ["--mode-file", mode_file]
    return subprocess.Popen(cmd)


def start_server_dual(
    host: str, port: int, left_file: str, right_file: str, mode_file: str | None = None
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        _SERVER_SCRIPT,
        "--host",
        host,
        "--port",
        str(port),
        "--left-file",
        left_file,
        "--right-file",
        right_file,
    ]
    if mode_file:
        cmd += ["--mode-file", mode_file]
    return subprocess.Popen(cmd)


def write_frame(path: str, img: np.ndarray, quality: int = 75):
    """Atomically write a JPEG frame to disk."""
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


# ---------------------------------------------------------------------------
# Model / device helpers
# ---------------------------------------------------------------------------


def _build_model(model: str, device):
    """Build the selected model runner and return (model_name, runner)."""
    if model == "yolov11s":
        from models.demos.yolov11s.runner.performant_runner import YOLOv11sPerformantRunner

        print("[init] Building YOLOv11sPerformantRunner...", flush=True)
        runner = YOLOv11sPerformantRunner(device)
        return "YOLOv11s", runner
    else:
        from models.demos.yolov8s.runner.performant_runner import YOLOv8sPerformantRunner

        print("[init] Building YOLOv8sPerformantRunner...", flush=True)
        runner = YOLOv8sPerformantRunner(device, 1)
        return "YOLOv8s", runner


def _device_params(model: str) -> dict:
    """Return CreateDevice kwargs for the chosen model."""
    if model == "yolov11s":
        from models.demos.yolov11s.common import YOLOV11S_L1_SMALL_SIZE

        return dict(l1_small_size=YOLOV11S_L1_SMALL_SIZE, trace_region_size=6434816, num_command_queues=2)
    else:
        from models.demos.yolov8s.common import YOLOV8S_L1_SMALL_SIZE

        return dict(l1_small_size=YOLOV8S_L1_SMALL_SIZE, trace_region_size=3211264, num_command_queues=2)


# ---------------------------------------------------------------------------
# Video source helpers
# ---------------------------------------------------------------------------


def _open_source(args):
    """Return (static_frame, cap) — one of them will be None."""
    static_frame = None
    cap = None
    if args.image:
        static_frame = cv2.imread(args.image)
        if static_frame is None:
            print(f"ERROR: cannot read image: {args.image}", file=sys.stderr)
            sys.exit(1)
        print(f"Static image: {args.image}  shape={static_frame.shape}")
    else:
        source = args.video if args.video else args.camera_index
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: cannot open video source: {source}", file=sys.stderr)
            sys.exit(1)
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {source}  fps={fps_src:.1f}  frames={total}")
    return static_frame, cap


def _read_frame(static_frame, cap, args):
    """Read next frame. Returns None only when source is exhausted."""
    if static_frame is not None:
        return static_frame
    ok, bgr = cap.read()
    if not ok:
        if args.video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = cap.read()
            if ok:
                return bgr
        return None
    return bgr


# ---------------------------------------------------------------------------
# Single-model inference loop (also used as --_worker subprocess)
# ---------------------------------------------------------------------------


def _run_single(args):
    static_frame, cap = _open_source(args)

    import ttnn

    print(f"[init] Opening device {args.device_id} for {args.model}...", flush=True)
    device = ttnn.CreateDevice(args.device_id, **_device_params(args.model))
    device.enable_program_cache()

    model_name, runner = _build_model(args.model, device)
    print(f"[init] {model_name} ready.", flush=True)

    coco_names = _load_coco_names()

    # Determine where to write frames
    frame_file = args._frame_file  # set when launched as a worker subprocess
    http_proc = None
    if not frame_file and args.serve:
        frame_file = os.path.join(tempfile.gettempdir(), f"{args.model}_frame.jpg")
        http_proc = start_server_single(args.host, args.port, frame_file)
        time.sleep(0.5)

    stop = False

    def _shutdown(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Running {model_name}... press Ctrl+C to stop.", flush=True)
    ema_dev_fps = 0.0
    first = True

    try:
        while not stop:
            bgr = _read_frame(static_frame, cap, args)
            if bgr is None:
                break

            inp = frame_to_tensor(bgr, skip_letterbox=args.pre_letterboxed)

            t_dev = time.perf_counter()
            preds = runner.run(torch_input_tensor=inp)
            # YOLOv11s runner uses blocking=False; sync before reading output
            if args.model == "yolov11s":
                ttnn.synchronize_device(device)
            dt_dev = time.perf_counter() - t_dev

            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds_torch = ttnn.to_torch(preds, dtype=torch.float32)
            annotated = nms_and_draw(preds_torch, bgr, args.conf, args.iou, coco_names)

            if first:
                ema_dev_fps = 1.0 / max(dt_dev, 1e-9)
                first = False
                print(f"[{model_name}] First inference: {dt_dev:.4f}s " f"({ema_dev_fps:.0f} device FPS)", flush=True)
            else:
                ema_dev_fps = _EMA_ALPHA * (1.0 / max(dt_dev, 1e-9)) + (1 - _EMA_ALPHA) * ema_dev_fps

            annotated = draw_hud(annotated, model_name, ema_dev_fps)

            if frame_file:
                write_frame(frame_file, annotated, args.jpeg_quality)
            else:
                cv2.imshow(f"{model_name} Demo", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        print(f"[{model_name}] Shutting down...", flush=True)
        if cap is not None:
            cap.release()
        if not frame_file:
            cv2.destroyAllWindows()
        if http_proc is not None:
            http_proc.terminate()
            try:
                http_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                http_proc.kill()
        # Only clean up temp file if we created it (not in worker mode)
        if frame_file and not args._worker:
            for f in (frame_file, frame_file + ".tmp"):
                try:
                    os.unlink(f)
                except OSError:
                    pass
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
        print(f"[{model_name}] Done.")


# ---------------------------------------------------------------------------
# Dual-model: launch two worker subprocesses + MJPEG server
# ---------------------------------------------------------------------------


def _wait_for_ready(proc: subprocess.Popen, name: str, timeout: int = 300):
    """Wait until the worker writes its first frame file (signals model ready)."""
    import select

    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print(f"[dual] {name} worker exited unexpectedly with code {proc.returncode}", flush=True)
            return False
        # Check stdout for the "ready" marker
        if proc.stdout and select.select([proc.stdout], [], [], 0.5)[0]:
            line = proc.stdout.readline()
            if line:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                if b"ready." in line:
                    return True
        else:
            time.sleep(0.5)
    print(f"[dual] WARNING: {name} did not become ready within {timeout}s", flush=True)
    return True  # continue anyway


def _pipe_output(proc: subprocess.Popen, name: str):
    """Forward subprocess stdout to parent stdout in a daemon thread."""

    try:
        for line in iter(proc.stdout.readline, b""):
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    except (ValueError, OSError):
        pass


def _run_dual(args):
    import threading as _threading

    tmpdir = tempfile.gettempdir()
    left_file = os.path.join(tmpdir, "yolov8s_frame.jpg")
    right_file = os.path.join(tmpdir, "yolov11s_frame.jpg")

    this_script = os.path.abspath(__file__)

    if args.video:
        source_args = ["--video", args.video]
    elif args.image:
        source_args = ["--image", args.image]
    else:
        source_args = ["--camera-index", str(args.camera_index)]

    common = (
        source_args
        + [
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--jpeg-quality",
            str(args.jpeg_quality),
            "--_worker",
        ]
        + (["--pre-letterboxed"] if args.pre_letterboxed else [])
    )

    # Each subprocess gets TT_VISIBLE_DEVICES so the UMD only opens that
    # one chip.  The env var remaps the visible chip to logical device 0.
    env_v8 = {**os.environ, "TT_VISIBLE_DEVICES": str(args.device_id)}
    env_v11 = {**os.environ, "TT_VISIBLE_DEVICES": str(args.device_id_v11s)}

    print(f"[dual] Launching YOLOv8s (TT_VISIBLE_DEVICES={args.device_id})...", flush=True)
    proc_v8 = subprocess.Popen(
        [sys.executable, "-u", this_script, "--model", "yolov8s", "--device-id", "0", "--_frame-file", left_file]
        + common,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env_v8,
    )

    print("[dual] Waiting for YOLOv8s to initialize...", flush=True)
    _wait_for_ready(proc_v8, "YOLOv8s")

    _threading.Thread(target=_pipe_output, args=(proc_v8, "YOLOv8s"), daemon=True).start()

    print(f"[dual] Launching YOLOv11s (TT_VISIBLE_DEVICES={args.device_id_v11s})...", flush=True)
    proc_v11 = subprocess.Popen(
        [sys.executable, "-u", this_script, "--model", "yolov11s", "--device-id", "0", "--_frame-file", right_file]
        + common,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env_v11,
    )

    print("[dual] Waiting for YOLOv11s to initialize...", flush=True)
    _wait_for_ready(proc_v11, "YOLOv11s")

    _threading.Thread(target=_pipe_output, args=(proc_v11, "YOLOv11s"), daemon=True).start()

    # Start MJPEG server
    http_proc = None
    if args.serve:
        http_proc = start_server_dual(args.host, args.port, left_file, right_file)
        time.sleep(0.5)
        print(f"[dual] MJPEG server on http://{args.host}:{args.port}/", flush=True)

    children = [proc_v8, proc_v11]
    if http_proc:
        children.append(http_proc)

    def _shutdown(*_):
        for p in children:
            try:
                p.terminate()
            except OSError:
                pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[dual] Running... press Ctrl+C to stop.", flush=True)

    try:
        # Wait for either worker to exit (or signal)
        while True:
            for p in (proc_v8, proc_v11):
                ret = p.poll()
                if ret is not None:
                    print(f"[dual] Worker pid={p.pid} exited with code {ret}", flush=True)
                    _shutdown()
                    raise SystemExit(ret)
            time.sleep(1)
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        print("[dual] Shutting down all subprocesses...", flush=True)
        for p in children:
            try:
                p.terminate()
            except OSError:
                pass
        for p in children:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        for f in (left_file, right_file, left_file + ".tmp", right_file + ".tmp"):
            try:
                os.unlink(f)
            except OSError:
                pass
        print("[dual] Done.")


# ---------------------------------------------------------------------------
# Unified demo: side-by-side ↔ large-model mode switching
# ---------------------------------------------------------------------------


def _terminate_proc(proc: subprocess.Popen | None, label: str = "", timeout: int = 15):
    """Gracefully terminate a subprocess, falling back to kill."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
    except OSError:
        return
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[unified] {label} did not exit in {timeout}s — killing", flush=True)
        proc.kill()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _read_mode_file(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip() or "side-by-side"
    except (FileNotFoundError, OSError):
        return "side-by-side"


def _write_mode_file(path: str, mode: str):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(mode)
    os.replace(tmp, path)


def _write_loading_placeholder(frame_file: str, text: str = "Loading..."):
    """Write a simple placeholder JPEG so the stream shows something during transitions."""
    canvas = np.full((480, 640, 3), 30, dtype=np.uint8)
    cv2.putText(canvas, text, (60, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 2, cv2.LINE_AA)
    write_frame(frame_file, canvas, 75)


def _compositor_thread(left_path: str, right_path: str, out_path: str, stop_event, quality: int = 75):
    """Read left+right frame files, composite side-by-side, write to out_path.

    Runs in a daemon thread so the HTTP server (which reads out_path) never
    needs to be restarted on mode switch.
    """
    left_mtime = right_mtime = 0.0
    left_img = right_img = None

    while not stop_event.is_set():
        changed = False
        for path, side in ((left_path, "L"), (right_path, "R")):
            try:
                mt = os.stat(path).st_mtime
            except (FileNotFoundError, OSError):
                continue
            prev = left_mtime if side == "L" else right_mtime
            if mt != prev:
                img = cv2.imread(path)
                if img is not None:
                    if side == "L":
                        left_img, left_mtime = img, mt
                    else:
                        right_img, right_mtime = img, mt
                    changed = True

        if changed:
            panels = [p for p in (left_img, right_img) if p is not None]
            if len(panels) == 2:
                lh, rh = panels[0].shape[0], panels[1].shape[0]
                th = min(lh, rh)
                l = panels[0] if lh == th else cv2.resize(panels[0], (int(panels[0].shape[1] * th / lh), th))
                r = panels[1] if rh == th else cv2.resize(panels[1], (int(panels[1].shape[1] * th / rh), th))
                composite = cv2.hconcat([l, r])
            elif panels:
                composite = panels[0]
            else:
                time.sleep(0.02)
                continue
            write_frame(out_path, composite, quality)

        time.sleep(0.02)


def _run_unified(args):
    import threading as _threading

    tmpdir = tempfile.gettempdir()
    left_file = os.path.join(tmpdir, "yolov8s_frame.jpg")
    right_file = os.path.join(tmpdir, "yolov11s_frame.jpg")
    # Single unified frame file — HTTP server reads only this, never restarts.
    unified_frame = os.path.join(tmpdir, "yolo_unified_frame.jpg")

    mode_file = args.mode_file
    if not mode_file:
        mode_file = os.path.join(tmpdir, "yolo_demo_mode.txt")
    _write_mode_file(mode_file, "side-by-side")

    this_script = os.path.abspath(__file__)

    if args.video:
        source_args = ["--video", args.video]
    elif args.image:
        source_args = ["--image", args.image]
    else:
        source_args = ["--camera-index", str(args.camera_index)]

    common_worker_args = (
        source_args
        + [
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--jpeg-quality",
            str(args.jpeg_quality),
            "--_worker",
        ]
        + (["--pre-letterboxed"] if args.pre_letterboxed else [])
    )

    env_v8 = {**os.environ, "TT_VISIBLE_DEVICES": str(args.device_id)}
    env_v11 = {**os.environ, "TT_VISIBLE_DEVICES": str(args.device_id_v11s)}

    # State tracking
    current_mode = "side-by-side"
    http_proc: subprocess.Popen | None = None
    proc_v8: subprocess.Popen | None = None
    proc_v11: subprocess.Popen | None = None
    proc_sahi: subprocess.Popen | None = None
    compositor_stop: _threading.Event | None = None
    compositor_t: _threading.Thread | None = None
    stop_event = _threading.Event()

    def _shutdown(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _start_side_by_side():
        nonlocal proc_v8, proc_v11, compositor_stop, compositor_t

        print("[unified] Starting side-by-side mode...", flush=True)

        proc_v8 = subprocess.Popen(
            [sys.executable, "-u", this_script, "--model", "yolov8s", "--device-id", "0", "--_frame-file", left_file]
            + common_worker_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_v8,
        )
        print("[unified] Waiting for YOLOv8s to initialize...", flush=True)
        _wait_for_ready(proc_v8, "YOLOv8s")
        _threading.Thread(target=_pipe_output, args=(proc_v8, "YOLOv8s"), daemon=True).start()

        proc_v11 = subprocess.Popen(
            [sys.executable, "-u", this_script, "--model", "yolov11s", "--device-id", "0", "--_frame-file", right_file]
            + common_worker_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_v11,
        )
        print("[unified] Waiting for YOLOv11s to initialize...", flush=True)
        _wait_for_ready(proc_v11, "YOLOv11s")
        _threading.Thread(target=_pipe_output, args=(proc_v11, "YOLOv11s"), daemon=True).start()

        compositor_stop = _threading.Event()
        compositor_t = _threading.Thread(
            target=_compositor_thread,
            args=(left_file, right_file, unified_frame, compositor_stop, args.jpeg_quality),
            daemon=True,
        )
        compositor_t.start()

        print("[unified] Side-by-side running.", flush=True)

    def _stop_side_by_side():
        nonlocal proc_v8, proc_v11, compositor_stop, compositor_t
        print("[unified] Stopping side-by-side workers...", flush=True)
        if compositor_stop:
            compositor_stop.set()
        compositor_stop = None
        compositor_t = None
        _terminate_proc(proc_v8, "YOLOv8s")
        _terminate_proc(proc_v11, "YOLOv11s")
        proc_v8 = None
        proc_v11 = None
        time.sleep(2)  # let kernel release TT device locks before new workers start

    def _start_large_model():
        nonlocal proc_sahi

        print("[unified] Starting large-model mode (YOLOv8L SAHI)...", flush=True)

        _write_loading_placeholder(unified_frame, "Loading YOLOv8L (4-device SAHI)...")

        video_1280 = args.video_1280
        if not video_1280:
            base = Path(args.video) if args.video else None
            if base:
                candidate = base.parent / base.name.replace("_3840_2160_", "_1280x1280_")
                if candidate.exists():
                    video_1280 = str(candidate)
            if not video_1280:
                candidate2 = Path(__file__).resolve().parent / "sample_images" / "14025986_1280x1280_29fps.mp4"
                if candidate2.exists():
                    video_1280 = str(candidate2)
            if not video_1280:
                print("[unified] WARNING: no 1280x1280 video found, using original video", flush=True)
                video_1280 = args.video

        env_sahi = {**os.environ, "TT_VISIBLE_DEVICES": "0,1,2,3"}

        proc_sahi = subprocess.Popen(
            [
                sys.executable,
                "-u",
                _SAHI_WORKER_SCRIPT,
                "--video",
                video_1280,
                "--frame-file",
                unified_frame,
                "--conf",
                str(args.conf),
                "--iou",
                str(args.iou),
                "--jpeg-quality",
                str(args.jpeg_quality),
                "--device-ids",
                "0,1,2,3",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env_sahi,
        )
        print("[unified] Waiting for SAHI worker to initialize...", flush=True)
        _wait_for_ready(proc_sahi, "SAHI")
        _threading.Thread(target=_pipe_output, args=(proc_sahi, "SAHI"), daemon=True).start()

        print("[unified] Large-model running.", flush=True)

    def _stop_large_model():
        nonlocal proc_sahi
        print("[unified] Stopping large-model worker...", flush=True)
        _terminate_proc(proc_sahi, "SAHI worker")
        proc_sahi = None
        time.sleep(2)  # let kernel release TT device locks before new workers start

    # --- Start the HTTP server once; it reads unified_frame and never restarts ---
    _write_loading_placeholder(unified_frame, "Starting YOLO demo...")
    http_proc = start_server_single(args.host, args.port, unified_frame, mode_file=mode_file)
    time.sleep(0.5)
    print(f"[unified] MJPEG server on http://{args.host}:{args.port}/", flush=True)

    # --- Initial launch: side-by-side ---
    _start_side_by_side()
    current_mode = "side-by-side"

    print("[unified] Running... press Ctrl+C to stop.", flush=True)
    print("[unified] Use the browser UI button to switch modes.", flush=True)

    try:
        while not stop_event.is_set():
            new_mode = _read_mode_file(mode_file)
            if new_mode not in ("side-by-side", "large-model"):
                time.sleep(0.5)
                continue
            if new_mode != current_mode:
                print(f"[unified] Mode change: {current_mode} -> {new_mode}", flush=True)
                # Write "loading:<target>" so the UI knows we're transitioning
                _write_mode_file(mode_file, f"loading:{new_mode}")
                if new_mode == "large-model":
                    _stop_side_by_side()
                    _write_loading_placeholder(unified_frame, "Loading YOLOv8L (4-device SAHI)...")
                    _start_large_model()
                elif new_mode == "side-by-side":
                    _stop_large_model()
                    _write_loading_placeholder(unified_frame, "Switching to Side-by-Side...")
                    _start_side_by_side()
                current_mode = new_mode
                _write_mode_file(mode_file, current_mode)

            if current_mode == "side-by-side":
                for p, name in [(proc_v8, "YOLOv8s"), (proc_v11, "YOLOv11s")]:
                    if p and p.poll() is not None:
                        print(f"[unified] {name} exited unexpectedly (code {p.returncode})", flush=True)
                        stop_event.set()
            elif current_mode == "large-model":
                if proc_sahi and proc_sahi.poll() is not None:
                    print(f"[unified] SAHI worker exited unexpectedly " f"(code {proc_sahi.returncode})", flush=True)
                    stop_event.set()

            time.sleep(0.5)

    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        print("[unified] Shutting down everything...", flush=True)
        if compositor_stop:
            compositor_stop.set()
        for p, label in [(proc_v8, "YOLOv8s"), (proc_v11, "YOLOv11s"), (proc_sahi, "SAHI"), (http_proc, "HTTP")]:
            _terminate_proc(p, label)

        for f in (
            left_file,
            right_file,
            unified_frame,
            left_file + ".tmp",
            right_file + ".tmp",
            unified_frame + ".tmp",
        ):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.unlink(mode_file)
        except OSError:
            pass
        print("[unified] Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    if args._worker:
        _run_single(args)
    elif args.unified:
        _run_unified(args)
    elif args.dual:
        _run_dual(args)
    else:
        _run_single(args)


if __name__ == "__main__":
    main()
