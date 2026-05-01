#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""qb2 camera demo supervisor.

Browser captures 1920x1080 webcam → WebRTC sendonly track to this server →
this process letterboxes each frame to 640x640 (for v8s/v11s) AND 1280x1280
(for v8L SAHI) and writes both as atomic JPEG files. Worker subprocesses
read whichever JPEG matches their mode, run inference, write JSON detections
back. The supervisor polls the dets files, packs them into one
DataChannel message per tick, and forwards to the browser. The browser
overlays boxes on its own local camera preview (no return video track).

Modes
-----
- side-by-side: yolov8s on device 0 + yolov11s on device 1, both reading
  /tmp/qb2_cam_640.jpg. Each worker is a unified_video_demo.py invoked with
  --frame-input-file + --dets-out-file.
- large-model:  yolov8l SAHI on devices 0..3, reading /tmp/qb2_cam_1280.jpg.
  Single worker (_qb2_v8l_worker.py) that opens a 2x2 mesh and tiles the
  1280x1280 input into 4 x 640x640.

Only ONE mode is active at any moment. Switching kills the old workers,
waits for chip locks to release (~2 s), then spawns the new ones.

Endpoints
---------
GET  /                 -> demo/launch_qb2.html
GET  /assets/<file>    -> demo/assets/<file>
POST /offer            -> WebRTC SDP exchange (browser camera in, dets DataChannel out)
POST /api/mode         -> {"mode": "side-by-side" | "large-model"}
GET  /api/status       -> {state, mode, fps, ...}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = Path(__file__).resolve().parent / "demo"
LAUNCH_HTML = DEMO_DIR / "launch_qb2.html"
ASSETS_DIR = DEMO_DIR / "assets"
UNIFIED_DEMO_SCRIPT = Path(__file__).resolve().parent / "unified_video_demo.py"
SAHI_WORKER_SCRIPT = Path(__file__).resolve().parent / "_qb2_v8l_worker.py"
# This QB2 box (2x P300, 4 BH chips) reports CUSTOM cluster type, so
# tt-metal demands an explicit fabric mesh graph descriptor at process
# init. The descriptor must match the chips visible to the process —
# multi-chip workers need the 4-chip p300_x2 graph; per-chip workers
# (TT_VISIBLE_DEVICES=N) only see one chip and need a 1×1 BH graph or
# the topology solver fails ("target graph with N nodes cannot be placed
# in global graph with 1 nodes"). Honor TT_MESH_GRAPH_DESC_PATH from
# the env if the caller already exported one.
MGD_4CHIP = REPO_ROOT / "tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto"
# p150 and p100 are byte-identical (1×1 BH); using p150 to match the
# convention of other yolo demos on this box.
MGD_1CHIP = REPO_ROOT / "tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto"

TMP = tempfile.gettempdir()
CAM_640_FILE = os.path.join(TMP, "qb2_cam_640.jpg")
CAM_1280_FILE = os.path.join(TMP, "qb2_cam_1280.jpg")
# Per-stream JPEGs for video-mode (each worker reads its own source).
CAM_V8S_640_FILE = os.path.join(TMP, "qb2_cam_v8s_640.jpg")
CAM_V11S_640_FILE = os.path.join(TMP, "qb2_cam_v11s_640.jpg")
CAM_V8L_640_FILE = os.path.join(TMP, "qb2_cam_v8l_640.jpg")
CAM_V11L_640_FILE = os.path.join(TMP, "qb2_cam_v11l_640.jpg")
DETS_V8S_FILE = os.path.join(TMP, "qb2_dets_v8s.json")
DETS_V11S_FILE = os.path.join(TMP, "qb2_dets_v11s.json")
DETS_V8L_FILE = os.path.join(TMP, "qb2_dets_v8l.json")
DETS_V8L_640_FILE = os.path.join(TMP, "qb2_dets_v8l_640.json")
DETS_V11L_640_FILE = os.path.join(TMP, "qb2_dets_v11l_640.json")

LETTERBOX_640 = 640
# 1216 (not 1280) so the 4-tile SAHI grid has a 64-px overlap zone on each
# axis: build_overlap_grid shifts the last row/col inward to start=576 when
# tile_w + col_start would exceed frame_w. Same source coverage, dramatically
# better cross-tile dedup because adjacent tiles fully share the seam region.
LETTERBOX_V8L = 1216
JPEG_Q_INPUT = 75  # input frames don't need to be archival quality


# ---------------------------------------------------------------------------
# Letterbox + atomic JPEG write
# ---------------------------------------------------------------------------


def _letterbox(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    gain = min(target / h, target / w)
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    if (w, h) != (new_w, new_h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_top = (target - new_h) // 2
    pad_bot = target - new_h - pad_top
    pad_lt = (target - new_w) // 2
    pad_rt = target - new_w - pad_lt
    return cv2.copyMakeBorder(img, pad_top, pad_bot, pad_lt, pad_rt, cv2.BORDER_CONSTANT, value=(114, 114, 114))


def _write_jpeg(path: str, img: np.ndarray, quality: int = JPEG_Q_INPUT) -> None:
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


def _read_dets(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _cleanup_files(*paths: str) -> None:
    for p in paths:
        for f in (p, p + ".tmp", p + ".meta", p + ".meta.tmp"):
            try:
                os.unlink(f)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# WebRTC ingress: video track in, frames pushed into asyncio.Queue
# ---------------------------------------------------------------------------


class IngressHub:
    """Owns the latest WebRTC frame + the active DataChannel for dets out."""

    def __init__(self) -> None:
        self.latest_frame: np.ndarray | None = None
        self.frame_event = asyncio.Event()
        self.frame_id = 0
        self.dch = None  # set when browser opens 'dets' DataChannel
        self.pcs: set[RTCPeerConnection] = set()

    async def consume_track(self, track) -> None:
        n = 0
        while True:
            try:
                frame = await track.recv()
            except Exception as e:
                print(f"[qb2] track ended after {n} frames: {e}", flush=True)
                return
            try:
                bgr = frame.to_ndarray(format="bgr24")
            except Exception as e:
                print(f"[qb2] frame convert failed: {e}", flush=True)
                continue
            if bgr is None or bgr.size == 0:
                continue
            self.latest_frame = bgr
            self.frame_id += 1
            self.frame_event.set()
            n += 1
            if n == 1:
                print(f"[qb2] first webcam frame: {bgr.shape[1]}x{bgr.shape[0]}", flush=True)

    def send_dets(self, payload: dict) -> bool:
        if self.dch is None or self.dch.readyState != "open":
            return False
        try:
            self.dch.send(json.dumps(payload, separators=(",", ":")))
            return True
        except Exception as e:
            print(f"[qb2] dch send error: {e}", flush=True)
            return False


# ---------------------------------------------------------------------------
# Frame writer task: letterbox latest frame to both sizes, write JPEGs
# ---------------------------------------------------------------------------


async def frame_writer_loop(hub: IngressHub, stop: asyncio.Event) -> None:
    """Pull latest frame, letterbox to 640 + 1280, write atomic JPEGs.

    Runs at native frame rate (set by the WebRTC track recv cadence). Both
    files are always written, even when the worker for one size isn't
    running — keeps mode switches snappy (the new worker has fresh input
    waiting the moment it's spawned).
    """
    print(f"[qb2] frame writer: -> {CAM_640_FILE} + {CAM_1280_FILE}", flush=True)
    while not stop.is_set():
        try:
            await asyncio.wait_for(hub.frame_event.wait(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        hub.frame_event.clear()
        bgr = hub.latest_frame
        if bgr is None:
            continue
        # Cheap on CPU: 1920x1080 → 640x640 + 1280x1280 letterbox is < 5 ms
        # combined; well under the 33 ms 30 fps budget.
        lb640 = _letterbox(bgr, LETTERBOX_640)
        lb_v8l = _letterbox(bgr, LETTERBOX_V8L)
        await asyncio.to_thread(_write_jpeg, CAM_640_FILE, lb640, JPEG_Q_INPUT)
        await asyncio.to_thread(_write_jpeg, CAM_1280_FILE, lb_v8l, JPEG_Q_INPUT)


# ---------------------------------------------------------------------------
# Video player loop: read a video file at its native FPS, write per-frame
# JPEGs to ``output_path``. Loops the source on EOF. Used in --video-mode in
# place of frame_writer_loop (no WebRTC ingress in that case).
# ---------------------------------------------------------------------------


def _write_frame_meta(meta_path: str, frame_idx: int, n_frames: int, fps: float) -> None:
    """Atomically publish the source frame index that matches the current
    JPEG. Workers read this after the JPEG to tag their dets with the
    source frame id; the browser uses that id (mod n_frames) to align dets
    with its <video> element's currentTime."""
    tmp = meta_path + ".tmp"
    try:
        with open(tmp, "w") as f:
            f.write(f"{frame_idx}\n{n_frames}\n{fps:.3f}\n")
        os.replace(tmp, meta_path)
    except OSError:
        pass


async def video_player_loop(
    label: str,
    src_path: str,
    output_path: str,
    target_size: int,
    stop: asyncio.Event,
) -> None:
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0.0 or fps > 240.0:
        fps = 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_dt = 1.0 / fps
    print(
        f"[qb2] video player [{label}]: {src_path} @ {fps:.1f} fps " f"({n_frames} frames) -> {output_path}",
        flush=True,
    )
    next_t = time.time()
    frame_idx = 0
    meta_path = output_path + ".meta"
    while not stop.is_set():
        ok, bgr = cap.read()
        if not ok or bgr is None:
            cap.release()
            cap = cv2.VideoCapture(src_path)
            frame_idx = 0
            continue
        if bgr.shape[0] != target_size or bgr.shape[1] != target_size:
            bgr = _letterbox(bgr, target_size)
        # Meta first, JPEG second: the worker triggers on JPEG mtime
        # change, so the meta written here is at-least-as-fresh as the
        # JPEG the worker is about to read.
        await asyncio.to_thread(_write_frame_meta, meta_path, frame_idx, n_frames, fps)
        await asyncio.to_thread(_write_jpeg, output_path, bgr, JPEG_Q_INPUT)
        frame_idx += 1
        next_t += frame_dt
        delay = next_t - time.time()
        if delay > 0:
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
        else:
            # Fell behind — reset cadence so we don't spin without sleeping.
            next_t = time.time()
    cap.release()


# ---------------------------------------------------------------------------
# Detection forwarder: poll worker dets files, send via DataChannel
# ---------------------------------------------------------------------------


async def dets_forward_loop(hub: IngressHub, mode_state: dict, stop: asyncio.Event) -> None:
    """Poll dets files for the active mode, batch into one DataChannel msg.

    Sends at ~30 Hz max (we don't need finer than the worker write cadence,
    and over-sending overwhelms the data channel buffer).
    """
    last_sent_id_v8s = -1
    last_sent_id_v11s = -1
    last_sent_id_v8l = -1
    last_sent_id_v8l_640 = -1
    last_sent_id_v11l_640 = -1

    def _per_model(d: dict) -> dict:
        # Pass through the source frame metadata so the browser can align
        # dets to its <video> element's currentTime. src_frame_id is the
        # video frame index (mod src_n_frames) the worker processed; if
        # the worker didn't emit it, browser falls back to "latest wins".
        out = {"fps": d.get("fps", 0), "dets": d.get("dets", [])}
        for k in ("src_frame_id", "src_n_frames", "src_fps"):
            if k in d:
                out[k] = d[k]
        return out

    while not stop.is_set():
        await asyncio.sleep(1 / 30)
        mode = mode_state.get("mode", "idle")
        if mode == "side-by-side":
            v8s = _read_dets(DETS_V8S_FILE)
            v11s = _read_dets(DETS_V11S_FILE)
            v8l_640 = _read_dets(DETS_V8L_640_FILE)
            v11l_640 = _read_dets(DETS_V11L_640_FILE)
            if v8s is None and v11s is None and v8l_640 is None and v11l_640 is None:
                continue
            payload: dict = {"k": "dets", "mode": mode, "input_res": LETTERBOX_640}
            if v8s and v8s.get("frame_id", -1) != last_sent_id_v8s:
                payload["v8s"] = _per_model(v8s)
                last_sent_id_v8s = v8s.get("frame_id", -1)
            if v11s and v11s.get("frame_id", -1) != last_sent_id_v11s:
                payload["v11s"] = _per_model(v11s)
                last_sent_id_v11s = v11s.get("frame_id", -1)
            if v8l_640 and v8l_640.get("frame_id", -1) != last_sent_id_v8l_640:
                payload["v8l_640"] = _per_model(v8l_640)
                last_sent_id_v8l_640 = v8l_640.get("frame_id", -1)
            if v11l_640 and v11l_640.get("frame_id", -1) != last_sent_id_v11l_640:
                payload["v11l_640"] = _per_model(v11l_640)
                last_sent_id_v11l_640 = v11l_640.get("frame_id", -1)
            if any(k in payload for k in ("v8s", "v11s", "v8l_640", "v11l_640")):
                hub.send_dets(payload)
        elif mode == "large-model":
            v8l = _read_dets(DETS_V8L_FILE)
            if v8l is None:
                continue
            if v8l.get("frame_id", -1) == last_sent_id_v8l:
                continue
            last_sent_id_v8l = v8l.get("frame_id", -1)
            payload = {
                "k": "dets",
                "mode": mode,
                "input_res": LETTERBOX_V8L,
                "v8l": _per_model(v8l),
            }
            hub.send_dets(payload)


# ---------------------------------------------------------------------------
# Mode controller: spawn/kill worker subprocesses
# ---------------------------------------------------------------------------


class ModeController:
    def __init__(self, conf: float, iou: float, video_mode: bool = False) -> None:
        self.conf = conf
        self.iou = iou
        self.video_mode = video_mode
        self.mode = "idle"
        self.target_mode: str | None = None
        self.workers: list[subprocess.Popen] = []
        self._lock = asyncio.Lock()
        self.last_error: str | None = None

    def _python(self) -> str:
        candidate = REPO_ROOT / "python_env" / "bin" / "python3"
        return str(candidate) if candidate.exists() else sys.executable

    def _env(self, visible_devices: str | None = None) -> dict:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{REPO_ROOT}:{existing}" if existing else str(REPO_ROOT)
        # Pick the MGD that matches what the worker will see: the SAHI
        # large-model path passes visible_devices=None (sees all 4 chips →
        # p300_x2); side-by-side workers pass a single device id (sees 1 chip
        # → p100, the 1×1 BH descriptor). Caller-supplied env wins.
        if not env.get("TT_MESH_GRAPH_DESC_PATH"):
            n_visible = len(visible_devices.split(",")) if visible_devices else 4
            mgd = MGD_1CHIP if n_visible == 1 else MGD_4CHIP
            if mgd.is_file():
                env["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)
        if visible_devices is not None:
            env["TT_VISIBLE_DEVICES"] = visible_devices
        return env

    def _spawn_side_by_side(self) -> None:
        # In video-mode each worker polls its own per-stream JPEG and we run
        # FOUR models (v8s + v11s + v8l_640 + v11l_640) on streams 1..4 across
        # chips 0..3. In webcam mode we keep the original 2-worker layout
        # (v8s + v11s, both reading the single supervisor-letterboxed JPEG).
        v8s_input = CAM_V8S_640_FILE if self.video_mode else CAM_640_FILE
        v11s_input = CAM_V11S_640_FILE if self.video_mode else CAM_640_FILE

        def _worker_argv(model: str, frame_input: str, dets_out: str) -> list[str]:
            return [
                self._python(),
                "-u",
                str(UNIFIED_DEMO_SCRIPT),
                "--_worker",
                "--model",
                model,
                # device-id is logical inside the TT_VISIBLE_DEVICES remap, so
                # all workers pass 0 — the env restricts them to one device.
                "--device-id",
                "0",
                "--frame-input-file",
                frame_input,
                "--dets-out-file",
                dets_out,
                "--conf",
                str(self.conf),
                "--iou",
                str(self.iou),
            ]

        v8s_cmd = _worker_argv("yolov8s", v8s_input, DETS_V8S_FILE)
        v11s_cmd = _worker_argv("yolov11s", v11s_input, DETS_V11S_FILE)
        print(f"[qb2] spawn yolov8s (TT_VISIBLE_DEVICES=0): {' '.join(v8s_cmd)}", flush=True)
        p_v8s = subprocess.Popen(v8s_cmd, env=self._env("0"), start_new_session=True)
        print(f"[qb2] spawn yolov11s (TT_VISIBLE_DEVICES=1): {' '.join(v11s_cmd)}", flush=True)
        p_v11s = subprocess.Popen(v11s_cmd, env=self._env("1"), start_new_session=True)
        workers = [p_v8s, p_v11s]

        if self.video_mode:
            v8l_640_cmd = _worker_argv("yolov8l_640", CAM_V8L_640_FILE, DETS_V8L_640_FILE)
            v11l_640_cmd = _worker_argv("yolov11l", CAM_V11L_640_FILE, DETS_V11L_640_FILE)
            print(f"[qb2] spawn yolov8l_640 (TT_VISIBLE_DEVICES=2): {' '.join(v8l_640_cmd)}", flush=True)
            p_v8l_640 = subprocess.Popen(v8l_640_cmd, env=self._env("2"), start_new_session=True)
            print(f"[qb2] spawn yolov11l (TT_VISIBLE_DEVICES=3): {' '.join(v11l_640_cmd)}", flush=True)
            p_v11l_640 = subprocess.Popen(v11l_640_cmd, env=self._env("3"), start_new_session=True)
            workers += [p_v8l_640, p_v11l_640]

        self.workers = workers

    def _spawn_large_model(self) -> None:
        # SAHI worker opens a 2x2 mesh; do NOT set TT_VISIBLE_DEVICES so
        # open_mesh_device picks 4 contiguous devices (0..3) from the system.
        # Merge params mirror the supervisor demo's camera transport
        # (demo_supervisor.py): WBF + IoS @ 0.6 threshold, plus seam-merge
        # to catch same-class boxes split across the x=640/y=640 boundaries.
        # conf=0.7 matches the camera-mode tuning (it's also above the
        # bfloat8 noise floor of 0.50, so floor noise is fully suppressed).
        cmd = [
            self._python(),
            "-u",
            str(SAHI_WORKER_SCRIPT),
            "--frame-input-file",
            CAM_1280_FILE,
            "--dets-out-file",
            DETS_V8L_FILE,
            "--conf",
            "0.7",
            "--iou",
            "0.45",
            "--merge-mode",
            "wbf",
            "--merge-match",
            "ios",
            # Aggressive merge for hard-cut 2×2 seams: low IoS (cross-tile
            # WBF fuses on weaker matches), wide seam tolerance, and a
            # near-zero perpendicular-overlap floor. --class-agnostic lets
            # the cross-tile WBF dedup boxes that picked different class
            # labels in adjacent tiles (typical for partially-occluded
            # objects straddling a seam).
            "--merge-threshold",
            "0.35",
            "--class-agnostic",
            "--seam-merge",
            "--seam-tol",
            "240",
            "--seam-perp-overlap-frac",
            "0.02",
        ]
        print(f"[qb2] spawn yolov8l-sahi: {' '.join(cmd)}", flush=True)
        p = subprocess.Popen(cmd, env=self._env(), start_new_session=True)
        self.workers = [p]

    async def _stop_workers(self, timeout: float = 5.0) -> None:
        if not self.workers:
            return
        print(f"[qb2] stopping {len(self.workers)} worker(s)...", flush=True)
        for p in self.workers:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        p.terminate()
                    except OSError:
                        pass
        t0 = time.time()
        while time.time() - t0 < timeout:
            if all(p.poll() is not None for p in self.workers):
                break
            await asyncio.sleep(0.1)
        for p in self.workers:
            if p.poll() is None:
                print(f"[qb2] worker pid={p.pid} didn't stop on SIGTERM — SIGKILL", flush=True)
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        p.kill()
                    except OSError:
                        pass
        # Reap
        for p in self.workers:
            try:
                p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                pass
        self.workers = []
        # Clean stale dets so the next mode doesn't see ghost detections
        _cleanup_files(
            DETS_V8S_FILE,
            DETS_V11S_FILE,
            DETS_V8L_FILE,
            DETS_V8L_640_FILE,
            DETS_V11L_640_FILE,
        )
        # tt-metal needs ~2 s for chip locks to release after worker exit.
        await asyncio.sleep(2.0)

    async def set_mode(self, target: str) -> dict:
        if target not in ("side-by-side", "large-model", "idle"):
            return {"ok": False, "error": f"unknown mode {target!r}"}
        async with self._lock:
            workers_alive = bool(self.workers) and all(p.poll() is None for p in self.workers)
            if target == self.mode and workers_alive:
                return {"ok": True, "mode": self.mode, "note": "no-op"}
            self.target_mode = target
            self.last_error = None
            await self._stop_workers()
            self.mode = "idle"
            if target == "idle":
                return {"ok": True, "mode": self.mode}
            try:
                if target == "side-by-side":
                    self._spawn_side_by_side()
                else:
                    self._spawn_large_model()
                self.mode = target
                return {"ok": True, "mode": self.mode}
            except Exception as e:  # noqa: BLE001
                self.last_error = repr(e)
                print(f"[qb2] spawn failed: {e!r}", flush=True)
                return {"ok": False, "mode": self.mode, "error": self.last_error}
            finally:
                self.target_mode = None

    def status(self) -> dict:
        alive = [p.poll() is None for p in self.workers]
        return {
            "mode": self.mode,
            "target_mode": self.target_mode,
            "workers_alive": sum(alive),
            "workers_total": len(self.workers),
            "last_error": self.last_error,
        }


# ---------------------------------------------------------------------------
# HTTP handlers
# ---------------------------------------------------------------------------


async def index_handler(_req: web.Request) -> web.Response:
    if not LAUNCH_HTML.exists():
        return web.Response(status=404, text=f"missing: {LAUNCH_HTML}\n")
    return web.Response(body=LAUNCH_HTML.read_bytes(), content_type="text/html", charset="utf-8")


async def asset_handler(req: web.Request) -> web.Response:
    name = req.match_info.get("name", "")
    safe = (ASSETS_DIR / name).resolve()
    if not safe.is_file() or ASSETS_DIR.resolve() not in safe.parents:
        return web.Response(status=404, text="not found\n")
    return web.FileResponse(safe)


async def video_handler(req: web.Request) -> web.Response:
    """Serve one of the configured video files for direct browser playback."""
    name = req.match_info.get("name", "")
    cfg: dict = req.app["cfg"]
    path_map: dict = cfg.get("video_paths", {})
    src = path_map.get(name)
    if not src:
        return web.Response(status=404, text="not found\n")
    p = Path(src)
    if not p.is_file():
        return web.Response(status=404, text=f"missing: {src}\n")
    return web.FileResponse(p)


async def api_config_handler(req: web.Request) -> web.Response:
    cfg: dict = req.app["cfg"]
    return web.json_response(
        {
            "video_mode": bool(cfg.get("video_mode", False)),
            "video_urls": cfg.get("video_urls", {}),
        }
    )


async def offer_handler(req: web.Request) -> web.Response:
    body = await req.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])
    pc = RTCPeerConnection(
        configuration=RTCConfiguration(iceServers=[RTCIceServer(urls="stun:stun.l.google.com:19302")])
    )
    hub: IngressHub = req.app["hub"]
    hub.pcs.add(pc)

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        print(f"[qb2] pc state={pc.connectionState}", flush=True)
        if pc.connectionState in ("failed", "closed"):
            try:
                await pc.close()
            except Exception:
                pass
            hub.pcs.discard(pc)

    @pc.on("track")
    def _on_track(track) -> None:
        if track.kind == "video":
            print(f"[qb2] video track id={track.id}", flush=True)
            asyncio.create_task(hub.consume_track(track))

    @pc.on("datachannel")
    def _on_dc(channel) -> None:
        print(f"[qb2] datachannel: {channel.label}", flush=True)
        if channel.label == "dets":
            hub.dch = channel

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def api_mode_handler(req: web.Request) -> web.Response:
    try:
        body = await req.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    mode = str(body.get("mode") or "").strip().lower()
    ctrl: ModeController = req.app["ctrl"]
    result = await ctrl.set_mode(mode)
    return web.json_response(result, status=200 if result.get("ok") else 400)


async def api_status_handler(req: web.Request) -> web.Response:
    ctrl: ModeController = req.app["ctrl"]
    hub: IngressHub = req.app["hub"]
    return web.json_response(
        {
            **ctrl.status(),
            "ingress_frames": hub.frame_id,
            "dch_open": hub.dch is not None and hub.dch.readyState == "open",
        }
    )


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


async def on_startup(app: web.Application) -> None:
    hub: IngressHub = app["hub"]
    ctrl: ModeController = app["ctrl"]
    stop: asyncio.Event = app["stop"]
    cfg: dict = app["cfg"]
    mode_state = {"mode": ctrl.mode}

    async def _track_mode() -> None:
        while not stop.is_set():
            mode_state["mode"] = ctrl.mode
            await asyncio.sleep(0.2)

    if cfg.get("video_mode"):
        # Video mode: independent player loops, no WebRTC frame ingress. Each
        # loop is independent so a mode switch finds fresh input ready (the
        # SAHI 1216-letterbox is also fed continuously, even while side-by-
        # side is active — keeps switches snappy without stutter).
        paths = cfg["video_paths"]
        app["task_player_v8s"] = asyncio.create_task(
            video_player_loop("v8s", paths["v8s"], CAM_V8S_640_FILE, LETTERBOX_640, stop)
        )
        app["task_player_v11s"] = asyncio.create_task(
            video_player_loop("v11s", paths["v11s"], CAM_V11S_640_FILE, LETTERBOX_640, stop)
        )
        if paths.get("v8l_640"):
            app["task_player_v8l_640"] = asyncio.create_task(
                video_player_loop("v8l_640", paths["v8l_640"], CAM_V8L_640_FILE, LETTERBOX_640, stop)
            )
        if paths.get("v11l_640"):
            app["task_player_v11l_640"] = asyncio.create_task(
                video_player_loop("v11l_640", paths["v11l_640"], CAM_V11L_640_FILE, LETTERBOX_640, stop)
            )
        if paths.get("v8l"):
            app["task_player_v8l"] = asyncio.create_task(
                video_player_loop("v8l", paths["v8l"], CAM_1280_FILE, LETTERBOX_V8L, stop)
            )
    else:
        app["task_writer"] = asyncio.create_task(frame_writer_loop(hub, stop))
    app["task_dets"] = asyncio.create_task(dets_forward_loop(hub, mode_state, stop))
    app["task_track_mode"] = asyncio.create_task(_track_mode())


async def on_cleanup(app: web.Application) -> None:
    stop: asyncio.Event = app["stop"]
    stop.set()
    ctrl: ModeController = app["ctrl"]
    hub: IngressHub = app["hub"]
    await ctrl._stop_workers()
    task_keys = (
        "task_writer",
        "task_player_v8s",
        "task_player_v11s",
        "task_player_v8l",
        "task_player_v8l_640",
        "task_player_v11l_640",
        "task_dets",
        "task_track_mode",
    )
    for k in task_keys:
        t = app.get(k)
        if t and not t.done():
            t.cancel()
    await asyncio.gather(*(pc.close() for pc in list(hub.pcs)), return_exceptions=True)
    hub.pcs.clear()
    _cleanup_files(
        CAM_640_FILE,
        CAM_1280_FILE,
        CAM_V8S_640_FILE,
        CAM_V11S_640_FILE,
        CAM_V8L_640_FILE,
        CAM_V11L_640_FILE,
        DETS_V8S_FILE,
        DETS_V11S_FILE,
        DETS_V8L_FILE,
        DETS_V8L_640_FILE,
        DETS_V11L_640_FILE,
    )


def build_app(
    conf: float,
    iou: float,
    *,
    video_mode: bool = False,
    video_v8s: str | None = None,
    video_v11s: str | None = None,
    video_v8l: str | None = None,
    video_v8l_640: str | None = None,
    video_v11l_640: str | None = None,
) -> web.Application:
    app = web.Application(client_max_size=64 * 1024)
    app["hub"] = IngressHub()
    app["ctrl"] = ModeController(conf=conf, iou=iou, video_mode=video_mode)
    app["stop"] = asyncio.Event()
    app["cfg"] = {
        "video_mode": video_mode,
        "video_paths": (
            {
                "v8s": video_v8s,
                "v11s": video_v11s,
                "v8l_640": video_v8l_640,
                "v11l_640": video_v11l_640,
                "v8l": video_v8l,
            }
            if video_mode
            else {}
        ),
        # Browser-facing URLs the launch page uses for <video src=...>.
        "video_urls": (
            {
                "v8s": "/video/v8s",
                "v11s": "/video/v11s",
                "v8l_640": "/video/v8l_640",
                "v11l_640": "/video/v11l_640",
                "v8l": "/video/v8l",
            }
            if video_mode
            else {}
        ),
    }
    app.router.add_get("/", index_handler)
    app.router.add_get("/assets/{name:[A-Za-z0-9_.\\-]+}", asset_handler)
    app.router.add_get("/video/{name:[A-Za-z0-9_.\\-]+}", video_handler)
    app.router.add_get("/api/config", api_config_handler)
    app.router.add_post("/offer", offer_handler)
    app.router.add_post("/api/mode", api_mode_handler)
    app.router.add_get("/api/status", api_status_handler)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="qb2 unified-camera demo supervisor.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument(
        "--video-mode",
        action="store_true",
        help="Replace WebRTC webcam ingress with two looping video files (one per model).",
    )
    default_v8s = str(REPO_ROOT / "models/demos/yolo_eval/sample_images/multi_stream_1280x720/1.mp4")
    default_v11s = str(REPO_ROOT / "models/demos/yolo_eval/sample_images/multi_stream_1280x720/2.mp4")
    default_v8l_640 = str(REPO_ROOT / "models/demos/yolo_eval/sample_images/multi_stream_1280x720/3.mp4")
    default_v11l_640 = str(REPO_ROOT / "models/demos/yolo_eval/sample_images/multi_stream_1280x720/6.mp4")
    default_v8l = str(REPO_ROOT / "models/demos/yolo_eval/sample_images/multi_stream_1280x720/7.mp4")
    p.add_argument("--video-v8s", default=default_v8s, help="Video file for YOLOv8s in --video-mode.")
    p.add_argument("--video-v11s", default=default_v11s, help="Video file for YOLOv11s in --video-mode.")
    p.add_argument(
        "--video-v8l-640",
        default=default_v8l_640,
        help="Video file for YOLOv8L 640x640 (chip 2) in --video-mode side-by-side.",
    )
    p.add_argument(
        "--video-v11l-640",
        default=default_v11l_640,
        help="Video file for YOLOv11L 640x640 (chip 3) in --video-mode side-by-side.",
    )
    p.add_argument(
        "--video-v8l",
        default=default_v8l,
        help="Video file for YOLOv8L SAHI (large-model) in --video-mode (uses the 1280x1280 source).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not LAUNCH_HTML.exists():
        print(f"[qb2] WARN: {LAUNCH_HTML} does not exist yet — / will 404 until it's created.", file=sys.stderr)
    if args.video_mode:
        for label, path in (
            ("v8s", args.video_v8s),
            ("v11s", args.video_v11s),
            ("v8l_640", args.video_v8l_640),
            ("v11l_640", args.video_v11l_640),
            ("v8l", args.video_v8l),
        ):
            if not Path(path).is_file():
                print(f"[qb2] FATAL: --video-{label.replace('_', '-')} not found: {path}", file=sys.stderr)
                sys.exit(2)
        print(
            "[qb2] video mode: "
            f"v8s <- {args.video_v8s}; v11s <- {args.video_v11s}; "
            f"v8l_640 <- {args.video_v8l_640}; v11l_640 <- {args.video_v11l_640}; "
            f"v8l <- {args.video_v8l}",
            flush=True,
        )
    app = build_app(
        args.conf,
        args.iou,
        video_mode=args.video_mode,
        video_v8s=args.video_v8s,
        video_v11s=args.video_v11s,
        video_v8l=args.video_v8l,
        video_v8l_640=args.video_v8l_640,
        video_v11l_640=args.video_v11l_640,
    )
    print(f"[qb2] launch page: http://{args.host}:{args.port}/", flush=True)
    web.run_app(app, host=args.host, port=args.port, print=None, access_log=None)


if __name__ == "__main__":
    main()
