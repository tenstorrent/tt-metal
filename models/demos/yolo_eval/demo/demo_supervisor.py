#!/usr/bin/env python3
"""Demo supervisor for the YOLOv8L SAHI-640 pipeline.

Owns ONE long-running child process (`yolov8l_sahi_640_pipelined.py`) and lets
the launch page swap its source/transport at runtime by killing+respawning the
child with the right CLI flags. Also serves the launch page + Tenstorrent logo
on the same origin so the browser's WebRTC offer can post to the pipeline at a
sibling URL without any CORS gymnastics.

Endpoints
---------
GET  /                       -> demo/launch.html
GET  /assets/<file>          -> demo/assets/<file>
GET  /api/status             -> {"transport","ready","frame_w","frame_h","n_tiles","uptime_s","pipeline_url"}
POST /api/source             -> body {"transport": "video"|"camera", "width"?: int, "height"?: int}
                                Restarts the pipeline with the matching flags.
                                Returns {"ok": true, "url": "http://<host>:9090/", "kind": "webrtc",
                                         "transport": ..., "frame_w": ..., "frame_h": ...,
                                         "tiles_x": ..., "tiles_y": ..., "n_tiles": ...}.

Transports
----------
video  -> --source file --serve --serve-codec h264
              --stream-bitrate <bitrate> --stream-keyint <keyint>
              --frame-width / --frame-height taken from the file via cv2.VideoCapture.
              The pipeline spawns _h264_server on the pipeline port; this process
              reverse-proxies `GET /stream` so the launch page stays same-origin.
camera -> --source webrtc --webrtc-delivery data_only
              --frame-width / --frame-height passed by the browser. The pipeline
              spawns webrtc_ingress on the pipeline port; `/stream` is not served
              (the browser draws boxes over its own local camera preview).

Sub-4K policy: dimensions are passed through verbatim. The pipeline's
build_overlap_grid + compute_mesh_shape pick `ceil(W/640) x ceil(H/640)` tiles
and the smallest sub-mesh that fits.
"""
from __future__ import annotations

import argparse
import asyncio
import ctypes
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from aiohttp import ClientSession, ClientTimeout, web


def _child_preexec() -> None:
    # Linux only. Ask the kernel to SIGTERM this child the instant the
    # supervisor process dies -- including on SIGKILL/crash where our
    # `on_cleanup` hook never runs. `start_new_session=True` below already
    # isolates the pipeline's process group for programmatic restart; this
    # closes the orphan-on-crash gap so `pgrep yolov8l_sahi` never outlives
    # the supervisor.
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM, 0, 0, 0)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


TILE = 640
DEMO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DEMO_DIR / "assets"
LAUNCH_HTML = DEMO_DIR / "launch.html"
PIPELINE_SCRIPT = DEMO_DIR.parent / "yolov8l_sahi_640_pipelined.py"
MULTI_PIPELINE_SCRIPT = DEMO_DIR.parent / "yolov8l_sahi_5frame_pipelined.py"
TT_METAL_DIR = Path(__file__).resolve().parents[4]

# Multi-stream topology (mirrors yolov8l_sahi_5frame_pipelined.py constants).
MULTI_N_STREAMS = 8
MULTI_TILES_PER_STREAM = 4
MULTI_STREAM_W = 1280
MULTI_STREAM_H = 1280


def _grid_for(w: int, h: int) -> dict:
    cols = max(1, math.ceil(w / TILE))
    rows = max(1, math.ceil(h / TILE))
    return {"tiles_x": cols, "tiles_y": rows, "n_tiles": cols * rows}


def _peek_video_meta(path: str) -> tuple[int, int, float, int]:
    """Return (width, height, fps, n_frames) for the source file.

    fps and n_frames let the browser key its detections buffer by
    `frame_id % n_frames` so a long-running pipeline (which loops its
    `FrameSource` internally and keeps `fc` monotonic) still lines up
    with the browser's `<video loop>` element — which resets its clock
    every cycle.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return (3840, 2160, 30.0, 0)
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        w, h = 3840, 2160
    if fps <= 0.0:
        fps = 30.0
    return (w, h, fps, max(0, n))


def _connect_host(host: str) -> str:
    """0.0.0.0 / :: are bind-only sentinels; treat them as localhost when CONNECTING."""
    if not host or host in ("0.0.0.0", "::", "::0"):
        return "127.0.0.1"
    return host


def _port_busy(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.25)
            return s.connect_ex((host or "127.0.0.1", port)) == 0
    except OSError:
        return False


async def _await_port_free(host: str, port: int, budget_s: float = 8.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < budget_s:
        if not _port_busy(host, port):
            return True
        await asyncio.sleep(0.2)
    return not _port_busy(host, port)


async def _await_pipeline_ready(host: str, port: int, budget_s: float = 90.0) -> bool:
    """The pipeline serves an HTML page on / once aiortc is up. We poll it.
    Model warmup + trace capture can easily take 30-60 s on first start.
    """
    url = f"http://{_connect_host(host)}:{port}/"
    t0 = time.time()
    async with ClientSession(timeout=ClientTimeout(total=2.0)) as sess:
        while time.time() - t0 < budget_s:
            try:
                async with sess.get(url) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


class Supervisor:
    def __init__(
        self,
        *,
        video_path: str,
        video_multi_path: str,
        pipeline_host: str,
        pipeline_port: int,
        default_bitrate: str,
        default_keyint: int,
        extra_pipeline_args: list[str],
    ) -> None:
        self.video_path = video_path
        self.video_multi_path = video_multi_path
        self.pipeline_host = pipeline_host
        self.pipeline_port = pipeline_port
        self.default_bitrate = default_bitrate
        self.default_keyint = default_keyint
        self.extra_pipeline_args = list(extra_pipeline_args)
        self.proc: Optional[subprocess.Popen] = None
        self.transport: str = "idle"
        self.frame_w: int = 0
        self.frame_h: int = 0
        self.fps: float = 0.0
        self.n_frames: int = 0
        self.start_t: float = 0.0
        self.tiles: dict = {"tiles_x": 0, "tiles_y": 0, "n_tiles": 0}
        # Multi-stream fields -- zero in single-stream transports.
        self.n_streams: int = 0
        self.tiles_per_stream: int = 0
        self._lock = asyncio.Lock()
        # State machine for the launch page to poll.  Transitions:
        # idle -> switching -> ready | error
        # ready -> switching (mode change) -> ready | error
        # error -> recovering (auto on init failure / manual /api/reset) -> idle
        self.state: str = "idle"
        self.last_error: Optional[str] = None
        self.pending_transport: Optional[str] = None
        self._restart_task: Optional[asyncio.Task] = None
        self._recover_task: Optional[asyncio.Task] = None
        self._consecutive_failures: int = 0
        # Bookkeeping for retry-after-recovery: when an init failure auto-
        # triggers chip recovery, _recover_chips uses this to re-issue the
        # original transport request once the reset finishes.  Cleared once
        # consumed (single retry only — we don't want infinite loops).
        self._pending_retry: Optional[tuple] = None

    @staticmethod
    def _bitrate_bps(val: str) -> str:
        """Convert '4M' / '4000k' / int-string to a bps integer string.

        The pipeline's --stream-bitrate expects bps. The supervisor CLI takes
        the human-readable form for backwards compatibility.
        """
        s = str(val).strip().lower()
        if not s:
            return "4000000"
        if s.endswith("m"):
            return str(int(float(s[:-1]) * 1_000_000))
        if s.endswith("k"):
            return str(int(float(s[:-1]) * 1_000))
        try:
            return str(int(float(s)))
        except ValueError:
            return "4000000"

    def _python(self) -> str:
        candidate = TT_METAL_DIR / "python_env/bin/python3"
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def _env(self) -> dict:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{TT_METAL_DIR}:{existing}" if existing else str(TT_METAL_DIR)
        # OpenCV's default FFmpeg decode is single-threaded; 4K H.264 caps
        # at ~30-45 FPS. 4 threads lets the decoder keep up with the 70+ FPS
        # pipeline (settings get picked up when cv2.VideoCapture opens).
        env.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;4")
        return env

    def _build_argv(self, transport: str, width: int, height: int) -> list[str]:
        # Multi-stream mode uses a different script; keep the same argv
        # shape (host/port/frame-width/frame-height) so the supervisor
        # readiness poll is unchanged.
        script = MULTI_PIPELINE_SCRIPT if transport == "multi" else PIPELINE_SCRIPT
        argv = [
            self._python(),
            str(script),
            "--host",
            self.pipeline_host,
            "--port",
            str(self.pipeline_port),
            "--frame-width",
            str(int(width)),
            "--frame-height",
            str(int(height)),
        ]
        if transport == "video":
            # Split delivery: browser plays the source MP4 natively via
            # byte-range GETs, and overlays boxes from /dets SSE. Pipeline
            # runs inference at full hardware speed (~70 fps) decoupled from
            # browser playback (30 fps native).
            argv += [
                "--source",
                "file",
                "--input",
                self.video_path,
                "--serve",
                "--serve-split",
                "--conf",
                "0.75",
                "--merge-mode",
                "nms",
                "--merge-match",
                "iou",
                "--class-agnostic",
                "--merge-threshold",
                "0.75",
            ]
        elif transport == "camera":
            # Live webcam: class-specific merge (NOT class-agnostic) so small
            # held objects (bottle/cup/phone) don't get absorbed into the
            # enclosing person's box by IoS merging. conf=0.5 catches those
            # objects at typical webcam confidence; per-tile NMS IoU=0.45 and
            # cross-tile merge IoU=0.45 keep same-class duplicates in check.
            argv += [
                "--source",
                "webrtc",
                "--webrtc-delivery",
                "data_only",
                "--conf",
                "0.7",
                "--iou",
                "0.45",
                "--merge-mode",
                "wbf",
                "--merge-match",
                "ios",
                "--merge-threshold",
                "0.6",
                "--seam-merge",
                "--seam-tol",
                "100",
            ]
        elif transport == "multi":
            # Multi-stream demo: 8 parallel 1280x1280 streams x 4 tiles each
            # = 32 tiles on the full 8x4 mesh. Split delivery only; same
            # detection tuning as single-stream video mode so confidence
            # behaviour is familiar.
            #
            # If --video-multi points at a directory, switch the pipeline to
            # `--inputs-dir`: each stream loops its own clip independently
            # (filename "1.mp4" → stream 0, "8.mp4" → stream 7).  Otherwise
            # fall back to the legacy single-source `--input` (broadcast 8x).
            video_multi_arg = (
                ["--inputs-dir", self.video_multi_path]
                if Path(self.video_multi_path).is_dir()
                else ["--input", self.video_multi_path]
            )
            argv += [
                *video_multi_arg,
                "--serve",
                "--serve-split",
                "--conf",
                "0.75",
                "--merge-mode",
                "nms",
                "--merge-match",
                "iou",
                "--class-agnostic",
                "--merge-threshold",
                "0.75",
            ]
        else:
            raise ValueError(f"unknown transport: {transport!r}")
        argv += self.extra_pipeline_args
        return argv

    async def stop(self, sig: int = signal.SIGTERM, timeout: float = 8.0) -> None:
        p = self.proc
        if p is None or p.poll() is not None:
            self.proc = None
            return
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                p.send_signal(sig)
            except ProcessLookupError:
                self.proc = None
                return
        t0 = time.time()
        while time.time() - t0 < timeout:
            if p.poll() is not None:
                self.proc = None
                return
            await asyncio.sleep(0.1)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                p.kill()
            except ProcessLookupError:
                pass
        self.proc = None

    def _status_payload(self) -> dict:
        """Snapshot used by both /api/status and the response from /api/source."""
        return {
            "state": self.state,
            "transport": self.transport,
            "pending_transport": self.pending_transport,
            "ready": self.state == "ready",
            "last_error": self.last_error,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "tiles_x": self.tiles.get("tiles_x", 0),
            "tiles_y": self.tiles.get("tiles_y", 0),
            "n_tiles": self.tiles.get("n_tiles", 0),
            "fps": self.fps,
            "n_frames": self.n_frames,
            "n_streams": self.n_streams,
            "tiles_per_stream": self.tiles_per_stream,
            # True when --video-multi is a directory (each stream loops its
            # own clip).  Browser uses this to decide whether to attach
            # 8 <video> elements (one per cell) or a single master video.
            "per_stream_sources": (self.transport == "multi" and Path(self.video_multi_path).is_dir()),
            "uptime_s": (time.time() - self.start_t) if self.start_t else 0.0,
            "url": f"http://{_connect_host(self.pipeline_host)}:{self.pipeline_port}/",
            "kind": "webrtc",
            # Legacy "ok" field; older callers checked status.ok rather than state.
            "ok": self.state == "ready",
        }

    async def request_restart(self, transport: str, width: int, height: int) -> dict:
        """Validate the request and kick off a background restart task.

        Returns immediately with state="switching" so the HTTP handler
        doesn't hold the connection open for 30-90 s while the pipeline
        spawns and chip init completes.  The launch page polls
        /api/status to learn when state becomes "ready" or "error".
        """
        async with self._lock:
            if self.state in ("switching", "recovering"):
                return {
                    "ok": False,
                    "state": self.state,
                    "error": f"busy ({self.state}); wait or POST /api/reset",
                }
            # Idempotent: same transport + dims + already ready -> just
            # echo current status, no spawn.  Mirrors the prior behavior.
            if (
                self.state == "ready"
                and self.transport == transport
                and self.proc is not None
                and self.proc.poll() is None
                and (
                    transport == "multi"
                    or width <= 0
                    or height <= 0
                    or (self.frame_w == int(width) and self.frame_h == int(height))
                )
            ):
                print(
                    f"[supervisor] /api/source no-op: already ready " f"transport={transport} (pid={self.proc.pid})",
                    flush=True,
                )
                return self._status_payload()
            # Cancel any stale task from prior run
            if self._restart_task and not self._restart_task.done():
                self._restart_task.cancel()
            self.state = "switching"
            self.pending_transport = transport
            self.last_error = None

        # Kick off background work; HTTP handler returns immediately.
        self._restart_task = asyncio.create_task(self._do_restart(transport, width, height))
        # Return a snapshot AFTER releasing the lock; state is now "switching".
        return self._status_payload()

    async def _do_restart(self, transport: str, width: int, height: int) -> None:
        """Background task that performs the actual restart.

        On success, sets state="ready".  On init failure, sets state="error"
        and triggers chip-health recovery.  Cancellation is tolerated.
        """
        try:
            ok = await self._restart_inner(transport, width, height)
        except asyncio.CancelledError:
            self.state = "idle"
            self.pending_transport = None
            raise
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            self.last_error = f"restart exception: {e!r}"
            self.pending_transport = None
            print(f"[supervisor] restart task failed: {e!r}", flush=True)
            ok = False
        finally:
            self.pending_transport = None

        if ok:
            self.state = "ready"
            self._consecutive_failures = 0
            self._pending_retry = None
        else:
            self.state = "error"
            self._consecutive_failures += 1
            print(
                f"[supervisor] restart FAILED (consecutive={self._consecutive_failures}): " f"{self.last_error}",
                flush=True,
            )
            # Auto-recover + retry on the FIRST init failure only — most
            # commonly a leftover hugepage/lock from the prior process.  If
            # the retry ALSO fails, leave state="error" with an informative
            # last_error so the launch page shows the failure to the user
            # instead of looping forever.
            if self._consecutive_failures == 1 and (self._recover_task is None or self._recover_task.done()):
                self._pending_retry = (transport, width, height)
                print(
                    f"[supervisor] auto-triggering chip recovery + retry of " f"transport={transport}",
                    flush=True,
                )
                self._recover_task = asyncio.create_task(self._recover_chips())
            else:
                self._pending_retry = None
                self.last_error = (
                    f"{self.last_error or 'init failed'} " f"(retry also failed; click ↻ Reset Chips and try again)"
                )
                print(
                    f"[supervisor] giving up after {self._consecutive_failures} " f"consecutive failures — state=error",
                    flush=True,
                )

    async def _restart_inner(self, transport: str, width: int, height: int) -> bool:
        async with self._lock:
            fps = 0.0
            n_frames = 0
            if transport == "video":
                w, h, fps, n_frames = _peek_video_meta(self.video_path)
                if width <= 0 or height <= 0:
                    width, height = w, h
            elif transport == "camera":
                if width <= 0 or height <= 0:
                    width, height = (1280, 720)
            elif transport == "multi":
                # Multi-stream dims are fixed at MULTI_STREAM_W x MULTI_STREAM_H.
                # fps / n_frames come from the multi-stream source file so the
                # browser's per-stream <video loop> can key dets by frame_id.
                # When --video-multi points at a directory, use any one of the
                # numbered files (they all share the same dims/fps post-transcode).
                meta_src = self.video_multi_path
                if Path(meta_src).is_dir():
                    for i in range(1, 9):
                        candidate = Path(meta_src) / f"{i}.mp4"
                        if candidate.exists():
                            meta_src = str(candidate)
                            break
                w, h, fps, n_frames = _peek_video_meta(meta_src)
                width, height = MULTI_STREAM_W, MULTI_STREAM_H
            else:
                self.last_error = f"unknown transport {transport}"
                return False

            # Idempotency check is now done in request_restart() before we
            # ever reach this code path; the inner restart always spawns.

            if transport == "multi":
                # Fixed topology for the multi-stream demo -- the launch page
                # uses (n_streams, tiles_per_stream) to build the 8-element
                # grid; n_tiles = 32 is informational.
                self.tiles = {
                    "tiles_x": 2,
                    "tiles_y": 2,
                    "n_tiles": MULTI_N_STREAMS * MULTI_TILES_PER_STREAM,
                }
            else:
                self.tiles = _grid_for(width, height)
            await self.stop()
            # Wait for port *and* chip lock — TT UMD holds a PCIe lock
            # under /dev/shm that can linger up to ~3 s after the owning
            # process exits. Giving stop() a longer budget avoids the
            # "Waiting for lock CHIP_IN_USE_*_PCIe" warning on spawn.
            await _await_port_free(self.pipeline_host, self.pipeline_port, 12.0)
            await asyncio.sleep(2.0)

            argv = self._build_argv(transport, width, height)
            print(f"[supervisor] spawn: {' '.join(argv)}", flush=True)
            self.proc = subprocess.Popen(  # noqa: S603
                argv,
                cwd=str(TT_METAL_DIR),
                env=self._env(),
                stdout=None,
                stderr=None,
                # Put the child in its own process group so we can SIGKILL the
                # whole tree on supervisor exit; otherwise a SIGKILL on the
                # supervisor leaves the pipeline orphaned and holding port 9090.
                start_new_session=True,
                # Kernel-level "die with the parent" backstop for the crash/
                # SIGKILL case where on_cleanup never fires.
                preexec_fn=_child_preexec,
            )
            self.transport = transport
            self.frame_w = int(width)
            self.frame_h = int(height)
            self.fps = fps if transport in ("video", "multi") else 0.0
            self.n_frames = n_frames if transport in ("video", "multi") else 0
            if transport == "multi":
                self.n_streams = MULTI_N_STREAMS
                self.tiles_per_stream = MULTI_TILES_PER_STREAM
            else:
                self.n_streams = 0
                self.tiles_per_stream = 0
            self.start_t = time.time()

        # Bump readiness timeout for the multi-stream pipeline — 32-chip init
        # + model load + trace capture is comfortably 60 s on a cold start.
        ready_budget = 150.0 if transport == "multi" else 90.0
        ready = await _await_pipeline_ready(self.pipeline_host, self.pipeline_port, ready_budget)
        if not ready:
            # Distinguish "subprocess died" from "process alive but didn't bind".
            poll = self.proc.poll() if self.proc else 0
            if poll is not None:
                self.last_error = f"pipeline subprocess exited rc={poll} during init"
            else:
                self.last_error = (
                    f"pipeline did not bind {self.pipeline_host}:{self.pipeline_port} within {ready_budget:.0f}s"
                )
        return bool(ready)

    async def _recover_chips(self) -> None:
        """Stop the pipeline and run `tt-smi -glx_reset_auto` in a worker.

        Triggered automatically after a pipeline init failure (most common
        cause: leftover hugepage mappings or chip locks from a prior
        unclean shutdown), or manually via POST /api/reset.  Sets state
        to "recovering" while running, then "idle" on success or "error"
        on failure.
        """
        async with self._lock:
            self.state = "recovering"
            self.last_error = None
        print("[supervisor] chip recovery: stopping pipeline + running tt-smi reset", flush=True)
        try:
            await self.stop(timeout=10.0)
        except Exception as e:  # noqa: BLE001
            print(f"[supervisor] stop during recovery raised: {e!r}", flush=True)

        # tt-smi -glx_reset_auto blocks for ~30 s while the trays power-cycle.
        # Run as a non-blocking subprocess so we don't peg the event loop.
        cmd = [self._python(), "-m", "tt_smi", "-glx_reset_auto"]
        # Fall back to the bare CLI if -m doesn't work in this env.
        smi_bin = TT_METAL_DIR / "python_env/bin/tt-smi"
        if smi_bin.exists():
            cmd = [str(smi_bin), "-glx_reset_auto"]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(TT_METAL_DIR),
                env=self._env(),
            )
            try:
                rc = await asyncio.wait_for(proc.wait(), timeout=180.0)
            except asyncio.TimeoutError:
                proc.kill()
                self.state = "error"
                self.last_error = "tt-smi reset timed out after 180s"
                print(f"[supervisor] {self.last_error}", flush=True)
                return
            if rc != 0:
                self.state = "error"
                self.last_error = f"tt-smi reset returned rc={rc}"
                print(f"[supervisor] {self.last_error}", flush=True)
                return
        except FileNotFoundError as e:
            self.state = "error"
            self.last_error = f"tt-smi not found: {e}"
            return

        # Recovery succeeded.  Two paths from here:
        #   1) A transport request triggered this recovery and saved a retry
        #      tuple.  Auto-re-issue that request so the launch page sees the
        #      cycle complete (state goes recovering -> switching -> ready).
        #   2) No retry pending (manual /api/reset, or 2nd consecutive
        #      failure where we gave up).  Settle at idle so the user can
        #      pick a transport from the launch page.
        retry = self._pending_retry
        self._pending_retry = None
        async with self._lock:
            self._consecutive_failures = 0
            self.transport = "idle"
            self.frame_w = 0
            self.frame_h = 0
            self.tiles = {"tiles_x": 0, "tiles_y": 0, "n_tiles": 0}
            self.start_t = 0.0
            if retry:
                # Stay in 'switching' so the launch page's awaitState() loop
                # doesn't briefly observe 'idle' and prematurely give up.
                self.state = "switching"
                self.pending_transport = retry[0]
            else:
                self.state = "idle"
        if retry:
            t, w, h = retry
            print(
                f"[supervisor] chip recovery complete; auto-retrying " f"transport={t} ({w}x{h})",
                flush=True,
            )
            # Cancel any stale task before kicking off a fresh _do_restart.
            if self._restart_task and not self._restart_task.done():
                self._restart_task.cancel()
            self._restart_task = asyncio.create_task(self._do_restart(t, w, h))
        else:
            print("[supervisor] chip recovery complete; state=idle", flush=True)


async def _api_status(req: web.Request) -> web.Response:
    sup: Supervisor = req.app["sup"]
    alive = sup.proc is not None and sup.proc.poll() is None
    listening = _port_busy(sup.pipeline_host, sup.pipeline_port)
    payload = sup._status_payload()
    payload.update(
        {
            "alive": alive,
            "listening": listening,
            "pipeline_url": payload["url"],
        }
    )
    return web.json_response(payload)


def _strip_charset(ct: str) -> str:
    """aiohttp's ClientSession refuses content_type values that include a charset
    parameter (e.g. 'application/json; charset=utf-8'). Same restriction applies
    when we hand the upstream Content-Type back into web.Response(content_type=...).
    Strip everything after the first ';'.
    """
    return (ct or "").split(";", 1)[0].strip() or "application/json"


async def _proxy_offer(req: web.Request) -> web.Response:
    """Same-origin proxy from the launch page to the pipeline's /offer endpoint.
    Lets the browser talk to the supervisor only (no cross-origin SDP POSTs).
    """
    sup: Supervisor = req.app["sup"]
    body = await req.read()
    target = f"http://{_connect_host(sup.pipeline_host)}:{sup.pipeline_port}/offer"
    upstream_ct = _strip_charset(req.headers.get("Content-Type", "application/json"))
    headers = {"Content-Type": upstream_ct}
    tok = req.headers.get("X-Auth-Token")
    if tok:
        headers["X-Auth-Token"] = tok
    last_err = None
    async with ClientSession(timeout=ClientTimeout(total=10.0)) as sess:
        # The pipeline is racing trace capture during a fresh restart; retry briefly.
        for _ in range(15):
            try:
                async with sess.post(target, data=body, headers=headers) as r:
                    txt = await r.read()
                    return web.Response(
                        status=r.status,
                        body=txt,
                        content_type=_strip_charset(r.headers.get("Content-Type", "application/json")),
                    )
            except Exception as e:  # noqa: BLE001
                last_err = e
                await asyncio.sleep(0.5)
    return web.json_response({"error": f"pipeline unreachable: {last_err}"}, status=502)


async def _proxy_passthrough(req: web.Request, upstream_path: str) -> web.StreamResponse:
    """Stream a GET through to the pipeline on `upstream_path`. Forwards
    Range/Accept headers and preserves upstream status, content-type,
    Content-Length / Accept-Ranges / Content-Range. Used for the raw
    source file (byte-range) and the /dets SSE stream.

    Only valid in video mode; camera mode doesn't expose these endpoints.
    """
    sup: Supervisor = req.app["sup"]
    if sup.transport not in ("video", "multi"):
        return web.json_response(
            {"error": f"{upstream_path} not available in transport={sup.transport!r}"},
            status=503,
        )
    target = f"http://{_connect_host(sup.pipeline_host)}:{sup.pipeline_port}{upstream_path}"
    fwd_headers = {}
    for h in ("Range", "Accept", "Accept-Encoding", "If-None-Match", "If-Modified-Since"):
        v = req.headers.get(h)
        if v:
            fwd_headers[h] = v
    # SSE needs an unbounded read deadline; byte-range reads finish fast but
    # may also be long for a 4K file over a slow sink — leave sock_read=None.
    timeout = ClientTimeout(total=None, sock_connect=5.0, sock_read=None)
    sess = ClientSession(timeout=timeout)
    try:
        upstream = await sess.get(target, headers=fwd_headers)
    except Exception as e:
        await sess.close()
        return web.json_response({"error": f"upstream unreachable: {e}"}, status=502)

    resp = web.StreamResponse(status=upstream.status, reason=upstream.reason)
    ct = _strip_charset(upstream.headers.get("Content-Type", "application/octet-stream"))
    resp.content_type = ct
    for h in ("Content-Length", "Accept-Ranges", "Content-Range", "Cache-Control", "X-Accel-Buffering"):
        v = upstream.headers.get(h)
        if v:
            resp.headers[h] = v
    resp.headers.setdefault("Cache-Control", "no-store")
    try:
        await resp.prepare(req)
        async for chunk in upstream.content.iter_any():
            if not chunk:
                continue
            await resp.write(chunk)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"[supervisor] {upstream_path} proxy error: {e}", flush=True)
    finally:
        try:
            upstream.release()
        except Exception:
            pass
        await sess.close()
    return resp


async def _proxy_source_mp4(req: web.Request) -> web.StreamResponse:
    # Direct file serve (no inter-process HTTP hop). Avoids Chrome
    # ERR_INVALID_HTTP_RESPONSE caused by Content-Length mismatches when the
    # upstream proxy connection dropped mid-range, and lets aiohttp handle
    # HEAD + Range natively instead of forwarding every HEAD as a full GET.
    sup: Supervisor = req.app["sup"]
    if sup.transport == "video":
        path = Path(sup.video_path)
    elif sup.transport == "multi":
        # Multi-stream may point at either a directory (per-stream files) or
        # a single file (legacy broadcast).  When it's a directory, the
        # browser should request /source-N.mp4 instead — return any of the
        # numbered files for back-compat consumers that still hit /source.mp4.
        mp = Path(sup.video_multi_path)
        if mp.is_dir():
            for i in range(1, 9):
                cand = mp / f"{i}.mp4"
                if cand.exists():
                    mp = cand
                    break
        path = mp
    else:
        return web.json_response(
            {"error": f"/source.mp4 not available in transport={sup.transport!r}"},
            status=503,
        )
    if not path.exists():
        return web.Response(status=404, text=f"source missing: {path}\n")
    resp = web.FileResponse(path=str(path), chunk_size=1 << 16)
    resp.content_type = "video/mp4"
    return resp


async def _proxy_source_mp4_n(req: web.Request) -> web.StreamResponse:
    """Per-stream source for the multi-stream demo.

    /source-1.mp4 .. /source-8.mp4 — only valid when --video-multi is a
    directory.  When --video-multi is a single file, all stream indices
    return that same file (so the launch page works either way).
    """
    sup: Supervisor = req.app["sup"]
    if sup.transport != "multi":
        return web.json_response(
            {"error": f"/source-N.mp4 only valid in transport=multi (got {sup.transport!r})"},
            status=503,
        )
    try:
        n = int(req.match_info.get("n", "0"))
    except ValueError:
        return web.Response(status=400, text="bad stream index\n")
    if n < 1 or n > MULTI_N_STREAMS:
        return web.Response(status=400, text=f"stream index out of range (1..{MULTI_N_STREAMS})\n")
    mp = Path(sup.video_multi_path)
    if mp.is_dir():
        path = mp / f"{n}.mp4"
    else:
        # Single-file fallback: same content for every stream.
        path = mp
    if not path.exists():
        return web.Response(status=404, text=f"source missing: {path}\n")
    resp = web.FileResponse(path=str(path), chunk_size=1 << 16)
    resp.content_type = "video/mp4"
    return resp


async def _proxy_dets(req: web.Request) -> web.StreamResponse:
    return await _proxy_passthrough(req, "/dets")


async def _proxy_stream(req: web.Request) -> web.StreamResponse:
    """Back-compat stub — split delivery replaced /stream with /source.mp4+/dets.
    Return 503 so any cached launch.html referencing /stream fails fast.
    """
    sup: Supervisor = req.app["sup"]
    return web.json_response(
        {"error": f"/stream not available in transport={sup.transport!r} (use /source.mp4 + /dets)"},
        status=503,
    )


async def _api_source(req: web.Request) -> web.Response:
    try:
        body = await req.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    transport = str(body.get("transport") or "").strip().lower()
    if transport not in ("video", "camera", "multi"):
        return web.json_response(
            {"ok": False, "error": "transport must be 'video', 'camera', or 'multi'"},
            status=400,
        )
    width = int(body.get("width") or 0)
    height = int(body.get("height") or 0)
    sup: Supervisor = req.app["sup"]
    out = await sup.request_restart(transport, width, height)
    # 202 Accepted while switching, 200 if the no-op idempotency path
    # returned an already-ready state, 409 if busy.
    if out.get("state") == "ready":
        code = 200
    elif out.get("state") == "switching":
        code = 202
    elif out.get("state") in ("recovering",):
        code = 409
    else:
        code = 200 if out.get("ok") else 503
    return web.json_response(out, status=code)


async def _api_reset(_req: web.Request) -> web.Response:
    """Manually trigger a chip reset (kills pipeline, runs tt-smi)."""
    sup: Supervisor = _req.app["sup"]
    if sup._recover_task and not sup._recover_task.done():
        return web.json_response(
            {"ok": False, "state": sup.state, "error": "recovery already in progress"},
            status=409,
        )
    sup._recover_task = asyncio.create_task(sup._recover_chips())
    return web.json_response(
        {"ok": True, "state": "recovering"},
        status=202,
    )


async def _index(_req: web.Request) -> web.Response:
    if not LAUNCH_HTML.exists():
        return web.Response(status=404, text="demo/launch.html not found\n")
    return web.Response(body=LAUNCH_HTML.read_bytes(), content_type="text/html", charset="utf-8")


async def _asset(req: web.Request) -> web.Response:
    name = req.match_info.get("name", "")
    safe = (ASSETS_DIR / name).resolve()
    if not safe.is_file() or ASSETS_DIR.resolve() not in safe.parents:
        return web.Response(status=404, text="not found\n")
    return web.FileResponse(safe)


async def _on_cleanup(app: web.Application) -> None:
    sup: Supervisor = app["sup"]
    await sup.stop(timeout=5.0)


def build_app(sup: Supervisor) -> web.Application:
    app = web.Application(client_max_size=64 * 1024)
    app["sup"] = sup
    app.router.add_get("/", _index)
    app.router.add_get("/api/status", _api_status)
    app.router.add_post("/api/source", _api_source)
    app.router.add_post("/api/reset", _api_reset)
    app.router.add_post("/offer", _proxy_offer)
    app.router.add_get("/stream", _proxy_stream)
    app.router.add_get("/source.mp4", _proxy_source_mp4)
    app.router.add_get(r"/source-{n:\d+}.mp4", _proxy_source_mp4_n)
    app.router.add_get("/dets", _proxy_dets)
    app.router.add_get("/assets/{name:[A-Za-z0-9_.\\-]+}", _asset)
    app.on_cleanup.append(_on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo supervisor for YOLOv8L SAHI-640 pipeline.")
    p.add_argument("--video", required=True, help="Path to the demo video (used in Video mode).")
    p.add_argument(
        "--video-multi",
        default=None,
        help="Path to the 1280x1280 video used in Multi-stream mode "
        "(default: sample_images/14052767_1280x1280_30fps.mp4 under the demo dir).",
    )
    p.add_argument("--bind", default="0.0.0.0:9100", help="host:port for the supervisor (default 0.0.0.0:9100).")
    p.add_argument("--pipeline-host", default="0.0.0.0", help="Host the pipeline binds (default 0.0.0.0).")
    p.add_argument("--pipeline-port", type=int, default=9090, help="Port the pipeline binds (default 9090).")
    p.add_argument(
        "--auto-start",
        choices=["video", "camera", "multi", "none"],
        default="video",
        help="Spawn this transport on boot (default 'video').",
    )
    p.add_argument(
        "--bitrate",
        default="16M",
        help="H.264 stream bitrate cap (default 16M — bumped from 4M to fix 1080p macroblocking on busy scenes).",
    )
    p.add_argument("--keyint", type=int, default=30, help="server_only H.264 keyframe interval (default 30).")
    p.add_argument(
        "--pipeline-arg",
        action="append",
        default=[],
        help="Extra --key value pairs forwarded verbatim to the pipeline (repeatable).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.video).exists():
        print(f"[supervisor] video not found: {args.video}", file=sys.stderr, flush=True)
        sys.exit(2)
    if not PIPELINE_SCRIPT.exists():
        print(f"[supervisor] pipeline script missing: {PIPELINE_SCRIPT}", file=sys.stderr, flush=True)
        sys.exit(2)

    # Multi-stream default: sibling sample file under the yolo_eval demo dir.
    video_multi = args.video_multi or str(DEMO_DIR.parent / "sample_images" / "14052767_1280x1280_30fps.mp4")
    if not Path(video_multi).exists():
        print(
            f"[supervisor] WARN: multi-stream video not found: {video_multi} "
            f"(Multi mode will fail until --video-multi points at a valid file)",
            file=sys.stderr,
            flush=True,
        )
    if not MULTI_PIPELINE_SCRIPT.exists():
        print(
            f"[supervisor] WARN: multi-stream pipeline missing: {MULTI_PIPELINE_SCRIPT}",
            file=sys.stderr,
            flush=True,
        )

    bind_host, _, bind_port_s = args.bind.partition(":")
    bind_port = int(bind_port_s or 9100)

    sup = Supervisor(
        video_path=str(Path(args.video).resolve()),
        video_multi_path=str(Path(video_multi).resolve()),
        pipeline_host=args.pipeline_host,
        pipeline_port=int(args.pipeline_port),
        default_bitrate=args.bitrate,
        default_keyint=int(args.keyint),
        extra_pipeline_args=list(args.pipeline_arg),
    )

    async def _bootstrap(app: web.Application) -> None:
        if args.auto_start in ("video", "camera", "multi"):
            print(f"[supervisor] auto-starting transport={args.auto_start}", flush=True)
            await sup.request_restart(args.auto_start, 0, 0)

    app = build_app(sup)
    app.on_startup.append(_bootstrap)

    print(f"[supervisor] launch page: http://{bind_host or '127.0.0.1'}:{bind_port}/", flush=True)
    print(f"[supervisor] pipeline URL: http://{args.pipeline_host or '127.0.0.1'}:{args.pipeline_port}/", flush=True)
    web.run_app(app, host=bind_host or "0.0.0.0", port=bind_port, print=None, access_log=None)


if __name__ == "__main__":
    main()
