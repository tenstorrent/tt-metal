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

TILE = 640
DEMO_DIR = Path(__file__).resolve().parent
ASSETS_DIR = DEMO_DIR / "assets"
LAUNCH_HTML = DEMO_DIR / "launch.html"
PIPELINE_SCRIPT = DEMO_DIR.parent / "yolov8l_sahi_640_pipelined.py"
TT_METAL_DIR = Path(__file__).resolve().parents[4]


def _grid_for(w: int, h: int) -> dict:
    cols = max(1, math.ceil(w / TILE))
    rows = max(1, math.ceil(h / TILE))
    return {"tiles_x": cols, "tiles_y": rows, "n_tiles": cols * rows}


def _peek_video_dims(path: str) -> tuple[int, int]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return (3840, 2160)
    cap = cv2.VideoCapture(path)
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return (3840, 2160)
    return (w, h)


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
        pipeline_host: str,
        pipeline_port: int,
        default_bitrate: str,
        default_keyint: int,
        extra_pipeline_args: list[str],
    ) -> None:
        self.video_path = video_path
        self.pipeline_host = pipeline_host
        self.pipeline_port = pipeline_port
        self.default_bitrate = default_bitrate
        self.default_keyint = default_keyint
        self.extra_pipeline_args = list(extra_pipeline_args)
        self.proc: Optional[subprocess.Popen] = None
        self.transport: str = "idle"
        self.frame_w: int = 0
        self.frame_h: int = 0
        self.start_t: float = 0.0
        self.tiles: dict = {"tiles_x": 0, "tiles_y": 0, "n_tiles": 0}
        self._lock = asyncio.Lock()

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
        argv = [
            self._python(),
            str(PIPELINE_SCRIPT),
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
            # Video (4K) runs cleanly with fast torchvision NMS — source is
            # high-quality, cross-class misclassifications are rare. NMM's
            # Python-loop merge adds ~5ms to BG and caps throughput, so use
            # the vectorized NMS path and keep raw FPS above the baseline.
            argv += [
                "--source",
                "file",
                "--input",
                self.video_path,
                "--serve",
                "--serve-codec",
                "h264",
                "--stream-bitrate",
                self._bitrate_bps(self.default_bitrate),
                "--stream-keyint",
                str(self.default_keyint),
                "--stream-fps",
                "75",
                # "--display-width", "1920",
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

    async def restart(self, transport: str, width: int, height: int) -> dict:
        async with self._lock:
            if transport == "video":
                if width <= 0 or height <= 0:
                    width, height = _peek_video_dims(self.video_path)
            elif transport == "camera":
                if width <= 0 or height <= 0:
                    width, height = (1280, 720)
            else:
                return {"ok": False, "error": f"unknown transport {transport}"}

            # Idempotency check: if the pipeline is already running the
            # requested transport+dims, skip the kill/spawn. Otherwise a page
            # reload triggers an unnecessary restart and the TT chip lock
            # races (old process still holds 'CHIP_IN_USE_*_PCIe' when the
            # new one tries to open the mesh, producing a 60 s warm-reset).
            alive = self.proc is not None and self.proc.poll() is None
            port_up = _port_busy(self.pipeline_host, self.pipeline_port)
            if (
                alive
                and port_up
                and self.transport == transport
                and self.frame_w == int(width)
                and self.frame_h == int(height)
            ):
                print(
                    f"[supervisor] restart no-op: already running transport={transport} "
                    f"{width}x{height} (pid={self.proc.pid})",
                    flush=True,
                )
                self.tiles = _grid_for(width, height)
                return {
                    "ok": True,
                    "url": f"http://{_connect_host(self.pipeline_host)}:{self.pipeline_port}/",
                    "kind": "webrtc",
                    "transport": transport,
                    "frame_w": self.frame_w,
                    "frame_h": self.frame_h,
                    "tiles_x": self.tiles["tiles_x"],
                    "tiles_y": self.tiles["tiles_y"],
                    "n_tiles": self.tiles["n_tiles"],
                }

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
            )
            self.transport = transport
            self.frame_w = int(width)
            self.frame_h = int(height)
            self.start_t = time.time()

        ready = await _await_pipeline_ready(self.pipeline_host, self.pipeline_port, 90.0)
        return {
            "ok": bool(ready),
            "url": f"http://{_connect_host(self.pipeline_host)}:{self.pipeline_port}/",
            "kind": "webrtc",
            "transport": transport,
            "frame_w": self.frame_w,
            "frame_h": self.frame_h,
            "tiles_x": self.tiles["tiles_x"],
            "tiles_y": self.tiles["tiles_y"],
            "n_tiles": self.tiles["n_tiles"],
        }


async def _api_status(req: web.Request) -> web.Response:
    sup: Supervisor = req.app["sup"]
    alive = sup.proc is not None and sup.proc.poll() is None
    listening = _port_busy(sup.pipeline_host, sup.pipeline_port)
    return web.json_response(
        {
            "transport": sup.transport,
            "ready": bool(alive and listening),
            "alive": alive,
            "frame_w": sup.frame_w,
            "frame_h": sup.frame_h,
            "tiles_x": sup.tiles.get("tiles_x", 0),
            "tiles_y": sup.tiles.get("tiles_y", 0),
            "n_tiles": sup.tiles.get("n_tiles", 0),
            "uptime_s": (time.time() - sup.start_t) if sup.start_t else 0.0,
            "pipeline_url": f"http://{_connect_host(sup.pipeline_host)}:{sup.pipeline_port}/",
        }
    )


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


async def _proxy_stream(req: web.Request) -> web.StreamResponse:
    """Reverse-proxy GET /stream to the pipeline's H.264 fMP4 stream.

    Only valid in video mode; camera mode has no /stream endpoint, so we
    return 503 to avoid a stale <video> element hanging on connection.
    Forwards Range headers so the browser can seek.
    """
    sup: Supervisor = req.app["sup"]
    if sup.transport != "video":
        return web.json_response(
            {"error": f"/stream not available in transport={sup.transport!r}"},
            status=503,
        )
    target = f"http://{_connect_host(sup.pipeline_host)}:{sup.pipeline_port}/stream"
    fwd_headers = {}
    for h in ("Range", "Accept", "Accept-Encoding"):
        v = req.headers.get(h)
        if v:
            fwd_headers[h] = v
    timeout = ClientTimeout(total=None, sock_connect=5.0, sock_read=None)
    sess = ClientSession(timeout=timeout)
    try:
        upstream = await sess.get(target, headers=fwd_headers)
    except Exception as e:
        await sess.close()
        return web.json_response({"error": f"stream unreachable: {e}"}, status=502)

    resp = web.StreamResponse(status=upstream.status, reason=upstream.reason)
    ct = _strip_charset(upstream.headers.get("Content-Type", "video/mp4"))
    resp.content_type = ct
    for h in ("Content-Length", "Accept-Ranges", "Content-Range", "Cache-Control"):
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
        print(f"[supervisor] /stream proxy error: {e}", flush=True)
    finally:
        try:
            upstream.release()
        except Exception:
            pass
        await sess.close()
    return resp


async def _api_source(req: web.Request) -> web.Response:
    try:
        body = await req.json()
    except Exception:
        return web.json_response({"ok": False, "error": "invalid json"}, status=400)
    transport = str(body.get("transport") or "").strip().lower()
    if transport not in ("video", "camera"):
        return web.json_response({"ok": False, "error": "transport must be 'video' or 'camera'"}, status=400)
    width = int(body.get("width") or 0)
    height = int(body.get("height") or 0)
    sup: Supervisor = req.app["sup"]
    out = await sup.restart(transport, width, height)
    code = 200 if out.get("ok") else 504
    return web.json_response(out, status=code)


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
    app.router.add_post("/offer", _proxy_offer)
    app.router.add_get("/stream", _proxy_stream)
    app.router.add_get("/assets/{name:[A-Za-z0-9_.\\-]+}", _asset)
    app.on_cleanup.append(_on_cleanup)
    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Demo supervisor for YOLOv8L SAHI-640 pipeline.")
    p.add_argument("--video", required=True, help="Path to the demo video (used in Video mode).")
    p.add_argument("--bind", default="0.0.0.0:9100", help="host:port for the supervisor (default 0.0.0.0:9100).")
    p.add_argument("--pipeline-host", default="0.0.0.0", help="Host the pipeline binds (default 0.0.0.0).")
    p.add_argument("--pipeline-port", type=int, default=9090, help="Port the pipeline binds (default 9090).")
    p.add_argument(
        "--auto-start",
        choices=["video", "camera", "none"],
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

    bind_host, _, bind_port_s = args.bind.partition(":")
    bind_port = int(bind_port_s or 9100)

    sup = Supervisor(
        video_path=str(Path(args.video).resolve()),
        pipeline_host=args.pipeline_host,
        pipeline_port=int(args.pipeline_port),
        default_bitrate=args.bitrate,
        default_keyint=int(args.keyint),
        extra_pipeline_args=list(args.pipeline_arg),
    )

    async def _bootstrap(app: web.Application) -> None:
        if args.auto_start in ("video", "camera"):
            print(f"[supervisor] auto-starting transport={args.auto_start}", flush=True)
            asyncio.create_task(sup.restart(args.auto_start, 0, 0))

    app = build_app(sup)
    app.on_startup.append(_bootstrap)

    print(f"[supervisor] launch page: http://{bind_host or '127.0.0.1'}:{bind_port}/", flush=True)
    print(f"[supervisor] pipeline URL: http://{args.pipeline_host or '127.0.0.1'}:{args.pipeline_port}/", flush=True)
    web.run_app(app, host=bind_host or "0.0.0.0", port=bind_port, print=None, access_log=None)


if __name__ == "__main__":
    main()
