#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Split-delivery server for file-mode video.

Serves the *original* MP4 file to the browser with byte-range support so
`<video src="/source.mp4">` can play it natively (no server re-encode),
and streams detection JSON (tagged with frame_id) over an SSE endpoint
so the browser can overlay boxes onto the playing video.

This decouples inference throughput from network bandwidth: pixels flow
at the source's native bitrate (once), detections flow at ~14 KB/s.

Routes:
    GET /            minimal HTML so the supervisor's readiness poll gets 200
    GET /healthz     JSON health probe
    GET /source.mp4  the video file, range-capable (web.FileResponse)
    GET /dets        Server-Sent Events stream of {k:"dets",frame_id,...} msgs

Run as an mp.Process launched by yolov8l_sahi_640_pipelined.py::

    ctx = mp.get_context("spawn")
    dets_q = ctx.Queue(128)
    proc = ctx.Process(
        target=run_server,
        args=(host, port, dets_q, source_path, frame_w, frame_h),
    )
    proc.start()
"""
from __future__ import annotations

import asyncio
import json
import signal
from pathlib import Path
from typing import Any

from aiohttp import web

_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLOv8L Split Video</title></head>
<body style="margin:0;background:#000;color:#fff;font:14px sans-serif">
<p style="padding:12px">Split video transport active. Open the launch page to use it.</p>
</body></html>
"""


async def _healthz(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _index(_req: web.Request) -> web.Response:
    return web.Response(body=_HTML.encode(), content_type="text/html", charset="utf-8")


async def _source(req: web.Request) -> web.StreamResponse:
    path: Path = req.app["source_path"]
    if not path.exists():
        return web.Response(status=404, text=f"source missing: {path}\n")
    # FileResponse handles Range, If-Modified-Since, ETag, keep-alive. Exactly
    # the HTTP features the browser <video> element needs to seek+play.
    resp = web.FileResponse(path=str(path), chunk_size=1 << 16)
    # Help the browser pick the right decoder.
    resp.content_type = "video/mp4"
    return resp


async def _dets_sse(req: web.Request) -> web.StreamResponse:
    """Stream detections over Server-Sent Events.

    `dets_q.get()` is blocking — run it in the default executor so the
    event loop stays responsive. Drop messages only if the client TCP
    buffer backs up (write_eof below catches the ConnectionReset).
    """
    dets_q = req.app["dets_q"]
    loop = asyncio.get_running_loop()
    resp = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering
            "Access-Control-Allow-Origin": "*",
        },
    )
    await resp.prepare(req)
    # Initial comment flushes headers through any upstream proxy.
    await resp.write(b": ok\n\n")
    sent = 0
    try:
        while True:
            # `run_in_executor(None, get)` lets the loop keep serving other
            # requests (including /source.mp4 ranges) while we wait on the
            # multiprocessing queue.
            msg = await loop.run_in_executor(None, dets_q.get)
            if msg is None:
                break
            try:
                payload = json.dumps(msg, separators=(",", ":"))
            except Exception:
                continue
            await resp.write(b"data: " + payload.encode("utf-8") + b"\n\n")
            sent += 1
            if sent == 1 or sent % 300 == 0:
                print(f"[split] dets sent={sent}", flush=True)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except Exception as e:
        print(f"[split] sse error after {sent} msgs: {e}", flush=True)
    try:
        await resp.write_eof()
    except Exception:
        pass
    print(f"[split] sse client disconnected (sent={sent})", flush=True)
    return resp


def _build_app(dets_q: Any, source_path: Path) -> web.Application:
    app = web.Application()
    app["dets_q"] = dets_q
    app["source_path"] = source_path
    app.router.add_get("/", _index)
    app.router.add_get("/healthz", _healthz)
    app.router.add_get("/source.mp4", _source)
    app.router.add_get("/dets", _dets_sse)
    return app


def run_server(
    host: str,
    port: int,
    dets_q: Any,
    source_path: str,
    frame_w: int,
    frame_h: int,
) -> None:
    """mp.Process entry point. Runs an aiohttp server until the parent exits."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    path = Path(source_path).resolve()
    print(
        f"[split] server on http://{host}:{port}/  " f"source={path} ({frame_w}x{frame_h})",
        flush=True,
    )
    try:
        app = _build_app(dets_q, path)
        web.run_app(app, host=host, port=port, print=None, access_log=None)
    except Exception as e:
        print(f"[split] server exit: {e}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    import sys

    ctx = mp.get_context("spawn")
    dq: Any = ctx.Queue(128)
    src = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sample.mp4"
    run_server("0.0.0.0", 9090, dq, src, 3840, 2160)
