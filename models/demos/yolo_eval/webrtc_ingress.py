#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Minimal WebRTC ingress bridge for live-camera mode.

Browser POSTs an SDP offer with a sendonly camera video track; we accept,
decode VideoFrames to BGR ndarrays, and push into `ingress_q` (mp.Queue)
for the pipeline's prep worker. Detections flow back via a single
RTCDataChannel named "dets" that drains `dets_q`.

Deliberately small — no return video track, no H.264 codec patch, no
STUN/TURN plumbing beyond aiortc defaults. Use `webrtc_bridge.py` (stash)
for the fancier return-track variant; this file is the "just works" path.

Run as an mp.Process launched by yolov8l_sahi_640_pipelined.py::

    ctx = mp.get_context("spawn")
    ingress_q = ctx.Queue(4)
    dets_q    = ctx.Queue(32)
    proc = ctx.Process(
        target=run_server,
        args=(host, port, ingress_q, dets_q, frame_w, frame_h),
    )
    proc.start()
"""
from __future__ import annotations

import asyncio
import json
import signal
import time
from typing import Any

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription

_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLOv8L Camera</title></head>
<body style="margin:0;background:#000;color:#fff;font:14px sans-serif">
<p style="padding:12px">Camera transport active. Open the launch page to use it.</p>
</body></html>
"""


async def _healthz(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True})


async def _index(_req: web.Request) -> web.Response:
    return web.Response(body=_HTML.encode(), content_type="text/html", charset="utf-8")


async def _consume_track(track: Any, ingress_q: Any) -> None:
    """Pull decoded VideoFrames from the browser's camera track, convert to
    BGR, and drop-oldest-on-full into ingress_q.
    """
    n = 0
    t0 = time.perf_counter()
    while True:
        try:
            frame = await track.recv()
        except Exception as e:  # track ended / connection closed
            print(f"[webrtc] track ended after {n} frames: {e}", flush=True)
            return
        try:
            bgr = frame.to_ndarray(format="bgr24")
        except Exception as e:
            print(f"[webrtc] frame convert failed: {e}", flush=True)
            continue
        if bgr is None or bgr.size == 0:
            continue
        try:
            ingress_q.put_nowait(bgr)
        except Exception:
            # queue full — drop oldest and retry
            try:
                ingress_q.get_nowait()
            except Exception:
                pass
            try:
                ingress_q.put_nowait(bgr)
            except Exception:
                pass
        n += 1
        if n == 1:
            dt = (time.perf_counter() - t0) * 1000
            print(
                f"[webrtc] first frame: {bgr.shape[1]}x{bgr.shape[0]} t={dt:.0f}ms",
                flush=True,
            )
        elif n % 90 == 0:
            print(f"[webrtc] frames received={n}", flush=True)


async def _pump_dets(channel: Any, dets_q: Any, pc: Any) -> None:
    """Drain dets_q (a multiprocessing.Queue) and forward each dict as a
    JSON text message over the DataChannel.

    `dets_q.get()` is blocking — run it on the default executor so the
    aiohttp event loop isn't stalled.
    """
    loop = asyncio.get_running_loop()
    print("[webrtc] dets pump started", flush=True)
    sent = 0
    while pc.connectionState not in ("failed", "closed", "disconnected"):
        try:
            msg = await loop.run_in_executor(None, dets_q.get)
        except Exception:
            break
        if msg is None:
            break
        if channel.readyState != "open":
            # Client closed or not ready — drop this dets batch and keep going.
            continue
        try:
            channel.send(json.dumps(msg))
            sent += 1
            if sent == 1 or sent % 300 == 0:
                print(f"[webrtc] dets sent={sent}", flush=True)
        except Exception as e:
            print(f"[webrtc] dc send error: {e}", flush=True)
            break
    print(f"[webrtc] dets pump stopped (sent={sent})", flush=True)


async def _offer(req: web.Request) -> web.Response:
    body = await req.json()
    offer = RTCSessionDescription(sdp=body["sdp"], type=body["type"])

    pc = RTCPeerConnection()
    req.app["pcs"].add(pc)
    ingress_q = req.app["ingress_q"]
    dets_q = req.app["dets_q"]

    @pc.on("connectionstatechange")
    async def _on_state() -> None:
        print(f"[webrtc] pc state={pc.connectionState}", flush=True)
        if pc.connectionState in ("failed", "closed"):
            try:
                await pc.close()
            except Exception:
                pass
            req.app["pcs"].discard(pc)

    @pc.on("track")
    def _on_track(track: Any) -> None:
        if track.kind == "video":
            print(f"[webrtc] track kind={track.kind} id={track.id}", flush=True)
            asyncio.create_task(_consume_track(track, ingress_q))

    @pc.on("datachannel")
    def _on_dc(channel: Any) -> None:
        print(f"[webrtc] datachannel opened: {channel.label}", flush=True)
        if channel.label == "dets":
            asyncio.create_task(_pump_dets(channel, dets_q, pc))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})


async def _on_cleanup(app: web.Application) -> None:
    pcs = app["pcs"]
    await asyncio.gather(*(pc.close() for pc in list(pcs)), return_exceptions=True)
    pcs.clear()


def _build_app(ingress_q: Any, dets_q: Any) -> web.Application:
    app = web.Application()
    app["ingress_q"] = ingress_q
    app["dets_q"] = dets_q
    app["pcs"] = set()
    app.router.add_get("/", _index)
    app.router.add_get("/healthz", _healthz)
    app.router.add_post("/offer", _offer)
    app.on_cleanup.append(_on_cleanup)
    return app


def run_server(
    host: str,
    port: int,
    ingress_q: Any,
    dets_q: Any,
    frame_w: int,
    frame_h: int,
) -> None:
    """mp.Process entry point. Runs an aiohttp server until the parent exits."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(
        f"[webrtc] bridge on http://{host}:{port}/  expect {frame_w}x{frame_h}",
        flush=True,
    )
    try:
        app = _build_app(ingress_q, dets_q)
        web.run_app(app, host=host, port=port, print=None, access_log=None)
    except Exception as e:
        print(f"[webrtc] server exit: {e}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    iq: Any = ctx.Queue(4)
    dq: Any = ctx.Queue(32)
    run_server("0.0.0.0", 9090, iq, dq, 1280, 720)
