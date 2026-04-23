#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
H.264 fMP4-over-HTTP streaming server.

Drop-in replacement for ``_mjpeg_server.py`` for WAN/remote-browser demos.
Encodes BGR canvases (NMS + drawn) with libx264 ultrafast/zerolatency and
streams fragmented MP4 via chunked HTTP — playable by any ``<video>`` tag.

Bandwidth vs MJPEG at 70 fps, 1920x1080 BGR:
  - MJPEG (TurboJPEG q75): ~4.8 MB/frame  ~2.8 Gbps (WAN-unusable)
  - H.264 ultrafast 4 Mbps:     ~8 KB/frame  ~4 Mbps    (comfortable over broadband)

Run as an ``mp.Process`` launched by ``yolov8l_sahi_640_pipelined.py``::

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=2)    # drop-oldest on back-pressure
    proc = ctx.Process(
        target=run_server,
        args=(frame_queue, host, port, width, height, fps, bitrate_bps, keyint),
    )
    proc.start()
    # BG worker pushes: frame_queue.put(canvas_bgr_contiguous)
"""
from __future__ import annotations

import errno
import os
import socket as _socket
import threading
import time
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import av
import numpy as np

# ---------------------------------------------------------------------------
# Shared latest-frame slot (single-writer / many-reader fan-out).
#
# One drain thread pops BGR frames from the producer mp.Queue into
# ``_latest_frame``. Each HTTP streaming handler waits on ``_cond`` for a
# bump of ``_latest_seq`` and encodes whatever is current. If two clients
# connect, each runs its own libx264 encoder against the same latest-frame
# slot — simple, correct, costs 1 encoder CPU per viewer (fine for demos).
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_cond = threading.Condition(_lock)
_latest_frame: np.ndarray | None = None
_latest_seq: int = 0
_stop = threading.Event()


def _drain_loop(frame_queue: Any, target_fps: int) -> None:
    """Pull BGR frames from the producer queue into the shared slot.

    Decimates to ``target_fps``: producers running faster than the stream
    rate (e.g. 75 fps pipeline into a 60 fps stream) have their extras
    dropped here before any encoder work happens.  Keeps encoder CPU cost
    bounded and the stream's frame-number PTS matching wall-clock real time.
    """
    global _latest_frame, _latest_seq
    min_interval = 0.95 / max(target_fps, 1)  # accept slightly eagerly
    last_accept = 0.0
    while not _stop.is_set():
        try:
            item = frame_queue.get(timeout=0.5)
        except Exception:
            continue
        if item is None:  # sentinel → shutdown
            _stop.set()
            with _cond:
                _cond.notify_all()
            return
        # Accept either:
        #   - BGR: ndim=3, shape[2]=3  (smoke-test path, bgr24)
        #   - YUV I420 planar: ndim=2, shape=(H*3/2, W)  (pipeline fast path)
        if not isinstance(item, np.ndarray):
            continue
        if item.ndim == 3 and item.shape[2] == 3:
            pass  # BGR
        elif item.ndim == 2:
            pass  # YUV I420 (height = canvas_h * 3/2)
        else:
            continue
        now = time.perf_counter()
        if now - last_accept < min_interval:
            continue
        last_accept = now
        with _cond:
            _latest_frame = item
            _latest_seq += 1
            _cond.notify_all()


# ---------------------------------------------------------------------------
# DVR recorder: long-lived encoder that writes fMP4 to disk in parallel to
# the live /stream encoders, so the browser can rewind via /recording.mp4.
# Uses fragmented MP4 (moov at start, no final-atom dependency) so the file
# is readable + seekable while still growing.
# ---------------------------------------------------------------------------
def _recorder_loop(
    out_path: str,
    width: int,
    height: int,
    target_fps: int,
    bitrate_bps: int,
    keyint: int,
    size_cap_bytes: int,
    thread_count: int,
) -> None:
    try:
        try:
            os.unlink(out_path)  # fresh file per pipeline launch
        except FileNotFoundError:
            pass
        container = av.open(
            out_path,
            mode="w",
            format="mp4",
            options={"movflags": "empty_moov+frag_keyframe+default_base_moof"},
        )
        stream = container.add_stream("libx264", rate=Fraction(target_fps, 1))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 0
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.thread_type = "SLICE"
        stream.codec_context.thread_count = thread_count
        maxrate_bps = max(int(bitrate_bps), 1_000_000)
        stream.codec_context.options = {
            "preset": "superfast",
            "tune": "zerolatency",
            "profile": "high",
            "level": "5.1",
            "crf": "14",
            "maxrate": str(maxrate_bps),
            "bufsize": str(maxrate_bps * 2),
            "g": str(keyint),
            "keyint_min": str(keyint),
            "sc_threshold": "0",
            "x264-params": "nal-hrd=none:repeat-headers=1",
        }
    except Exception as e:
        print(f"[h264] recorder init failed: {e}", flush=True)
        return

    print(
        f"[h264] recording → {out_path} (cap={size_cap_bytes/1e9:.1f} GB, " f"{thread_count} threads)",
        flush=True,
    )

    last_seq = 0
    t_start = time.perf_counter()
    last_pts = -1
    last_dts = -1
    frame_count = 0
    capped = False
    try:
        while not _stop.is_set() and not capped:
            with _cond:
                while _latest_seq == last_seq and not _stop.is_set():
                    if not _cond.wait(timeout=1.0):
                        continue
                if _stop.is_set():
                    break
                frame = _latest_frame
                last_seq = _latest_seq
            if frame is None:
                continue

            fmt = "bgr24" if frame.ndim == 3 else "yuv420p"
            av_frame = av.VideoFrame.from_ndarray(frame, format=fmt)
            pts = int((time.perf_counter() - t_start) * 90000)
            if pts <= last_pts:
                pts = last_pts + 1
            last_pts = pts
            av_frame.pts = pts
            av_frame.time_base = stream.time_base

            for packet in stream.encode(av_frame):
                if packet.dts is not None and packet.dts <= last_dts:
                    packet.dts = last_dts + 1
                    if packet.pts is not None and packet.pts < packet.dts:
                        packet.pts = packet.dts
                if packet.dts is not None:
                    last_dts = packet.dts
                container.mux(packet)

            frame_count += 1
            # Check size cap roughly every 10s (not every frame — os.stat isn't free).
            if frame_count % max(target_fps * 10, 10) == 0:
                try:
                    if os.path.getsize(out_path) > size_cap_bytes:
                        print(
                            f"[h264] recording size cap hit "
                            f"({os.path.getsize(out_path)/1e9:.2f} GB) — stopping recorder",
                            flush=True,
                        )
                        capped = True
                except Exception:
                    pass
    except Exception as e:
        print(f"[h264] recorder error: {e}", flush=True)
    finally:
        try:
            for packet in stream.encode():  # flush
                container.mux(packet)
        except Exception:
            pass
        try:
            container.close()
        except Exception:
            pass
        print("[h264] recorder closed", flush=True)


# ---------------------------------------------------------------------------
# File-like wrapper so PyAV's MP4 muxer can write directly to an HTTP
# response's wfile. No intermediate BytesIO buffering → minimum latency.
# ---------------------------------------------------------------------------
class _ResponseIO:
    def __init__(self, handler: BaseHTTPRequestHandler) -> None:
        self._handler = handler
        self._closed = False

    def write(self, data: bytes) -> int:
        if self._closed:
            return 0
        try:
            self._handler.wfile.write(data)
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._closed = True
            raise
        return len(data)

    def flush(self) -> None:
        if self._closed:
            return
        try:
            self._handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self._closed = True

    def close(self) -> None:
        self._closed = True

    def seekable(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Minimal landing page — a single <video> tag that plays /stream.
# ---------------------------------------------------------------------------
_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLOv8L SAHI-640</title>
<style>
html,body{margin:0;padding:0;background:#000;height:100%}
#v{width:100%;height:100%;object-fit:contain;background:#000}
</style></head>
<body>
<video id="v" autoplay muted playsinline src="/stream"></video>
</body></html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default access log
        pass

    def do_GET(self):
        # Strip query string (the DVR toggle appends ?t=... for cache-busting).
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_html()
        elif path == "/stream":
            self._serve_h264()
        elif path == "/recording.mp4":
            self._serve_recording()
        elif path == "/healthz":
            self._serve_health()
        else:
            self.send_error(404)

    def _serve_recording(self):
        """Serve the growing fMP4 file with HTTP Range so the browser timeline works."""
        path = getattr(self.server, "recording_path", None)
        if not path:
            self.send_error(404, "recording disabled")
            return
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            self.send_error(404, "no recording yet")
            return
        # Don't serve a barely-started file — a browser that tries to parse
        # a too-small fMP4 before the first moof lands will choke and refuse
        # to reload; better to 404 and have the user click DVR again.
        if size < 65536:
            self.send_error(404, "recording not ready — try again in a second")
            return

        range_header = self.headers.get("Range", "")
        start, end = 0, size - 1
        status = 200
        if range_header.startswith("bytes="):
            spec = range_header[6:].split(",", 1)[0]  # first range only
            lo, _, hi = spec.partition("-")
            try:
                if lo:
                    start = int(lo)
                if hi:
                    end = int(hi)
            except ValueError:
                pass
            status = 206
        start = max(0, min(start, size - 1))
        end = min(end, size - 1)
        if end < start:
            end = start
        length = end - start + 1

        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    buf = f.read(min(64 * 1024, remaining))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    remaining -= len(buf)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def _serve_html(self):
        body = _HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_health(self):
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_h264(self):
        srv = self.server  # type: ignore[attr-defined]
        width = srv.width
        height = srv.height
        target_fps = srv.target_fps
        bitrate_bps = srv.bitrate_bps
        keyint = srv.keyint

        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()

        io_obj = _ResponseIO(self)

        try:
            container = av.open(
                io_obj,
                mode="w",
                format="mp4",
                # fMP4 for live streaming:
                #   empty_moov            → moov carries no samples (required for fragmented)
                #   frag_keyframe         → start a new fragment at each keyframe
                #   default_base_moof     → correct relative offsets in moof
                options={"movflags": "empty_moov+frag_keyframe+default_base_moof"},
            )
            stream = container.add_stream("libx264", rate=Fraction(target_fps, 1))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            # CRF-driven rate control — quality is the anchor, not bitrate.
            # At 4K, ultrafast+baseline+CBR@10Mbps visibly softens the image;
            # superfast+high+CRF 18 is visually near-lossless at similar cost.
            # bitrate_bps becomes a peak/maxrate cap (WAN safety), not a target.
            stream.bit_rate = 0
            # 90 kHz — the MPEG timebase. Fine-grained enough that no two packets
            # rescale onto the same integer DTS in the mp4 muxer's internal tick
            # (using 1/target_fps here races with x264's DTS offsets and trips
            # "non monotonically increasing dts" after ~3s of streaming).
            stream.time_base = Fraction(1, 90000)
            # Enable slice-parallel threading: frame-threading would add N frames
            # of latency, but slice-threading splits a single frame across cores
            # with zero extra buffering. 6 threads keeps 4K superfast real-time
            # while leaving enough cores free for the pipeline's BG worker
            # (NMS + draw + cvtColor) to sustain >70 FPS on EPYC 9354P.
            stream.codec_context.thread_type = "SLICE"
            stream.codec_context.thread_count = 6
            # superfast: huge quality win over ultrafast at the same bitrate.
            # profile=high: enables CABAC + 8×8 transform (baseline disables both);
            #   universally supported by browsers in 2026 including Safari/mobile.
            # crf=16: midpoint between "visually lossless" (14) and "excellent"
            #   (18) — sharper than 18 on detail, still WAN-friendly bitrate.
            # maxrate/bufsize: cap the peak rate so CRF can't blow a WAN link.
            maxrate_bps = max(int(bitrate_bps), 1_000_000)
            stream.codec_context.options = {
                "preset": "superfast",
                "tune": "zerolatency",
                "profile": "high",
                "level": "5.1",  # 4K60 needs level 5.1
                "crf": "18",
                "maxrate": str(maxrate_bps),
                "bufsize": str(maxrate_bps * 2),
                "g": str(keyint),
                "keyint_min": str(keyint),
                "sc_threshold": "0",  # forbid extra keyframes from scene detection
                "x264-params": "nal-hrd=none:repeat-headers=1",
            }
        except Exception as e:
            print(f"[h264] encoder init failed: {e}", flush=True)
            return

        last_seq = 0
        # Wall-clock PTS in 90 kHz ticks. This keeps playback timing correct
        # regardless of encoder throughput — if the encoder can't keep up with
        # target_fps, the stream plays back at the real rate without speedup.
        t_start = time.perf_counter()
        last_pts = -1
        last_dts = -1  # mp4 muxer requires strictly increasing DTS
        try:
            while not _stop.is_set():
                with _cond:
                    while _latest_seq == last_seq and not _stop.is_set():
                        if not _cond.wait(timeout=1.0):
                            continue
                    if _stop.is_set():
                        break
                    frame_bgr = _latest_frame
                    last_seq = _latest_seq

                if frame_bgr is None:
                    continue

                # Accept BGR (ndim=3) from smoke test or YUV I420 planar
                # (ndim=2) from the pipeline BG worker (pre-converted to
                # skip libx264's internal BGR→YUV at 4K).
                fmt = "bgr24" if frame_bgr.ndim == 3 else "yuv420p"
                av_frame = av.VideoFrame.from_ndarray(frame_bgr, format=fmt)
                pts = int((time.perf_counter() - t_start) * 90000)
                if pts <= last_pts:
                    pts = last_pts + 1
                last_pts = pts
                av_frame.pts = pts
                av_frame.time_base = stream.time_base

                for packet in stream.encode(av_frame):
                    # libx264 occasionally emits packets whose DTS equals the
                    # previous packet's after rescale; the mp4 muxer rejects
                    # them and kills the stream. Force strict +1 on collision.
                    if packet.dts is not None and packet.dts <= last_dts:
                        packet.dts = last_dts + 1
                        if packet.pts is not None and packet.pts < packet.dts:
                            packet.pts = packet.dts
                    if packet.dts is not None:
                        last_dts = packet.dts
                    container.mux(packet)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            if not isinstance(e, OSError) or e.errno != errno.EPIPE:
                print(f"[h264] stream error: {e}", flush=True)
        finally:
            try:
                for packet in stream.encode():  # flush encoder
                    container.mux(packet)
            except Exception:
                pass
            try:
                container.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class _Server(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, addr, handler, *, width, height, target_fps, bitrate_bps, keyint, recording_path=None):
        # Configure BEFORE super().__init__ — bind may trigger immediate
        # connection attempts whose handler reads these attributes.
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.bitrate_bps = bitrate_bps
        self.keyint = keyint
        self.recording_path = recording_path
        super().__init__(addr, handler)

    def server_bind(self):
        self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        try:
            self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass
        super().server_bind()


def run_server(
    frame_queue: Any,
    host: str,
    port: int,
    width: int,
    height: int,
    target_fps: int = 60,
    bitrate_bps: int = 4_000_000,
    keyint: int = 60,
    # DVR off by default — it adds a second 4K encoder + disk writer that
    # competed with the live encoder and caused stream buffering on the
    # consuming browser. Pass a path explicitly to opt back in.
    recording_path: str | None = None,
    recording_size_cap_bytes: int = 15 * 1024 * 1024 * 1024,
    recording_thread_count: int = 4,
) -> None:
    """mp.Process entry point. Blocks until parent dies or a sentinel arrives."""
    import signal as _signal

    _signal.signal(_signal.SIGINT, _signal.SIG_IGN)

    drain = threading.Thread(target=_drain_loop, args=(frame_queue, target_fps), daemon=True)
    drain.start()

    # DVR recorder runs in-process on its own thread alongside the HTTP server.
    # Doubles encoder cost vs live-only, but keeps a separate seekable artifact
    # for /recording.mp4. Disable by passing recording_path=None.
    if recording_path:
        rec = threading.Thread(
            target=_recorder_loop,
            args=(
                recording_path,
                width,
                height,
                target_fps,
                bitrate_bps,
                keyint,
                recording_size_cap_bytes,
                recording_thread_count,
            ),
            daemon=True,
        )
        rec.start()

    server = _Server(
        (host, port),
        _Handler,
        width=width,
        height=height,
        target_fps=target_fps,
        bitrate_bps=bitrate_bps,
        keyint=keyint,
        recording_path=recording_path,
    )
    print(
        f"[h264] server on http://{host}:{port}/  "
        f"({width}x{height}@{target_fps}fps, {bitrate_bps/1e6:.1f} Mbps, keyint={keyint})"
        + (f"  DVR={recording_path}" if recording_path else "  DVR=off"),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop.set()
        with _cond:
            _cond.notify_all()
        try:
            server.server_close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Standalone smoke test: synthetic frames, no inference required.
# ---------------------------------------------------------------------------
def _smoke_test():
    """Synthetic producer for quick browser-side sanity checks.

    Run:  python models/demos/yolo_eval/_h264_server.py --smoke --port 9091
    """
    import argparse
    import multiprocessing as mp

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9091)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=4_000_000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue(maxsize=2)
    p = ctx.Process(
        target=run_server,
        args=(q, args.host, args.port, args.width, args.height, args.fps, args.bitrate, args.fps * 2),
        daemon=True,
    )
    p.start()

    # Generate animated synthetic frames so the browser has something to render.
    t0 = time.time()
    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    period = 1.0 / args.fps
    next_t = time.perf_counter()
    try:
        while True:
            phase = (time.time() - t0) * 60
            r = int(128 + 127 * np.sin(phase * 0.02))
            g = int(128 + 127 * np.sin(phase * 0.027 + 1.0))
            b = int(128 + 127 * np.sin(phase * 0.031 + 2.0))
            frame[:] = (b, g, r)
            # moving bar
            x = int(phase * 4) % args.width
            frame[:, max(0, x - 20) : x + 20] = 255
            try:
                q.put_nowait(frame.copy())
            except Exception:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(frame.copy())
                except Exception:
                    pass
            next_t += period
            sleep_for = next_t - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_t = time.perf_counter()
    except KeyboardInterrupt:
        q.put_nowait(None)
        p.join(timeout=2)


if __name__ == "__main__":
    _smoke_test()
