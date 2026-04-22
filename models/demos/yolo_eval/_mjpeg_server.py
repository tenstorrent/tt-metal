#!/usr/bin/env python3
"""
Standalone MJPEG HTTP server with optional mode-switching UI.

Frame sources:
  --frame-file FILE              Single-pane: serve one JPEG file as-is.
  --left-file L --right-file R   Dual-pane: read two JPEGs, composite side-by-side.

Mode switching (unified demo):
  --mode-file FILE               Enables mode-toggle UI + /api/mode endpoint.

Launched as a subprocess to keep the HTTP server isolated from inference.
"""
import json
import os
import socket as _socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

_jpeg_lock = threading.Lock()
_jpeg_data = b""  # last-sent JPEG (fallback if FIFO is empty)
_jpeg_seq = 0
# Small FIFO of unique frames waiting to be sent.  Absorbs producer jitter
# so a 20-30 ms pause on the prep side doesn't translate into a held display
# frame.  Each display tick pops one; if the FIFO is empty we re-send the
# last frame.  Bounded so we don't drift arbitrarily far behind live.
_jpeg_fifo: deque = deque()
_fifo_max = 2  # ~33 ms added latency at 60 Hz; absorbs most producer pauses

_mode_file: str | None = None
_send_hz: float = 60.0  # MJPEG send pacing clock (default 60 Hz to match displays)


# ---------------------------------------------------------------------------
# Single-file reader
# ---------------------------------------------------------------------------


def _single_reader(path: str):
    global _jpeg_seq
    last_mtime = 0.0
    last_size = 0
    while True:
        try:
            st = os.stat(path)
            if st.st_mtime != last_mtime or st.st_size != last_size:
                with open(path, "rb") as f:
                    data = f.read()
                if data:
                    with _jpeg_lock:
                        _jpeg_fifo.append(data)
                        # Drop oldest if over capacity (keep freshest N).
                        while len(_jpeg_fifo) > _fifo_max:
                            _jpeg_fifo.popleft()
                        _jpeg_seq += 1
                    last_mtime = st.st_mtime
                    last_size = st.st_size
        except (FileNotFoundError, OSError):
            pass
        time.sleep(0.0005)  # 0.5 ms poll — catch writes even if bursty


# ---------------------------------------------------------------------------
# Dual-file reader (composites left + right side-by-side)
# ---------------------------------------------------------------------------


def _dual_reader(left_path: str, right_path: str):
    global _jpeg_data, _jpeg_seq
    import cv2

    left_mtime = right_mtime = 0.0
    left_img = right_img = None

    while True:
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
            ok, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                with _jpeg_lock:
                    _jpeg_data = buf.tobytes()
                    _jpeg_seq += 1

        time.sleep(0.003)


# ---------------------------------------------------------------------------
# Mode-file helpers
# ---------------------------------------------------------------------------


def _read_mode() -> str:
    """Read the current mode from the mode file (default: side-by-side)."""
    if not _mode_file:
        return "side-by-side"
    try:
        with open(_mode_file) as f:
            return f.read().strip() or "side-by-side"
    except (FileNotFoundError, OSError):
        return "side-by-side"


def _write_mode(mode: str):
    if not _mode_file:
        return
    tmp = _mode_file + ".tmp"
    with open(tmp, "w") as f:
        f.write(mode)
    os.replace(tmp, _mode_file)


# ---------------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head>
<title>YOLO Demo</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#e6edf3;font-family:-apple-system,BlinkMacSystemFont,
  "Segoe UI",Helvetica,Arial,sans-serif;display:flex;flex-direction:column;
  align-items:center;min-height:100vh}
#banner{width:100%;text-align:center;padding:10px 0;font-size:1.1rem;
  font-weight:600;color:#f0c040;background:#1a1e26;display:none}
#stream-wrap{flex:1;display:flex;justify-content:center;align-items:center;padding:8px}
#stream-wrap img{max-width:100%;max-height:calc(100vh - 100px);border-radius:4px}
#controls{padding:12px;display:__CONTROLS_DISPLAY__}
#mode-btn{padding:10px 28px;font-size:1rem;font-weight:600;border:none;
  border-radius:6px;cursor:pointer;color:#fff;background:#238636;transition:background .2s}
#mode-btn:hover{background:#2ea043}
#mode-btn.large-model{background:#1f6feb}
#mode-btn.large-model:hover{background:#388bfd}
</style>
</head>
<body>
<div id="banner"></div>
<div id="stream-wrap"><img id="stream" src="/stream"></div>
<div id="controls">
  <button id="mode-btn" onclick="toggleMode()">Switch to Large Model Mode</button>
</div>
<script>
let currentMode = "side-by-side";
const btn = document.getElementById("mode-btn");
const banner = document.getElementById("banner");

function updateUI(mode, loading) {
  currentMode = mode;
  if (mode === "side-by-side") {
    btn.textContent = "Switch to Large Model Mode";
    btn.className = "";
  } else {
    btn.textContent = "Switch to Side-by-Side";
    btn.className = "large-model";
  }
  if (loading) {
    banner.style.display = "block";
    banner.textContent = loading;
    btn.disabled = true;
  } else {
    banner.style.display = "none";
    btn.disabled = false;
  }
}

async function toggleMode() {
  const next = currentMode === "side-by-side" ? "large-model" : "side-by-side";
  const loadMsg = next === "large-model"
    ? "Loading YOLOv8L (4-device SAHI)..."
    : "Switching to Side-by-Side...";
  updateUI(next, loadMsg);
  try {
    const resp = await fetch("/api/mode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({mode: next})
    });
    const data = await resp.json();
    if (data.mode) currentMode = data.mode;
  } catch(e) { console.error(e); }
}

async function pollMode() {
  try {
    const resp = await fetch("/api/mode");
    const data = await resp.json();
    if (data.loading) {
      btn.disabled = true;
      banner.style.display = "block";
    } else {
      if (data.mode && data.mode !== currentMode) {
        updateUI(data.mode, null);
      }
      banner.style.display = "none";
      btn.disabled = false;
    }
  } catch(e) {}
}
setInterval(pollMode, 1500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/stream":
            self._serve_mjpeg()
        elif self.path == "/api/mode":
            self._get_mode()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/mode":
            self._post_mode()
        else:
            self.send_error(404)

    def _serve_html(self):
        display = "block" if _mode_file else "none"
        html = _HTML_TEMPLATE.replace("__CONTROLS_DISPLAY__", display)
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def _serve_mjpeg(self):
        global _jpeg_data
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        # Strict send clock + small FIFO.  Each tick pops the oldest queued
        # unique frame (absorbs producer bursts and pauses); on empty, re-send
        # the last frame.  Together this gives a smooth, jitter-free stream
        # even when the prep pipeline has 20-30 ms variance.
        PACE = 1.0 / _send_hz
        next_send = time.perf_counter()
        try:
            while True:
                now = time.perf_counter()
                delay = next_send - now
                if delay > 0:
                    time.sleep(delay)
                elif delay < -2 * PACE:
                    next_send = now
                with _jpeg_lock:
                    if _jpeg_fifo:
                        jpeg = _jpeg_fifo.popleft()
                        _jpeg_data = jpeg
                    else:
                        jpeg = _jpeg_data
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                next_send += PACE
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _get_mode(self):
        raw = _read_mode()
        if raw.startswith("loading:"):
            mode = raw.split(":", 1)[1]
            loading = True
        else:
            mode = raw
            loading = False
        body = json.dumps({"mode": mode, "loading": loading}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _post_mode(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw)
        except json.JSONDecodeError:
            req = {}
        new_mode = req.get("mode", "side-by-side")
        if new_mode not in ("side-by-side", "large-model"):
            new_mode = "side-by-side"
        _write_mode(new_mode)
        body = json.dumps({"mode": new_mode, "ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _mode_file, _send_hz
    import argparse

    p = argparse.ArgumentParser(description="Standalone MJPEG HTTP server.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9090)
    p.add_argument("--frame-file", default=None, help="Single-pane JPEG file.")
    p.add_argument("--left-file", default=None, help="Left-pane JPEG (dual mode).")
    p.add_argument("--right-file", default=None, help="Right-pane JPEG (dual mode).")
    p.add_argument("--mode-file", default=None, help="Shared file for mode switching (enables toggle UI).")
    p.add_argument("--send-hz", type=float, default=60.0, help="Strict send clock in Hz (match display refresh).")
    args = p.parse_args()

    _mode_file = args.mode_file
    _send_hz = args.send_hz

    dual = args.left_file and args.right_file
    if not dual and not args.frame_file:
        p.error("Provide --frame-file OR --left-file + --right-file")

    if dual:
        t = threading.Thread(target=_dual_reader, args=(args.left_file, args.right_file), daemon=True)
    else:
        t = threading.Thread(target=_single_reader, args=(args.frame_file,), daemon=True)
    t.start()

    class _Server(ThreadingMixIn, HTTPServer):
        allow_reuse_address = True
        daemon_threads = True

        def server_bind(self):
            self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
            try:
                self.socket.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEPORT, 1)
            except (AttributeError, OSError):
                pass
            super().server_bind()

    server = _Server((args.host, args.port), _Handler)
    mode = "dual" if dual else "single"
    print(f"[http] MJPEG server pid={os.getpid()} ({mode}) on http://{args.host}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
