# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import sys
import subprocess
from pathlib import Path
import socket
import threading
import argparse
from http.server import HTTPServer, SimpleHTTPRequestHandler
import asyncio
import websockets
import json
import signal

from loguru import logger

from tt_metal.tools.profiler.common import (
    PROFILER_ARTIFACTS_DIR,
    PROFILER_WASM_DIR,
    PROFILER_WASM_TRACE_FILE_NAME,
    PROFILER_WASM_TRACES_DIR,
)

clients = set()


def _kill_previous_server_process():
    try:
        output = subprocess.check_output(["ps", "-eo", "pid,cmd"]).decode()
        for line in output.splitlines():
            if __file__ in line and "python" in line:
                pid = int(line.strip().split(None, 1)[0])
                if pid != os.getpid():
                    logger.info(f"Killing previous server process with PID {pid}")
                    os.kill(pid, signal.SIGKILL)
    except Exception as e:
        logger.warning(f"Could not kill previous server process: {e}")


def launch_server_subprocess(directory=None, port=None, daemon=True):
    logger.info("Launching tracy web app GUI server subprocess...")
    _kill_previous_server_process()
    log_path = PROFILER_ARTIFACTS_DIR / "tracy_wasm_gui_server.log"
    cmd = [sys.executable, __file__]
    if directory is not None:
        cmd += ["--dir", directory]
    if port is not None:
        cmd += ["--port", str(port)]
    logger.info(f"Running command: {' '.join(cmd)}")
    log_file = open(log_path, "a", buffering=1)  # line-buffered
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        close_fds=True,
    )
    logger.info(f"Started server with PID {process.pid}, logging to {log_path}")
    return process


def is_server_running(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return False  # Port is free
        except OSError:
            return True  # Port is in use


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):
        print(f"[{self.client_address[0]}] {format % args}")

    def do_GET(self):
        # Serve /traces as a JSON list of available trace files (from PROFILER_WASM_DIR/traces)
        traces_dir = PROFILER_WASM_TRACES_DIR
        if self.path == "/traces":
            try:
                files = []
                print(f"[DEBUG] traces_dir: {traces_dir}")
                if os.path.isdir(traces_dir):
                    files = [f for f in os.listdir(traces_dir) if f.endswith(".tracy")]
                    print(f"[DEBUG] Found trace files: {files}")
                    files.sort(reverse=True)
                else:
                    print(f"[DEBUG] traces_dir does not exist: {traces_dir}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(files).encode("utf-8"))
            except Exception as e:
                print(f"[DEBUG] Exception in /traces: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"[]")
            return
        # Serve /traces/<filename> for downloading a trace file (from PROFILER_WASM_DIR/traces)
        if self.path.startswith("/traces/"):
            import urllib.parse

            filename = self.path[len("/traces/") :]
            # Remove query string if present
            filename = filename.split("?", 1)[0]
            filename = urllib.parse.unquote(filename)
            print(f"[DEBUG] /traces/ raw filename: {repr(filename)}")
            file_path = os.path.join(traces_dir, filename)
            print(f"[DEBUG] /traces/ resolved file_path: '{file_path}'")
            # List directory contents for debugging
            try:
                dir_listing = os.listdir(traces_dir)
                print(f"[DEBUG] /traces/ directory listing: {dir_listing}")
            except Exception as e:
                print(f"[DEBUG] /traces/ could not list directory: {e}")
            # Only allow .tracy files, no path traversal
            if not filename.endswith(".tracy") or "/" in filename or "\\" in filename:
                print(f"[DEBUG] /traces/ rejected filename: '{filename}'")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename")
                return
            if not os.path.isfile(file_path):
                print(f"[DEBUG] /traces/ file not found: '{file_path}'")
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
                return
            print(f"[DEBUG] /traces/ serving file: '{file_path}'")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f"attachment; filename={filename}")
            self.end_headers()
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        # New: /set-embed-tracy/<filename> copies the selected trace to embed.tracy for default loading
        if self.path.startswith("/set-embed-tracy/"):
            import urllib.parse

            filename = self.path[len("/set-embed-tracy/") :]
            filename = urllib.parse.unquote(filename)
            if not filename.endswith(".tracy") or "/" in filename or "\\" in filename:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename")
                return
            src_path = os.path.join(traces_dir, filename)
            dst_path = os.path.join(PROFILER_WASM_DIR, PROFILER_WASM_TRACE_FILE_NAME)
            import shutil

            try:
                # Remove embed.tracy if it is a symlink
                if os.path.islink(dst_path):
                    os.unlink(dst_path)
                shutil.copyfile(src_path, dst_path)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
        # Default: serve as normal static file (from PROFILER_WASM_DIR)
        return super().do_GET()


async def notify_clients():
    print("Embed file changed, notifying clients...")
    if clients:
        print(f"Notifying {clients} clients")
        await asyncio.wait([client.send("reload") for client in clients])


async def watch_embed_file():
    last_mtime = None
    while True:
        try:
            mtime = os.path.getmtime(PROFILER_WASM_TRACE_FILE_NAME)
            if last_mtime is None:
                last_mtime = mtime
            elif mtime != last_mtime:
                last_mtime = mtime
                await notify_clients()
        except FileNotFoundError:
            pass
        await asyncio.sleep(1)


async def ws_handler(websocket):
    clients.add(websocket)
    try:
        async for _ in websocket:
            pass
    finally:
        clients.remove(websocket)


async def websocket_main(ws_port):
    ws_server = await websockets.serve(ws_handler, "0.0.0.0", ws_port)
    print(f"WebSocket server running on ws://0.0.0.0:{ws_port}")
    await watch_embed_file()


def start_websocket_server(ws_port):
    asyncio.run(websocket_main(ws_port))


def run_server(directory, port):
    ws_port = port + 1
    if is_server_running(port):
        print(f"Server is already running on HTTP port {port}. Exiting.")
        return
    if is_server_running(ws_port):
        print(f"Server is already running on WebSocket port {ws_port}. Exiting.")
        return

    os.chdir(directory)
    print(f"Serving WASM from {directory} on http://0.0.0.0:{port} ...")
    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=start_websocket_server, args=(ws_port,), daemon=True)
    ws_thread.start()

    # Start HTTP server
    HTTPServer(("0.0.0.0", port), CORSRequestHandler).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve WASM files with COOP/COEP headers.")
    parser.add_argument(
        "--dir", type=str, default=PROFILER_WASM_DIR, help="Directory to serve (default: current directory)"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    args = parser.parse_args()
    run_server(args.dir, args.port)
