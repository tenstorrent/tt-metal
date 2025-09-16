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

from loguru import logger

from tt_metal.tools.profiler.common import PROFILER_ARTIFACTS_DIR, PROFILER_WASM_DIR, PROFILER_WASM_TRACE_FILE_NAME

clients = set()


def _kill_previous_server_process():
    import subprocess
    import signal
    import os

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
