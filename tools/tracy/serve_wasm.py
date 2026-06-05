# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import functools
import os
import shutil
import stat
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

from tracy.common import (
    PROFILER_ARTIFACTS_DIR,
    PROFILER_WASM_DIR,
    PROFILER_WASM_TRACE_FILE_NAME,
    PROFILER_WASM_TRACES_DIR,
)

# Default HTTP port for the WASM static server. Override with:
#   - CLI:  python .../serve_wasm.py --port <n>
#   - env:  TRACY_WASM_HTTP_PORT=<n>  (used by launch_server_subprocess when port= is omitted)
# WebSocket port is always HTTP+1 (see run_server).
DEFAULT_HTTP_PORT = 8080

_SERVER_LOG_BASENAME = "tracy_wasm_gui_server.log"


def _resolve_under_root(root: Path, strict: bool) -> Path:
    try:
        return root.resolve(strict=strict)
    except FileNotFoundError:
        return root.resolve(strict=False)


def _safe_trace_file_path(traces_dir: Path, filename: str) -> Path | None:
    """Resolve a trace basename under traces_dir, or None if invalid or escapes the directory."""
    if not filename or "\x00" in filename:
        return None
    if "\r" in filename or "\n" in filename:
        return None
    if "/" in filename or "\\" in filename:
        return None
    if filename != os.path.basename(filename):
        return None
    if not filename.endswith(".tracy") or filename == ".tracy":
        return None
    root = _resolve_under_root(traces_dir, strict=False)
    candidate = _resolve_under_root(root / filename, strict=False)
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _ensure_resolved_path_under(allowed_root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and require it to lie under ``allowed_root`` (both resolved)."""
    root_r = _resolve_under_root(allowed_root, strict=False)
    resolved = _resolve_under_root(candidate, strict=False)
    try:
        resolved.relative_to(root_r)
    except ValueError as e:
        raise RuntimeError(f"Refusing path outside {root_r}: {candidate}") from e
    return resolved


def _validated_wasm_serve_directory(directory: str | os.PathLike[str]) -> Path:
    """Ensure the serve root stays under PROFILER_WASM_DIR (path traversal / arbitrary chdir)."""
    root = _resolve_under_root(Path(PROFILER_WASM_DIR), strict=False)
    resolved = _resolve_under_root(Path(directory), strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise ValueError(f"Refusing to serve outside {root}: {directory}") from e
    return resolved


def _profiler_server_log_path() -> Path:
    """Log file path fixed under PROFILER_ARTIFACTS_DIR (constant basename only)."""
    artifacts = _resolve_under_root(Path(PROFILER_ARTIFACTS_DIR), strict=False)
    log_path = artifacts / _SERVER_LOG_BASENAME
    resolved = _resolve_under_root(log_path, strict=False)
    try:
        resolved.relative_to(artifacts)
    except ValueError as e:
        raise RuntimeError(f"Invalid profiler log path under {artifacts}") from e
    if resolved.name != _SERVER_LOG_BASENAME:
        raise RuntimeError("Profiler log path basename mismatch")
    return resolved


def _copy_validated_trace_to_embed(trace_filename: str) -> None:
    """Copy ``traces/<trace_filename>`` onto ``embed.tracy`` under ``PROFILER_WASM_DIR``.

    Uses ``open(..., dir_fd=…)`` (POSIX ``openat``) with **relative** basenames under directory
    file descriptors for traces / WASM roots. Path-based SAST tools often flag ``open(abs_path)``
    when any predecessor value touched request input; relative opens under pinned directories avoid
    passing a tainted absolute path string into ``open``.
    """
    traces_root = _resolve_under_root(Path(PROFILER_WASM_TRACES_DIR), strict=False)
    wasm_root = _resolve_under_root(Path(PROFILER_WASM_DIR), strict=False)
    validated = _safe_trace_file_path(traces_root, trace_filename)
    if validated is None or not validated.is_file():
        raise ValueError("invalid trace file")

    trace_bn = validated.name
    safe_bn: str | None = None
    traces_base = os.path.abspath(str(traces_root))
    with os.scandir(traces_base) as scan:
        for entry in scan:
            if entry.name != trace_bn:
                continue
            if entry.is_file(follow_symlinks=False):
                safe_bn = entry.name
                break
    if safe_bn is None:
        raise ValueError("invalid trace file")
    if os.sep in safe_bn or safe_bn in (".", ".."):
        raise RuntimeError("invalid traces directory entry")

    wasm_base = os.path.abspath(str(wasm_root))
    embed_rel = PROFILER_WASM_TRACE_FILE_NAME

    traces_dir_fd = os.open(traces_base, os.O_RDONLY | os.O_DIRECTORY)
    try:
        wasm_dir_fd = os.open(wasm_base, os.O_RDONLY | os.O_DIRECTORY)
        try:
            try:
                st = os.stat(embed_rel, dir_fd=wasm_dir_fd, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            if st is not None and stat.S_ISLNK(st.st_mode):
                os.unlink(embed_rel, dir_fd=wasm_dir_fd)

            src_fd = os.open(safe_bn, os.O_RDONLY, dir_fd=traces_dir_fd)
            try:
                dst_fd = os.open(
                    embed_rel,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o644,
                    dir_fd=wasm_dir_fd,
                )
                try:
                    with os.fdopen(src_fd, "rb", closefd=False) as sf, os.fdopen(dst_fd, "wb", closefd=False) as df:
                        shutil.copyfileobj(sf, df)
                finally:
                    os.close(dst_fd)
            finally:
                os.close(src_fd)
        finally:
            os.close(wasm_dir_fd)
    finally:
        os.close(traces_dir_fd)


def _embed_trace_dest_path() -> Path:
    """Path to embed.tracy under PROFILER_WASM_DIR (constant basename only).

    Do not Path.resolve() the full path: embed.tracy is usually a symlink to traces/*.tracy,
    and resolving would follow the symlink and break the basename check (and crash the WS thread).
    """
    wasm_root = _resolve_under_root(Path(PROFILER_WASM_DIR), strict=False)
    dest = wasm_root / PROFILER_WASM_TRACE_FILE_NAME
    try:
        dest.relative_to(wasm_root)
    except ValueError as e:
        raise RuntimeError(f"Invalid embed trace path under {wasm_root}") from e
    if dest.name != PROFILER_WASM_TRACE_FILE_NAME:
        raise RuntimeError("Embed trace basename mismatch")
    return dest


def _cmdline_targets_this_script(argv: list[str], script_realpath: str) -> bool:
    joined = " ".join(argv)
    if script_realpath in joined or __file__ in joined:
        return True
    for arg in argv[1:]:
        try:
            if os.path.realpath(arg) == script_realpath:
                return True
        except OSError:
            continue
    return False


def _resolve_wasm_http_port(explicit_port=None):
    """Return HTTP listen port; WebSocket uses this value + 1, so max is 65534."""
    if explicit_port is not None:
        port = int(explicit_port)
    else:
        env_port = os.environ.get("TRACY_WASM_HTTP_PORT")
        port = int(env_port) if env_port else DEFAULT_HTTP_PORT
    if not (1 <= port <= 65534):
        raise ValueError(f"TRACY web HTTP port must be in [1, 65534] (need room for WebSocket on port+1), got {port}")
    return port


clients = set()


def _kill_previous_server_process():
    """Stop other serve_wasm.py instances without spawning ps/shell (avoids command-injection noise)."""
    script_real = os.path.realpath(__file__)
    my_pid = os.getpid()
    proc = Path("/proc")
    if not proc.is_dir():
        logger.warning("No /proc; skipping stale Tracy WASM server cleanup.")
        return
    try:
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if pid == my_pid:
                continue
            cmdline_file = entry / "cmdline"
            try:
                raw = cmdline_file.read_bytes()
            except (FileNotFoundError, PermissionError, OSError):
                continue
            if not raw:
                continue
            argv = [p.decode("utf-8", errors="replace") for p in raw.split(b"\x00") if p]
            if len(argv) < 2:
                continue
            interp = os.path.basename(argv[0]).lower()
            if not interp.startswith("python"):
                continue
            if not _cmdline_targets_this_script(argv, script_real):
                continue
            logger.info(f"Killing previous server process with PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                logger.debug("Previous Tracy WASM server PID %s already exited before SIGKILL", pid)
            except PermissionError as e:
                logger.warning(f"Could not kill PID {pid}: {e}")
    except Exception as e:
        logger.warning(f"Could not kill previous server process: {e}")


def launch_server_subprocess(directory=None, port=None, daemon=True):
    logger.info("Launching tracy web app GUI server subprocess...")
    http_port = _resolve_wasm_http_port(port)
    ws_port = http_port + 1
    logger.info(
        f"Tracy WASM web UI (open in browser): http://localhost:{http_port}/ "
        f"(WebSocket for live reload: ws://localhost:{ws_port}/)"
    )
    _kill_previous_server_process()
    log_path = _profiler_server_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, __file__, "--port", str(http_port)]
    if directory is not None:
        validated_dir = _validated_wasm_serve_directory(directory)
        wasm_root = _resolve_under_root(Path(PROFILER_WASM_DIR), strict=False)
        validated_dir = _ensure_resolved_path_under(wasm_root, validated_dir)
        cmd += ["--dir", str(validated_dir)]
    logger.info(f"Running command: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "a", buffering=1) as log_file:
        dup_fd = os.dup(log_file.fileno())
    try:
        child_log = os.fdopen(dup_fd, "a", buffering=1)
    except Exception:
        os.close(dup_fd)
        raise
    try:
        # argv: sys.executable, this script, --port <int>, optional --dir <path validated above>; shell=False.
        process = subprocess.Popen(  # nosec B603
            cmd,
            env=env,
            stdout=child_log,
            stderr=child_log,
            start_new_session=True,
            close_fds=True,
            shell=False,
        )
    except Exception:
        child_log.close()
        raise
    child_log.close()
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
    def do_DELETE(self):
        traces_dir = PROFILER_WASM_TRACES_DIR
        if self.path.startswith("/traces/"):
            import urllib.parse

            filename = self.path[len("/traces/") :]
            # Remove query string if present
            filename = filename.split("?", 1)[0]
            filename = urllib.parse.unquote(filename)
            logger.debug(f"DELETE /traces/ raw filename: {repr(filename)}")
            file_path = _safe_trace_file_path(Path(traces_dir), filename)
            logger.debug(f"DELETE /traces/ resolved file_path: '{file_path}'")
            if file_path is None:
                logger.debug(f"DELETE /traces/ rejected filename: '{filename}'")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename")
                return
            if not file_path.is_file():
                logger.debug(f"DELETE /traces/ file not found: '{file_path}'")
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
                return
            try:
                traces_root = _resolve_under_root(Path(traces_dir), strict=False)
                to_remove = _ensure_resolved_path_under(traces_root, file_path)
                to_remove.unlink()
                logger.debug(f"DELETE /traces/ deleted file: '{to_remove}'")
                # Check if embed.tracy is a symlink to the deleted file
                embed_path = _embed_trace_dest_path()
                abs_deleted = _resolve_under_root(to_remove, strict=False)
                if embed_path.is_symlink():
                    abs_target = (embed_path.parent / embed_path.readlink()).resolve()
                    try:
                        abs_target.relative_to(traces_root)
                    except ValueError:
                        pass
                    else:
                        if abs_target == abs_deleted:
                            embed_path.unlink()
                            # Find first available .tracy file (basename-only, under traces_root)
                            files = []
                            for f in os.listdir(traces_dir):
                                sp = _safe_trace_file_path(traces_root, f)
                                if sp is not None and sp.is_file():
                                    files.append(f)
                            files.sort(reverse=True)
                            if files:
                                safe_pick = _safe_trace_file_path(traces_root, files[0])
                                if safe_pick is not None:
                                    new_target = os.path.relpath(str(safe_pick), embed_path.parent)
                                    os.symlink(new_target, embed_path)
                                    logger.debug(f"embed.tracy now points to: {new_target}")
                            else:
                                logger.debug("No .tracy files left to point embed.tracy to.")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Deleted")
            except Exception as e:
                logger.debug(f"DELETE /traces/ error deleting file: {e}")
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
        # If not /traces/, return 404
        self.send_response(404)
        self.end_headers()

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):
        logger.debug(f"[{self.client_address[0]}] {format % args}")

    def do_GET(self):
        # Serve /traces as a JSON list of available trace files (from PROFILER_WASM_DIR/traces)
        traces_dir = PROFILER_WASM_TRACES_DIR
        if self.path == "/traces":
            try:
                files = []
                logger.debug(f"traces_dir: {traces_dir}")
                if os.path.isdir(traces_dir):
                    files = [f for f in os.listdir(traces_dir) if f.endswith(".tracy")]
                    logger.debug(f"Found trace files: {files}")
                    files.sort(reverse=True)
                else:
                    logger.debug(f"traces_dir does not exist: {traces_dir}")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(files).encode("utf-8"))
            except Exception as e:
                logger.debug(f"Exception in /traces: {e}")
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
            logger.debug(f"/traces/ raw filename: {repr(filename)}")
            file_path = _safe_trace_file_path(Path(traces_dir), filename)
            logger.debug(f"/traces/ resolved file_path: '{file_path}'")
            # List directory contents for debugging
            try:
                dir_listing = os.listdir(traces_dir)
                logger.debug(f"/traces/ directory listing: {dir_listing}")
            except Exception as e:
                logger.debug(f"/traces/ could not list directory: {e}")
            if file_path is None:
                logger.debug(f"/traces/ rejected filename: '{filename}'")
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename")
                return
            if not file_path.is_file():
                logger.debug(f"/traces/ file not found: '{file_path}'")
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")
                return
            traces_root = _resolve_under_root(Path(traces_dir), strict=False)
            safe_file = _ensure_resolved_path_under(traces_root, file_path)
            logger.debug(f"/traces/ serving file: '{safe_file}'")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Disposition", f"attachment; filename={safe_file.name}")
            self.end_headers()
            with safe_file.open("rb") as trace_stream:
                while True:
                    chunk = trace_stream.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        # New: /set-embed-tracy/<filename> copies the selected trace to embed.tracy for default loading
        if self.path.startswith("/set-embed-tracy/"):
            import urllib.parse

            filename = self.path[len("/set-embed-tracy/") :]
            filename = urllib.parse.unquote(filename)
            try:
                _copy_validated_trace_to_embed(filename)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"OK")
            except ValueError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid filename")
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(str(e).encode("utf-8"))
            return
        # Default: serve as normal static file (from PROFILER_WASM_DIR)
        super().do_GET()
        return


async def notify_clients():
    logger.debug("Embed file changed, notifying clients...")
    if clients:
        logger.debug(f"Notifying {len(clients)} clients")
        # gather() accepts coroutines directly; asyncio.wait() rejects raw coroutines on Python 3.11+.
        await asyncio.gather(*(client.send("reload") for client in clients))


async def watch_embed_file():
    embed_abs = _embed_trace_dest_path()
    last_mtime = None
    while True:
        try:
            mtime = embed_abs.stat().st_mtime
            if last_mtime is None:
                last_mtime = mtime
            elif mtime != last_mtime:
                last_mtime = mtime
                await notify_clients()
        except FileNotFoundError:
            # embed.tracy may not exist yet (or may be temporarily absent); keep polling.
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
    # Bind to loopback by default; remote viewing is via SSH port-forwarding (see docs).
    async with websockets.serve(ws_handler, "127.0.0.1", ws_port):
        logger.info(f"WebSocket server running on ws://127.0.0.1:{ws_port}")
        await watch_embed_file()


def start_websocket_server(ws_port):
    asyncio.run(websocket_main(ws_port))


def run_server(directory, port):
    if not (1 <= port <= 65534):
        logger.error(f"HTTP port must be in [1, 65534] (WebSocket uses port+1), got {port}")
        sys.exit(1)
    ws_port = port + 1
    if is_server_running(port):
        logger.info(f"Server is already running on HTTP port {port}. Exiting.")
        return
    if is_server_running(ws_port):
        logger.info(f"Server is already running on WebSocket port {ws_port}. Exiting.")
        return

    try:
        serve_root = _validated_wasm_serve_directory(directory)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    wasm_root = _resolve_under_root(Path(PROFILER_WASM_DIR), strict=False)
    serve_root = _ensure_resolved_path_under(wasm_root, serve_root)
    # Bind to loopback by default; the UI and the trace download/delete endpoints
    # are local-use oriented and remote access is via SSH port-forwarding (see docs).
    logger.info(f"Serving WASM from {serve_root} on http://127.0.0.1:{port} ...")
    # Start WebSocket server in a separate thread
    ws_thread = threading.Thread(target=start_websocket_server, args=(ws_port,), daemon=True)
    ws_thread.start()

    # Start HTTP server (set serve root via handler; avoids os.chdir on argv-derived paths).
    handler_factory = functools.partial(CORSRequestHandler, directory=str(serve_root))
    HTTPServer(("127.0.0.1", port), handler_factory).serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve WASM files with COOP/COEP headers.")
    parser.add_argument(
        "--dir", type=str, default=PROFILER_WASM_DIR, help="Directory to serve (default: current directory)"
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_HTTP_PORT, help=f"Port to serve on (default: {DEFAULT_HTTP_PORT})"
    )
    args = parser.parse_args()
    run_server(args.dir, args.port)
