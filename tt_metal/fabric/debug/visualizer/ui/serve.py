#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Serves the fabric debug visualizer, and the captures under a folder, on a lab host or a Mac.

Usage: serve.py [--files-path <folder>] [--port 8765] [--remote] [--no-browser]

<folder> is a logs folder, where manifest, capture, and decode artifacts are stored in a per-run folder. It defaults to
TT_METAL_LOGS_PATH.

The server listens only on 127.0.0.1 and opens the browser. With --remote, for a host you reached over ssh, it prints
the ssh command that forwards the port to your machine instead.

Endpoints:
  GET /                the UI's files
  GET /api/captures    every capture found, with its runs and rank files
  GET /files/<path>    a decoded file or a .bin that /api/captures listed; handles Range requests, and sends decoded
                       files gzip-compressed when the browser accepts it
"""

import argparse
import errno
import getpass
import gzip
import json
import os
import re
import socket
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

UI_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PORT = 8765
NUM_PORTS_TO_TRY = 10

DECODED_FILE_PATTERN = re.compile(r"^fabric_decoded_rank_(\d+)_of_(\d+)\.json$")
# How deep a decoded file may sit below the folder searched. A folder of several logs folders puts it at
# <logs>/generated/fabric_debug/runs/<run_id>/captures/<capture_id>/, 7 folders down.
MAX_SEARCH_DEPTH = 10
MAX_DIRECTORIES_SEARCHED = 20000

UI_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
}
COPY_CHUNK_BYTES = 1 << 20


def choose_search_root(path):
    """Returns the folder to search for captures: a logs folder's runs folder when it has one, else `path` itself."""
    for runs_directory in (path / "generated" / "fabric_debug" / "runs", path / "fabric_debug" / "runs"):
        if runs_directory.is_dir():
            return runs_directory
    return path


def find_decoded_files(search_root):
    """Returns every decoded file at captures/<capture_id>/ below `search_root`, and whether the search stopped early.

    The folder itself may also be a capture folder, or a run folder that holds captures/.
    """
    decoded_files = []
    num_directories_searched = 0
    pending = [(search_root, 0)]
    while pending:
        directory, depth = pending.pop()
        num_directories_searched += 1
        if num_directories_searched > MAX_DIRECTORIES_SEARCHED:
            return sorted(decoded_files), True
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        is_capture_directory = directory.parent.name == "captures"
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_file() and is_capture_directory and DECODED_FILE_PATTERN.match(entry.name):
                decoded_files.append(Path(entry.path))
            elif entry.is_dir(follow_symlinks=False) and not is_capture_directory and depth < MAX_SEARCH_DEPTH:
                pending.append((Path(entry.path), depth + 1))
    return sorted(decoded_files), False


def describe_file(path, url_root):
    """Returns the name, URL and size of `path`, served at files/<path relative to url_root>."""
    relative_path = path.relative_to(url_root).as_posix()
    description = {"name": path.name, "url": "files/" + quote(relative_path), "size_bytes": path.stat().st_size}
    return description, relative_path


def list_captures(search_root):
    """Groups the decoded files under `search_root` into captures, runs and ranks.

    A capture is one capture_id under one runs folder: an original's runs/<run_id>/captures/<capture_id>/, or a
    copy's <capture_id>/<run_id>/captures/<capture_id>/. Returns the listing and every servable file by relative path.
    """
    decoded_files, search_stopped_early = find_decoded_files(search_root)
    servable_files = {}
    captures = {}
    for decoded_file in decoded_files:
        capture_directory = decoded_file.parent
        run_directory = capture_directory.parent.parent
        capture_key = (str(run_directory.parent), capture_directory.name)
        capture = captures.setdefault(
            capture_key,
            {
                "capture_id": capture_directory.name,
                "folder": str(run_directory.parent),
                "runs": {},
            },
        )
        run = capture["runs"].setdefault(
            run_directory.name,
            {"run_id": run_directory.name, "folder": str(run_directory), "ranks": []},
        )
        rank_number, world_size = (int(number) for number in DECODED_FILE_PATTERN.match(decoded_file.name).groups())
        decoded, decoded_relative_path = describe_file(decoded_file, search_root)
        servable_files[decoded_relative_path] = decoded_file
        bin_file = capture_directory / f"fabric_capture_rank_{rank_number}_of_{world_size}.bin"
        bin_description = None
        if bin_file.is_file():
            bin_description, bin_relative_path = describe_file(bin_file, search_root)
            servable_files[bin_relative_path] = bin_file
        run["ranks"].append(
            {"rank": rank_number, "world_size": world_size, "decoded": decoded, "bin": bin_description}
        )

    listing = []
    for capture in captures.values():
        runs = sorted(capture["runs"].values(), key=lambda run: run["run_id"])
        for run in runs:
            run["ranks"].sort(key=lambda rank: rank["rank"])
        listing.append({**capture, "runs": runs})
    # Capture ids are UTC timestamps, so the newest sorts last.
    listing.sort(key=lambda capture: (capture["capture_id"], capture["folder"]), reverse=True)
    return {
        "root": str(search_root),
        "search_stopped_early": search_stopped_early,
        "captures": listing,
    }, servable_files


def parse_byte_range(range_header, file_size):
    """Returns (start, end) inclusive for a single `bytes=` range, or None when it can't be satisfied."""
    match = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", range_header)
    if not match or not (match.group(1) or match.group(2)):
        return None
    if match.group(1):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else file_size - 1
    else:
        suffix_length = int(match.group(2))
        start = max(0, file_size - suffix_length)
        end = file_size - 1
    end = min(end, file_size - 1)
    if start > end:
        return None
    return start, end


class CaptureCatalog:
    """The captures under the search root, rescanned when the page asks for the list, and decoded files gzipped once."""

    def __init__(self, search_root):
        self.search_root = search_root
        self.lock = threading.Lock()
        self.listing, self.servable_files = list_captures(search_root)
        self.gzipped_files = {}

    def rescan(self):
        listing, servable_files = list_captures(self.search_root)
        with self.lock:
            self.listing, self.servable_files = listing, servable_files
        return listing

    def find_servable_file(self, relative_path):
        with self.lock:
            return self.servable_files.get(relative_path)

    def read_gzipped(self, path):
        """Returns `path` gzip-compressed, from memory while the file is unchanged."""
        stat = path.stat()
        file_version = (stat.st_mtime_ns, stat.st_size)
        with self.lock:
            cached = self.gzipped_files.get(path)
            if cached and cached[0] == file_version:
                return cached[1]
            compressed = gzip.compress(path.read_bytes(), compresslevel=6)
            self.gzipped_files[path] = (file_version, compressed)
            return compressed


class VisualizerRequestHandler(BaseHTTPRequestHandler):
    server_version = "fabric-debug-visualizer"
    catalog = None

    def do_GET(self):
        self.handle_request(send_body=True)

    def do_HEAD(self):
        self.handle_request(send_body=False)

    def handle_request(self, send_body):
        url_path = unquote(urlsplit(self.path).path)
        if url_path == "/api/captures":
            self.send_json(self.catalog.rescan(), send_body)
        elif url_path.startswith("/files/"):
            self.send_capture_file(url_path[len("/files/") :], send_body)
        else:
            self.send_ui_file(url_path, send_body)

    def send_json(self, value, send_body):
        body = json.dumps(value, indent=2).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def send_ui_file(self, url_path, send_body):
        relative_path = url_path.lstrip("/") or "index.html"
        path = (UI_DIRECTORY / relative_path).resolve()
        if UI_DIRECTORY not in path.parents or path.suffix not in UI_CONTENT_TYPES or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", UI_CONTENT_TYPES[path.suffix])
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def send_capture_file(self, relative_path, send_body):
        # Only files /api/captures listed are served, looked up by exact path, so no path leads outside them.
        path = self.catalog.find_servable_file(relative_path)
        if path is None:
            self.catalog.rescan()
            path = self.catalog.find_servable_file(relative_path)
        if path is None or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "not a decoded file or .bin of a capture this server found")
            return
        content_type = "application/json; charset=utf-8" if path.suffix == ".json" else "application/octet-stream"
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header is None and path.suffix == ".json" and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = self.catalog.read_gzipped(path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if send_body:
                self.wfile.write(body)
            return

        start, end = 0, file_size - 1
        status = HTTPStatus.OK
        if range_header is not None:
            byte_range = parse_byte_range(range_header, file_size)
            if byte_range is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            start, end = byte_range
            status = HTTPStatus.PARTIAL_CONTENT
        num_bytes = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(num_bytes))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()
        if not send_body:
            return
        with path.open("rb") as file:
            file.seek(start)
            remaining = num_bytes
            while remaining > 0:
                chunk = file.read(min(COPY_CHUNK_BYTES, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, format, *args):
        # Only errors: the page makes a Range request for every raw-bytes view.
        pass

    def log_error(self, format, *args):
        sys.stderr.write(f"serve.py: {self.address_string()} {format % args}\n")


def bind_server(port):
    """Binds 127.0.0.1 at `port`, or the next free port of the few after it."""
    for candidate_port in range(port, port + NUM_PORTS_TO_TRY):
        try:
            return ThreadingHTTPServer(("127.0.0.1", candidate_port), VisualizerRequestHandler)
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    sys.exit(f"serve.py: ports {port} to {port + NUM_PORTS_TO_TRY - 1} are all in use; pick another with --port")


def describe_captures(listing):
    lines = []
    for capture in listing["captures"]:
        num_ranks = sum(len(run["ranks"]) for run in capture["runs"])
        num_runs = len(capture["runs"])
        lines.append(
            f"  {capture['capture_id']}  {num_runs} run{'s' if num_runs != 1 else ''}, "
            f"{num_ranks} rank{'s' if num_ranks != 1 else ''}  {capture['folder']}"
        )
    return lines


def print_startup(listing, port, remote):
    url = f"http://localhost:{port}/"
    num_captures = len(listing["captures"])
    print(f"fabric debug visualizer: {num_captures} capture{'s' if num_captures != 1 else ''} under {listing['root']}")
    for line in describe_captures(listing):
        print(line)
    if num_captures == 0:
        print("  none found; pass --files-path <logs folder>")
    if listing["search_stopped_early"]:
        print(f"  stopped after {MAX_DIRECTORIES_SEARCHED} folders; pass a --files-path closer to the captures")
    print()
    if remote:
        user = getpass.getuser()
        host = socket.gethostname()
        print("On your own machine, forward the port:")
        print()
        print(f"    ssh -N -L {port}:localhost:{port} {user}@{host}")
        print()
        print("  through a jump host, add -J <jump host>, for example: ssh -N -J exabox -L ...")
        print(f"  if {port} is taken on your machine, change the first {port} and the URL's port to match")
        print()
        print(f"then open {url}")
    else:
        print(f"Open {url}")
    print("Ctrl-C stops the server.")
    sys.stdout.flush()


def main(argv):
    parser = argparse.ArgumentParser(
        description="Serves the fabric debug visualizer and the captures under a folder.",
    )
    parser.add_argument(
        "--files-path",
        type=Path,
        help="logs folder, copied capture folder, or run or capture folder (default: TT_METAL_LOGS_PATH)",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default {DEFAULT_PORT})")
    parser.add_argument(
        "--remote",
        action="store_true",
        help="this host was reached over ssh: print the port forward to run on your machine instead of opening a browser",
    )
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    options = parser.parse_args(argv)

    files_path = options.files_path
    if files_path is None:
        if not os.environ.get("TT_METAL_LOGS_PATH"):
            sys.exit("serve.py: TT_METAL_LOGS_PATH isn't set; set it, or pass --files-path <folder>")
        files_path = Path(os.environ["TT_METAL_LOGS_PATH"])
    path = files_path.expanduser().resolve()
    if not path.is_dir():
        sys.exit(f"serve.py: {path} is not a folder")

    VisualizerRequestHandler.catalog = CaptureCatalog(choose_search_root(path))
    server = bind_server(options.port)
    port = server.server_address[1]
    print_startup(VisualizerRequestHandler.catalog.listing, port, options.remote)
    if not options.remote and not options.no_browser:
        threading.Timer(0.3, webbrowser.open, args=(f"http://localhost:{port}/",)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nserve.py: stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
