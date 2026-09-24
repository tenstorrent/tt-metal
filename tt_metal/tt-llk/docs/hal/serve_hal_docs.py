#!/usr/bin/env python3
"""Serve the Blackhole HAL documentation on a configurable TCP port.

Examples:
    python3 docs/hal/serve_hal_docs.py
    python3 docs/hal/serve_hal_docs.py --port 8080
    python3 docs/hal/serve_hal_docs.py --host 127.0.0.1 --port 8080
    python3 docs/hal/serve_hal_docs.py --regenerate --port 8080

The default bind address, 0.0.0.0, makes the server reachable through any
network interface. Only the documentation pages and source files linked by
those pages are served; the rest of the checkout is not exposed.

The server also persists descriptor and field naming suggestions in
``cfg_naming_suggestions.json`` and page edits in ``document_edits.json``. It
supports Python 3.6 and newer; regenerating the field catalog requires Python
3.9 or newer.
"""

import argparse
import datetime
import json
import os
import posixpath
import socket
import subprocess
import sys
import threading
import uuid
from html.parser import HTMLParser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[2]
HAL_DOCS = Path(__file__).resolve().parent
INDEX_URL = "/docs/hal/index.html"
FAVICON_URL = "/docs/hal/favicon.svg"
SUGGESTIONS_URL = "/__suggestions__"
SUGGESTIONS_FILE = HAL_DOCS / "cfg_naming_suggestions.json"
MAX_SUGGESTION_BODY_BYTES = 16 * 1024
DOCUMENT_URL = "/__document__"
DOCUMENT_EDITS_FILE = HAL_DOCS / "document_edits.json"
MAX_DOCUMENT_BODY_BYTES = 2 * 1024 * 1024
DOCUMENTS = (
    HAL_DOCS / "index.html",
    HAL_DOCS / "cfg_hal_interface.html",
    HAL_DOCS / "cfg_field_catalog.html",
    HAL_DOCS / "address_counters_hal_interface.html",
    HAL_DOCS / "math_counters_hal_interface.html",
    HAL_DOCS / "gpr_ops_hal_interface.html",
    HAL_DOCS / "atomic_hal_interface.html",
    HAL_DOCS / "mop_hal_interface.html",
    HAL_DOCS / "replay_hal_interface.html",
    HAL_DOCS / "sync_hal_interface.html",
    HAL_DOCS / "unpack_hal_interface.html",
    HAL_DOCS / "fpu_hal_interface.html",
)
REFRESH_LOCK = threading.Lock()
SUGGESTIONS_LOCK = threading.Lock()
DOCUMENT_EDITS_LOCK = threading.Lock()


def editable_document_urls():
    """Return the exact public URL accepted by the page-edit endpoint."""
    return frozenset(
        "/{}".format(document.relative_to(ROOT).as_posix()) for document in DOCUMENTS
    )


def empty_document_edits():
    return {"version": 1, "pages": {}}


def load_document_edits():
    """Load the durable page-edit store, treating a missing store as empty."""
    if not DOCUMENT_EDITS_FILE.exists():
        return empty_document_edits()
    payload = json.loads(DOCUMENT_EDITS_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported document-edit file format")
    if not isinstance(payload.get("pages"), dict):
        raise ValueError("document-edit store must contain a pages object")
    return payload


def write_json_atomically(path, payload):
    """Atomically replace a JSON file in the documentation directory."""
    temporary = path.with_name(".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def empty_suggestions():
    return {"version": 1, "suggestions": []}


def load_suggestions():
    """Load the durable naming-suggestion store, treating a missing store as empty."""
    if not SUGGESTIONS_FILE.exists():
        return empty_suggestions()
    payload = json.loads(SUGGESTIONS_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("unsupported naming-suggestion file format")
    if not isinstance(payload.get("suggestions"), list):
        raise ValueError("naming-suggestion store must contain a suggestions list")
    return payload


def write_suggestions(payload):
    """Atomically replace the durable naming-suggestion store."""
    write_json_atomically(SUGGESTIONS_FILE, payload)


def suggestion_text(payload, key, maximum, required=False):
    value = payload.get(key, "")
    if not isinstance(value, str):
        raise ValueError("{} must be text".format(key))
    value = value.strip()
    if required and not value:
        raise ValueError("{} is required".format(key))
    if len(value) > maximum:
        raise ValueError("{} must be at most {} characters".format(key, maximum))
    return value


def validated_suggestion(payload):
    """Validate client data and return the record written to disk."""
    if not isinstance(payload, dict):
        raise ValueError("request body must be a JSON object")
    target_type = suggestion_text(payload, "target_type", 16, required=True)
    if target_type not in {"descriptor", "field"}:
        raise ValueError("target_type must be descriptor or field")

    record = {
        "id": uuid.uuid4().hex,
        "target_type": target_type,
        "target_id": suggestion_text(payload, "target_id", 500, required=True),
        "family": suggestion_text(payload, "family", 32, required=True),
        "descriptor": suggestion_text(payload, "descriptor", 500, required=True),
        "field": suggestion_text(payload, "field", 200),
        "current_name": suggestion_text(payload, "current_name", 500, required=True),
        "suggested_name": suggestion_text(
            payload, "suggested_name", 200, required=True
        ),
        "suggested_by": suggestion_text(payload, "suggested_by", 100),
        "rationale": suggestion_text(payload, "rationale", 1000),
        "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
    }
    if target_type == "field" and not record["field"]:
        raise ValueError("field is required for field suggestions")
    if target_type == "descriptor" and record["field"]:
        raise ValueError("field must be empty for descriptor suggestions")
    return record


class LinkCollector(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.links = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag in {"a", "link"}:
            target = attributes.get("href")
        elif tag == "script":
            target = attributes.get("src")
        else:
            return
        if target:
            self.links.append(target)


def request_path(value):
    """Return a decoded, normalized absolute URL path without query data."""
    path = posixpath.normpath(unquote(urlsplit(value).path))
    return path if path.startswith("/") else "/{}".format(path)


def discover_public_files():
    """Build the allowlist from the documentation's local links."""
    public = {
        "/{}".format(document.relative_to(ROOT).as_posix()) for document in DOCUMENTS
    }

    for document in DOCUMENTS:
        collector = LinkCollector()
        collector.feed(document.read_text(encoding="utf-8"))
        document_url = "/{}".format(document.relative_to(ROOT).as_posix())

        for link in collector.links:
            parsed = urlsplit(link)
            if parsed.scheme or parsed.netloc or link.startswith("#"):
                continue
            target = request_path(urljoin(document_url, link))
            filesystem_target = (ROOT / target.lstrip("/")).resolve()
            try:
                filesystem_target.relative_to(ROOT)
            except ValueError:
                continue
            if filesystem_target.is_file():
                public.add(target)

    return frozenset(public)


class DocumentationHandler(SimpleHTTPRequestHandler):
    public_files = frozenset()

    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = request_path(self.path)
        if path == DOCUMENT_URL:
            self.get_document_edit()
            return
        if path == SUGGESTIONS_URL:
            try:
                with SUGGESTIONS_LOCK:
                    payload = load_suggestions()
                self.send_json(200, payload)
            except (OSError, UnicodeError, ValueError) as error:
                self.send_json(500, {"ok": False, "message": str(error)})
            return
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        path = request_path(self.path)
        if path == DOCUMENT_URL:
            self.save_document_edit()
            return
        if path == SUGGESTIONS_URL:
            self.save_suggestion()
            return
        if path != "/__refresh__":
            self.send_error(404, "Unknown documentation action")
            return
        if not REFRESH_LOCK.acquire(False):
            self.send_json(
                409, {"ok": False, "message": "A refresh is already running"}
            )
            return

        try:
            if sys.version_info < (3, 9):
                self.send_json(
                    500,
                    {
                        "ok": False,
                        "message": "Refreshing the field catalog requires Python 3.9 or newer",
                    },
                )
                return

            result = subprocess.run(
                [sys.executable, str(HAL_DOCS / "gen_cfg_field_catalog.py")],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            if result.returncode != 0:
                self.send_json(
                    500,
                    {
                        "ok": False,
                        "message": "Field catalog generation failed",
                        "output": result.stdout,
                    },
                )
                return

            DocumentationHandler.public_files = discover_public_files()
            self.send_json(
                200,
                {
                    "ok": True,
                    "message": "Documentation regenerated",
                    "published_files": len(DocumentationHandler.public_files),
                },
            )
        except (OSError, UnicodeError) as error:
            self.send_json(500, {"ok": False, "message": str(error)})
        finally:
            REFRESH_LOCK.release()

    def document_path_from_query(self):
        query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
        values = query.get("path", [])
        if len(values) != 1:
            raise ValueError("path query parameter is required")
        path = request_path(values[0])
        if path not in editable_document_urls():
            raise ValueError("document is not editable")
        return path

    def get_document_edit(self):
        try:
            path = self.document_path_from_query()
            with DOCUMENT_EDITS_LOCK:
                payload = load_document_edits()
                page = payload["pages"].get(path)
            self.send_json(200, {"ok": True, "path": path, "document": page})
        except ValueError as error:
            self.send_json(400, {"ok": False, "message": str(error)})
        except (OSError, UnicodeError) as error:
            self.send_json(500, {"ok": False, "message": str(error)})

    def read_json_body(self, maximum):
        content_type = (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length")
        if not 0 < content_length <= maximum:
            raise ValueError(
                "request body must be between 1 and {} bytes".format(maximum)
            )
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def save_document_edit(self):
        try:
            request = self.read_json_body(MAX_DOCUMENT_BODY_BYTES)
            if not isinstance(request, dict):
                raise ValueError("request body must be a JSON object")
            raw_path = request.get("path", "")
            if not isinstance(raw_path, str):
                raise ValueError("path must be text")
            path = request_path(raw_path)
            if path not in editable_document_urls():
                raise ValueError("document is not editable")
            content = request.get("content")
            if not isinstance(content, str):
                raise ValueError("content must be Markdown text")
            if not content.strip():
                raise ValueError("content must not be empty")
            if len(content.encode("utf-8")) > MAX_DOCUMENT_BODY_BYTES:
                raise ValueError("document is too large")

            page = {
                "content": content,
                "updated_at": datetime.datetime.utcnow()
                .replace(microsecond=0)
                .isoformat()
                + "Z",
            }
            with DOCUMENT_EDITS_LOCK:
                payload = load_document_edits()
                payload["pages"][path] = page
                write_json_atomically(DOCUMENT_EDITS_FILE, payload)
            self.send_json(200, {"ok": True, "path": path, "document": page})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.send_json(400, {"ok": False, "message": str(error)})
        except (OSError, UnicodeError) as error:
            self.send_json(500, {"ok": False, "message": str(error)})

    def save_suggestion(self):
        content_type = (
            self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        )
        if content_type != "application/json":
            self.send_json(
                415, {"ok": False, "message": "Content-Type must be application/json"}
            )
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"ok": False, "message": "Invalid Content-Length"})
            return
        if not 0 < content_length <= MAX_SUGGESTION_BODY_BYTES:
            self.send_json(
                413,
                {
                    "ok": False,
                    "message": "Suggestion body must be between 1 and {} bytes".format(
                        MAX_SUGGESTION_BODY_BYTES
                    ),
                },
            )
            return

        try:
            raw_body = self.rfile.read(content_length).decode("utf-8")
            suggestion = validated_suggestion(json.loads(raw_body))
            with SUGGESTIONS_LOCK:
                payload = load_suggestions()
                payload["suggestions"].append(suggestion)
                write_suggestions(payload)
            self.send_json(201, {"ok": True, "suggestion": suggestion})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.send_json(400, {"ok": False, "message": str(error)})
        except (OSError, UnicodeError) as error:
            self.send_json(500, {"ok": False, "message": str(error)})

    def send_head(self):
        path = request_path(self.path)
        if path == "/":
            self.send_response(302)
            self.send_header("Location", INDEX_URL)
            self.end_headers()
            return None
        if path == "/favicon.ico":
            self.send_response(302)
            self.send_header("Location", FAVICON_URL)
            self.end_headers()
            return None
        if path not in self.public_files:
            self.send_error(
                404, "This file is not part of the published HAL documentation"
            )
            return None

        target = (ROOT / path.lstrip("/")).resolve()
        try:
            target.relative_to(ROOT)
        except ValueError:
            self.send_error(
                404, "This file is not part of the published HAL documentation"
            )
            return None
        if not target.is_file():
            self.send_error(404, "Documentation file not found")
            return None
        return SimpleHTTPRequestHandler.send_head(self)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        SimpleHTTPRequestHandler.end_headers(self)

    def guess_type(self, path):
        if Path(path).suffix in {".h", ".py", ".yaml"}:
            return "text/plain; charset=utf-8"
        return SimpleHTTPRequestHandler.guess_type(self, path)

    def copyfile(self, source, outputfile):
        try:
            SimpleHTTPRequestHandler.copyfile(self, source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers commonly cancel an in-flight response during navigation.
            pass


class DocumentationServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def port_number(value):
    port = int(value)
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 0 and 65535")
    return port


def local_ipv4_addresses():
    addresses = set()
    try:
        for result in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(result[4][0])
    except socket.gaierror:
        pass

    # A UDP connect chooses the machine's outbound interface without sending
    # application data or requiring the destination to answer.
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe.connect(("192.0.2.1", 9))
            addresses.add(probe.getsockname()[0])
        finally:
            probe.close()
    except OSError:
        pass

    return sorted(address for address in addresses if not address.startswith("127."))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Serve Blackhole HAL documentation and persist CFG naming suggestions.",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="address to bind (default: 0.0.0.0; use 127.0.0.1 for port-forward-only access)",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=port_number,
        help="TCP port to bind (default: 8000; 0 chooses a free port)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="regenerate cfg_field_catalog.html before starting (requires Python 3.9+)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.regenerate:
        if sys.version_info < (3, 9):
            raise SystemExit("--regenerate requires Python 3.9 or newer")
        subprocess.run(
            [sys.executable, str(HAL_DOCS / "gen_cfg_field_catalog.py")],
            cwd=str(ROOT),
            check=True,
        )

    for document in DOCUMENTS:
        if not document.is_file():
            raise SystemExit("documentation file not found: {}".format(document))

    DocumentationHandler.public_files = discover_public_files()
    os.chdir(str(ROOT))
    try:
        server = DocumentationServer((args.host, args.port), DocumentationHandler)
    except OSError as error:
        raise SystemExit(
            "cannot listen on {}:{}: {}".format(args.host, args.port, error)
        )

    host, port = server.server_address[:2]
    print(
        "Serving {} documentation files".format(len(DocumentationHandler.public_files)),
        flush=True,
    )
    print("Local:       http://127.0.0.1:{}/".format(port), flush=True)
    if host in {"0.0.0.0", "::"}:
        for address in local_ipv4_addresses():
            print("Network:     http://{}:{}/".format(address, port), flush=True)
        print(
            "Port-forward remote port {}, then open http://127.0.0.1:{}/".format(
                port, port
            ),
            flush=True,
        )
        print(
            "Warning: this is an unauthenticated HTTP server; use it only on a trusted network.",
            flush=True,
        )
    elif host.startswith("127.") or host == "::1":
        print("Loopback only: a machine-name URL will not work.", flush=True)
        print(
            "Port-forward remote port {}, then open http://127.0.0.1:{}/".format(
                port, port
            ),
            flush=True,
        )
    else:
        print("Bound to:    http://{}:{}/".format(host, port), flush=True)
    print("Press Ctrl-C to stop.", flush=True)
    print(
        "Use Refresh in the page toolbar after editing the documentation.", flush=True
    )
    print(
        "Naming suggestions are saved in {}.".format(
            SUGGESTIONS_FILE.relative_to(ROOT)
        ),
        flush=True,
    )
    print(
        "Page edits are saved in {}.".format(DOCUMENT_EDITS_FILE.relative_to(ROOT)),
        flush=True,
    )

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
