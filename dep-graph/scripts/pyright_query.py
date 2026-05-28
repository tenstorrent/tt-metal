"""Minimal Pyright LSP client for batch type queries.

Talks to `pyright-langserver --stdio` over JSON-RPC. For each `(file, line,
character)` position we request a hover, parse Pyright's response to
extract the inferred type, and accumulate the results.

Used by py_index to fill in receiver_type for parameters Pyright can
infer but our static propagator can't — typical case: a function whose
parameter type comes from its call sites (inter-procedural inference).

Usage:
    client = PyrightClient(workspace_root)
    client.start()
    types = client.hover_at_positions([(file_path, line, col), ...])
    client.shutdown()

The protocol is straightforward:
  - LSP framing: each message is `Content-Length: N\r\n\r\n<json>`
  - We use synchronous request/response (no async event loop)
  - Pyright sends a torrent of `window/logMessage` notifications during
    startup — we drain them until our `initialize` response comes back.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path


HOVER_TYPE_PATTERN = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


@dataclass
class HoverResult:
    """Pyright's hover response, parsed for the type signature."""
    raw_value: str | None
    inferred_type: str | None     # extracted from the ```python block
    error: str | None = None


class PyrightClient:
    def __init__(self, workspace_root: Path, python_version: str = "3.10") -> None:
        self.workspace_root = workspace_root.resolve()
        self.python_version = python_version
        self.proc: subprocess.Popen | None = None
        self._next_id = 0
        self._response_buf: dict[int, dict] = {}
        self._stderr_buf: list[str] = []
        self._reader_thread: threading.Thread | None = None
        self._reader_done = False

    def start(self) -> None:
        self.proc = subprocess.Popen(
            ["pyright-langserver", "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.workspace_root),
        )
        self._reader_thread = threading.Thread(target=self._stderr_drain, daemon=True)
        self._reader_thread.start()
        self._initialize()

    def _stderr_drain(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        for line in self.proc.stderr:
            self._stderr_buf.append(line.decode("utf-8", errors="replace"))

    def _send(self, method: str, params: dict, is_notification: bool = False) -> dict | None:
        assert self.proc is not None and self.proc.stdin is not None
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if not is_notification:
            self._next_id += 1
            msg["id"] = self._next_id
        body = json.dumps(msg).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()
        if is_notification:
            return None
        return self._read_until_id(self._next_id)

    def _read_message(self) -> dict | None:
        assert self.proc is not None and self.proc.stdout is not None
        # Read Content-Length header.
        content_length: int | None = None
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return None
            line = line.decode("utf-8", errors="replace").strip()
            if line.lower().startswith("content-length:"):
                content_length = int(line.split(":", 1)[1].strip())
            elif line == "":
                break  # end of headers
        if content_length is None:
            return None
        body = self.proc.stdout.read(content_length)
        return json.loads(body)

    def _read_until_id(self, target_id: int) -> dict | None:
        """Read messages until we see a response with the target id.
        Notifications and unrelated responses are stashed/ignored."""
        if target_id in self._response_buf:
            return self._response_buf.pop(target_id)
        while True:
            msg = self._read_message()
            if msg is None:
                return None
            if "id" in msg and "method" not in msg:
                # It's a response.
                mid = msg["id"]
                if mid == target_id:
                    return msg
                self._response_buf[mid] = msg
            # Notifications (no id or with method) are ignored.

    def _initialize(self) -> None:
        root_uri = self.workspace_root.as_uri()
        resp = self._send("initialize", {
            "processId": None,
            "rootUri": root_uri,
            "workspaceFolders": [{"uri": root_uri, "name": "workspace"}],
            "capabilities": {
                "textDocument": {
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                },
            },
            "initializationOptions": {
                "settings": {
                    "python": {"pythonVersion": self.python_version},
                    "pyright": {"useLibraryCodeForTypes": True},
                }
            },
        })
        if resp is None or "error" in resp:
            raise RuntimeError(f"pyright initialize failed: {resp}")
        # Send `initialized` notification.
        self._send("initialized", {}, is_notification=True)

    def open_file(self, file_path: Path) -> None:
        """Sync the file's content to pyright (required before hovers)."""
        text = file_path.read_text(encoding="utf-8", errors="replace")
        self._send("textDocument/didOpen", {
            "textDocument": {
                "uri": file_path.resolve().as_uri(),
                "languageId": "python",
                "version": 1,
                "text": text,
            },
        }, is_notification=True)

    def hover(self, file_path: Path, line: int, character: int) -> HoverResult:
        """Request a hover at the given 0-indexed position."""
        resp = self._send("textDocument/hover", {
            "textDocument": {"uri": file_path.resolve().as_uri()},
            "position": {"line": line, "character": character},
        })
        if resp is None:
            return HoverResult(None, None, error="no response")
        if "error" in resp:
            return HoverResult(None, None, error=str(resp["error"]))
        result = resp.get("result")
        if not result:
            return HoverResult(None, None)
        contents = result.get("contents")
        if isinstance(contents, dict):
            raw = contents.get("value", "")
        elif isinstance(contents, str):
            raw = contents
        elif isinstance(contents, list) and contents:
            raw = contents[0] if isinstance(contents[0], str) else contents[0].get("value", "")
        else:
            raw = ""
        # Pyright wraps type info in ```python ... ``` fenced blocks.
        match = HOVER_TYPE_PATTERN.search(raw)
        inferred = match.group(1).strip() if match else None
        return HoverResult(raw, inferred)

    def shutdown(self) -> None:
        if self.proc is None:
            return
        try:
            self._send("shutdown", {})
            self._send("exit", {}, is_notification=True)
        except Exception:
            pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        self.proc = None


def main() -> int:
    """Smoke test: hover at a known position in a file passed on argv."""
    if len(sys.argv) < 4:
        print("usage: pyright_query.py <workspace_root> <file.py> <line> <char>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    file_path = Path(sys.argv[2])
    line = int(sys.argv[3])
    char = int(sys.argv[4]) if len(sys.argv) >= 5 else 0
    client = PyrightClient(root)
    client.start()
    try:
        client.open_file(file_path)
        result = client.hover(file_path, line, char)
        print(f"raw: {result.raw_value!r}")
        print(f"inferred_type: {result.inferred_type!r}")
        return 0
    finally:
        client.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
