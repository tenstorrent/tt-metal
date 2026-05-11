# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end check: ``python -m tracy`` wraps a minimal trace pytest, then the Tracy
WASM static server answers HTTP as expected (COOP/COEP, assets, ``/traces``).

Uses stdlib HTTP only. Requires a Tracy-enabled build, ``tracy-capture``, profiler
WASM artifacts under ``build/profiler/build_wasm``, and silicon (same bar as
``test_trace_runs``).

Inner pytest target is kept as ``test_trace_runs`` so this file does not recurse
into itself.

SSH port forwarding: use **fixed** ports so ``LocalForward`` matches the server.
Default HTTP port is **8080** (WebSocket **8081**). Set ``TRACY_WASM_HTTP_PORT``
to override (must leave room for ``port+1``). Prefer ``127.0.0.1`` in SSH and in
the browser (see IPv6 ``localhost`` pitfalls).

The test **does not** stop the WASM server afterward (Tracy leaves it up for the
GUI). Tearing it down here caused SSH ``LocalForward`` probes to hit closed ports
and spam ``connection refused`` until clients gave up. The next run **begins** by
clearing ``port`` / ``port+1`` so reruns stay deterministic.

To force cleanup after assertions (e.g. shared CI runners), set
``TRACY_WASM_HTTP_TEST_TEARDOWN=1``. Otherwise stop the server yourself if needed
(e.g. ``fuser -k 8080/tcp`` / ``8081/tcp``).

Run (from repo root, with ``PYTHONPATH`` including ``tools`` like other pytest runs):

  pytest tests/ttnn/tracy/test_tracy_wasm_http_integration.py -v

  pytest tests/ttnn/tracy/test_tracy_wasm_http_integration.py::test_tracy_wasm_gui_http_after_tracy_capture -v
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _wasm_http_port() -> int:
    """Match Tracy WASM defaults / SSH ``-L 8080:127.0.0.1:8080 -L 8081:127.0.0.1:8081``."""
    return int(os.environ.get("TRACY_WASM_HTTP_PORT", "8080"))


def _header_ci(headers, name: str) -> str | None:
    want = name.lower()
    for k, v in headers.items():
        if k.lower() == want:
            return v
    return None


def _http_get(url: str, timeout: float):
    with urlopen(url, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, resp.headers, body


def _wait_server_ready(base: str, attempts: int, delay_s: float) -> None:
    health = base.rstrip("/") + "/"
    last_err: Exception | None = None
    for _ in range(attempts):
        try:
            _http_get(health, timeout=5.0)
            return
        except (URLError, HTTPError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(delay_s)
    raise AssertionError(f"Tracy WASM server not reachable at {health!r}") from last_err


def _kill_tcp_port(port: int) -> None:
    fuser = shutil.which("fuser")
    if fuser:
        subprocess.run([fuser, "-k", f"{port}/tcp"], capture_output=True, timeout=15, check=False)
        return
    lsof = shutil.which("lsof")
    if lsof:
        r = subprocess.run([lsof, "-ti", f":{port}"], capture_output=True, text=True, timeout=10, check=False)
        for line in r.stdout.strip().splitlines():
            try:
                os.kill(int(line.strip()), signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass


def _assert_tracy_wasm_http_ok(base: str) -> None:
    status, headers, body = _http_get(base, timeout=30.0)
    assert status == 200
    assert _header_ci(headers, "Cross-Origin-Opener-Policy") == "same-origin"
    assert _header_ci(headers, "Cross-Origin-Embedder-Policy") == "require-corp"
    assert b"tracy-profiler" in body.lower()

    root = base.rstrip("/") + "/"
    for path, min_len in (("tracy-profiler.js", 500), ("tracy-profiler.wasm", 5000)):
        st, _, data = _http_get(root + path, timeout=60.0)
        assert st == 200, path
        assert len(data) >= min_len, path

    st, th, tbody = _http_get(root + "traces", timeout=30.0)
    assert st == 200
    ct = _header_ci(th, "Content-Type") or ""
    assert "application/json" in ct
    json.loads(tbody.decode("utf-8"))


@pytest.mark.timeout(900)
def test_tracy_wasm_gui_http_after_tracy_capture():
    wasm_dir = REPO_ROOT / "build" / "profiler" / "build_wasm"
    if not wasm_dir.is_dir():
        pytest.skip(f"Missing WASM serve root: {wasm_dir}")

    port = _wasm_http_port()
    if not (1 <= port <= 65534):
        pytest.fail("TRACY_WASM_HTTP_PORT must be in [1, 65534] (WebSocket uses port+1).")
    base = f"http://127.0.0.1:{port}/"
    _kill_tcp_port(port)
    _kill_tcp_port(port + 1)

    inner_target = "tests/ttnn/tracy/test_trace_runs.py::test_with_ops_single_core[100-5]"
    cmd = [
        sys.executable,
        "-m",
        "tracy",
        "--web-app-port",
        str(port),
        "-m",
        "pytest",
        inner_target,
        "-q",
        "--tb=short",
    ]

    env = os.environ.copy()
    tools_path = str(REPO_ROOT / "tools")
    env["PYTHONPATH"] = tools_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else tools_path
    env.setdefault("TT_METAL_HOME", str(REPO_ROOT))

    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, timeout=900)
        assert (
            proc.returncode == 0
        ), "Tracy capture / pytest leg failed; WASM server is only started after a successful capture."

        _wait_server_ready(base, attempts=max(3, int(30.0 / 0.25)), delay_s=0.25)
        _assert_tracy_wasm_http_ok(base)
    finally:
        if os.environ.get("TRACY_WASM_HTTP_TEST_TEARDOWN", "").strip().lower() in ("1", "true", "yes"):
            _kill_tcp_port(port)
            _kill_tcp_port(port + 1)
