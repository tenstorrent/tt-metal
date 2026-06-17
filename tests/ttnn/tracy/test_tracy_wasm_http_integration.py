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

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
                # Best-effort cleanup: the PID may be unparsable, already gone, or owned by
                # another user. None of these should fail the test, so ignore them.
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


# ---------------------------------------------------------------------------
# Live-reload checks (server-side; no silicon / browser / WASM build required).
#
# The Tracy WASM page opens a WebSocket back to the server and reloads when
# embed.tracy changes. The server derives the WebSocket port as HTTP-port + 1,
# so the page must do the same instead of hard-coding it -- a hard-coded port
# silently breaks live-reload on any non-default --port / TRACY_WASM_HTTP_PORT
# (the page then reload-loops every 2s via ws.onclose).
# ---------------------------------------------------------------------------

TOOLS_DIR = REPO_ROOT / "tools"
SERVE_WASM = TOOLS_DIR / "tracy" / "serve_wasm.py"
INDEX_HTML_SRC = REPO_ROOT / "tt_metal" / "third_party" / "tracy" / "profiler" / "wasm" / "index.html"


def test_index_html_ws_port_derived_from_http_port():
    """The live-reload WebSocket port must track the HTTP port, not be hard-coded."""
    if not INDEX_HTML_SRC.is_file():
        pytest.skip(f"Tracy WASM index.html source not found: {INDEX_HTML_SRC}")
    html = INDEX_HTML_SRC.read_text(encoding="utf-8")
    assert "new WebSocket(" in html, "expected a live-reload WebSocket in index.html"
    assert "location.port" in html, (
        "WebSocket port must be derived from location.port (HTTP-port + 1), "
        "otherwise live-reload breaks on any non-default --port / TRACY_WASM_HTTP_PORT"
    )
    assert (
        "':8081'" not in html and ":8081'" not in html and '":8081"' not in html
    ), "WebSocket port 8081 is hard-coded; derive it from location.port instead"


def test_index_html_reload_returns_to_default_trace():
    """On a new-capture reload, the page must drop ?trace= and show the default (newest) trace."""
    if not INDEX_HTML_SRC.is_file():
        pytest.skip(f"Tracy WASM index.html source not found: {INDEX_HTML_SRC}")
    html = INDEX_HTML_SRC.read_text(encoding="utf-8")
    assert "'reload'" in html, "expected a WebSocket 'reload' handler"
    # The reload handler must navigate to the bare path (location.pathname) so a pinned
    # ?trace= is cleared and embed.tracy loads; a plain location.reload() keeps the query.
    assert "location.pathname" in html, (
        "new-capture reload must return to location.pathname (drop ?trace=), "
        "otherwise it re-loads the pinned trace instead of the newest capture"
    )


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_free_port_pair(start: int = 8100) -> int:
    """Return an http port P >= start such that P and P+1 (the WebSocket port) are both free."""
    for base in range(start, 8600, 2):
        if _port_free(base) and _port_free(base + 1):
            return base
    # No pair found: skip explicitly so the function never falls through to an implicit
    # None return (pytest.skip raises Skipped, so this is unreachable as a return path).
    raise pytest.skip.Exception("No free HTTP/WebSocket port pair available")


def _run_in_serve_wasm_env(snippet: str, tt_metal_home: Path, timeout: float = 30.0) -> subprocess.CompletedProcess:
    """Run a Python snippet that imports tracy.serve_wasm, with TT_METAL_HOME pinned to a temp tree."""
    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal_home)
    env["PYTHONPATH"] = str(TOOLS_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-c", snippet], env=env, capture_output=True, text=True, timeout=timeout)


def _wait_http_ready_sock(port: int, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise AssertionError(f"serve_wasm HTTP port {port} never became ready")


def _make_wasm_serve_dir(tmp_path: Path) -> Path:
    """Create a minimal serveable build_wasm tree under tmp_path and return it."""
    wasm_dir = tmp_path / "build" / "profiler" / "build_wasm"
    (wasm_dir / "traces").mkdir(parents=True, exist_ok=True)
    # Minimal static asset so the HTTP root is serveable.
    (wasm_dir / "index.html").write_text("<!doctype html><title>tracy-profiler</title>", encoding="utf-8")
    return wasm_dir


def _spawn_serve_wasm(wasm_dir: Path, tt_metal_home: Path, port: int) -> subprocess.Popen:
    """Start serve_wasm.py over wasm_dir. PROFILER_WASM_DIR is derived from TT_METAL_HOME."""
    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal_home)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(TOOLS_DIR) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.Popen(
        [sys.executable, str(SERVE_WASM), "--port", str(port), "--dir", str(wasm_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _terminate_serve_wasm(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    if out and "Traceback" in out:
        print(out)


def _write_trace(traces_dir: Path, name: str, mtime: float) -> None:
    """Drop a .tracy file with explicit mtime so recency order is deterministic."""
    p = traces_dir / name
    p.write_bytes(b"\x00trace\x00" * 8)
    os.utime(p, (mtime, mtime))


async def _await_reload(ws_uri: str, embed_path: Path, *, recv_timeout_s: float) -> str:
    import websockets

    async with websockets.connect(ws_uri) as ws:
        # Let the server-side watcher establish its mtime baseline (polls every 1s)
        # and let this client register before we change the file.
        await asyncio.sleep(2.5)
        # Mutate embed.tracy: rewrite content and bump mtime so the change is
        # unambiguous regardless of filesystem mtime granularity.
        embed_path.write_bytes(b"reload-trigger" * 64)
        st = embed_path.stat()
        os.utime(embed_path, (st.st_atime, st.st_mtime + 5))
        return await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)


@pytest.mark.timeout(120)
def test_ws_reload_broadcast_on_embed_change(tmp_path):
    """serve_wasm broadcasts 'reload' over the WebSocket when embed.tracy changes."""
    pytest.importorskip("websockets")
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    embed_path = wasm_dir / "embed.tracy"
    embed_path.write_bytes(b"initial")

    port = _find_free_port_pair()
    proc = _spawn_serve_wasm(wasm_dir, tmp_path, port)
    try:
        _wait_http_ready_sock(port)
        msg = asyncio.run(_await_reload(f"ws://127.0.0.1:{port + 1}/", embed_path, recv_timeout_s=10.0))
        assert msg == "reload", f"expected 'reload' broadcast, got {msg!r}"
    finally:
        _terminate_serve_wasm(proc)


@pytest.mark.timeout(120)
def test_traces_listing_ordered_newest_first_by_mtime(tmp_path):
    """GET /traces orders by mtime, not by filename (the cmd-name prefix must not win)."""
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    traces_dir = wasm_dir / "traces"
    # Names chosen so lexicographic-reverse order contradicts chronological order:
    # alphabetically "zzz" > "mmm" > "aaa", but here "aaa" is newest.
    _write_trace(traces_dir, "zzz_oldcmd_2026_01_01_00_00_00.tracy", mtime=1000.0)
    _write_trace(traces_dir, "mmm_midcmd_2026_01_01_00_00_00.tracy", mtime=2000.0)
    _write_trace(traces_dir, "aaa_newcmd_2026_01_01_00_00_00.tracy", mtime=3000.0)

    port = _find_free_port_pair()
    proc = _spawn_serve_wasm(wasm_dir, tmp_path, port)
    try:
        _wait_http_ready_sock(port)
        _, _, body = _http_get(f"http://127.0.0.1:{port}/traces", timeout=30.0)
        listing = json.loads(body.decode("utf-8"))
    finally:
        _terminate_serve_wasm(proc)

    assert listing == [
        "aaa_newcmd_2026_01_01_00_00_00.tracy",
        "mmm_midcmd_2026_01_01_00_00_00.tracy",
        "zzz_oldcmd_2026_01_01_00_00_00.tracy",
    ], f"expected newest-first by mtime, got {listing}"


@pytest.mark.timeout(120)
def test_delete_repoints_embed_to_newest_remaining_by_mtime(tmp_path):
    """Deleting the embedded trace repoints embed.tracy at the newest *remaining* one."""
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    traces_dir = wasm_dir / "traces"
    # Two survivors whose name order contradicts mtime order: "z_old" sorts last
    # alphabetically-reversed-first, but "a_new" is the newest by mtime.
    _write_trace(traces_dir, "z_old_2026_01_01_00_00_00.tracy", mtime=1000.0)
    _write_trace(traces_dir, "a_new_2026_01_01_00_00_00.tracy", mtime=3000.0)
    # The currently-embedded trace, which we will delete.
    _write_trace(traces_dir, "current_2026_01_01_00_00_00.tracy", mtime=2000.0)
    embed_path = wasm_dir / "embed.tracy"
    embed_path.symlink_to("traces/current_2026_01_01_00_00_00.tracy")

    port = _find_free_port_pair()
    proc = _spawn_serve_wasm(wasm_dir, tmp_path, port)
    try:
        _wait_http_ready_sock(port)
        req = Request(f"http://127.0.0.1:{port}/traces/current_2026_01_01_00_00_00.tracy", method="DELETE")
        with urlopen(req, timeout=30.0) as resp:
            assert resp.status == 200, resp.status
            resp.read()
    finally:
        _terminate_serve_wasm(proc)

    assert embed_path.is_symlink(), "embed.tracy should remain a symlink after repoint"
    assert (
        os.readlink(embed_path) == "traces/a_new_2026_01_01_00_00_00.tracy"
    ), f"embed.tracy should repoint to the newest remaining trace, got {os.readlink(embed_path)!r}"


def test_point_embed_at_trace_is_always_a_symlink(tmp_path):
    """point_embed_at_trace makes embed.tracy a relative symlink into traces/, even over a regular file.

    embed.tracy must always be a symlink so the live-reload watcher follows it and DELETE can
    detect/advance the active trace. This guards against reintroducing a regular-file copy (which
    leaves the deleted trace as a stale default and orphans its bytes).
    """
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    (wasm_dir / "traces" / "cap_2026_01_01_00_00_00.tracy").write_bytes(b"data")
    embed_path = wasm_dir / "embed.tracy"
    # Pre-existing *regular file* to prove it gets replaced by a symlink.
    embed_path.write_bytes(b"old-regular-file")

    res = _run_in_serve_wasm_env(
        "from tracy.serve_wasm import point_embed_at_trace\n" "point_embed_at_trace('cap_2026_01_01_00_00_00.tracy')\n",
        tmp_path,
    )
    assert res.returncode == 0, res.stderr

    assert embed_path.is_symlink(), "embed.tracy must be a symlink, not a regular-file copy"
    assert os.readlink(embed_path) == "traces/cap_2026_01_01_00_00_00.tracy"
    assert not (wasm_dir / "embed.tracy.partial").exists(), "temp symlink must not be left behind"


# ---------------------------------------------------------------------------
# Server lifecycle: a new capture must reuse a running server (let the live-reload
# watcher push the refresh), not kill+restart it, and the stale-server cleanup must
# be scoped to its own port so it can't reap a sibling GUI on another port.
# ---------------------------------------------------------------------------


def test_argv_http_port_parsing():
    """_argv_http_port reads --port from a serve_wasm argv (used to port-scope the kill)."""
    from tracy.serve_wasm import DEFAULT_HTTP_PORT, _argv_http_port

    assert _argv_http_port(["python", "serve_wasm.py", "--port", "8123", "--dir", "/x"]) == 8123
    assert _argv_http_port(["python", "serve_wasm.py", "--port=8200"]) == 8200
    assert _argv_http_port(["python", "serve_wasm.py", "--dir", "/x"]) == DEFAULT_HTTP_PORT


@pytest.mark.timeout(120)
def test_launch_reuses_running_server_instead_of_restarting(tmp_path):
    """launch_server_subprocess must reuse a server already serving the port (no kill/respawn)."""
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    (wasm_dir / "embed.tracy").write_bytes(b"initial")
    port = _find_free_port_pair()
    proc = _spawn_serve_wasm(wasm_dir, tmp_path, port)
    try:
        _wait_http_ready_sock(port)
        snippet = (
            "from tracy.serve_wasm import launch_server_subprocess\n"
            f"r = launch_server_subprocess(port={port})\n"
            "print('REUSED' if r is None else 'SPAWNED:%s' % r.pid)\n"
        )
        res = _run_in_serve_wasm_env(snippet, tmp_path)
        assert res.returncode == 0, res.stderr
        assert "REUSED" in res.stdout, f"expected reuse, got stdout={res.stdout!r} stderr={res.stderr!r}"
        # The original server must still be alive -- not killed and restarted.
        time.sleep(0.5)
        assert proc.poll() is None, "running server was killed/restarted instead of reused"
    finally:
        _terminate_serve_wasm(proc)


@pytest.mark.timeout(120)
def test_kill_previous_server_is_port_scoped(tmp_path):
    """Stale-server cleanup must only reap the matching port, never a sibling on another port."""
    if not SERVE_WASM.is_file():
        pytest.skip(f"serve_wasm.py not found: {SERVE_WASM}")

    wasm_dir = _make_wasm_serve_dir(tmp_path)
    port = _find_free_port_pair()
    other_port = _find_free_port_pair(start=port + 2)
    assert other_port != port
    proc = _spawn_serve_wasm(wasm_dir, tmp_path, port)
    try:
        _wait_http_ready_sock(port)

        # Cleanup targeting a DIFFERENT port must leave our server untouched.
        res = _run_in_serve_wasm_env(
            f"from tracy.serve_wasm import _kill_previous_server_process as k; k({other_port})", tmp_path
        )
        assert res.returncode == 0, res.stderr
        time.sleep(0.5)
        assert proc.poll() is None, "server on another port must survive a port-scoped cleanup"

        # Cleanup targeting OUR port must reap it.
        res2 = _run_in_serve_wasm_env(
            f"from tracy.serve_wasm import _kill_previous_server_process as k; k({port})", tmp_path
        )
        assert res2.returncode == 0, res2.stderr
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline and proc.poll() is None:
            time.sleep(0.2)
        assert proc.poll() is not None, "port-scoped cleanup must reap the matching-port server"
    finally:
        _terminate_serve_wasm(proc)
