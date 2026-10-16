# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat server behind uvicorn (``tools/qwen38_asgi.py``) without a device: the container settings from an
environment (the device nodes, the profile, the context, the server flags, the refusals), the composed argv parses
with the server's own parser and a flag the server does not know is refused there, the runtime identity of a tree
without git history, the lifespan (the server runs as a child process: ``READY`` gates the startup, SIGTERM + wait on the shutdown with
the exit status logged, a child that ends before ``READY`` fails the startup, a child that ends on its own ends the
process with its status) and the forwarding (a request's bytes reach the server unchanged, a streamed reply
arrives as it is written, 503 before ``READY``, a client's hang-up closes the server's connection)."""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_asgi as asgi
from models.demos.blackhole.qwen38_flash_next.tools import runtime_admission

# -- the settings ----------------------------------------------------------------------------------------------------


def _devices(tmp_path: Path, nodes) -> Path:
    root = tmp_path / "tenstorrent"
    root.mkdir(exist_ok=True)
    for entry in root.iterdir():
        entry.unlink()
    for node in nodes:
        (root / str(node)).touch()
    return root


def _settings(tmp_path: Path, nodes=(4, 5, 6, 7), **environ: str) -> asgi.Qwen38ContainerServer:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir(exist_ok=True)
    environment = {"QWEN38_CACHE_ROOT": str(tmp_path / "tensor-cache"), **environ}
    return asgi.Qwen38ContainerServer.from_environment(
        environment, device_root=_devices(tmp_path, nodes), checkpoint=checkpoint, now="20260926T000000Z"
    )


def test_settings_on_four_other_nodes_move_the_line_profile_and_parse_with_the_server(tmp_path) -> None:
    server = _settings(tmp_path, QWEN38_ALLOCATED_CONTEXT="65536")
    argv = list(server.argv)
    assert argv[argv.index("--device-nodes") + 1] == "4,5,6,7"
    assert argv[argv.index("--allocated-context") + 1] == "65536"
    assert argv[argv.index("--acceptance-prompts") + 1] == str(asgi.ACCEPTANCE_PROMPTS)
    assert argv[argv.index("--component-cache-root") + 1].endswith("caches/c65536-p150-line/components")
    assert argv[argv.index("--routed-bf4-scratch-root") + 1].endswith("caches/bf4-experts")
    assert argv[argv.index("--evidence") + 1] == str(server.evidence)
    assert argv[argv.index("--host") + 1] == "127.0.0.1" and argv[argv.index("--port") + 1] == "18000"
    assert "--prefill-slab" not in argv  # the slab and --mtp are alternatives; the default form is the MTP chain
    assert (
        server.environment["TT_VISIBLE_DEVICES"] == "0,1,2,3"
    )  # UMD's indices over the four nodes the container holds
    assert server.environment["QWEN38_HARDWARE_MODE"] == "diagnostic_non_promoting"
    assert server.environment["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert "TT_MESH_GRAPH_DESC_PATH" not in server.environment  # the line profile has no descriptor
    assert server.profile.device_nodes == (4, 5, 6, 7) and server.profile.host == "p150-line"
    # the server derives the same visible set when it takes --device-nodes on a host that presents exactly these nodes,
    # and the node numbers on a host that presents more (the historical hosts, nodes 0..7)
    moved = hardware_profiles.P150_LINE.with_device_nodes((4, 5, 6, 7), present=(4, 5, 6, 7))
    assert (moved.device_nodes, moved.visible_devices) == ((4, 5, 6, 7), "0,1,2,3")
    host = hardware_profiles.P150_LINE.with_device_nodes((4, 5, 6, 7), present=range(8))
    assert (host.device_nodes, host.visible_devices) == ((4, 5, 6, 7), "4,5,6,7")
    assert hardware_profiles.present_device_nodes(tmp_path / "absent") == ()
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as chat_server

    parsed = chat_server._parser().parse_args(argv)  # the server's own parser admits every composed flag
    assert (parsed.mtp, parsed.sampling, parsed.require_json_96) == (4, True, True)
    assert parsed.stall_seconds == 300.0 and parsed.device_nodes == "4,5,6,7" and parsed.allocated_context == 65536
    assert asgi.ACCEPTANCE_PROMPTS.is_dir() and (asgi.ACCEPTANCE_PROMPTS / "SHA256SUMS").is_file()


def test_settings_on_the_profile_nodes_pass_no_device_nodes_and_a_ring_profile_exports_its_descriptor(tmp_path) -> None:
    server = _settings(tmp_path, nodes=(0, 1, 2, 3), QWEN38_HARDWARE_PROFILE="tt-quietbox")
    assert "--device-nodes" not in server.argv
    assert server.environment["TT_VISIBLE_DEVICES"] == "0,1,2,3"
    descriptor = Path(server.environment["TT_MESH_GRAPH_DESC_PATH"])
    assert descriptor.is_file() and descriptor.name == hardware_profiles.QUIETBOX.mesh_graph_descriptor
    assert server.argv[server.argv.index("--hardware-profile") + 1] == "tt-quietbox"


@pytest.mark.parametrize(
    "nodes, environ, match",
    [
        ((4, 5, 6), {}, "exactly four chips"),
        ((4, 5, 6, 7, 8), {}, "exactly four chips"),
        ((4, 5, 6, 7), {"QWEN38_ALLOCATED_CONTEXT": "40000"}, "QWEN38_ALLOCATED_CONTEXT must be one of"),
        ((4, 5, 6, 7), {"QWEN38_HARDWARE_PROFILE": "nowhere"}, "QWEN38_HARDWARE_PROFILE 'nowhere' is not one of"),
        ((4, 5, 6, 7), {"QWEN38_INNER_PORT": "80"}, "QWEN38_INNER_PORT must be a port"),
        ((4, 5, 6, 7), {"QWEN38_STOP_SECONDS": "0"}, "QWEN38_STOP_SECONDS must be positive"),
        ((4, 5, 6, 7), {"QWEN38_SERVER_ARGS": "--mtp '4"}, "QWEN38_SERVER_ARGS does not shell-split"),
    ],
)
def test_settings_refuse_what_they_cannot_serve(expect_error, tmp_path, nodes, environ, match) -> None:
    with expect_error(asgi.Qwen38ContainerError, match=match):
        _settings(tmp_path, nodes=nodes, **environ)


def test_a_flag_the_server_does_not_know_is_refused_by_its_parser_not_dropped(tmp_path, capsys, monkeypatch) -> None:
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as chat_server

    server = _settings(tmp_path, QWEN38_SERVER_ARGS="--mtp 4 --sampling --no-such-flag")
    assert "--no-such-flag" in server.argv  # composed as given ...
    with pytest.raises(SystemExit) as refused:  # allow-pytest.raises: inspect the captured exception object
        chat_server._parser().parse_args(list(server.argv))  # ... and refused where the server parses
    assert refused.value.code == 2 and "--no-such-flag" in capsys.readouterr().err
    # the server's own admission rules stay at the parser: the slab and --mtp combine (the 128-row MTP chunk
    # extension's slab form runs the slab's MTP rows inside the slab body; the chain admits the pair when it opens), so
    # the pair parses with both values set and nothing dropped
    server = _settings(tmp_path, QWEN38_SERVER_ARGS="--mtp 4 --sampling --prefill-slab 2048")
    args = chat_server._parser().parse_args(list(server.argv))
    assert args.mtp == 4 and args.prefill_slab == 2048 and args.sampling


def test_the_checkpoint_comes_from_the_hub_cache_or_the_explicit_directory(tmp_path, monkeypatch) -> None:
    explicit = tmp_path / "ckpt"
    assert asgi.checkpoint_directory({"QWEN38_CHECKPOINT": str(explicit)}) == explicit
    seen = {}

    def fake_snapshot_download(repo_id, *, revision, local_files_only):
        seen.update(repo_id=repo_id, revision=revision, local_files_only=local_files_only)
        return str(tmp_path / "snapshot")

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    assert asgi.checkpoint_directory({"HF_MODEL": "org/model"}) == tmp_path / "snapshot"
    assert seen == {"repo_id": "org/model", "revision": asgi.PINNED_CHECKPOINT_REVISION, "local_files_only": True}
    assert asgi.checkpoint_directory({}) == tmp_path / "snapshot" and seen["repo_id"] == asgi.DEFAULT_HF_MODEL


# -- the runtime identity of a tree without git history ------------------------------------------------------------


def test_a_tree_without_git_history_is_admitted_only_with_its_head_declared(
    expect_error, tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv(runtime_admission.DECLARED_HEAD_VARIABLE, raising=False)
    with expect_error(runtime_admission.RuntimeAdmissionError, match="without git history"):
        runtime_admission.git_identity(tmp_path)
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, "abc")
    with expect_error(runtime_admission.RuntimeAdmissionError, match="must be lowercase 40-hex"):
        runtime_admission.git_identity(tmp_path)
    head = "78ae83c92c420ae1cd4aa52b5a4b8217961531cb"
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, head)
    identity = runtime_admission.git_identity(tmp_path)
    assert identity == {"repo": str(tmp_path), "head": head, "tree": None, "dirty": False, "source": "declared"}


def test_a_checkout_with_history_refuses_a_declared_head_that_differs(expect_error, monkeypatch) -> None:
    root = runtime_admission.REPO_ROOT
    if subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True).returncode:
        pytest.skip("not a git checkout")
    monkeypatch.delenv(runtime_admission.DECLARED_HEAD_VARIABLE, raising=False)
    identity = runtime_admission.git_identity(root)
    assert "source" not in identity and len(identity["head"]) == 40
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, identity["head"])
    assert runtime_admission.git_identity(root)["head"] == identity["head"]
    monkeypatch.setenv(runtime_admission.DECLARED_HEAD_VARIABLE, "0" * 40)
    with expect_error(runtime_admission.RuntimeAdmissionError, match="differs from the checkout head"):
        runtime_admission.git_identity(root)


# -- the lifespan and the forwarding, against a stand-in server ------------------------------------------------------


class _StandIn(http.server.BaseHTTPRequestHandler):
    """A loopback stand-in for the chat server: ``GET /health`` echoes a document, ``POST /v1/chat/completions``
    records the body it received and streams SSE events; the server keeps what it saw."""

    protocol_version = "HTTP/1.0"

    def log_message(self, *_args) -> None:
        pass

    def do_GET(self) -> None:
        body = json.dumps({"status": "ok", "path": self.path, "accept": self.headers.get("Accept")}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.server.seen.append({"path": self.path, "body": body, "content_type": self.headers.get("Content-Type")})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        events = int(self.headers.get("X-Events", "3"))
        try:
            for index in range(events):
                self.wfile.write(f"data: {json.dumps({'index': index})}\n\n".encode())
                self.wfile.flush()
                self.server.written.append(time.monotonic())
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except OSError as error:
            self.server.errors.append(f"{type(error).__name__}")


@pytest.fixture
def stand_in():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StandIn)
    server.seen, server.written, server.errors = [], [], []
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def _container(tmp_path: Path, port: int, stop_seconds: float = 5.0) -> asgi.Qwen38ContainerServer:
    evidence = tmp_path / "run"
    return asgi.Qwen38ContainerServer(
        profile=hardware_profiles.P150_LINE,
        argv=("--port", str(port)),
        environment={"TT_VISIBLE_DEVICES": "0,1,2,3"},
        evidence=evidence,
        inner_port=port,
        stop_seconds=stop_seconds,
        directories=(evidence / "tmp",),
    )


# A stand-in for the server process: waits, writes READY (the port from its argv) unless told to fail, then serves until
# SIGTERM, on which it exits 0 (the real server's drain ends the same way); ``fail`` ends it with that status instead.
FAKE_SERVER = r"""
import json, os, signal, sys, time
evidence, ready_after, status, write_ready = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else "0"
stop = []
signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
time.sleep(ready_after)
if not write_ready:
    sys.exit(status)
open(os.path.join(evidence, "READY"), "w").write(json.dumps({"port": int(port), "visible": os.environ.get("TT_VISIBLE_DEVICES")}) + chr(10))
deadline = time.time() + 30
while not stop and time.time() < deadline:
    time.sleep(0.05)
sys.exit(status)
"""


def _spawn_fake(ready_after: float = 0.3, status: int = 0, write_ready: bool = True):
    def spawn(server: asgi.Qwen38ContainerServer):
        command = [
            sys.executable,
            "-c",
            FAKE_SERVER,
            str(server.evidence),
            str(ready_after),
            str(status),
            "1" if write_ready else "0",
            *server.argv,
        ]
        return subprocess.Popen(command, env={**os.environ, **server.environment}, stdin=subprocess.DEVNULL)

    return spawn


async def _http(app, method: str, path: str, body: bytes = b"", headers=(), disconnect_after: int | None = None):
    """Drive one request through the ASGI callable; returns the status, the headers, the body messages and their
    arrival times.  ``disconnect_after`` delivers ``http.disconnect`` once that many body messages arrived."""

    request_sent = False
    disconnected = asyncio.Event()
    sent: list[dict] = []
    arrivals: list[float] = []

    async def receive():
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)
        if message["type"] == "http.response.body":
            arrivals.append(time.monotonic())
            bodies = [m for m in sent if m["type"] == "http.response.body"]
            if disconnect_after is not None and len(bodies) >= disconnect_after:
                disconnected.set()

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": [(k.encode(), v.encode()) for k, v in headers],
    }
    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    bodies = [m["body"] for m in sent if m["type"] == "http.response.body"]
    return start["status"], dict(start["headers"]), bodies, arrivals


def test_spawn_server_runs_the_server_module_as_a_child_with_the_composed_environment(tmp_path, monkeypatch) -> None:
    seen = {}

    class Recorded:
        pid = 4242
        returncode = None

        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        seen.update(command=command, **kwargs)
        return Recorded()

    monkeypatch.setattr(asgi.subprocess, "Popen", fake_popen)
    monkeypatch.delenv("TT_VISIBLE_DEVICES", raising=False)
    container = _container(tmp_path, 18000)
    process = asgi.spawn_server(container)
    assert process.pid == 4242
    assert seen["command"][:3] == [sys.executable, "-m", asgi.SERVER_MODULE] and seen["command"][3:] == [
        "--port",
        "18000",
    ]
    assert seen["env"]["TT_VISIBLE_DEVICES"] == "0,1,2,3" and seen["env"]["PATH"] == os.environ["PATH"]
    assert "TT_VISIBLE_DEVICES" not in os.environ  # the wrapper's own environment is untouched
    assert seen["stdin"] is subprocess.DEVNULL


def test_lifespan_ready_gates_the_startup_and_the_shutdown_signals_and_waits(tmp_path, stand_in) -> None:
    container = _container(tmp_path, stand_in.server_address[1])
    environment_before = dict(os.environ)
    app = asgi.Qwen38ASGIApp(settings=lambda: container, spawn=_spawn_fake(0.3), on_server_exit=lambda status: None)

    async def scenario():
        status, headers, bodies, _ = await _http(app, "GET", "/health")
        assert status == 503 and json.loads(b"".join(bodies))["error"]["code"] == "unavailable"
        messages = asyncio.Queue()
        sent = []
        lifespan = asyncio.create_task(
            app({"type": "lifespan"}, messages.get, lambda m: sent.append(m) or asyncio.sleep(0))
        )
        await messages.put({"type": "lifespan.startup"})
        started = time.monotonic()
        while not sent:
            assert not lifespan.done(), lifespan.exception()
            await asyncio.sleep(0.02)
        assert sent[0] == {"type": "lifespan.startup.complete"}
        assert time.monotonic() - started >= 0.25  # not before READY was written
        assert app.ready and container.ready_marker.is_file() and (container.evidence / "tmp").is_dir()
        ready = json.loads(container.ready_marker.read_text())
        assert ready == {"port": stand_in.server_address[1], "visible": "0,1,2,3"}  # the child got the environment ...
        assert dict(os.environ) == environment_before  # ... this process did not
        assert app.process.poll() is None
        # forwarded once ready
        status, headers, bodies, _ = await _http(app, "GET", "/health", headers=[("accept", "application/json")])
        assert status == 200 and headers[b"content-type"] == b"application/json"
        assert json.loads(b"".join(bodies)) == {"status": "ok", "path": "/health", "accept": "application/json"}
        assert b"connection" not in headers and b"transfer-encoding" not in headers
        await messages.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(lifespan, 10)
        assert sent[1] == {"type": "lifespan.shutdown.complete"}
        assert app.stopping and app.process.returncode == 0  # SIGTERM -> the child's clean exit, observed
        status, _, bodies, _ = await _http(app, "GET", "/health")
        assert status == 503 and "stopping" in json.loads(b"".join(bodies))["error"]["message"]

    asyncio.run(scenario())


def test_a_child_that_ends_before_ready_fails_the_startup_with_its_status(tmp_path, stand_in) -> None:
    container = _container(tmp_path, stand_in.server_address[1])
    app = asgi.Qwen38ASGIApp(
        settings=lambda: container, spawn=_spawn_fake(0.05, status=3, write_ready=False), on_server_exit=lambda s: None
    )

    async def scenario():
        messages = asyncio.Queue()
        sent = []
        await messages.put({"type": "lifespan.startup"})
        await app({"type": "lifespan"}, messages.get, lambda m: sent.append(m) or asyncio.sleep(0))
        assert sent[0]["type"] == "lifespan.startup.failed" and "status 3 before READY" in sent[0]["message"]
        assert not app.ready and not container.ready_marker.exists()

    asyncio.run(scenario())


def test_a_child_that_ends_on_its_own_ends_the_process_with_its_status(tmp_path, stand_in) -> None:
    container = _container(tmp_path, stand_in.server_address[1])
    exits = []
    app = asgi.Qwen38ASGIApp(settings=lambda: container, spawn=_spawn_fake(0.1), on_server_exit=exits.append)

    async def scenario():
        messages = asyncio.Queue()
        sent = []
        lifespan = asyncio.create_task(
            app({"type": "lifespan"}, messages.get, lambda m: sent.append(m) or asyncio.sleep(0))
        )
        await messages.put({"type": "lifespan.startup"})
        while not sent:
            await asyncio.sleep(0.02)
        assert sent[0]["type"] == "lifespan.startup.complete" and app.ready
        os.kill(app.process.pid, 15)  # the child ends on its own (the real server: the stall watchdog's exit 1)
        deadline = time.monotonic() + 10
        while not exits and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert exits == [0] and not app.ready  # the fake exits 0 on SIGTERM; the status is passed through as read
        await messages.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(lifespan, 10)
        assert sent[1] == {"type": "lifespan.shutdown.complete"}

    asyncio.run(scenario())


def test_forwarding_passes_the_bytes_through_and_streams_as_the_server_writes(tmp_path, stand_in) -> None:
    container = _container(tmp_path, stand_in.server_address[1])
    app = asgi.Qwen38ASGIApp(settings=lambda: container, spawn=_spawn_fake(), on_server_exit=lambda s: None)
    app.server, app.ready = container, True  # as after the lifespan startup
    request = json.dumps(
        {"messages": [{"role": "user", "content": "hi"}], "not_a_field": 1, "response_format": {"type": "json_object"}}
    ).encode()

    async def scenario():
        status, headers, bodies, arrivals = await _http(
            app,
            "POST",
            "/v1/chat/completions",
            request,
            headers=[("content-type", "application/json"), ("x-events", "4")],
        )
        assert status == 200 and headers[b"content-type"] == b"text/event-stream"
        assert stand_in.seen[-1] == {
            "path": "/v1/chat/completions",
            "body": request,
            "content_type": "application/json",
        }
        text = b"".join(bodies).decode()
        assert text == "".join(f"data: {json.dumps({'index': i})}\n\n" for i in range(4)) + "data: [DONE]\n\n"
        assert bodies[-1] == b"" and len(bodies) >= 3  # streamed in pieces, the empty final message closes it
        assert arrivals[0] < stand_in.written[-1]  # the first piece reached the client before the server wrote its last

    asyncio.run(scenario())


def test_a_client_hang_up_closes_the_servers_connection(tmp_path, stand_in) -> None:
    container = _container(tmp_path, stand_in.server_address[1])
    app = asgi.Qwen38ASGIApp(settings=lambda: container, spawn=_spawn_fake(), on_server_exit=lambda s: None)
    app.server, app.ready = container, True

    async def scenario():
        status, _, bodies, _ = await _http(
            app, "POST", "/v1/chat/completions", b"{}", headers=[("x-events", "60")], disconnect_after=1
        )
        assert status == 200 and len(bodies) < 60
        deadline = time.monotonic() + 5
        while not stand_in.errors and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        assert stand_in.errors, "the stand-in's write did not fail after the client hung up"

    asyncio.run(scenario())
