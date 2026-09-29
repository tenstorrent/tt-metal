# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The chat server as an ASGI application: the form a container that starts its model with uvicorn can run.

    python -m uvicorn --lifespan on models.demos.blackhole.qwen38_flash_next.tools.qwen38_asgi:app

starts ``tools/qwen38_chat_server.py`` as a child process (``python -m`` the server module with the composed
arguments and environment: the server keeps its own main thread, its signal handlers and its device teardown exactly
as the shell launcher runs it, and this process never touches a device) and forwards every request to it over the
loopback.  The lifespan startup composes the server's arguments from the environment, starts it, and returns only
once the server has written its ``READY`` record (the mesh opened, the chain captured, the acceptance records
replayed and the ``json`` record matched), so uvicorn's ``Application startup complete`` line means what ``READY``
means; a server that ends before the record fails the startup with its status.  The lifespan shutdown sends the
server SIGTERM (its drain, then the chain's release and the mesh close in the server's own order) and waits for its
exit, logged with its status; a server that ends on its own (the stall watchdog's exit 1) ends this process with the
same status, so the container exits as the server did.  Requests are forwarded byte for byte: this module parses nothing of a request or a reply,
so the server's rules apply unchanged (what the server cannot honour it refuses with HTTP 400; nothing is dropped
here), a streaming reply reaches the client as the server writes it, and a client that hangs up closes the
server's connection too, so the device never runs a request for nobody.  Before ``READY`` and after the stop every
request is answered HTTP 503 in the server's error shape.

The environment (a tt-model serve profile sets it; every value is checked before the server starts):

    QWEN38_HARDWARE_PROFILE   the server's hardware profile (default ``p150-line``; ``tt-quietbox``, ``tt-quietbox-2``)
    QWEN38_ALLOCATED_CONTEXT  the resident context, 32768 (default), 65536, 131072 or 262144
    QWEN38_SERVER_ARGS        further server flags, shell-split (default ``--mtp 4 --sampling --stall-seconds 300
                              --require-json-96``); a flag the server does not know is refused by its parser
    QWEN38_CACHE_ROOT         the cache root (default ``/tensor-cache``): the launcher's layout, ``caches/<label>/
                              {components,model-io}``, ``caches/bf4-experts`` shared by every context, ``runs/``
    QWEN38_CHECKPOINT         the checkpoint directory; unset, the pinned revision of ``HF_MODEL`` (default
                              ``Qwen/Qwen3.8-Flash-Next``) already in the local Hugging Face hub cache
    QWEN38_INNER_PORT         the loopback port the server listens on behind uvicorn (default 18000)
    QWEN38_STOP_SECONDS       how long the shutdown waits for the server's exit after SIGTERM (default 300)
    QWEN38_TT_METAL_SHA       the runtime identity of a tree without git history (``tools/runtime_admission.py``)

The device nodes are the ``/dev/tenstorrent`` entries the container was given: exactly four, the profile's own or
another four (``--device-nodes``; ``TT_VISIBLE_DEVICES`` is then UMD's indices 0..3 over them, the cluster descriptor
keeps their node numbers).  The server's environment (``TT_VISIBLE_DEVICES``, the mesh graph
descriptor, ``TT_METAL_TRACE_ALLOC_TRACKING``, ``QWEN38_HARDWARE_MODE``) is given to the child, as the shell launcher
sets it; this process's own environment is not changed.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_QSA_CACHE_CAPACITIES

MODEL_DIR = Path(__file__).resolve().parents[1]
SERVER_MODULE = "models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_server"
ACCEPTANCE_PROMPTS = MODEL_DIR / "tools" / "acceptance" / "greedy-prompts"
DEFAULT_HF_MODEL = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_PROFILE = "p150-line"
DEFAULT_ALLOCATED_CONTEXT = 32768
DEFAULT_SERVER_ARGS = "--mtp 4 --sampling --stall-seconds 300 --require-json-96"
DEFAULT_CACHE_ROOT = "/tensor-cache"
DEFAULT_INNER_PORT = 18000
DEFAULT_STOP_SECONDS = 300.0
DEVICE_ROOT = Path("/dev/tenstorrent")
READY_POLL_SECONDS = 0.5
STARTUP_LOG_SECONDS = 60.0  # a progress line while the server starts (the first start converts for half an hour)
# The forwarded connection has no read timeout: a long prompt's prefill answers nothing for minutes, and a wedge is
# the server's stall watchdog's to end (it exits; this process then answers 503 and stops).
HOP_BY_HOP = frozenset(
    (
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
    )
)
CHUNK = 64 << 10


class Qwen38ContainerError(ValueError):
    """The environment does not describe a server this module can start."""


def _log(event: str, **fields: Any) -> None:
    print(
        json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": f"asgi_{event}", **fields}),
        flush=True,
    )


def device_nodes(device_root: Path = DEVICE_ROOT) -> tuple[int, ...]:
    """The ``/dev/tenstorrent/<N>`` nodes this process can see, sorted: the four chips the container was given."""

    if not device_root.is_dir():
        raise Qwen38ContainerError(f"{device_root}: no device nodes (the container was started without --device)")
    return hardware_profiles.present_device_nodes(device_root)


def checkpoint_directory(environ: Mapping[str, str]) -> Path:
    """``QWEN38_CHECKPOINT`` when set; else the pinned revision of ``HF_MODEL`` in the local hub cache, which the
    container's host prefetched (nothing is downloaded here)."""

    explicit = environ.get("QWEN38_CHECKPOINT")
    if explicit:
        return Path(explicit)
    repo_id = environ.get("HF_MODEL") or DEFAULT_HF_MODEL
    from huggingface_hub import snapshot_download  # the hub cache layout is the library's

    try:
        return Path(snapshot_download(repo_id, revision=PINNED_CHECKPOINT_REVISION, local_files_only=True))
    except Exception as error:  # noqa: BLE001  the library's own error classes vary by version
        raise Qwen38ContainerError(
            f"the checkpoint {repo_id} at {PINNED_CHECKPOINT_REVISION} is not in the local Hugging Face hub cache "
            f"({type(error).__name__}: {error}); set QWEN38_CHECKPOINT to the checkpoint directory instead"
        ) from error


@dataclass(frozen=True)
class Qwen38ContainerServer:
    """What the lifespan starts: the server's argv, the environment it checks, its run directory and its port."""

    profile: hardware_profiles.ResidentHardwareProfile
    argv: tuple[str, ...]
    environment: Mapping[str, str]
    evidence: Path
    inner_port: int
    stop_seconds: float
    directories: tuple[Path, ...] = field(default=())

    @property
    def ready_marker(self) -> Path:
        return self.evidence / "READY"

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        device_root: Path = DEVICE_ROOT,
        checkpoint: Path | None = None,
        now: str | None = None,
    ) -> "Qwen38ContainerServer":
        environ = dict(os.environ if environ is None else environ)
        table = hardware_profiles.hardware_profile_table()
        name = environ.get("QWEN38_HARDWARE_PROFILE") or DEFAULT_PROFILE
        if name not in table:
            raise Qwen38ContainerError(f"QWEN38_HARDWARE_PROFILE {name!r} is not one of {sorted(table)}")
        profile = table[name]
        nodes = device_nodes(device_root)
        if len(nodes) != 4:
            raise Qwen38ContainerError(
                f"{device_root} holds {len(nodes)} device node(s) {list(nodes)}: the server needs exactly four chips"
            )
        if nodes != tuple(profile.device_nodes):
            # the four nodes are all this container presents: TT_VISIBLE_DEVICES takes UMD's indices 0..3 (the server derives
            # the same from /dev/tenstorrent when it takes --device-nodes; the cluster descriptor keeps the node numbers)
            try:
                profile = profile.with_device_nodes(nodes, present=nodes)
            except hardware_profiles.HardwareProfileError as error:
                raise Qwen38ContainerError(f"device nodes {list(nodes)}: {error}") from error
        context_text = environ.get("QWEN38_ALLOCATED_CONTEXT") or str(DEFAULT_ALLOCATED_CONTEXT)
        if not context_text.isdigit() or int(context_text) not in RESIDENT_QSA_CACHE_CAPACITIES:
            raise Qwen38ContainerError(
                f"QWEN38_ALLOCATED_CONTEXT must be one of {list(RESIDENT_QSA_CACHE_CAPACITIES)}, got {context_text!r}"
            )
        context = int(context_text)
        port_text = environ.get("QWEN38_INNER_PORT") or str(DEFAULT_INNER_PORT)
        if not port_text.isdigit() or not 1024 <= int(port_text) <= 65535:
            raise Qwen38ContainerError(f"QWEN38_INNER_PORT must be a port in [1024, 65535], got {port_text!r}")
        stop_text = environ.get("QWEN38_STOP_SECONDS") or str(DEFAULT_STOP_SECONDS)
        try:
            stop_seconds = float(stop_text)
        except ValueError as error:
            raise Qwen38ContainerError(f"QWEN38_STOP_SECONDS must be a number of seconds, got {stop_text!r}") from error
        if stop_seconds <= 0:
            raise Qwen38ContainerError(f"QWEN38_STOP_SECONDS must be positive, got {stop_text!r}")
        try:
            extra = shlex.split(environ.get("QWEN38_SERVER_ARGS", DEFAULT_SERVER_ARGS))
        except ValueError as error:
            raise Qwen38ContainerError(f"QWEN38_SERVER_ARGS does not shell-split: {error}") from error
        cache_root = Path(environ.get("QWEN38_CACHE_ROOT") or DEFAULT_CACHE_ROOT)
        label = f"c{context}-{profile.host}"
        caches = cache_root / "caches" / label
        stamp = now or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        evidence = cache_root / "runs" / f"q38-chat-server-{profile.host}-{stamp}-{os.getpid()}"
        checkpoint = checkpoint if checkpoint is not None else checkpoint_directory(environ)
        argv = [
            "--hardware-profile",
            name,
            "--checkpoint",
            str(checkpoint),
            "--component-cache-root",
            str(caches / "components"),
            "--routed-bf4-scratch-root",
            str(cache_root / "caches" / "bf4-experts"),
            "--model-io-cache-root",
            str(caches / "model-io"),
            "--phase-log",
            str(evidence / "phase-markers.jsonl"),
            "--evidence",
            str(evidence),
            "--allocated-context",
            str(context),
            "--host",
            "127.0.0.1",
            "--port",
            str(int(port_text)),
            "--acceptance-prompts",
            str(ACCEPTANCE_PROMPTS),
        ]
        if nodes != tuple(table[name].device_nodes):
            argv += ["--device-nodes", ",".join(str(node) for node in nodes)]
        argv += extra
        environment = {
            "TT_VISIBLE_DEVICES": profile.visible_devices,
            "QWEN38_HARDWARE_MODE": "diagnostic_non_promoting",
            "TT_METAL_TRACE_ALLOC_TRACKING": "1",
            "TMPDIR": str(evidence / "tmp"),
            "OMP_NUM_THREADS": environ.get("OMP_NUM_THREADS") or "4",
        }
        descriptor = hardware_profiles.mesh_graph_descriptor_path(profile)
        if descriptor is not None:
            environment["TT_MESH_GRAPH_DESC_PATH"] = str(descriptor)
        return cls(
            profile=profile,
            argv=tuple(argv),
            environment=environment,
            evidence=evidence,
            inner_port=int(port_text),
            stop_seconds=stop_seconds,
            directories=(
                caches / "components",
                caches / "model-io",
                cache_root / "caches" / "bf4-experts",
                evidence / "tmp",
            ),
        )


def _error_document(message: str, code: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": "server_error", "param": None, "code": code}}).encode(
        "utf-8"
    )


class ServerProcess(Protocol):
    """What the lifespan needs of the server's process (``subprocess.Popen`` has it)."""

    pid: int
    returncode: int | None

    def poll(self) -> int | None:
        ...

    def send_signal(self, signum: int) -> None:
        ...

    def wait(self, timeout: float | None = None) -> int:
        ...


def spawn_server(server: Qwen38ContainerServer) -> ServerProcess:
    """The chat server as a child of this process: its own main thread and signal handlers; its output is this
    process's (the container's log); its environment is this one plus the server's."""

    command = [sys.executable, "-m", SERVER_MODULE, *server.argv]
    return subprocess.Popen(command, env={**os.environ, **server.environment}, stdin=subprocess.DEVNULL)


def _exit_with(status: int) -> None:
    """The server ended on its own: this process ends with its status so the container exits as the server did (the
    stall watchdog's exit 1 asks a supervisor for a restart).  No device state lives in this process."""

    os._exit(status if status else 1)


class Qwen38ASGIApp:
    """The ASGI callable: the lifespan runs the chat server as a child process, HTTP requests are forwarded to it."""

    def __init__(
        self,
        *,
        settings: Callable[[], Qwen38ContainerServer] = Qwen38ContainerServer.from_environment,
        spawn: Callable[[Qwen38ContainerServer], ServerProcess] = spawn_server,
        on_server_exit: Callable[[int], None] = _exit_with,
    ) -> None:
        self._settings = settings
        self._spawn = spawn
        self._on_server_exit = on_server_exit
        self.server: Qwen38ContainerServer | None = None
        self.process: ServerProcess | None = None
        self.ready = False
        self.stopping = False

    # -- lifespan ----------------------------------------------------------------------------------------------------

    async def startup(self) -> None:
        server = self._settings()
        for directory in server.directories:
            directory.mkdir(parents=True, exist_ok=True)
        server.evidence.mkdir(parents=True, exist_ok=True)
        self.server = server
        _log(
            "server_starting",
            argv=list(server.argv),
            environment=dict(server.environment),
            evidence=str(server.evidence),
        )
        self.process = process = self._spawn(server)
        _log("server_spawned", pid=process.pid)
        started = time.monotonic()
        last_line = started
        while not server.ready_marker.exists():
            status = process.poll()
            if status is not None:
                raise RuntimeError(f"the chat server ended with status {status} before READY (its log above says why)")
            await asyncio.sleep(READY_POLL_SECONDS)
            if time.monotonic() - last_line >= STARTUP_LOG_SECONDS:
                last_line = time.monotonic()
                _log("server_starting_still", seconds=round(last_line - started))
        self.ready = True
        ready = json.loads(server.ready_marker.read_text(encoding="utf-8"))
        _log(
            "server_ready",
            seconds=round(time.monotonic() - started, 1),
            port=server.inner_port,
            pid=process.pid,
            ready=ready,
        )
        asyncio.ensure_future(self._watch())

    async def _watch(self) -> None:
        """The server ended without a stop from this process: end this process with its status."""

        process = self.process
        while process is not None and not self.stopping:
            status = process.poll()
            if status is not None:
                _log("server_exited", status=status, pid=process.pid)
                self.ready = False
                self._on_server_exit(status)
                return
            await asyncio.sleep(2.0)

    async def shutdown(self) -> None:
        self.stopping = True
        server, process = self.server, self.process
        if server is None or process is None:
            return
        if process.poll() is not None:
            _log("server_stopped", status=process.returncode, note="already ended")
            return
        _log("server_stopping", pid=process.pid, stop_seconds=server.stop_seconds)
        process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + server.stop_seconds
        while process.poll() is None and time.monotonic() < deadline:
            await asyncio.sleep(READY_POLL_SECONDS)
        if process.poll() is None:
            _log("server_stop_timeout", pid=process.pid, stop_seconds=server.stop_seconds)
            return
        _log("server_stopped", status=process.returncode)

    async def _lifespan(self, receive: Callable, send: Callable) -> None:
        message = await receive()
        assert message["type"] == "lifespan.startup", message
        try:
            await self.startup()
        except Exception as error:  # noqa: BLE001  uvicorn reports the message and exits
            _log("startup_failed", error=f"{type(error).__name__}: {error}")
            await send({"type": "lifespan.startup.failed", "message": f"{type(error).__name__}: {error}"})
            return
        await send({"type": "lifespan.startup.complete"})
        message = await receive()
        assert message["type"] == "lifespan.shutdown", message
        try:
            await self.shutdown()
        except Exception as error:  # noqa: BLE001
            _log("shutdown_failed", error=f"{type(error).__name__}: {error}")
            await send({"type": "lifespan.shutdown.failed", "message": f"{type(error).__name__}: {error}"})
            return
        await send({"type": "lifespan.shutdown.complete"})

    # -- http --------------------------------------------------------------------------------------------------------

    async def _refuse(self, send: Callable, status: int, message: str, code: str) -> None:
        body = _error_document(message, code)
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _forward(self, scope: Mapping[str, Any], receive: Callable, send: Callable) -> None:
        server = self.server
        if server is None or not self.ready or self.stopping:
            await self._refuse(
                send,
                503,
                "the model server is not ready" if not self.stopping else "the model server is stopping",
                "unavailable",
            )
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body += message.get("body", b"")
            if not message.get("more_body"):
                break
        headers: dict[str, str] = {}
        for raw_name, raw_value in scope.get("headers", ()):
            name = raw_name.decode("latin-1")
            if name.lower() in HOP_BY_HOP or name.lower() in ("host", "content-length"):
                continue
            headers[name] = raw_value.decode("latin-1")
        if body or scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            headers["Content-Length"] = str(len(body))
        target = scope["path"] + (("?" + scope["query_string"].decode("latin-1")) if scope.get("query_string") else "")
        loop = asyncio.get_running_loop()
        connection = http.client.HTTPConnection("127.0.0.1", server.inner_port, timeout=None)

        def open_response() -> http.client.HTTPResponse:
            connection.request(scope["method"], target, body=bytes(body) if body else None, headers=headers)
            return connection.getresponse()

        try:
            response = await loop.run_in_executor(None, open_response)
        except OSError as error:
            _log("forward_failed", target=target, error=f"{type(error).__name__}: {error}")
            await self._refuse(send, 503, f"the model server did not answer: {error}", "unavailable")
            connection.close()
            return
        response_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in response.getheaders()
            if name.lower() not in HOP_BY_HOP
        ]
        await send({"type": "http.response.start", "status": response.status, "headers": response_headers})
        disconnect = asyncio.ensure_future(self._wait_disconnect(receive))
        try:
            while True:
                read = loop.run_in_executor(None, response.read1, CHUNK)
                done, _pending = await asyncio.wait({read, disconnect}, return_when=asyncio.FIRST_COMPLETED)
                if disconnect in done and read not in done:
                    # the client hung up: close the server's connection so its next write fails and the request
                    # ends there as "disconnected" (the device never runs a request for nobody)
                    _log("client_disconnected", target=target)
                    sock = connection.sock
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except OSError:
                            pass
                    connection.close()
                    await read  # the read returns or raises now that the socket is closed
                    return
                chunk = read.result()
                if not chunk:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
                    return
                await send({"type": "http.response.body", "body": bytes(chunk), "more_body": True})
        except OSError as error:
            _log("forward_ended", target=target, error=f"{type(error).__name__}: {error}")
            try:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
            except Exception:  # noqa: BLE001  the client is gone too
                pass
        finally:
            disconnect.cancel()
            connection.close()

    @staticmethod
    async def _wait_disconnect(receive: Callable) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return

    async def __call__(self, scope: Mapping[str, Any], receive: Callable, send: Callable) -> None:
        kind = scope["type"]
        if kind == "lifespan":
            await self._lifespan(receive, send)
        elif kind == "http":
            await self._forward(scope, receive, send)
        elif kind == "websocket":
            await send({"type": "websocket.close", "code": 1003})
        else:
            raise RuntimeError(f"unsupported ASGI scope type {kind!r}")


app = Qwen38ASGIApp()


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m ...qwen38_asgi [--host H] [--port P]``: uvicorn with the lifespan on, the same command the
    container runs."""

    import argparse

    parser = argparse.ArgumentParser(description="the chat server behind uvicorn (see the module docstring)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    import uvicorn

    uvicorn.run(f"{__name__}:app", host=args.host, port=args.port, lifespan="on", log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
