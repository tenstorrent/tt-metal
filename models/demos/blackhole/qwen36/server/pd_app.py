# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill/decode disaggregated Qwen3.8-27B on a Blackhole P150x8: the bundle's supervisor + proxy.

tt-model (kind ``tt-dit-server``) starts the image with ``python -m uvicorn --lifespan on <this module>:app`` and
treats uvicorn's ``Application startup complete`` as READY.  This app's lifespan startup launches two ``vllm serve``
children on the two fully connected 2x2 halves of the board (each a 1x4 mesh, TP=4):

* **P** (``kv_producer``): prefill only, chips ``QWEN36_PD_PREFILL_CHIPS`` (default 0,1,6,7).  It prefills the prompt,
  exports the request's paged KV blocks and its Gated-DeltaNet recurrent/conv state, and returns a 1-token completion
  carrying ``kv_transfer_params``.
* **D** (``kv_consumer``): decode, chips ``QWEN36_PD_DECODE_CHIPS`` (default 2,3,4,5).  It pulls that state over
  Mooncake's transfer engine (TCP, in-container loopback), imports it into a free batch row and continues decoding.

Both run ``vllm_tt_plugin.kv_connector.tt_mooncake_connector.TTMooncakeConnector``.  The proxy (this app) is the
public OpenAI-compatible surface: ``/v1/chat/completions`` and ``/v1/completions`` go to P (``max_tokens=1``) and
then stream from D; ``/v1/models``, ``/health``, ``/version`` and anything else under ``/v1`` pass through to D.
``GET /`` describes the running stack.  Either half can also be queried directly on its own port inside the
container (``QWEN36_PD_PREFILL_PORT`` / ``QWEN36_PD_DECODE_PORT``) for experiments.

The pure parts (config, argv/env construction) import without fastapi/httpx so the image's verify lines and the
host-side tests can exercise them; ``app`` is built on first access.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

try:
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    from starlette.background import BackgroundTask
except ImportError:  # the pure parts stay importable without the HTTP stack
    httpx = FastAPI = Request = JSONResponse = Response = StreamingResponse = BackgroundTask = None  # type: ignore

BUNDLE_NAME = "qwen3.8-27b-pd4x4-p150x8"
MODEL_ID = "Qwen/Qwen3.8-27B"
CONNECTOR_MODULE = "vllm_tt_plugin.kv_connector.tt_mooncake_connector"
CONNECTOR_NAME = "TTMooncakeConnector"
REPO_ROOT = Path(__file__).resolve().parents[4]  # .../tt-metal

# The two fully connected 2x2 halves of the P150x8's 2x4 ethernet grid (rows 6-7-4-5 / 0-1-2-3).
DEFAULT_PREFILL_CHIPS = (0, 1, 6, 7)
DEFAULT_DECODE_CHIPS = (2, 3, 4, 5)
# vLLM's --additional-config for the TT platform, the same one the plain p300x2 bundle serves with.
TT_ADDITIONAL_CONFIG = {
    "tt": {
        "fabric_config": "FABRIC_1D",
        "trace_region_size": 1073741824,
        "l1_small_size": 24576,
        "sample_on_device_mode": "decode_only",
    }
}
# Environment the model code expects (mirrors the p300x2 bundle's serve.env).  TT_DECODE_BUCKETING=0: decode
# bucketing together with on-device sampling corrupts served state on this tt-metal (see PLAN.md of the port).
CHILD_ENV = {
    "ARCH_NAME": "blackhole",
    "TT_QWEN35_TEXT_VER": "qwen36_blackhole",
    "QWEN36_MAX_TOKENS_ALL_USERS": "525312",
    "VLLM_RPC_TIMEOUT": "900000",
    "VLLM_CONFIGURE_LOGGING": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "TT_DECODE_BUCKETING": "0",
}
STRIPPED_REQUEST_HEADERS = frozenset({"host", "content-length", "connection", "transfer-encoding"})
STRIPPED_RESPONSE_HEADERS = frozenset({"content-length", "transfer-encoding", "connection"})


class BundleConfigError(ValueError):
    pass


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise BundleConfigError(f"{name}={raw!r} is not an integer") from error


def _env_chips(env: Mapping[str, str], name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        chips = tuple(int(x) for x in raw.split(","))
    except ValueError as error:
        raise BundleConfigError(f"{name}={raw!r}: expected comma-separated chip ids") from error
    if len(chips) != 4 or len(set(chips)) != 4:
        raise BundleConfigError(f"{name}={raw!r}: exactly four distinct chips make a 1x4 half")
    return chips


@dataclass(frozen=True)
class BundleConfig:
    """Everything the lifespan needs, resolved from the environment once."""

    weights: str = MODEL_ID
    prefill_chips: tuple[int, ...] = DEFAULT_PREFILL_CHIPS
    decode_chips: tuple[int, ...] = DEFAULT_DECODE_CHIPS
    mesh_device: str = "P150x4"
    prefill_port: int = 8100
    decode_port: int = 8200
    side_channel_port: int = 18100
    prefill_max_num_seqs: int = 8
    decode_max_num_seqs: int = 32
    max_model_len: int = 262144
    block_size: int = 64
    metal_cache: Path = Path("/cache")
    offline: bool = True
    log_seconds: int = 30
    # first boot: P and D would otherwise convert the same weight cache at the same time
    serial_cold_boot: bool = True
    extra_vllm_args: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "BundleConfig":
        env = os.environ if env is None else env
        prefill = _env_chips(env, "QWEN36_PD_PREFILL_CHIPS", DEFAULT_PREFILL_CHIPS)
        decode = _env_chips(env, "QWEN36_PD_DECODE_CHIPS", DEFAULT_DECODE_CHIPS)
        if set(prefill) & set(decode):
            raise BundleConfigError(f"prefill chips {prefill} and decode chips {decode} overlap")
        ports = {
            "QWEN36_PD_PREFILL_PORT": _env_int(env, "QWEN36_PD_PREFILL_PORT", 8100),
            "QWEN36_PD_DECODE_PORT": _env_int(env, "QWEN36_PD_DECODE_PORT", 8200),
            "QWEN36_PD_SIDE_PORT": _env_int(env, "QWEN36_PD_SIDE_PORT", 18100),
        }
        if len(set(ports.values())) != 3:
            raise BundleConfigError(f"the internal ports must differ: {ports}")
        extra = env.get("QWEN36_PD_VLLM_ARGS", "").split()
        return cls(
            weights=env.get("HF_MODEL") or MODEL_ID,
            prefill_chips=prefill,
            decode_chips=decode,
            mesh_device=env.get("QWEN36_PD_MESH_DEVICE") or "P150x4",
            prefill_port=ports["QWEN36_PD_PREFILL_PORT"],
            decode_port=ports["QWEN36_PD_DECODE_PORT"],
            side_channel_port=ports["QWEN36_PD_SIDE_PORT"],
            prefill_max_num_seqs=_env_int(env, "QWEN36_PD_PREFILL_MAX_NUM_SEQS", 8),
            decode_max_num_seqs=_env_int(env, "QWEN36_PD_DECODE_MAX_NUM_SEQS", 32),
            max_model_len=_env_int(env, "QWEN36_PD_MAX_MODEL_LEN", 262144),
            # tt-model exports TT_METAL_CACHE=/cache; outside a container fall back to tt-metal's own default
            metal_cache=Path(env.get("TT_METAL_CACHE") or Path.home() / ".cache" / "tt-metal-cache"),
            offline=env.get("HF_HUB_OFFLINE", "1") not in ("0", "false", "False"),
            log_seconds=_env_int(env, "QWEN36_PD_LOG_SECONDS", 30),
            serial_cold_boot=env.get("QWEN36_PD_SERIAL_COLD_BOOT", "1") not in ("0", "false", "False"),
            extra_vllm_args=tuple(extra),
        )


def kv_transfer_config(config: BundleConfig, role: str) -> dict[str, Any]:
    """The ``--kv-transfer-config`` of one half.  P publishes its side channel (block/state hand-off requests from D)
    on loopback; both use Mooncake's TCP transport, the only one a driverless host has."""
    extra: dict[str, Any] = {"mooncake_protocol": "tcp"}
    if role == "prefill":
        extra.update({"side_channel_host": "127.0.0.1", "side_channel_port": config.side_channel_port})
    return {
        "kv_connector": CONNECTOR_NAME,
        "kv_connector_module_path": CONNECTOR_MODULE,
        "kv_role": "kv_producer" if role == "prefill" else "kv_consumer",
        "kv_connector_extra_config": extra,
    }


def vllm_argv(config: BundleConfig, role: str) -> list[str]:
    """``vllm serve`` of one half, via the venv's interpreter so no PATH lookup is involved."""
    if role not in ("prefill", "decode"):
        raise ValueError(f"role must be prefill or decode, not {role!r}")
    port = config.prefill_port if role == "prefill" else config.decode_port
    max_num_seqs = config.prefill_max_num_seqs if role == "prefill" else config.decode_max_num_seqs
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        config.weights,
        "--max-model-len",
        str(config.max_model_len),
        "--max-num-seqs",
        str(max_num_seqs),
        "--block-size",
        str(config.block_size),
        "--additional-config",
        json.dumps(TT_ADDITIONAL_CONFIG),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--reasoning-parser",
        "qwen3",
        "--max-num-batched-tokens",
        str(config.max_model_len),
        "--kv-transfer-config",
        json.dumps(kv_transfer_config(config, role)),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    argv += list(config.extra_vllm_args)
    return argv


def _cuda_stub_dirs() -> list[str]:
    """Mooncake's engine links libcuda.so.1, libcudart.so.12, libcurl.so.4 and rdma-core (libibverbs/libmlx5); on a
    driverless, RDMA-less host they come from wheels (tt-mooncake-sysdeps: libcuda/libcurl stubs + rdma-core copies;
    nvidia-cuda-runtime-cu12: libcudart) whose lib dirs are not on the loader path."""
    dirs: list[str] = []
    try:
        import tt_mooncake_sysdeps  # type: ignore

        dirs.append(tt_mooncake_sysdeps.lib_dir())
    except ImportError:
        pass
    try:
        import nvidia.cuda_runtime  # type: ignore

        for p in nvidia.cuda_runtime.__path__:
            lib = Path(p) / "lib"
            if lib.is_dir():
                dirs.append(str(lib))
    except ImportError:
        pass
    return dirs


def vllm_env(config: BundleConfig, role: str, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of one half: its chips (UMD's TT_VISIBLE_DEVICES makes the four a P150x4 cluster), its own JIT
    kernel cache under TT_METAL_CACHE (two processes must not build into one), the model's knobs."""
    env = dict(os.environ if base is None else base)
    env.update(CHILD_ENV)
    chips = config.prefill_chips if role == "prefill" else config.decode_chips
    env["TT_VISIBLE_DEVICES"] = ",".join(str(c) for c in chips)
    env["MESH_DEVICE"] = config.mesh_device
    env["HF_MODEL"] = config.weights
    env["HF_HUB_OFFLINE"] = "1" if config.offline else "0"
    env["TT_METAL_CACHE"] = str(config.metal_cache / role)
    env["PYTHONUNBUFFERED"] = "1"
    stub_dirs = _cuda_stub_dirs()
    if stub_dirs:
        prior = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(stub_dirs + ([prior] if prior else []))
    # tt-dit-server's launcher exports the mesh SHAPE of the profile's 8-chip SKU; each half is a 1x4
    env.pop("QWEN36_PD_MESH_SHAPE", None)
    return env


def weight_cache_is_warm(env: Mapping[str, str] | None = None) -> bool:
    """Whether a converted 1x4 weight cache already exists under TT_CACHE_PATH (the model writes
    ``.../tensor_cache_<dtype>_mesh1x4``).  On a cold cache P and D must not both convert at once."""
    env = os.environ if env is None else env
    root = env.get("TT_CACHE_PATH")
    if not root or not Path(root).is_dir():
        return False
    for path in Path(root).rglob("tensor_cache_*_mesh1x4"):
        if path.is_dir() and any(path.iterdir()):
            return True
    return False


def _log(event: str, **fields: Any) -> None:
    record = {"ts": round(time.time(), 3), "bundle": BUNDLE_NAME, "event": event, **fields}
    print(json.dumps(record, default=str), flush=True)


def exit_status(code: int | None) -> dict[str, Any]:
    status: dict[str, Any] = {"status": code}
    if code is None:
        status["outcome"] = "running"
    elif code == 0:
        status["outcome"] = "exited"
    elif code > 0:
        status["outcome"] = "failed"
    else:
        status["outcome"] = "signal"
        try:
            status["signal"] = signal.Signals(-code).name
        except ValueError:
            status["signal"] = str(-code)
    return status


def filtered_headers(headers: Mapping[str, str] | Any, stripped: frozenset[str]) -> list[tuple[str, str]]:
    return [(k, v) for k, v in headers.items() if k.lower() not in stripped]


class VllmHalf:
    """One ``vllm serve`` child: start, wait for its ``/health``, stop with a drain."""

    def __init__(self, config: BundleConfig, role: str) -> None:
        self.config = config
        self.role = role
        self.port = config.prefill_port if role == "prefill" else config.decode_port
        self.argv = vllm_argv(config, role)
        self.env = vllm_env(config, role)
        self.process: subprocess.Popen | None = None
        self.started_at = 0.0
        self.ready_seconds: float | None = None
        self.exit_reported = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        Path(self.env["TT_METAL_CACHE"]).mkdir(parents=True, exist_ok=True)
        _log(
            "start",
            role=self.role,
            command=self.argv,
            exported={k: self.env[k] for k in sorted(self.env) if k.startswith(("TT_", "MESH_", "QWEN36_", "HF_"))},
        )
        self.started_at = time.monotonic()
        # stdout/stderr inherited: the halves' logs are the container's log
        self.process = subprocess.Popen(self.argv, cwd=str(REPO_ROOT), env=self.env, stdin=subprocess.DEVNULL)

    def poll(self) -> int | None:
        return None if self.process is None else self.process.poll()

    async def wait_ready(self, client: "httpx.AsyncClient") -> None:
        """Poll ``/health`` and the process; log every ``log_seconds`` so a 15-minute first boot (JIT kernel
        compile + weight conversion) shows progress.  No timeout of its own: tt-model's readiness watch bounds it."""
        if self.process is None:
            raise RuntimeError(f"{self.role} was not started")
        next_log = 0.0
        while True:
            code = self.process.poll()
            if code is not None:
                status = self.report_exit()
                raise RuntimeError(f"the {self.role} half {status['outcome']} with status {code} before READY")
            try:
                response = await client.get(f"{self.base_url}/health", timeout=5.0)
                if response.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            now = time.monotonic()
            if now >= next_log:
                _log("waiting", role=self.role, seconds=round(now - self.started_at))
                next_log = now + self.config.log_seconds
            await asyncio.sleep(1.0)
        self.ready_seconds = round(time.monotonic() - self.started_at, 1)
        _log("ready", role=self.role, seconds=self.ready_seconds, url=self.base_url)

    def stop(self, grace_seconds: float = 120.0) -> int | None:
        """SIGTERM (vLLM drains and closes the mesh), SIGKILL after the grace."""
        process = self.process
        if process is None:
            return None
        if process.poll() is None:
            _log("stop", role=self.role, signal="SIGTERM", grace_seconds=grace_seconds)
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=grace_seconds)
            except subprocess.TimeoutExpired:
                _log("stop", role=self.role, signal="SIGKILL")
                process.kill()
                process.wait(timeout=30)
        self.exit_reported = True
        _log("stopped", role=self.role, **exit_status(process.returncode))
        return process.returncode

    def report_exit(self) -> dict[str, Any]:
        status = exit_status(self.poll())
        if status["status"] is not None and not self.exit_reported:
            self.exit_reported = True
            _log("child_exited", role=self.role, **status)
        return status


@dataclass
class Stack:
    config: BundleConfig
    prefill: VllmHalf
    decode: VllmHalf
    boot: dict[str, Any] = field(default_factory=dict)

    def dead_half(self) -> VllmHalf | None:
        for half in (self.prefill, self.decode):
            if half.poll() is not None:
                return half
        return None


def bundle_description(config: BundleConfig, stack: Stack | None) -> dict[str, Any]:
    def half(h: VllmHalf | None) -> dict[str, Any] | None:
        if h is None:
            return None
        return {
            "chips": list(config.prefill_chips if h.role == "prefill" else config.decode_chips),
            "port": h.port,
            "max_num_seqs": h.argv[h.argv.index("--max-num-seqs") + 1],
            "ready_seconds": h.ready_seconds,
            "exit": exit_status(h.poll()),
        }

    return {
        "bundle": BUNDLE_NAME,
        "model": config.weights,
        "topology": {
            "hardware": "p150x8",
            "prefill": {"mesh": config.mesh_device, "chips": list(config.prefill_chips), "tensor_parallel": 4},
            "decode": {"mesh": config.mesh_device, "chips": list(config.decode_chips), "tensor_parallel": 4},
            "transfer": "mooncake-transfer-engine (tcp, loopback)",
            "connector": f"{CONNECTOR_MODULE}.{CONNECTOR_NAME}",
        },
        "limits": {"max_model_len": config.max_model_len, "block_size": config.block_size},
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
            "health": "/health",
        },
        "halves": None if stack is None else {"prefill": half(stack.prefill), "decode": half(stack.decode)},
        "boot": None if stack is None else stack.boot,
    }


def prefill_request(req_data: dict[str, Any]) -> dict[str, Any]:
    """The request as P sees it: one token, non-streaming, asking for its KV to be handed on."""
    data = dict(req_data)
    data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }
    data["stream"] = False
    data["max_tokens"] = 1
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 1
    data.pop("stream_options", None)
    data.pop("min_tokens", None)
    data.pop("min_completion_tokens", None)
    return data


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi, httpx and uvicorn are required to serve (pip install fastapi uvicorn httpx)")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config = BundleConfig.from_env()
        _log("config", **{k: v for k, v in bundle_description(config, None).items() if k not in ("halves", "boot")})
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=128),
        )
        stack = Stack(config, VllmHalf(config, "prefill"), VllmHalf(config, "decode"))
        warm = weight_cache_is_warm()
        serial = config.serial_cold_boot and not warm
        stack.boot = {"weight_cache_warm": warm, "serial": serial, "started_at": time.time()}
        _log("boot", **stack.boot)
        t0 = time.monotonic()
        try:
            if serial:
                # cold cache: let one half convert and write the weight cache before the other reads it
                stack.prefill.start()
                await stack.prefill.wait_ready(client)
                stack.decode.start()
                await stack.decode.wait_ready(client)
            else:
                stack.prefill.start()
                stack.decode.start()
                await asyncio.gather(stack.prefill.wait_ready(client), stack.decode.wait_ready(client))
        except BaseException:
            stack.prefill.stop(grace_seconds=60.0)
            stack.decode.stop(grace_seconds=60.0)
            await client.aclose()
            raise
        stack.boot["ready_seconds"] = round(time.monotonic() - t0, 1)
        _log("stack_ready", **stack.boot)
        app.state.config = config
        app.state.stack = stack
        app.state.client = client
        try:
            yield
        finally:
            await client.aclose()
            # P first: it holds no request state D still needs once the client streams stop
            await asyncio.to_thread(stack.prefill.stop)
            await asyncio.to_thread(stack.decode.stop)

    app = FastAPI(title=BUNDLE_NAME, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def _dead(stack: Stack) -> JSONResponse | None:
        half = stack.dead_half()
        if half is None:
            return None
        status = half.report_exit()
        return JSONResponse(
            {
                "error": {
                    "message": f"the {half.role} half {status['outcome']} with status {status['status']}; "
                    "restart the server",
                    "type": "server_error",
                }
            },
            status_code=503,
        )

    @app.get("/")
    async def root(request: Request) -> JSONResponse:
        return JSONResponse(bundle_description(request.app.state.config, request.app.state.stack))

    async def passthrough(request: Request, path: str) -> Response:
        """Anything that is not a generation request goes to D (both halves serve the same model)."""
        client: httpx.AsyncClient = request.app.state.client
        stack: Stack = request.app.state.stack
        dead = _dead(stack)
        if dead is not None:
            return dead
        body = await request.body()
        upstream = client.build_request(
            request.method,
            f"{stack.decode.base_url}{path}",
            headers=filtered_headers(request.headers, STRIPPED_REQUEST_HEADERS),
            content=body if body else None,
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError as error:
            return JSONResponse(
                {
                    "error": {
                        "message": f"the decode half is unreachable: {type(error).__name__}: {error}",
                        "type": "server_error",
                    }
                },
                status_code=502,
            )
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=dict(filtered_headers(response.headers, STRIPPED_RESPONSE_HEADERS)),
            media_type=response.headers.get("content-type"),
            background=BackgroundTask(response.aclose),
        )

    async def disaggregated(request: Request, path: str) -> Response:
        """Prefill on P (one token, returns kv_transfer_params), then the full generation streamed from D."""
        client: httpx.AsyncClient = request.app.state.client
        stack: Stack = request.app.state.stack
        dead = _dead(stack)
        if dead is not None:
            return dead
        try:
            req_data = await request.json()
        except ValueError as error:
            return JSONResponse(
                {"error": {"message": f"invalid JSON body: {error}", "type": "invalid_request_error"}}, status_code=400
            )
        if not isinstance(req_data, dict):
            return JSONResponse(
                {"error": {"message": "the body must be a JSON object", "type": "invalid_request_error"}},
                status_code=400,
            )
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        headers = {"X-Request-Id": request_id}
        auth = request.headers.get("authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            p_response = await client.post(
                f"{stack.prefill.base_url}{path}", json=prefill_request(req_data), headers=headers
            )
        except httpx.HTTPError as error:
            return JSONResponse(
                {
                    "error": {
                        "message": f"the prefill half is unreachable: {type(error).__name__}: {error}",
                        "type": "server_error",
                    }
                },
                status_code=502,
            )
        if p_response.status_code != 200:
            # a rejected request (bad params, too long, ...) is reported as P reported it
            return Response(
                content=p_response.content,
                status_code=p_response.status_code,
                media_type=p_response.headers.get("content-type"),
            )
        params = p_response.json().get("kv_transfer_params") or {}
        if not params.get("do_remote_prefill"):
            return JSONResponse(
                {
                    "error": {
                        "message": "the prefill half returned no kv_transfer_params; is its KV connector active?",
                        "type": "server_error",
                    }
                },
                status_code=502,
            )
        d_data = dict(req_data)
        d_data["kv_transfer_params"] = params
        upstream = client.build_request("POST", f"{stack.decode.base_url}{path}", json=d_data, headers=headers)
        try:
            d_response = await client.send(upstream, stream=True)
        except httpx.HTTPError as error:
            return JSONResponse(
                {
                    "error": {
                        "message": f"the decode half is unreachable: {type(error).__name__}: {error}",
                        "type": "server_error",
                    }
                },
                status_code=502,
            )
        # aiter_raw passes the SSE stream through chunk by chunk; a client that hangs up closes the upstream behind it
        return StreamingResponse(
            d_response.aiter_raw(),
            status_code=d_response.status_code,
            headers=dict(filtered_headers(d_response.headers, STRIPPED_RESPONSE_HEADERS)),
            media_type=d_response.headers.get("content-type"),
            background=BackgroundTask(d_response.aclose),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Response:
        return await disaggregated(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request) -> Response:
        return await disaggregated(request, "/v1/completions")

    @app.api_route("/health", methods=["GET", "HEAD"])
    async def health(request: Request) -> Response:
        return await passthrough(request, "/health")

    @app.api_route("/version", methods=["GET"])
    async def version(request: Request) -> Response:
        return await passthrough(request, "/version")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "HEAD", "OPTIONS"])
    async def v1(request: Request, path: str) -> Response:
        return await passthrough(request, f"/v1/{path}")

    return app


_APP = None


def __getattr__(name: str):
    """``app`` is built on first access (PEP 562): uvicorn's ``pd_app:app`` and the image's verify line get the
    FastAPI application, while the pure parts above import in a checkout without fastapi installed."""
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)
