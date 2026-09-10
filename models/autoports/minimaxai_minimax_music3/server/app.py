# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-style audio endpoint for MiniMax-Music3 on one Blackhole chip (stage 08).

Launched the way tt-model-manager's ``TtDitServerLauncher`` launches every ``tt-dit-server`` app::

    python -m uvicorn --host 0.0.0.0 --port 8000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app

with ``HF_MODEL`` (an HF repo id or a local snapshot directory), ``MM3_MESH_SHAPE`` (``RxC``; v1 is single-chip and
always opens a 1x1 mesh on device 0 - the shape it was given is only logged), ``MESH_DEVICE`` (logged),
``TT_DIT_CACHE_DIR`` (DiT weight cache) and ``HF_HOME``. The ASGI lifespan resolves the weights, opens the device, loads
``MiniMaxMusic3Pipeline`` with the optimized dtype policy, warms every trace with a tiny song and only then yields, so
uvicorn's ``Application startup complete`` line means the chip is claimed and the model is warm.

Endpoints (the SGLang-Omni contract from the model card):

* ``POST /v1/audio/speech`` - JSON ``{model, input (lyrics), instructions (caption), response_format: "wav", seed,
  max_new_tokens (frames at 25 fps, default 1500), num_inference_steps (30), stream: false}`` -> ``audio/wav``
  (44.1 kHz stereo 16-bit PCM). ``audio_duration`` (seconds) is accepted as an alias of ``max_new_tokens``.
  ``stream: true`` -> 501. Limits: prompt <= 5000 tokens, frames <= 9000, steps 1..200.
* ``GET /health`` -> ``{"status": "ok", "model", "device", "warm": true, "busy"}``.
* ``GET /v1/models`` -> the OpenAI list shape.

One generation at a time (``asyncio.Lock``); every device call (load, warm-up, generation, shutdown) runs on ONE
dedicated worker thread so the event loop keeps answering ``/health`` while the chip is busy. ``MM3_REQUEST_TIMEOUT_S``
bounds a request (504); the generation itself cannot be interrupted, so the lock stays held until the device is idle.
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.autoports.minimaxai_minimax_music3.tt.constants import FRAME_RATE, MAX_AUDIO_FRAMES, MAX_PROMPT_TOKENS
from models.autoports.minimaxai_minimax_music3.tt.denoiser import DEFAULT_STEPS
from models.autoports.minimaxai_minimax_music3.tt.prompt import PromptEncoder

MODEL_ID = "MiniMaxAI/MiniMax-Music3"
DEFAULT_FRAMES = 1500  # 60 s, the model card's default `max_new_tokens`
MAX_STEPS = 200
# Raw MiniMax checkpoints the diffusers layout does not use (about 20 GB): never downloaded by the server.
WEIGHTS_IGNORE_PATTERNS = ("flowmatching_vae.pth", "dav.pth", "qwen_7B/*", "assets/*", "figures/*", "scripts/*")
WARMUP_PROMPT = "Genre: ambient. A quiet pad."
WARMUP_LYRICS = "[verse]\nla la la"

MODEL_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("TT_DIT_CACHE_DIR", str(MODEL_DIR / "generated" / "tt_dit_cache"))


# ----------------------------------------------------------------------------- configuration
def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    return float(raw) if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


class Settings:
    """Everything the launcher / operator controls through the environment."""

    def __init__(self) -> None:
        self.hf_model = os.environ.get("HF_MODEL", "").strip() or os.environ.get("MM3_WEIGHTS", "").strip()
        self.mesh_shape = os.environ.get("MM3_MESH_SHAPE", "1x1").strip() or "1x1"
        self.mesh_device_name = os.environ.get("MESH_DEVICE", "").strip()
        self.dtype_policy = os.environ.get("MM3_DTYPE_POLICY", "optimized").strip() or "optimized"
        self.request_timeout_s = _env_float("MM3_REQUEST_TIMEOUT_S", 1800.0)
        self.warmup_seconds = _env_float("MM3_WARMUP_SECONDS", 2.0)
        self.warmup_steps = _env_int("MM3_WARMUP_STEPS", 4)
        self.trace_region_size = _env_int("MM3_TRACE_REGION_SIZE", 200_000_000)
        self.torch_threads = _env_int("MM3_TORCH_THREADS", max(8, (os.cpu_count() or 8) - 4))


def resolve_weights(hf_model: str) -> Path:
    """``HF_MODEL`` -> the snapshot directory holding ``language_model/``, ``transformer/`` ... .

    A local directory is used as is (``$MM3_WEIGHTS/language_model``, which common.sh exports as ``HF_MODEL`` for
    tt_transformers, resolves to its parent). Anything else is an HF repo id and goes through ``snapshot_download``
    with the raw ``.pth`` / ``qwen_7B`` checkpoints excluded.
    """
    if not hf_model:
        raise RuntimeError("HF_MODEL is not set (HF repo id or local snapshot directory of MiniMaxAI/MiniMax-Music3)")
    local = Path(hf_model).expanduser()
    if local.is_dir():
        if (local / "language_model" / "config.json").is_file():
            return local.resolve()
        if local.name == "language_model" and (local.parent / "transformer").is_dir():
            return local.parent.resolve()
        raise RuntimeError(f"HF_MODEL={hf_model!r} is a directory but not a MiniMax-Music3 snapshot")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", hf_model):
        raise RuntimeError(f"HF_MODEL={hf_model!r} is neither an existing directory nor an HF repo id")
    from huggingface_hub import snapshot_download

    revision = os.environ.get("MM3_HF_REVISION") or None
    logger.info(f"downloading {hf_model} (revision {revision or 'main'}) without the raw .pth / qwen_7B checkpoints")
    path = snapshot_download(hf_model, revision=revision, ignore_patterns=list(WEIGHTS_IGNORE_PATTERNS))
    return Path(path).resolve()


# ----------------------------------------------------------------------------- the model behind the app
class Engine:
    """The device, the pipeline and the one worker thread that is allowed to touch them."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mm3-device")
        self.lock = asyncio.Lock()
        self.mesh = None
        self.pipe = None
        self.weights_dir: Optional[Path] = None
        self.warm = False
        self.busy = False
        self.started_at = time.time()
        self.requests_served = 0
        self.last_request: Optional[Dict[str, Any]] = None
        self.device_info: Dict[str, Any] = {}
        self.load_log: Dict[str, Any] = {}
        # Host tokenizer for request validation, separate from the pipeline's so the event loop never shares the
        # tokenizer's mutable BPE cache with the device thread.
        self.prompt_encoder: Optional[PromptEncoder] = None

    # -- device thread -------------------------------------------------------------------------------------------
    async def run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    def _start(self) -> None:
        import ttnn

        s = self.settings
        torch.set_num_threads(s.torch_threads)
        self.weights_dir = resolve_weights(s.hf_model)
        logger.info(f"weights: {self.weights_dir}")
        if s.mesh_shape != "1x1":
            logger.warning(
                f"MM3_MESH_SHAPE={s.mesh_shape} requested; v1 is single-chip and opens a 1x1 mesh on device 0"
            )
        logger.info(
            f"MESH_DEVICE={s.mesh_device_name or '(unset)'} MM3_MESH_SHAPE={s.mesh_shape} -> 1x1 mesh, device 0"
        )
        t0 = time.perf_counter()
        # No fabric on this multi-chip host (see tests/conftest.py); a 1x1 mesh does not need it.
        self.mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=s.trace_region_size)
        try:
            self._load_and_warm(t0)
        except BaseException:
            # A failed start must not depend on process teardown to release the chip.
            logger.exception("startup failed; closing the device")
            self._stop()
            raise

    def _load_and_warm(self, t0: float) -> None:
        import ttnn
        from models.autoports.minimaxai_minimax_music3.tt.pipeline import MiniMaxMusic3Pipeline

        s = self.settings
        self.mesh.enable_program_cache()
        self.device_info = {
            "mesh_shape": "1x1",
            "requested_mesh_shape": s.mesh_shape,
            "mesh_device": s.mesh_device_name or None,
            "device_ids": [int(i) for i in self.mesh.get_device_ids()],
            "arch": str(ttnn.get_arch_name()),
            "visible_devices": os.environ.get("TT_METAL_VISIBLE_DEVICES"),
            "hostname": socket.gethostname(),
        }
        self.pipe = MiniMaxMusic3Pipeline.load(self.mesh, str(self.weights_dir), dtype_policy=s.dtype_policy)
        self.prompt_encoder = PromptEncoder(str(self.weights_dir / "tokenizer"))
        self.load_log = dict(self.pipe.load_log)
        self.load_log["open_device_and_load_s"] = time.perf_counter() - t0
        # Warm-up song: the smallest window shape, its DiT trace, the vocoder worker round trip and the wav encoder,
        # so the first real request sees a fully warm server (`load` already captured the AR traces and the
        # 200-frame-window DiT trace).
        t0 = time.perf_counter()
        out = self.pipe.generate(
            WARMUP_PROMPT,
            WARMUP_LYRICS,
            audio_duration=s.warmup_seconds,
            seed=0,
            num_inference_steps=s.warmup_steps,
            keep_latents=False,
        )
        encode_wav(out["audio"], out["sampling_rate"])
        self.load_log["warmup_song_s"] = time.perf_counter() - t0
        self.load_log["warmup_song_frames"] = int(out["frames"])
        self.warm = True
        logger.info(
            f"server warm: {out['frames']}-frame warm-up song in {self.load_log['warmup_song_s']:.1f} s; "
            f"load {self.load_log.get('load_total_s', 0):.0f} s"
        )

    def _stop(self) -> None:
        import ttnn

        self.warm = False
        try:
            if self.pipe is not None:
                self.pipe.release()
        finally:
            self.pipe = None
            if self.mesh is not None:
                ttnn.close_mesh_device(self.mesh)
                self.mesh = None
        logger.info("device closed")

    def _generate(self, req: "SpeechRequest", frames: int, steps: int) -> Dict[str, Any]:
        assert self.pipe is not None
        return self.pipe.generate(
            req.instructions,
            req.input,
            max_frames=frames,
            audio_duration=frames / FRAME_RATE,
            seed=req.seed,
            num_inference_steps=steps,
            keep_latents=False,
        )

    # -- public ----------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.run(self._start)

    async def stop(self) -> None:
        await self.run(self._stop)
        self.executor.shutdown(wait=True)

    def prompt_tokens(self, instructions: str, lyrics: str) -> int:
        """Host-side tokenization (milliseconds): raises ``ValueError`` for empty parts or more than 5000 tokens."""
        assert self.prompt_encoder is not None
        return int(self.prompt_encoder.encode(instructions, lyrics).shape[1])

    async def generate_locked(self, req: "SpeechRequest", frames: int, steps: int) -> Dict[str, Any]:
        """Run one generation with the lock held for exactly as long as the device is busy."""
        async with self.lock:
            self.busy = True
            try:
                out = await self.run(self._generate, req, frames, steps)
            finally:
                self.busy = False
            self.requests_served += 1
            return out


# ----------------------------------------------------------------------------- request / response
class SpeechRequest(BaseModel):
    """The model card's ``/v1/audio/speech`` body (``input`` = lyrics, ``instructions`` = music description)."""

    model_config = ConfigDict(extra="ignore")

    model: Optional[str] = None
    input: str = Field(..., description="lyrics; structure tags such as [verse] on their own lines")
    instructions: str = Field(..., description="music description / caption")
    response_format: str = "wav"
    seed: Optional[int] = Field(None, ge=0, le=2**63 - 1)
    max_new_tokens: Optional[int] = Field(None, ge=1, description="audio frames at 25 fps (default 1500)")
    audio_duration: Optional[float] = Field(None, gt=0, description="seconds; alias of max_new_tokens / 25")
    num_inference_steps: Optional[int] = Field(None, ge=1)
    stream: bool = False
    speed: Optional[float] = None  # OpenAI field, ignored
    voice: Optional[str] = None  # OpenAI field, ignored

    @model_validator(mode="after")
    def _check(self) -> "SpeechRequest":
        if not self.input.strip():
            raise ValueError("`input` (the lyrics) must be a non-empty string")
        if not self.instructions.strip():
            raise ValueError("`instructions` (the music description) must be a non-empty string")
        if self.response_format.lower() != "wav":
            raise ValueError(f'response_format {self.response_format!r} is not supported; only "wav"')
        if self.max_new_tokens is not None and self.audio_duration is not None:
            if int(self.audio_duration * FRAME_RATE) != self.max_new_tokens:
                raise ValueError("`max_new_tokens` and `audio_duration` disagree; pass one of them")
        return self

    def frames(self) -> int:
        if self.max_new_tokens is not None:
            frames = int(self.max_new_tokens)
        elif self.audio_duration is not None:
            frames = int(self.audio_duration * FRAME_RATE)
        else:
            frames = DEFAULT_FRAMES
        if frames < 1:
            raise ValueError(
                f"`audio_duration` {self.audio_duration} is shorter than one audio frame (1 / {FRAME_RATE:.0f} s)"
            )
        if frames > MAX_AUDIO_FRAMES:
            raise ValueError(
                f"`max_new_tokens` {frames} exceeds the maximum of {MAX_AUDIO_FRAMES} frames ({MAX_AUDIO_FRAMES / FRAME_RATE:.0f} s)"
            )
        return frames

    def steps(self) -> int:
        steps = int(self.num_inference_steps) if self.num_inference_steps is not None else DEFAULT_STEPS
        if not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"`num_inference_steps` must be in 1..{MAX_STEPS}, got {steps}")
        return steps


def encode_wav(audio: np.ndarray, sampling_rate: int) -> bytes:
    """``[2, S]`` float32 in [-1, 1] -> 16-bit PCM stereo RIFF/WAVE bytes."""
    buf = io.BytesIO()
    sf.write(buf, np.ascontiguousarray(audio.T), int(sampling_rate), format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _error(status: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type}})


# ----------------------------------------------------------------------------- the app
settings = Settings()
engine = Engine(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"MiniMax-Music3 server starting: HF_MODEL={settings.hf_model!r} policy={settings.dtype_policy}")
    await engine.start()
    try:
        yield
    finally:
        logger.info("MiniMax-Music3 server shutting down")
        await engine.stop()


app = FastAPI(title="MiniMax-Music3 on Tenstorrent", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if engine.warm else "loading",
        "model": MODEL_ID,
        "device": engine.device_info,
        "warm": engine.warm,
        "busy": engine.busy,
        "dtype_policy": settings.dtype_policy,
        "weights_dir": str(engine.weights_dir) if engine.weights_dir else None,
        "uptime_s": round(time.time() - engine.started_at, 1),
        "requests_served": engine.requests_served,
        "last_request": engine.last_request,
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "created": int(engine.started_at),
                "owned_by": "MiniMaxAI",
                "capabilities": {"audio": True, "response_formats": ["wav"], "stream": False},
            }
        ],
    }


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest, request: Request):
    if req.stream:
        return _error(501, "streaming is not supported; send stream: false", "not_implemented")
    if req.model and req.model not in (MODEL_ID, MODEL_ID.split("/")[-1], settings.hf_model):
        return _error(404, f"model {req.model!r} is not served here; use {MODEL_ID!r}", "not_found_error")
    if not engine.warm:
        return _error(503, "model is still loading", "server_error")
    try:
        frames = req.frames()
        steps = req.steps()
        prompt_tokens = await asyncio.to_thread(engine.prompt_tokens, req.instructions, req.input)
    except ValueError as exc:
        return _error(400, str(exc))
    if prompt_tokens > MAX_PROMPT_TOKENS:  # build_text_ids already raises; kept as the documented limit
        return _error(400, f"prompt has {prompt_tokens} tokens; the maximum is {MAX_PROMPT_TOKENS}")

    t0 = time.perf_counter()
    task = asyncio.ensure_future(engine.generate_locked(req, frames, steps))
    try:
        out = await asyncio.wait_for(asyncio.shield(task), timeout=settings.request_timeout_s)
    except asyncio.TimeoutError:
        logger.error(f"request timed out after {settings.request_timeout_s:.0f} s; the device finishes the song first")

        def _abandoned(done: "asyncio.Future"):
            exc = done.exception() if not done.cancelled() else None
            if exc is not None:
                logger.error(f"abandoned generation failed: {exc!r}")
            else:
                logger.warning("abandoned generation finished; its audio was discarded")
            engine.last_request = {
                "frames": frames,
                "num_inference_steps": steps,
                "outcome": "timeout",
                "error": repr(exc) if exc else None,
            }

        task.add_done_callback(_abandoned)
        return _error(504, f"generation exceeded MM3_REQUEST_TIMEOUT_S={settings.request_timeout_s:.0f}", "timeout")
    except Exception as exc:  # the pipeline failed: report it, keep serving
        logger.exception("generation failed")
        return _error(500, f"generation failed: {type(exc).__name__}: {exc}", "server_error")
    wav = encode_wav(out["audio"], out["sampling_rate"])
    seconds = out["audio"].shape[-1] / out["sampling_rate"]
    total = time.perf_counter() - t0
    engine.last_request = {
        "frames": int(out["frames"]),
        "requested_frames": frames,
        "audio_seconds": round(seconds, 3),
        "seed": int(out["seed"]),
        "num_inference_steps": steps,
        "prompt_tokens": prompt_tokens,
        "stopped_by": out["stopped_by"],
        "generation_s": round(float(out["timings"]["total"]), 2),
        "wall_s": round(total, 2),
    }
    logger.info(f"served {seconds:.2f} s of audio in {total:.1f} s: {engine.last_request}")
    headers = {
        "Content-Disposition": 'attachment; filename="minimax_music3.wav"',
        "X-MM3-Frames": str(int(out["frames"])),
        "X-MM3-Seed": str(int(out["seed"])),
        "X-MM3-Stopped-By": str(out["stopped_by"]),
        "X-MM3-Prompt-Tokens": str(prompt_tokens),
        "X-MM3-Generation-Seconds": f"{out['timings']['total']:.2f}",
        "X-MM3-Sampling-Rate": str(int(out["sampling_rate"])),
    }
    return Response(content=wav, media_type="audio/wav", headers=headers)


@app.exception_handler(HTTPException)
async def _http_exception(request: Request, exc: HTTPException):
    return _error(exc.status_code, str(exc.detail))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    """Body validation failures in the OpenAI error shape (422 like FastAPI's default, one readable message)."""
    parts = []
    for err in exc.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()) if x != "body")
        msg = err.get("msg", "invalid")
        parts.append(f"{loc}: {msg}" if loc else msg)
    return _error(422, "; ".join(parts) or "invalid request body")
