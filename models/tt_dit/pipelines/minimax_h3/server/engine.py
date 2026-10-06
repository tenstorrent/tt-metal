# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Request validation, the job queue, and the one worker thread that owns the mesh.

The shape of this is the same as every other tt-dit server: HTTP handlers only ever touch the queue
and the job table, and exactly one thread opens the mesh, builds the pipeline and runs generations.
A diffusion mesh cannot be shared between requests, so concurrency here means *queueing*, not
parallelism, and the contract says so explicitly (``queue_position``, 429 when full).

Three things are specific to MiniMax-H3 and worth reading before changing anything:

* **Cancel is cooperative and lives in the event callback.** The pipeline fires `DenoiseStep` after
  every forward; the callback raises `Cancelled` from inside the pipeline when the job's flag is
  set. That is only safe because `trace_denoise` is off on both served presets ((1,1) and (1,4)) --
  on a traced preset the unwind would happen inside a trace region. If a preset here ever turns
  tracing on, cancel has to become a between-requests operation.
* **The hashes are of the raw tensors, not of the mp4.** x264 and AAC are not bit-reproducible
  across builds and flags, so a determinism check made on the container bytes answers a question
  about the encoder. `frames_sha256` is the uint8 ``[F, H, W, 3]`` the encoder is *fed* and
  `audio_sha256` is the float32 stereo the wav is quantized from -- both taken before any lossy step.
* **A freshly written weight cache has to be healed before it is loaded.** See `_heal_new_caches`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import itertools
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .config import (
    AUDIO_CHANNELS,
    CANVAS_MULTIPLE,
    FPS,
    FRAMES_MIN,
    LONG_EDGE_MAX,
    SAMPLE_RATE,
    SHORT_EDGE_MIN,
    ServerConfig,
    align_frames,
    canvas_for_aspect,
    steps_to_grid_points,
)

log = logging.getLogger("minimax_h3.server")

#: Job phases, in the only order they may be observed in. `Job.set_phase` refuses to go backwards.
PHASES = (
    "queued",
    "encoding",
    "encoding_image",
    "denoising",
    "decoding_video",
    "decoding_audio",
    "muxing",
    "completed",
)
#: Pipeline section name -> the phase it presents as.
SECTION_PHASE = {
    "encoder": "encoding",
    "vae_encode": "encoding_image",
    "denoising": "denoising",
    "vae": "decoding_video",
    "audio": "decoding_audio",
}
#: Pipeline section name -> the `timings` key it accumulates into.
SECTION_TIMING = {
    "encoder": "encode_s",
    "vae_encode": "encode_s",
    "denoising": "denoise_s",
    "vae": "vae_s",
    "audio": "audio_s",
}
#: Fraction of `progress` already spent when a phase begins. Denoising interpolates across its share.
PHASE_PROGRESS = {
    "queued": 0.0,
    "encoding": 0.02,
    "encoding_image": 0.05,
    "denoising": 0.08,
    "decoding_video": 0.80,
    "decoding_audio": 0.92,
    "muxing": 0.97,
    "completed": 1.0,
}

MAX_PROMPT_CHARS = 8000


class Cancelled(Exception):
    """Raised inside the pipeline, from the event callback, to abandon a running job."""


# --------------------------------------------------------------------------- request


class ImagePrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(..., description="base64 PNG/JPEG, or a data: URL carrying one")
    position: Literal["first", "last"] = "first"

    def decode(self):
        """-> PIL.Image. Accepts a bare base64 payload or a ``data:image/png;base64,…`` URL."""
        from PIL import Image

        payload = self.image.strip()
        if payload.startswith("data:"):
            _header, _, payload = payload.partition(",")
        try:
            raw = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image_prompts[{self.position}].image is not valid base64") from exc
        try:
            return Image.open(io.BytesIO(raw))
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"image_prompts[{self.position}].image is not a readable PNG/JPEG") from exc


class GenerationRequest(BaseModel):
    """`POST /v1/videos/generations`, validated and normalized against one `ServerConfig`.

    Pydantic enforces the types and `extra="forbid"` turns an unknown field into a 422; everything
    that depends on the served profile (canvas limits, the frame grid, the step ceiling) is applied
    in `normalize`, which is a plain function so it can be unit-tested without a request.
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str
    mode: Literal["t2va", "fl2va"] | None = None
    width: int | None = None
    height: int | None = None
    aspect_ratio: str = "16:9"
    num_frames: int | None = None
    duration_seconds: float | None = None
    steps: int | None = None
    seed: int = 0
    image_prompts: list[ImagePrompt] | None = None

    @model_validator(mode="after")
    def _check(self) -> "GenerationRequest":
        prompt = self.prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt is {len(prompt)} characters, the limit is {MAX_PROMPT_CHARS}")
        images = self.image_prompts or []
        if len(images) > 2:
            raise ValueError("at most two image prompts (one 'first', one 'last')")
        if len({i.position for i in images}) != len(images):
            raise ValueError("image prompts must have distinct positions")
        mode = self.mode or ("fl2va" if images else "t2va")
        if mode == "fl2va" and not images:
            raise ValueError("mode 'fl2va' requires at least one image prompt")
        if mode == "t2va" and images:
            raise ValueError("mode 't2va' takes no image prompts")
        if (self.width is None) != (self.height is None):
            raise ValueError("pass both width and height, or neither")
        if self.num_frames is not None and self.duration_seconds is not None:
            raise ValueError("pass num_frames or duration_seconds, not both")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        return self


@dataclass(frozen=True)
class NormalizedRequest:
    """What the engine actually renders: every field resolved, nothing left to a default."""

    mode: str
    prompt: str
    width: int
    height: int
    num_frames: int
    steps: int
    seed: int
    images: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "prompt": self.prompt,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "steps": self.steps,
            "seed": self.seed,
        }


def normalize(req: GenerationRequest, cfg: ServerConfig) -> NormalizedRequest:
    """Resolve a validated request against the served profile. Raises `ValueError` -> 422."""
    images = req.image_prompts or []
    mode = req.mode or ("fl2va" if images else "t2va")

    if req.width is not None:
        width, height = int(req.width), int(req.height)
    else:
        width, height = canvas_for_aspect(req.aspect_ratio, min(cfg.width, cfg.height))
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(f"canvas {width}x{height} must be a multiple of {CANVAS_MULTIPLE} on both axes")
    short, long = min(width, height), max(width, height)
    if not SHORT_EDGE_MIN <= short <= cfg.short_edge_max:
        raise ValueError(f"short edge {short} is outside [{SHORT_EDGE_MIN}, {cfg.short_edge_max}]")
    if long > LONG_EDGE_MAX:
        raise ValueError(f"long edge {long} exceeds {LONG_EDGE_MAX}")

    if req.duration_seconds is not None:
        requested = int(round(req.duration_seconds * FPS))
    elif req.num_frames is not None:
        requested = int(req.num_frames)
    else:
        requested = cfg.num_frames
    if requested < 1:
        raise ValueError(f"num_frames must be positive, got {requested}")
    num_frames = align_frames(requested)
    if num_frames < FRAMES_MIN:
        raise ValueError(f"{requested} frames snaps to {num_frames}, below the minimum {FRAMES_MIN}")
    if num_frames > cfg.frames_max:
        raise ValueError(f"{requested} frames snaps to {num_frames}, above this profile's maximum {cfg.frames_max}")

    steps = cfg.steps if req.steps is None else int(req.steps)
    if not 1 <= steps <= cfg.steps_max:
        raise ValueError(f"steps {steps} is outside [1, {cfg.steps_max}]")

    decoded = tuple((i.position, i.decode()) for i in images)
    return NormalizedRequest(
        mode=mode,
        prompt=req.prompt.strip(),
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        seed=int(req.seed),
        images=decoded,
    )


# --------------------------------------------------------------------------- job


_IDS = itertools.count(1)


@dataclass
class Job:
    req: NormalizedRequest
    id: str = field(default_factory=lambda: f"vid_{uuid.uuid4().hex[:20]}")
    seq: int = field(default_factory=lambda: next(_IDS))
    status: str = "queued"
    phase: str = "queued"
    progress: float = 0.0
    step: int = 0
    total_steps: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    timings: dict = field(
        default_factory=lambda: {
            "encode_s": 0.0,
            "denoise_s": 0.0,
            "vae_s": 0.0,
            "audio_s": 0.0,
            "mux_s": 0.0,
            "total_s": 0.0,
        }
    )
    result: dict | None = None
    path: str | None = None
    error: str | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- state transitions (all monotonic; see the `progress` contract in docs/API.md)
    def set_phase(self, phase: str) -> None:
        with self._lock:
            if PHASES.index(phase) < PHASES.index(self.phase):
                return
            self.phase = phase
            self.progress = max(self.progress, PHASE_PROGRESS[phase])

    def set_step(self, step: int, total: int) -> None:
        with self._lock:
            self.total_steps = max(self.total_steps, int(total))
            self.step = max(self.step, int(step))
            span = PHASE_PROGRESS["decoding_video"] - PHASE_PROGRESS["denoising"]
            share = PHASE_PROGRESS["denoising"] + span * (self.step / max(self.total_steps, 1))
            self.progress = min(max(self.progress, share), PHASE_PROGRESS["decoding_video"])

    def add_timing(self, key: str, seconds: float) -> None:
        with self._lock:
            self.timings[key] = round(self.timings.get(key, 0.0) + seconds, 3)

    @property
    def elapsed_s(self) -> float:
        start = self.started_at or self.created_at
        return round((self.completed_at or time.time()) - start, 2)

    def to_dict(self, queue_position: int = 0) -> dict:
        return {
            "id": self.id,
            "object": "video.generation",
            "status": self.status,
            "phase": self.phase,
            "progress": round(self.progress, 4),
            "step": self.step,
            "total_steps": self.total_steps or self.req.steps,
            "queue_position": queue_position,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_s": self.elapsed_s,
            "request": self.req.as_dict(),
            "timings": dict(self.timings),
            "result": self.result,
            "error": self.error,
        }

    def to_alias(self) -> dict:
        """Eric Zietlow's QB2 service view of the same job (`GET /jobs/{id}`)."""
        status = {"completed": "done", "failed": "error"}.get(self.status, self.status)
        return {
            "job_id": self.id,
            "status": status,
            "phase": self.phase,
            "step": self.step,
            "total_steps": self.total_steps or self.req.steps,
            "elapsed": self.elapsed_s,
            "error": self.error,
            "result_path": self.path,
        }


# --------------------------------------------------------------------------- media


def ffmpeg_exe() -> str:
    """The ffmpeg this server encodes with.

    `imageio-ffmpeg`'s bundled static build, because the tt-model runtime image has no system
    ffmpeg; a system binary is used only if the bundle is missing.
    """
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("no ffmpeg: install imageio-ffmpeg or put ffmpeg on PATH") from None
        return exe


def write_mp4(frames, pcm, dest: Path, fps: int = FPS, sample_rate: int = SAMPLE_RATE) -> int:
    """``frames`` uint8 [F, H, W, 3] + ``pcm`` float32 [N, 2] -> h264/yuv420p + AAC mp4. Returns bytes.

    Two passes, as in the bring-up's `run_e2e_h3.py`, and deliberately WITHOUT ``-shortest``: the AAC
    frame granularity makes the encoded audio a hair shorter than the video, and ``-shortest`` then
    truncates the video to match -- a 56-frame clip muxes to 53 and fails a frame-count check for a
    reason that has nothing to do with the model.
    """
    import wave

    import numpy as np

    exe = ffmpeg_exe()
    height, width = int(frames.shape[1]), int(frames.shape[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
        silent = Path(tmp) / "video.mp4"
        wav = Path(tmp) / "audio.wav"
        with wave.open(str(wav), "wb") as handle:
            handle.setnchannels(int(pcm.shape[1]))
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes((np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
        _run(
            [
                exe,
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                str(silent),
            ],
            frames.tobytes(),
        )
        _run(
            [
                exe,
                "-y",
                "-v",
                "error",
                "-i",
                str(silent),
                "-i",
                str(wav),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                str(sample_rate),
                "-ac",
                str(AUDIO_CHANNELS),
                str(dest),
            ]
        )
    return dest.stat().st_size


def _run(cmd: list[str], data: bytes | None = None) -> None:
    out = subprocess.run(cmd, input=data, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"{Path(cmd[0]).name} failed ({out.returncode}): {out.stderr[-600:].decode(errors='replace')}"
        )


# --------------------------------------------------------------------------- engine


class H3Engine:
    """One worker thread, one mesh, a FIFO queue in front of it."""

    def __init__(self, cfg: ServerConfig, log_fn: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.log = log_fn or log.info
        self.pipeline = None
        self.mesh_device = None
        self._fabric = False
        self.ready = threading.Event()
        self.stopping = threading.Event()
        self.dead: str | None = None
        self.started = time.time()
        self.boot_s: dict[str, float] = {}
        self.served = 0
        self.build_info: dict = {}
        self._q: "queue.Queue[Job | None]" = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._current: Job | None = None
        self._thread = threading.Thread(target=self._run, name="h3-worker", daemon=True)

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread.start()

    def wait_ready(self) -> None:
        """Block until the pipeline is loaded AND warm. Raises if the load failed.

        The lifespan awaits this, so uvicorn's "Application startup complete" is printed exactly when
        the server can generate -- and a load failure propagates out of the lifespan, which is what
        makes the container exit non-zero instead of serving 503s forever.
        """
        while not self.ready.wait(0.5):
            if self.dead:
                raise RuntimeError(f"engine failed to start: {self.dead}")
            if self.stopping.is_set():
                raise RuntimeError("engine stopped before it became ready")

    def shutdown(self, timeout: float = 240.0) -> None:
        """Stop accepting, cancel the running job at its next step, close the mesh.

        Called from the lifespan's shutdown half, i.e. on SIGTERM. Doing the cancel *and* the mesh
        close here is what keeps `tt-model stop` from needing a SIGKILL and a chip reset.
        """
        self.stopping.set()
        with self._lock:
            pending = [j for j in self._jobs.values() if not j.done.is_set()]
        for job in pending:
            job.cancel.set()
        self._q.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            self.log("worker did not stop in time; closing the mesh from the main thread")
        self._close_mesh()

    def _close_mesh(self) -> None:
        if self.mesh_device is None:
            return
        import ttnn

        device, self.mesh_device = self.mesh_device, None
        try:
            ttnn.close_mesh_device(device)
        finally:
            if self._fabric:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        self.log("mesh closed")

    # ---------------------------------------------------------------- load
    def _open_mesh(self):
        import ttnn

        cfg = self.cfg
        self._fabric = cfg.mesh_rows * cfg.mesh_cols > 1
        if self._fabric:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        return ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(cfg.mesh_rows, cfg.mesh_cols), l1_small_size=cfg.l1_small_size
        )

    def _cache_dirs(self) -> set[str]:
        root = self.cfg.cache_dir
        found: set[str] = set()
        if not root or not os.path.isdir(root):
            return found
        for base, _dirs, files in os.walk(root):
            if any(f.endswith(".tensorbin") for f in files):
                found.add(base)
        return found

    def _heal_new_caches(self, before: set[str]) -> list[str]:
        """Rewrite any weight cache this build just created, onto fresh inodes, before it is loaded.

        Measured during this bring-up: a cache directory written by tt-metal's own cache writer
        livelocks the FIRST device load of it -- ~100 % system time, ~2.1 M minor faults/s, no I/O,
        and it never returns. A byte-identical copy at another path loads in milliseconds, so a
        copy-over-itself rewrite is the repair (seconds, against a hang that does not end). The
        mechanism is still unexplained; this heals it rather than claiming to explain it.

        It only ever touches directories that did not exist before this process built the pipeline,
        so a warm mounted cache is never rewritten.
        """
        fresh = sorted(self._cache_dirs() - before)
        for directory in fresh:
            for name in sorted(os.listdir(directory)):
                path = Path(directory) / name
                if not path.is_file():
                    continue
                spare = path.with_suffix(path.suffix + ".heal")
                shutil.copyfile(path, spare)
                os.replace(spare, path)
        if fresh:
            self.log(f"healed {len(fresh)} freshly written weight-cache directories")
        return fresh

    def _load(self) -> None:
        from ..pipeline_minimax_h3 import MiniMaxH3Pipeline

        cfg = self.cfg
        self.log(
            f"profile {cfg.profile} on mesh {cfg.mesh_shape}: "
            f"{cfg.width}x{cfg.height} x{cfg.num_frames}, {cfg.steps} steps, "
            f"quant={cfg.dit_quant_profile}, text_encoder={cfg.text_encoder_device}"
        )
        t0 = time.time()
        self.mesh_device = self._open_mesh()
        self.boot_s["mesh_open_s"] = round(time.time() - t0, 1)
        self.log(f"mesh {cfg.mesh_shape} open in {self.boot_s['mesh_open_s']} s")

        self.log(
            f"Loading pipeline from {cfg.weights_dir or 'the Hugging Face cache'} "
            f"(adapter {Path(cfg.turbo_file).name if cfg.turbo_file else 'none'})"
        )
        before = self._cache_dirs()
        t0 = time.time()
        kwargs: dict[str, Any] = {}
        if cfg.precomputed_adaln is not None:
            kwargs["precomputed_adaln"] = bool(cfg.precomputed_adaln)
        if cfg.text_encoder_device is not None:
            kwargs["text_encoder_device"] = cfg.text_encoder_device
        self.pipeline = MiniMaxH3Pipeline.create_pipeline(
            mesh_device=self.mesh_device,
            weights_dir=cfg.weights_dir,
            vae_output_type=cfg.vae_output_type,
            dit_quant_profile=cfg.dit_quant_profile,
            lora_path=cfg.turbo_file,
            lora_strength=cfg.lora_strength,
            video_shift=cfg.video_shift,
            audio_shift=cfg.audio_shift,
            **kwargs,
        )
        self.boot_s["build_s"] = round(time.time() - t0, 1)
        self._heal_new_caches(before)

        profile = self.pipeline.dit_quant_profile
        handle = getattr(self.pipeline, "_lora_handle", None)
        host_scales = getattr(self.pipeline, "_lora_host_scales", None)
        self.build_info = {
            "coresident": bool(self.pipeline.coresident),
            "dit_quant_profile": None if profile is None else profile.name,
            "dit_quant_cache_tag": None if profile is None else profile.cache_tag,
            "precomputed_adaln": bool(self.pipeline.precomputed_adaln),
            "text_encoder_device": self.pipeline.text_encoder_device,
            "video_shift": self.pipeline.video_shift,
            "audio_shift": self.pipeline.audio_shift,
            "adapter": Path(cfg.turbo_file).name if cfg.turbo_file else None,
            # What the adapter is actually worth, read back rather than assumed: a Turbo file
            # published without an `alpha` falls back to scale 1.0, i.e. ~16x its siblings. On the
            # host-fuse path (a quantized DiT) there is no device handle and the fused per-tensor
            # scales are the record instead.
            "lora_effective_scale": getattr(handle, "scale", None) if handle is not None else None,
            "lora_host_fused_tensors": len(host_scales) if isinstance(host_scales, dict) else 0,
        }
        self.log(f"pipeline built in {self.boot_s['build_s']} s: {json.dumps(self.build_info, sort_keys=True)}")

        if cfg.warmup:
            self.log(
                f"Warm up: Compiling the served shape {cfg.width}x{cfg.height} x{cfg.num_frames} "
                f"at {cfg.steps} steps"
            )
            t0 = time.time()
            warm = NormalizedRequest(
                mode="t2va",
                prompt=cfg.warmup_prompt,
                width=cfg.width,
                height=cfg.height,
                num_frames=cfg.num_frames,
                steps=cfg.steps,
                seed=0,
            )
            self._generate(warm, Job(req=warm))
            self.boot_s["warmup_s"] = round(time.time() - t0, 1)
            self.log(f"Warm up complete in {self.boot_s['warmup_s']} s")
        self.boot_s["total_s"] = round(time.time() - self.started, 1)

    # ---------------------------------------------------------------- worker
    def _run(self) -> None:
        try:
            self._load()
        except BaseException as exc:  # noqa: BLE001 - any load failure must reach the lifespan
            self.dead = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            self.log(f"ENGINE LOAD FAILED: {self.dead}")
            self._close_mesh()
            return
        self.ready.set()
        self.log(f"engine ready after {self.boot_s.get('total_s')} s")

        while True:
            job = self._q.get()
            if job is None:
                break
            with self._lock:
                self._current = job
            try:
                if job.cancel.is_set() or self.stopping.is_set():
                    self._finish(job, "cancelled")
                    continue
                job.status = "running"
                job.started_at = time.time()
                self._serve(job)
                self._finish(job, "completed")
                self.served += 1
            except Cancelled:
                self.log(f"job {job.id} cancelled at step {job.step}/{job.total_steps}")
                self._finish(job, "cancelled")
            except BaseException as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                self.log(f"job {job.id} failed: {detail}\n{traceback.format_exc()}")
                job.error = detail[:2000]
                self._finish(job, "failed")
                if _is_device_fault(detail):
                    self.dead = detail
                    self.ready.clear()
                    self.log("device fault: engine marked dead (health -> 503)")
                    break
            finally:
                with self._lock:
                    self._current = None
        self.log("worker loop exited")
        self._close_mesh()

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        job.completed_at = time.time()
        job.timings["total_s"] = round(job.completed_at - (job.started_at or job.created_at), 3)
        if status == "completed":
            job.set_phase("completed")
            job.progress = 1.0
        else:
            # `cancelled` / `failed` are terminal statuses, not points on the phase line, so they are
            # assigned rather than pushed through `set_phase` (which only moves along PHASES).
            job.phase = status
        job.done.set()

    # ---------------------------------------------------------------- generation
    def _events(self, job: Job):
        """The pipeline event callback: phases, step progress, per-section timings, and cancel."""
        from ...events import DenoiseStep, SectionEnd, SectionStart

        opened: dict[str, float] = {}

        def on_event(event) -> None:
            if job.cancel.is_set() or self.stopping.is_set():
                raise Cancelled()
            if isinstance(event, SectionStart):
                opened[event.name] = time.time()
                phase = SECTION_PHASE.get(event.name)
                if phase:
                    job.set_phase(phase)
            elif isinstance(event, SectionEnd):
                started = opened.pop(event.name, None)
                key = SECTION_TIMING.get(event.name)
                if started is not None and key:
                    job.add_timing(key, time.time() - started)
            elif isinstance(event, DenoiseStep):
                job.set_step(event.step, event.total)

        return on_event

    def _generate(self, req: NormalizedRequest, job: Job) -> dict | None:
        """One generation. Returns the raw media, or None for the warm-up pass."""
        import numpy as np
        import torch

        import ttnn

        images = {pos: img for pos, img in req.images}
        output = self.pipeline(
            req.prompt,
            image=images.get("first"),
            last_image=images.get("last"),
            height=req.height,
            width=req.width,
            num_frames=req.num_frames,
            num_inference_steps=steps_to_grid_points(req.steps),
            seed=req.seed,
            on_event=self._events(job),
        )
        ttnn.synchronize_device(self.mesh_device)
        if output.video_format == "yuv420":
            raise RuntimeError("the server needs rgb frames; the profile must set vae_output_type uint8/float")
        frames = (output.video[0].permute(1, 2, 3, 0).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
        audio = output.audio[0] if output.audio.ndim == 3 else output.audio
        pcm = np.ascontiguousarray(np.clip(audio.T.numpy(), -1.0, 1.0), dtype=np.float32)
        return {"frames": frames, "pcm": pcm, "sample_rate": int(output.sampling_rate)}

    def _serve(self, job: Job) -> None:
        req = job.req
        self.log(
            f"job {job.id} start: {req.mode} {req.width}x{req.height} x{req.num_frames}, "
            f"{req.steps} steps, seed {req.seed}"
        )
        media = self._generate(req, job)
        if job.cancel.is_set():
            raise Cancelled()

        job.set_phase("muxing")
        t0 = time.time()
        frames, pcm = media["frames"], media["pcm"]
        dest = Path(self.cfg.out_dir) / f"{job.id}.mp4"
        size = write_mp4(frames, pcm, dest, fps=FPS, sample_rate=media["sample_rate"])
        job.add_timing("mux_s", time.time() - t0)
        job.path = str(dest)
        self._prune_outputs()

        job.result = {
            "url": f"/v1/videos/generations/{job.id}/download",
            "width": int(frames.shape[2]),
            "height": int(frames.shape[1]),
            "num_frames": int(frames.shape[0]),
            "fps": FPS,
            "duration_s": round(frames.shape[0] / FPS, 3),
            "sample_rate": media["sample_rate"],
            "audio_channels": int(pcm.shape[1]),
            "bytes": size,
            "frames_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "audio_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(),
        }
        self.log(
            f"job {job.id} done in {job.elapsed_s} s: {job.result['num_frames']} frames, "
            f"{size} bytes, phases {json.dumps(job.timings, sort_keys=True)}"
        )

    def _prune_outputs(self) -> None:
        out = Path(self.cfg.out_dir)
        clips = sorted(out.glob("vid_*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in clips[self.cfg.out_keep :]:
            stale.unlink(missing_ok=True)

    # ---------------------------------------------------------------- queue
    def submit(self, req: NormalizedRequest) -> Job:
        """FIFO enqueue. Raises `queue.Full` -> 429, `RuntimeError` -> 503."""
        if self.stopping.is_set():
            raise RuntimeError("server is shutting down")
        if not self.ready.is_set():
            raise RuntimeError(self.dead or "engine is still loading")
        job = Job(req=req)
        with self._lock:
            if self._pending_locked() >= self.cfg.queue_max:
                raise queue.Full(f"queue is full ({self.cfg.queue_max})")
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._gc_locked()
        self._q.put(job)
        return job

    def _pending_locked(self) -> int:
        return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))

    def _gc_locked(self) -> None:
        finished = [i for i in self._order if self._jobs[i].done.is_set()]
        for job_id in finished[: max(0, len(finished) - self.cfg.job_history)]:
            self._order.remove(job_id)
            self._jobs.pop(job_id, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 32) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order)][:limit]

    def queue_position(self, job: Job) -> int:
        """0 for a running or finished job, else its place in line (1 = next)."""
        if job.status != "queued":
            return 0
        with self._lock:
            waiting = [self._jobs[i] for i in self._order if self._jobs[i].status == "queued"]
        return waiting.index(job) + 1 if job in waiting else 0

    def cancel(self, job: Job) -> Job:
        """Queued -> cancelled at once; running -> cancelled at the next denoise step."""
        job.cancel.set()
        if job.status == "queued":
            # The worker will see the flag when it dequeues; mark it now so the API answers
            # truthfully straight away, and so a queued job that is never dequeued (shutdown) is
            # still reported as cancelled rather than queued forever.
            self._finish(job, "cancelled")
        return job

    @property
    def queue_depth(self) -> int:
        with self._lock:
            return self._pending_locked()

    def health(self) -> dict:
        return {
            "status": "ok" if self.ready.is_set() else ("error" if self.dead else "starting"),
            "ready": self.ready.is_set(),
            "model": self.cfg.model_id,
            "profile": self.cfg.profile,
            "mesh": self.cfg.mesh_shape,
            "queue_depth": self.queue_depth,
            "uptime_s": round(time.time() - self.started, 1),
            "requests_served": self.served,
            "boot_s": dict(self.boot_s),
            "build": dict(self.build_info),
            "error": None if self.dead is None else self.dead.splitlines()[0],
        }


_DEVICE_FAULT = re.compile(
    r"TT_THROW|TT_FATAL|TIMEOUT|device timeout|Fabric Router Sync|" r"Ethernet handshake|watcher", re.IGNORECASE
)


def _is_device_fault(detail: str) -> bool:
    """A failure the mesh does not recover from, as opposed to a bad request that reached the model."""
    return bool(_DEVICE_FAULT.search(detail))
