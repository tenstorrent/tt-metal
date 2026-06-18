# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import uuid
import time
import json
import queue
import argparse
import asyncio
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import io
import base64

from utils.logger import setup_logger
from utils.image_utils import pil_to_base64, ndarray_to_b64npy, b64npy_to_ndarray
from utils.cache_utils import validate_cache, log_cache_info

from device_specs import DeviceClass, validate_model_board


def parse_args():
    """Parse command-line arguments for the unified inference server"""
    parser = argparse.ArgumentParser(description="Unified Inference Server (SDXL / SD3.5 Large / Wan2.2 T2V)")
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=["sdxl", "sd35", "wan22"],
        help="Model to serve: 'sdxl', 'sd35', or 'wan22'",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        help="Tenstorrent board / cluster (e.g. t3k, p150, p300, p300x2, p150x4, p150x8). "
        "Required for --model sdxl and --model wan22. Ignored for --model sd35.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (SDXL: 1 worker; SD35/WAN: fewer steps, no trace)",
    )
    parser.add_argument("--workers", type=int, help="Override number of workers")
    parser.add_argument("--steps", type=int, help="Override number of inference steps")
    parser.add_argument("--guidance", type=float, help="Override guidance scale")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--host", type=str, help="Override server host")
    args = parser.parse_args()

    # --board is required for SDXL and Wan2.2; SD3.5 keeps its current flag-less invocation.
    if args.model in ("sdxl", "wan22"):
        if args.board is None:
            parser.error(f"--board is required when --model {args.model} (e.g. --board p300x2)")
        try:
            args.board = DeviceClass.from_string(args.board)
        except ValueError as e:
            parser.error(str(e))
        try:
            validate_model_board(args.model, args.board)
        except ValueError as e:
            parser.error(str(e))
    else:
        args.board = None

    return args


# Parse arguments and propagate dev-mode environment variables before config
# objects are created (config reads env vars at instantiation time).
args = parse_args()
if args.dev:
    if args.model == "sdxl":
        os.environ["SDXL_DEV_MODE"] = "true"
    elif args.model == "sd35":
        os.environ["SD35_DEV_MODE"] = "true"
    elif args.model == "wan22":
        os.environ["WAN_DEV_MODE"] = "true"

# Build the appropriate config
if args.model == "sd35":
    from sd35_config import SD35Config

    config = SD35Config()
    model_label = "SD3.5 Large"
    model_kind = "image"
elif args.model == "wan22":
    from wan_config import WanConfig

    config = WanConfig(board=args.board)
    model_label = "Wan2.2 T2V"
    model_kind = "video"
else:
    from sdxl_config import SDXLConfig

    config = SDXLConfig(board=args.board)
    model_label = "SDXL"
    model_kind = "image"

# Apply CLI overrides after config is constructed
if args.workers is not None:
    config.num_workers = args.workers
if args.steps is not None:
    config.num_inference_steps = args.steps
if args.guidance is not None:
    config.guidance_scale = args.guidance
if args.port is not None:
    config.server_port = args.port
if args.host is not None:
    config.server_host = args.host

logger = setup_logger("Server")
if config.dev_mode:
    logger.info(f"DEV MODE ENABLED ({model_label})")

# ---------------------------------------------------------------------------
# Pydantic models — superset of SDXL + SD35 fields
# ---------------------------------------------------------------------------


class ImageRequest(BaseModel):
    """Image generation request.

    All fields are accepted for both models. SD3.5 ignores prompt_2,
    negative_prompt_2, and guidance_rescale. SDXL ignores no SD35-only fields
    (there are none — the field sets are merged).
    """

    prompt: str
    negative_prompt: Optional[str] = ""

    # SDXL dual text-encoder fields (ignored by SD35)
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)

    # Shared generation parameters
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    """Image generation response"""

    images: List[str]  # Base64-encoded JPEG images
    inference_time: float
    model: str  # Model label for client-side detection ("SDXL" or "SD3.5 Large")


class VideoRequest(BaseModel):
    """Wan2.2 T2V request."""

    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    num_frames: Optional[int] = Field(default=None, ge=5, le=257)
    height: Optional[int] = Field(default=None, ge=64, le=2048)
    width: Optional[int] = Field(default=None, ge=64, le=2048)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    guidance_scale_2: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    flow_shift: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    boundary_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seed: Optional[int] = None
    # LoRA passthrough — Wan2.2 has two experts, so adapters are specified per
    # expert: high_lora_path -> high-noise (transformer), low_lora_path -> low-noise
    # (transformer_2). Either or both may be set; lora_scale applies to both.
    high_lora_path: Optional[str] = None
    low_lora_path: Optional[str] = None
    lora_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class VideoResponse(BaseModel):
    """Wan2.2 T2V response — frames as base64-encoded PNGs (T frames)."""

    frames: List[str]  # base64 PNG per frame
    width: int
    height: int
    num_frames: int
    inference_time: float
    model: str


# ---------------------------------------------------------------------------
# Staged-op request/response models (additive — used by the ComfyUI HTTP nodes)
#
# Tensors cross as base64 NumPy .npy payloads: {"b64": str, "shape": [...], "dtype": str}.
# These endpoints currently target the SDXL runner only.
# ---------------------------------------------------------------------------


class DenoiseRequest(BaseModel):
    """Staged denoise (KSampler) request — returns latents, not images."""

    prompt: str
    negative_prompt: Optional[str] = ""
    prompt_2: Optional[str] = None
    negative_prompt_2: Optional[str] = None
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=30.0)
    guidance_rescale: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seed: Optional[int] = None
    # LoRA passthrough (mirrors tt-media-server image request fields)
    lora_path: Optional[str] = None
    lora_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class TensorResponse(BaseModel):
    """A single tensor payload plus timing/model metadata."""

    latent: Optional[Dict[str, Any]] = None
    image: Optional[Dict[str, Any]] = None
    inference_time: float
    model: str
    # Optional LoRA application status (SDXL denoise). None when no adapter was
    # requested. When present, reports whether the adapter was applied and why it
    # may have been skipped/partially applied.
    lora: Optional[Dict[str, Any]] = None


class VaeDecodeRequest(BaseModel):
    latent: Dict[str, Any]  # base64 .npy payload, [B, C, H, W]


class VaeEncodeRequest(BaseModel):
    image: Dict[str, Any]  # base64 .npy payload, [B, H, W, C]


# Video-staged models (wan22): mirror the SDXL staged contract but carry the wan
# generation knobs. The split maps to WanRunner.denoise / WanRunner.vae_decode.


class VideoDenoiseRequest(BaseModel):
    """Staged wan22 denoise (Wan Sampler) request — returns latents, not frames."""

    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    guidance_scale: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    guidance_scale_2: Optional[float] = Field(default=None, ge=1.0, le=20.0)
    flow_shift: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    boundary_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seed: Optional[int] = None
    # LoRA passthrough — per-expert adapters (see VideoRequest for semantics).
    high_lora_path: Optional[str] = None
    low_lora_path: Optional[str] = None
    lora_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)


class VideoVaeDecodeRequest(BaseModel):
    latent: Dict[str, Any]  # base64 .npy payload, [B, z_dim, F, H, W]


# ---------------------------------------------------------------------------
# Global server state
# ---------------------------------------------------------------------------

task_queue = None
result_queue = None
error_queue = None
progress_queue = None
workers = []


# ---------------------------------------------------------------------------
# Lifespan: start workers, wait for warmup, yield, shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifespan — start workers, wait for warmup, then shutdown cleanly.

    Overlapped initialization strategy (SDXL):
      - Start worker 0, wait for kernel compilation signal, then start worker 1, etc.
      - Kernel compilation must be sequential (shared write to program cache).
      - Program-cache warmup can overlap between workers (per-device, no shared writes).
      - Reduces T3K startup from ~18-20 min → ~10-12 min for 4 workers.

    For SD35 (num_workers=1):
      - The kernel_ready loop (range(0)) is a no-op — no interleaving required.
      - The warmup wait still applies to the single worker.
    """
    global task_queue, result_queue, error_queue, progress_queue, workers

    logger.info(f"Starting {model_label} server...")

    # Log and validate HuggingFace / model cache state
    log_cache_info(logger)
    is_valid, issues = validate_cache()
    if not is_valid:
        logger.warning("Cache integrity issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("If startup fails, consider running with --clear-cache")
    else:
        logger.info("Cache validation passed")

    # Deferred import: must not initialize ttnn in the main process
    from worker import device_worker_process

    # Create inter-process queues
    task_queue = mp.Queue(maxsize=config.max_queue_size)
    result_queue = mp.Queue()
    warmup_signal_queue = mp.Queue()
    kernel_ready_queue = mp.Queue()
    error_queue = mp.Queue()
    # Carries per-task progress events (DenoiseStep/Section) from workers to the
    # streaming endpoints. Unbounded so a slow HTTP consumer never blocks a worker.
    progress_queue = mp.Queue()

    logger.info(f"Starting {config.num_workers} worker(s) with overlapped initialization...")

    for i in range(config.num_workers):
        logger.info(f"Starting worker {i}...")
        worker = mp.Process(
            target=device_worker_process,
            args=(
                i,
                task_queue,
                result_queue,
                warmup_signal_queue,
                kernel_ready_queue,
                error_queue,
                config,
                progress_queue,
            ),
            daemon=False,
        )
        worker.start()
        workers.append(worker)

        # Wait for kernel compilation before launching the next worker.
        # For SD35 (num_workers=1) this loop body never executes (range(0)).
        if i < config.num_workers - 1:
            logger.info(f"Waiting for worker {i} kernel compilation before starting next worker...")
            timeout = time.time() + 300  # 5-minute per-worker kernel compilation limit
            kernel_ready_received = False

            while time.time() < timeout:
                try:
                    worker_id = kernel_ready_queue.get(timeout=1)
                    logger.info(f"Worker {worker_id} kernel compilation complete, starting next worker...")
                    kernel_ready_received = True
                    break
                except Exception:
                    if not worker.is_alive():
                        logger.error(f"Worker {i} died during kernel compilation")
                        raise RuntimeError(f"Worker {i} died during kernel compilation")

            if not kernel_ready_received:
                logger.error(f"Worker {i} kernel compilation timeout!")
                raise RuntimeError(f"Worker {i} kernel compilation timeout")

    # Wait for ALL workers to complete full warmup (kernel compile + trace capture)
    logger.info("All workers started. Waiting for warmup completion...")
    warmups_received = 0
    # 1200s = 20 min — SD35 trace capture can take ~10-15 min on cold start
    timeout = time.time() + 1200

    while warmups_received < config.num_workers and time.time() < timeout:
        try:
            worker_id = warmup_signal_queue.get(timeout=1)
            warmups_received += 1
            logger.info(f"Worker {worker_id} warmup complete ({warmups_received}/{config.num_workers})")
        except Exception:
            # Check for crashed workers
            for idx, w in enumerate(workers):
                if not w.is_alive():
                    logger.error(f"Worker {idx} died during warmup")
                    raise RuntimeError(f"Worker {idx} died during warmup")

    if warmups_received < config.num_workers:
        logger.error(f"Warmup timeout! Only {warmups_received}/{config.num_workers} workers ready")
        raise RuntimeError("Warmup timeout")

    logger.info(f"All workers ready. {model_label} server is accepting requests.")

    yield

    # Graceful shutdown: send None sentinel to each worker
    logger.info("Shutting down workers...")
    for _ in range(config.num_workers):
        task_queue.put(None)

    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()

    logger.info("Server shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title=f"{model_label} Inference Server", lifespan=lifespan)


def _frames_to_base64_pngs(frames):
    """Convert numpy uint8 frames (T,H,W,3) to a list of base64-encoded PNGs."""
    from PIL import Image

    out = []
    for i in range(frames.shape[0]):
        img = Image.fromarray(frames[i], mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


@app.post("/video/generations", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """Generate a video from a text prompt (Wan2.2)."""
    if model_kind != "video":
        raise HTTPException(status_code=400, detail=f"This server is serving '{model_label}', not a video model")

    task_id = str(uuid.uuid4())
    try:
        task_queue.put((task_id, request.dict()), timeout=5)
    except Exception:
        raise HTTPException(status_code=503, detail="Task queue is full")

    start_time = time.time()
    timeout = start_time + config.inference_timeout_seconds

    while time.time() < timeout:
        try:
            result = result_queue.get(timeout=1)
            if result["task_id"] == task_id:
                frames_np = result["images"][0]  # (T,H,W,3) uint8
                T, H, W, _ = frames_np.shape
                frames_b64 = _frames_to_base64_pngs(frames_np)
                return VideoResponse(
                    frames=frames_b64,
                    width=W,
                    height=H,
                    num_frames=T,
                    inference_time=result["inference_time"],
                    model=model_label,
                )
            else:
                result_queue.put(result)
        except Exception:
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
            except Exception:
                pass

    raise HTTPException(status_code=408, detail="Request timeout")


@app.post("/image/generations", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    """Generate an image from a text prompt.

    Returns a base64-encoded JPEG along with inference time and the model label.
    """
    if model_kind != "image":
        raise HTTPException(status_code=400, detail=f"This server is serving '{model_label}', use /video/generations")

    task_id = str(uuid.uuid4())

    try:
        task_queue.put((task_id, request.dict()), timeout=5)
    except Exception:
        raise HTTPException(status_code=503, detail="Task queue is full")

    # Poll result queue until our task_id comes back or we time out
    start_time = time.time()
    timeout = start_time + config.inference_timeout_seconds

    while time.time() < timeout:
        try:
            result = result_queue.get(timeout=1)

            if result["task_id"] == task_id:
                base64_images = [pil_to_base64(img) for img in result["images"]]
                return ImageResponse(
                    images=base64_images,
                    inference_time=result["inference_time"],
                    model=model_label,
                )
            else:
                # Result belongs to a different task — put it back
                result_queue.put(result)

        except Exception:
            # Check error queue on timeout
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
            except Exception:
                pass

    raise HTTPException(status_code=408, detail="Request timeout")


def _submit_and_wait(request_dict: dict, timeout_seconds: Optional[float] = None) -> dict:
    """Enqueue a task and poll result_queue for its matching result.

    Shared by the staged endpoints. Mirrors the polling/error-handling pattern of
    the image/video handlers. Returns the worker's result dict.
    """
    task_id = str(uuid.uuid4())
    request_dict["op"] = request_dict.get("op", "generate")
    try:
        task_queue.put((task_id, request_dict), timeout=5)
    except Exception:
        raise HTTPException(status_code=503, detail="Task queue is full")

    deadline = time.time() + (timeout_seconds or config.inference_timeout_seconds)
    while time.time() < deadline:
        try:
            result = result_queue.get(timeout=1)
            if result.get("task_id") == task_id:
                return result
            else:
                result_queue.put(result)  # belongs to another task
        except Exception:
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
                raise HTTPException(status_code=500, detail=f"Worker error: {error.get('error', error)}")
            except HTTPException:
                raise
            except Exception:
                pass
    raise HTTPException(status_code=408, detail="Request timeout")


def _stream_denoise_response(req: dict, *, timeout_seconds: float = None):
    """Enqueue a denoise task and stream progress events as NDJSON.

    Shared by /video/denoise_stream and /latent/denoise_stream. Sets
    stream_progress=True so the worker publishes DenoiseStep/Section events to
    progress_queue; the blocking endpoints leave it unset and never drain it.

    Yields one JSON object per line:
      - progress events: {"type": "section_start"|"section_end"|"denoise_step", ...}
      - terminal event:  {"type": "result", "latent": <b64npy>, "inference_time": float}
      - on failure:      {"type": "error", "detail": str}
    """
    task_id = str(uuid.uuid4())
    req["op"] = "denoise"
    req["stream_progress"] = True
    try:
        task_queue.put((task_id, req), timeout=5)
    except Exception:
        raise HTTPException(status_code=503, detail="Task queue is full")

    deadline = time.time() + max(timeout_seconds or config.inference_timeout_seconds, 3600.0)

    async def event_stream():
        while time.time() < deadline:
            drained_any = False

            # 1) Forward any progress events for this task.
            while True:
                try:
                    ev = progress_queue.get_nowait()
                except queue.Empty:
                    break
                drained_any = True
                if ev.get("task_id") == task_id:
                    yield json.dumps(ev) + "\n"

            # 2) Check for the terminal result.
            try:
                result = result_queue.get_nowait()
            except queue.Empty:
                result = None

            if result is not None:
                if result.get("task_id") == task_id:
                    yield json.dumps(
                        {
                            "type": "result",
                            "latent": ndarray_to_b64npy(result["tensor"]),
                            "inference_time": result.get("inference_time", 0.0),
                            "model": model_label,
                            "lora": result.get("lora"),
                        }
                    ) + "\n"
                    return
                else:
                    # Belongs to another task — put it back for its handler.
                    result_queue.put(result)

            # 3) Surface worker errors.
            try:
                err = error_queue.get_nowait()
                logger.error(f"Worker error: {err}")
                yield json.dumps({"type": "error", "detail": f"Worker error: {err.get('error', err)}"}) + "\n"
                return
            except queue.Empty:
                pass

            if not drained_any:
                # Nothing pending; yield control so we don't busy-spin the event loop.
                await asyncio.sleep(0.05)

        yield json.dumps({"type": "error", "detail": "Request timeout"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


def _require_staged_support():
    """Staged ops are currently implemented for the SDXL runner only."""
    if args.model != "sdxl":
        raise HTTPException(
            status_code=400,
            detail=f"Staged ops (denoise/vae) are only supported for --model sdxl, not '{args.model}'. "
            f"Use /image/generations for the full pipeline.",
        )


@app.post("/latent/denoise", response_model=TensorResponse)
async def latent_denoise(request: DenoiseRequest):
    """Staged denoise (KSampler): run encode + UNet loop, return latents."""
    _require_staged_support()
    req = request.dict()
    req["op"] = "denoise"
    result = _submit_and_wait(req)
    return TensorResponse(
        latent=ndarray_to_b64npy(result["tensor"]),
        inference_time=result.get("inference_time", 0.0),
        model=model_label,
        lora=result.get("lora"),
    )


@app.post("/latent/denoise_stream")
async def latent_denoise_stream(request: DenoiseRequest):
    """Staged SDXL denoise (KSampler) with streaming per-step progress (NDJSON).

    Mirrors /video/denoise_stream. The blocking /latent/denoise endpoint is
    preserved for older clients.
    """
    _require_staged_support()
    return _stream_denoise_response(request.dict())


@app.post("/vae/decode", response_model=TensorResponse)
async def vae_decode(request: VaeDecodeRequest):
    """Staged VAE decode: latents [B, C, H, W] -> image [B, H, W, C] in [0, 1]."""
    _require_staged_support()
    latents = b64npy_to_ndarray(request.latent)
    result = _submit_and_wait({"op": "vae_decode", "latent": latents})
    return TensorResponse(
        image=ndarray_to_b64npy(result["tensor"]),
        inference_time=result.get("inference_time", 0.0),
        model=model_label,
    )


@app.post("/vae/encode", response_model=TensorResponse)
async def vae_encode(request: VaeEncodeRequest):
    """Staged VAE encode: image [B, H, W, C] in [0, 1] -> latents [B, C, H, W]."""
    _require_staged_support()
    images = b64npy_to_ndarray(request.image)
    result = _submit_and_wait({"op": "vae_encode", "image": images})
    return TensorResponse(
        latent=ndarray_to_b64npy(result["tensor"]),
        inference_time=result.get("inference_time", 0.0),
        model=model_label,
    )


def _require_video_staged_support():
    """Staged video ops (denoise/vae) are served only by a video model (wan22)."""
    if model_kind != "video":
        raise HTTPException(
            status_code=400,
            detail=f"Staged video ops (denoise/vae) require a video model, not '{model_label}'.",
        )


@app.post("/video/denoise", response_model=TensorResponse)
async def video_denoise(request: VideoDenoiseRequest):
    """Staged wan22 denoise (Wan Sampler): encode + denoise loop, return latents."""
    _require_video_staged_support()
    req = request.dict()
    req["op"] = "denoise"
    # Video denoise runs the full sampling loop — allow the long video timeout.
    result = _submit_and_wait(req, timeout_seconds=max(config.inference_timeout_seconds, 3600.0))
    return TensorResponse(
        latent=ndarray_to_b64npy(result["tensor"]),
        inference_time=result.get("inference_time", 0.0),
        model=model_label,
    )


@app.post("/video/denoise_stream")
async def video_denoise_stream(request: VideoDenoiseRequest):
    """Staged wan22 denoise with streaming progress (NDJSON).

    Yields one JSON object per line:
      - progress events: {"type": "section_start"|"section_end"|"denoise_step", ...}
      - terminal event:  {"type": "result", "latent": <b64npy>, "inference_time": float}
      - on failure:      {"type": "error", "detail": str}

    The blocking /video/denoise endpoint is preserved for older clients.
    """
    _require_video_staged_support()
    return _stream_denoise_response(request.dict())


@app.post("/video/vae_decode", response_model=TensorResponse)
async def video_vae_decode(request: VideoVaeDecodeRequest):
    """Staged wan22 VAE decode: latents [B, z_dim, F, H, W] -> frames [T, H, W, C] in [0, 1]."""
    _require_video_staged_support()
    latents = b64npy_to_ndarray(request.latent)
    result = _submit_and_wait({"op": "vae_decode", "latent": latents})
    return TensorResponse(
        image=ndarray_to_b64npy(result["tensor"]),
        inference_time=result.get("inference_time", 0.0),
        model=model_label,
    )


@app.get("/health")
async def health_check():
    """Server health check — includes model label for client-side detection"""
    alive_workers = sum(1 for w in workers if w.is_alive())
    return {
        "status": "healthy" if alive_workers > 0 else "unhealthy",
        "model": model_label,
        "workers_alive": alive_workers,
        "workers_total": config.num_workers,
        "queue_size": task_queue.qsize() if task_queue else 0,
    }


@app.get("/metrics")
async def get_metrics():
    """Server metrics endpoint"""
    return {
        "model": model_label,
        "task_queue_size": task_queue.qsize() if task_queue else 0,
        "result_queue_size": result_queue.qsize() if result_queue else 0,
        "error_queue_size": error_queue.qsize() if error_queue else 0,
        "workers_alive": sum(1 for w in workers if w.is_alive()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.server_host, port=config.server_port, log_level="info")
