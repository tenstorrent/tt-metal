# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import uuid
import time
import queue
import argparse
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from utils.logger import setup_logger
from utils.image_utils import pil_to_base64
from utils.video_utils import frames_to_mp4, mp4_to_base64
from utils.cache_utils import validate_cache, log_cache_info


def parse_args():
    """Parse command-line arguments for the unified inference server"""
    parser = argparse.ArgumentParser(
        description="Unified Image/Video Generation Inference Server (SDXL / SD3.5 Large / WAN 2.2)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sdxl",
        choices=["sdxl", "sd35", "wan"],
        help=(
            "Model to serve: 'sdxl' (T3K, 4 workers), 'sd35' (LoudBox, 1 worker × 2x4 mesh), "
            "or 'wan' (LoudBox, 1 worker × 2x4 mesh, text-to-video)"
        ),
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode (SDXL: 1 worker; SD35: fewer steps, no trace; WAN: fewer steps)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip warmup inference for WAN (useful for one-off runs where 2nd-inference speedup doesn't matter)",
    )
    parser.add_argument("--workers", type=int, help="Override number of workers")
    parser.add_argument("--steps", type=int, help="Override number of inference steps")
    parser.add_argument("--guidance", type=float, help="Override guidance scale")
    parser.add_argument("--port", type=int, help="Override server port")
    parser.add_argument("--host", type=str, help="Override server host")
    return parser.parse_args()


# Parse arguments and propagate dev-mode environment variables before config
# objects are created (config reads env vars at instantiation time).
args = parse_args()
if args.dev:
    if args.model == "sdxl":
        os.environ["SDXL_DEV_MODE"] = "true"
    elif args.model == "sd35":
        os.environ["SD35_DEV_MODE"] = "true"
    elif args.model == "wan":
        os.environ["WAN_DEV_MODE"] = "true"
if args.no_warmup:
    if args.model == "wan":
        os.environ["WAN_SKIP_WARMUP"] = "true"

# Build the appropriate config
if args.model == "sd35":
    from sd35_config import SD35Config

    config = SD35Config()
    model_label = "SD3.5 Large"
elif args.model == "wan":
    from wan_config import WANConfig

    config = WANConfig()
    model_label = "WAN 2.2"
else:
    from sdxl_config import SDXLConfig

    config = SDXLConfig()
    model_label = "SDXL"

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
    """Video generation request (WAN 2.2 T2V)"""

    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = Field(default=None, ge=1, le=200)
    seed: Optional[int] = None


class VideoResponse(BaseModel):
    """Video generation response"""

    video: str  # Base64-encoded MP4
    inference_time: float
    model: str  # Model label for client-side detection
    num_frames: int
    fps: int


# ---------------------------------------------------------------------------
# Global server state
# ---------------------------------------------------------------------------

task_queue = None
result_queue = None
error_queue = None
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
    global task_queue, result_queue, error_queue, workers

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


@app.post("/image/generations", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    """Generate an image from a text prompt.

    Returns a base64-encoded JPEG along with inference time and the model label.
    Not available when the server is running the WAN 2.2 video model.
    """
    if args.model == "wan":
        raise HTTPException(
            status_code=400,
            detail="Image generation is not available for the WAN 2.2 model. Use POST /video/generations instead.",
        )

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
        except queue.Empty:
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
            except queue.Empty:
                pass
            continue

        if result["task_id"] == task_id:
            try:
                base64_images = [pil_to_base64(img) for img in result["images"]]
                return ImageResponse(
                    images=base64_images,
                    inference_time=result["inference_time"],
                    model=model_label,
                )
            except Exception as exc:
                logger.error(f"Image encoding failed: {exc}")
                raise HTTPException(status_code=500, detail=f"Image encoding failed: {exc}")
        else:
            # Result belongs to a different task — put it back
            result_queue.put(result)

    raise HTTPException(status_code=408, detail="Request timeout")


@app.post("/video/generations", response_model=VideoResponse)
async def generate_video(request: VideoRequest):
    """Generate a video from a text prompt (WAN 2.2 T2V only).

    Returns a base64-encoded MP4 along with inference time, model label,
    frame count, and fps.
    """
    if args.model != "wan":
        raise HTTPException(
            status_code=400,
            detail=f"Video generation is only available for the WAN 2.2 model. Current model: {model_label}.",
        )

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
        except queue.Empty:
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
            except queue.Empty:
                pass
            continue

        if result["task_id"] == task_id:
            try:
                video_path = frames_to_mp4(result["images"], fps=config.fps)
                try:
                    encoded_video = mp4_to_base64(video_path)
                finally:
                    try:
                        os.remove(video_path)
                    except Exception:
                        pass
                return VideoResponse(
                    video=encoded_video,
                    inference_time=result["inference_time"],
                    model=model_label,
                    num_frames=config.num_frames,
                    fps=config.fps,
                )
            except Exception as exc:
                logger.error(f"Video encoding failed: {exc}")
                raise HTTPException(status_code=500, detail=f"Video encoding failed: {exc}")
        else:
            # Result belongs to a different task — put it back
            result_queue.put(result)

    raise HTTPException(status_code=408, detail="Request timeout")


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
