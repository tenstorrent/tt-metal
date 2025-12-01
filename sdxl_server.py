# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import sys
import uuid
import time
import argparse
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sdxl_config import SDXLConfig
from sdxl_worker import device_worker_process
from utils.logger import setup_logger
from utils.image_utils import pil_to_base64
from utils.cache_utils import validate_cache, log_cache_info


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="SDXL Inference Server")
    parser.add_argument("--dev", action="store_true", help="Enable dev mode (1 worker, fast warmup)")
    parser.add_argument("--workers", type=int, help="Override number of workers")
    parser.add_argument("--steps", type=int, help="Override inference steps for warmup")
    return parser.parse_args()


# Parse arguments and set environment variables before config creation
args = parse_args()
if args.dev:
    os.environ["SDXL_DEV_MODE"] = "true"


# Request/Response models
class ImageRequest(BaseModel):
    """Image generation request model"""

    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = Field(default=50, ge=12, le=100)
    guidance_scale: Optional[float] = Field(default=5.0, ge=1.0, le=20.0)
    guidance_rescale: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    """Image generation response model"""

    images: List[str]  # Base64-encoded images
    inference_time: float


# Global state
config = SDXLConfig()

# Apply CLI overrides
if args.workers is not None:
    config.num_workers = args.workers
if args.steps is not None:
    config.num_inference_steps = args.steps

logger = setup_logger("SDXLServer")
if config.dev_mode:
    logger.info("DEV MODE ENABLED: Single worker, reduced warmup steps")
task_queue = None
result_queue = None
error_queue = None
workers = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage server lifespan - startup and shutdown

    Based on: /home/tt-admin/tt-inference-server/tt-media-server/main.py
    """
    global task_queue, result_queue, error_queue, workers

    logger.info("Starting SDXL server...")

    # Log cache information
    log_cache_info(logger)

    # Validate cache integrity
    is_valid, issues = validate_cache()
    if not is_valid:
        logger.warning("Cache integrity issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("If startup fails, consider running with --clear-cache")
    else:
        logger.info("Cache validation passed")

    # Create queues
    task_queue = mp.Queue(maxsize=config.max_queue_size)
    result_queue = mp.Queue()
    warmup_signal_queue = mp.Queue()
    kernel_ready_queue = mp.Queue()  # Signal for kernel compilation complete (allows overlapped startup)
    error_queue = mp.Queue()

    # Start workers with OVERLAPPED initialization
    # Kernel compilation must be sequential (writes to shared cache)
    # But program cache warmup can overlap (per-device, no shared writes)
    # This reduces startup from ~18-20 min to ~10-12 min for 4 workers
    logger.info(f"Starting {config.num_workers} worker(s) with overlapped initialization...")

    for i in range(config.num_workers):
        logger.info(f"Starting worker {i}...")
        worker = mp.Process(
            target=device_worker_process,
            args=(i, task_queue, result_queue, warmup_signal_queue, kernel_ready_queue, error_queue, config),
            daemon=False,
        )
        worker.start()
        workers.append(worker)

        # Wait for kernel compilation (not full warmup) before starting next worker
        # This allows overlapping program cache warmup between workers
        if i < config.num_workers - 1:  # Don't wait after last worker
            logger.info(f"Waiting for worker {i} kernel compilation (next worker can start after)...")
            timeout = time.time() + 300  # 5 minute timeout for kernel compilation
            kernel_ready_received = False

            while time.time() < timeout:
                try:
                    worker_id = kernel_ready_queue.get(timeout=1)
                    logger.info(f"Worker {worker_id} kernel compilation complete, starting next worker...")
                    kernel_ready_received = True
                    break
                except:
                    # Check if worker crashed
                    if not worker.is_alive():
                        logger.error(f"Worker {i} died during kernel compilation")
                        raise RuntimeError(f"Worker {i} died during kernel compilation")

            if not kernel_ready_received:
                logger.error(f"Worker {i} kernel compilation timeout!")
                raise RuntimeError(f"Worker {i} kernel compilation timeout")

    # Now wait for ALL workers to complete full warmup
    logger.info("All workers started. Waiting for warmup completion...")
    warmups_received = 0
    timeout = time.time() + 600  # 10 minute timeout for remaining warmups

    while warmups_received < config.num_workers and time.time() < timeout:
        try:
            worker_id = warmup_signal_queue.get(timeout=1)
            warmups_received += 1
            logger.info(f"Worker {worker_id} warmup complete ({warmups_received}/{config.num_workers})")
        except:
            # Check if any worker crashed
            for idx, w in enumerate(workers):
                if not w.is_alive():
                    logger.error(f"Worker {idx} died during warmup")
                    raise RuntimeError(f"Worker {idx} died during warmup")

    if warmups_received < config.num_workers:
        logger.error(f"Warmup timeout! Only {warmups_received}/{config.num_workers} workers ready")
        raise RuntimeError(f"Warmup timeout")

    logger.info("All workers ready. Server is accepting requests.")

    yield

    # Shutdown
    logger.info("Shutting down workers...")
    for _ in range(config.num_workers):
        task_queue.put(None)  # Shutdown signal

    for worker in workers:
        worker.join(timeout=10)
        if worker.is_alive():
            worker.terminate()

    logger.info("Server shutdown complete")


# Create app
app = FastAPI(title="SDXL Inference Server", lifespan=lifespan)


@app.post("/image/generations", response_model=ImageResponse)
async def generate_image(request: ImageRequest):
    """
    Generate image from text prompt

    Args:
        request: Image generation request

    Returns:
        ImageResponse with base64-encoded images
    """
    task_id = str(uuid.uuid4())

    # Put task in queue
    try:
        task_queue.put((task_id, request.dict()), timeout=5)
    except:
        raise HTTPException(status_code=503, detail="Task queue is full")

    # Poll for result
    start_time = time.time()
    timeout = start_time + config.inference_timeout_seconds

    while time.time() < timeout:
        try:
            result = result_queue.get(timeout=1)

            if result["task_id"] == task_id:
                # Encode images to base64
                base64_images = [pil_to_base64(img) for img in result["images"]]

                return ImageResponse(images=base64_images, inference_time=result["inference_time"])
        except:
            # Check error queue
            try:
                error = error_queue.get_nowait()
                logger.error(f"Worker error: {error}")
            except:
                pass

    raise HTTPException(status_code=408, detail="Request timeout")


@app.get("/health")
async def health_check():
    """Server health check endpoint"""
    alive_workers = sum(1 for w in workers if w.is_alive())
    return {
        "status": "healthy" if alive_workers > 0 else "unhealthy",
        "workers_alive": alive_workers,
        "workers_total": config.num_workers,
        "queue_size": task_queue.qsize() if task_queue else 0,
    }


@app.get("/metrics")
async def get_metrics():
    """Server metrics endpoint"""
    return {
        "task_queue_size": task_queue.qsize() if task_queue else 0,
        "result_queue_size": result_queue.qsize() if result_queue else 0,
        "error_queue_size": error_queue.qsize() if error_queue else 0,
        "workers_alive": sum(1 for w in workers if w.is_alive()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.server_host, port=config.server_port, log_level="info")
