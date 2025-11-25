# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import uuid
import time
import multiprocessing as mp
from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sdxl_config import SDXLConfig
from sdxl_worker import device_worker_process
from utils.logger import setup_logger
from utils.image_utils import pil_to_base64


# Request/Response models
class ImageRequest(BaseModel):
    """Image generation request model"""

    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: Optional[int] = Field(default=50, ge=12, le=100)
    guidance_scale: Optional[float] = Field(default=5.0, ge=1.0, le=20.0)
    seed: Optional[int] = None


class ImageResponse(BaseModel):
    """Image generation response model"""

    images: List[str]  # Base64-encoded images
    inference_time: float


# Global state
config = SDXLConfig()
logger = setup_logger("SDXLServer")
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

    # Create queues
    task_queue = mp.Queue(maxsize=config.max_queue_size)
    result_queue = mp.Queue()
    warmup_signal_queue = mp.Queue()
    error_queue = mp.Queue()

    # Start workers ONE AT A TIME to avoid JIT cache corruption
    # All workers compile to the same cache directory, so concurrent compilation causes errors
    logger.info(f"Starting {config.num_workers} worker(s) sequentially...")

    for i in range(config.num_workers):
        logger.info(f"Starting worker {i}...")
        worker = mp.Process(
            target=device_worker_process,
            args=(i, task_queue, result_queue, warmup_signal_queue, error_queue, config),
            daemon=False,
        )
        worker.start()
        workers.append(worker)

        # Wait for THIS worker to complete warmup before starting next
        logger.info(f"Waiting for worker {i} to complete warmup (this may take 5-10 minutes for first worker)...")
        timeout = time.time() + 600  # 10 minute timeout per worker
        warmup_received = False

        while time.time() < timeout:
            try:
                worker_id = warmup_signal_queue.get(timeout=1)
                logger.info(f"Worker {worker_id} warmup complete ({i + 1}/{config.num_workers})")
                warmup_received = True
                break
            except:
                # Check if worker crashed
                if not worker.is_alive():
                    logger.error(f"Worker {i} died during warmup")
                    raise RuntimeError(f"Worker {i} died during warmup")

        if not warmup_received:
            logger.error(f"Worker {i} warmup timeout!")
            raise RuntimeError(f"Worker {i} warmup timeout")

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
