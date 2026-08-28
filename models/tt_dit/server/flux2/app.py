# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone uvicorn + FastAPI server for FLUX.2 text-to-image generation.

One process owns one mesh device and one warm pipeline. Generation is serialized on a
single executor thread (both a ttnn requirement and the reason concurrent requests
queue rather than collide), while the event loop stays responsive for health checks,
status polls and downloads.

Run it with ``run_server.sh``, or directly::

    python -m models.tt_dit.server.flux2.app
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger

from .config import ServerConfig
from .jobs import JobLimitError, JobStore
from .pipeline_holder import PipelineHolder
from .schemas import GenerateRequest, HealthResponse, JobResponse

cfg = ServerConfig.from_env()
holder = PipelineHolder(cfg)
store = JobStore(cfg, holder)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("startup: warming FLUX.2 pipeline (this can take several minutes)")
    await store.start(holder)
    logger.info("startup: ready")
    yield
    logger.info("shutdown: draining and releasing the device")
    await store.shutdown()


app = FastAPI(
    title="FLUX.2 on Tenstorrent",
    description="Text-to-image generation with FLUX.2 served on Tenstorrent hardware.",
    lifespan=lifespan,
)


def _resolve_params(req: GenerateRequest) -> dict:
    """Fill unset fields from the server config and reject shape changes.

    The pipeline is built and traced for one resolution, so honouring a different one
    per request would silently rebuild it. Rejecting is the honest response.
    """
    for name, requested in (("height", req.height), ("width", req.width)):
        configured = getattr(cfg, name)
        if requested is not None and requested != configured:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{name}={requested} does not match the served resolution "
                    f"({configured}). This server warms one shape; start another with "
                    f"FLUX2_{name.upper()} to serve a different one."
                ),
            )

    return {
        "prompt": req.prompt,
        "num_inference_steps": req.num_inference_steps or cfg.default_steps,
        "guidance_scale": req.guidance_scale if req.guidance_scale is not None else cfg.default_guidance,
        "seed": req.seed,
    }


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ready = holder.is_ready()
    return HealthResponse(
        status="ready" if ready else "initializing",
        ready=ready,
        model=cfg.checkpoint,
        mesh_shape=list(cfg.mesh_shape),
        height=cfg.height,
        width=cfg.width,
    )


@app.post("/generate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
async def generate(req: GenerateRequest) -> JobResponse:
    if not holder.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pipeline is still initializing; poll /health",
        )
    try:
        job = store.create(_resolve_params(req))
    except JobLimitError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
    return JobResponse(**job.to_public_dict())


@app.get("/jobs")
async def list_jobs() -> dict:
    return {"jobs": store.list()}


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job {job_id}")
    return JobResponse(**job.to_public_dict())


@app.get("/jobs/{job_id}/image")
async def download(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job {job_id}")
    if job.result_path is None or not os.path.exists(job.result_path):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"job {job_id} is {job.status.value}; no image to download",
        )
    return FileResponse(job.result_path, media_type="image/png", filename=f"{job_id}.png")


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel(job_id: str) -> JobResponse:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"unknown job {job_id}")
    ok, reason = store.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=reason)
    return JobResponse(**job.to_public_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=cfg.host, port=cfg.port)
