# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FastAPI app for MiniMax-H3 on Tenstorrent, served by tt-model's ``tt-dit-server`` kind:

    python -m uvicorn --host 0.0.0.0 --port <port> --lifespan on \
        models.tt_dit.pipelines.minimax_h3.server.app:app

The pipeline is loaded AND warmed inside the lifespan, so uvicorn's "Application startup complete"
means "ready to generate" -- that line is what `tt-model serve` waits for, and a server that printed
it while still compiling would hand the first real request a multi-minute timeout. A load failure
raises out of the lifespan, so the container exits non-zero rather than serving 503s.

Routes are the contract in docs/API.md: the `/v1/videos/generations` job API, plus the aliases
(`/generate`, `/jobs/{id}`, `/result/{id}`, `/cancel/{id}`, `/healthz`, `/capabilities`).
"""

from __future__ import annotations

import logging
import os
import queue
import sys
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .config import build_config
from .engine import GenerationRequest, H3Engine, ImagePrompt, Job, normalize

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout, force=True
)
log = logging.getLogger("minimax_h3.server")

CONFIG = build_config()
ENGINE = H3Engine(CONFIG, log_fn=log.info)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    log.info(
        f"MiniMax-H3 server starting: profile {CONFIG.profile}, mesh {CONFIG.mesh_shape}, "
        f"queue_max {CONFIG.queue_max}, out_dir {CONFIG.out_dir}"
    )
    os.makedirs(CONFIG.out_dir, exist_ok=True)
    ENGINE.start()
    await run_in_threadpool(ENGINE.wait_ready)
    try:
        yield
    finally:
        log.info("shutting down: cancelling work and closing the mesh")
        t0 = time.time()
        await run_in_threadpool(ENGINE.shutdown)
        log.info(f"shutdown complete in {time.time() - t0:.1f} s")


app = FastAPI(title="MiniMax-H3 on Tenstorrent", version="1.0.0", lifespan=lifespan)


async def auth(request: Request) -> None:
    """Bearer auth when `H3_API_KEY` is set.

    Applied to every generating route INCLUDING the aliases. The contract only names `/v1/*`, but an
    unauthenticated `/generate` would be an authentication bypass for the same work, so the aliases
    are covered too; the two health routes stay open so a probe needs no credentials.
    """
    if CONFIG.api_key and request.headers.get("authorization") != f"Bearer {CONFIG.api_key}":
        raise HTTPException(401, "invalid or missing API key")


PROTECTED = [Depends(auth)]


def _job_view(job: Job) -> dict:
    return job.to_dict(queue_position=ENGINE.queue_position(job))


def _require(job_id: str) -> Job:
    job = ENGINE.get(job_id)
    if job is None:
        raise HTTPException(404, f"no job {job_id}")
    return job


def _submit(body: GenerationRequest) -> Job:
    try:
        req = normalize(body, CONFIG)
    except ValueError as exc:
        # A profile-dependent rule (canvas limits, frame grid, step ceiling) is still a request
        # error, so it answers 422 like the schema-level ones rather than 400.
        raise HTTPException(422, str(exc)) from None
    try:
        return ENGINE.submit(req)
    except queue.Full:
        raise HTTPException(429, f"queue is full ({CONFIG.queue_max} jobs queued or running)") from None
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from None


# --------------------------------------------------------------------------- health / discovery


@app.get("/v1/health")
async def health():
    payload = ENGINE.health()
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


@app.get("/healthz")
async def healthz():
    ready = ENGINE.ready.is_set()
    return JSONResponse({"ok": ready, "ready": ready}, status_code=200 if ready else 503)


@app.get("/v1/models", dependencies=PROTECTED)
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": CONFIG.model_id,
                "object": "model",
                "created": int(ENGINE.started),
                "owned_by": "tenstorrent",
                "task": "text-image-to-audio-video",
                "profile": CONFIG.profile,
                "mesh": CONFIG.mesh_shape,
            }
        ],
    }


@app.get("/v1/capabilities", dependencies=PROTECTED)
async def capabilities():
    return CONFIG.capabilities


@app.get("/capabilities")
async def capabilities_alias():
    return CONFIG.capabilities


# --------------------------------------------------------------------------- jobs


@app.post("/v1/videos/generations", status_code=202, dependencies=PROTECTED)
async def submit(body: GenerationRequest):
    return JSONResponse(_job_view(_submit(body)), status_code=202)


@app.get("/v1/videos/generations", dependencies=PROTECTED)
async def list_jobs(limit: int = 32):
    return {"object": "list", "data": [_job_view(j) for j in ENGINE.recent(limit)]}


@app.get("/v1/videos/generations/{job_id}", dependencies=PROTECTED)
async def get_job(job_id: str):
    return _job_view(_require(job_id))


@app.delete("/v1/videos/generations/{job_id}", dependencies=PROTECTED)
async def cancel_job(job_id: str):
    return _job_view(ENGINE.cancel(_require(job_id)))


@app.get("/v1/videos/generations/{job_id}/download", dependencies=PROTECTED)
async def download(job_id: str):
    job = _require(job_id)
    if job.status != "completed" or not job.path or not os.path.exists(job.path):
        raise HTTPException(409, f"job {job_id} is {job.status}")
    return FileResponse(job.path, media_type="video/mp4", filename=f"{job_id}.mp4")


# --------------------------------------------------------------------------- aliases
# Eric Zietlow's QB2 service shape, so a client written against that server works unchanged.


class GenerateAlias(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    width: int | None = None
    height: int | None = None
    num_frames: int | None = None
    steps: int | None = None
    seed: int = 0
    mode: str | None = None
    first_frame: str | None = Field(None, description="path to an image ON THE SERVER")
    last_frame: str | None = None

    def to_request(self) -> GenerationRequest:
        """Read any server-side keyframe paths off disk and build the real request."""
        import base64

        images = []
        for path, position in ((self.first_frame, "first"), (self.last_frame, "last")):
            if not path:
                continue
            if not os.path.isfile(path):
                raise HTTPException(422, f"{position}_frame {path!r} is not a file on this server")
            images.append(ImagePrompt(image=base64.b64encode(open(path, "rb").read()).decode(), position=position))
        fields = {
            "prompt": self.prompt,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "num_frames": self.num_frames,
            "steps": self.steps,
            "image_prompts": images or None,
        }
        if self.mode:
            fields["mode"] = self.mode
        try:
            return GenerationRequest(**fields)
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False)) from None


@app.post("/generate", dependencies=PROTECTED)
async def generate_alias(body: GenerateAlias):
    return {"job_id": _submit(body.to_request()).id}


@app.get("/jobs/{job_id}", dependencies=PROTECTED)
async def job_alias(job_id: str):
    return _require(job_id).to_alias()


@app.get("/result/{job_id}", dependencies=PROTECTED)
async def result_alias(job_id: str):
    return await download(job_id)


@app.post("/cancel/{job_id}", dependencies=PROTECTED)
async def cancel_alias(job_id: str):
    ENGINE.cancel(_require(job_id))
    return {"cancelled": True}
