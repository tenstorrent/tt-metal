"""FastAPI app for MiniMax Music 3 on Tenstorrent. Served by tt-model's `tt-dit-server` kind:
    python -m uvicorn --host 0.0.0.0 --port <port> --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
The model loads in the lifespan; uvicorn's "Application startup complete" is the readiness line.

Routes: GET/POST /v1/health · GET /v1/models · POST /v1/audio/speech (SGLang-Omni / OpenAI-compatible, synchronous)
        POST /v1/music/jobs · GET /v1/music/jobs/{id} · GET /v1/music/jobs/{id}/audio · DELETE /v1/music/jobs/{id}
"""
from __future__ import annotations

import logging
import os
import queue
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from models.autoports.minimaxai_minimax_music3.config import HF_REPO_ID, VOCODER_SAMPLE_RATE
from models.autoports.minimaxai_minimax_music3.server import audio_io
from models.autoports.minimaxai_minimax_music3.server.engine import Music3Engine
from models.autoports.minimaxai_minimax_music3.server.schemas import JobStatus, JobSubmitted, SpeechRequest

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("minimax_music3")
engine = Music3Engine(log=log.info)
API_KEY = os.environ.get("MUSIC3_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    await run_in_threadpool(engine.wait_ready)
    yield


app = FastAPI(title="MiniMax Music 3 on Tenstorrent", version="0.1.0", lifespan=lifespan)


async def auth(request: Request):
    if API_KEY and request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid or missing API key")


def _submit(req: SpeechRequest):
    if req.stream:
        raise HTTPException(
            400, "streaming is not supported by this model (the reference server rejects it too); set stream=false"
        )
    if not audio_io.available_formats().get(req.response_format):
        raise HTTPException(400, f"format {req.response_format!r} not available on this server")
    try:
        return engine.submit(req)
    except queue.Full:
        raise HTTPException(429, "server busy; try again")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


def _encode(job) -> Response:
    audio, st = job.result
    sr = job.req.sample_rate
    out = audio_io.resample(audio, VOCODER_SAMPLE_RATE, sr)
    data, ctype = audio_io.encode(out, job.req.response_format, sr)
    return Response(
        data,
        media_type=ctype,
        headers={
            "Content-Disposition": f"attachment; filename=music.{job.req.response_format}",
            "X-Music3-Frames": str(st["frames"]),
            "X-Music3-Audio-Seconds": f"{st['audio_s']:.3f}",
            "X-Music3-Generation-Seconds": f"{st['seconds']:.3f}",
            "X-Music3-RTF": f"{st['rtf']:.3f}",
            "X-Music3-Sample-Rate": str(sr),
            "X-Music3-Ended": str(bool(st["ended"])).lower(),
            "X-Music3-Job": job.id,
        },
    )


@app.api_route("/v1/health", methods=["GET", "POST"])
async def health():
    h = engine.health()
    return JSONResponse(h, status_code=200 if h["status"] == "ok" else 503)


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": os.environ.get("HF_MODEL_ID", HF_REPO_ID),
                "object": "model",
                "created": int(engine.started),
                "owned_by": "MiniMaxAI",
                "served_on": "tenstorrent",
                "mesh": engine.health()["mesh"],
                "task": "text-to-music",
            }
        ],
    }


@app.post("/v1/audio/speech", dependencies=[Depends(auth)])
async def speech(req: SpeechRequest, request: Request):
    job = _submit(req)
    while not await run_in_threadpool(job.done.wait, 1.0):
        if await request.is_disconnected():
            job.cancel.set()
            raise HTTPException(499, "client disconnected")
    if job.status == "error":
        raise HTTPException(500, job.error)
    if job.status == "cancelled":
        raise HTTPException(499, "cancelled")
    return _encode(job)


@app.post("/v1/music/jobs", dependencies=[Depends(auth)], response_model=JobSubmitted)
async def jobs_submit(req: SpeechRequest):
    job = _submit(req)
    return JobSubmitted(id=job.id, status=job.status, queue_position=engine.q.qsize())


@app.get("/v1/music/jobs/{job_id}", dependencies=[Depends(auth)], response_model=JobStatus)
async def jobs_status(job_id: str):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return JobStatus(**job.to_status())


@app.get("/v1/music/jobs/{job_id}/audio", dependencies=[Depends(auth)])
async def jobs_audio(job_id: str):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job.status != "done":
        raise HTTPException(409, f"job is {job.status}")
    return _encode(job)


@app.delete("/v1/music/jobs/{job_id}", dependencies=[Depends(auth)])
async def jobs_cancel(job_id: str):
    job = engine.jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    job.cancel.set()
    return {"id": job.id, "status": "cancelling" if not job.done.is_set() else job.status}
