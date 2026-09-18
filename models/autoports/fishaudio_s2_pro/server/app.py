"""FastAPI app for Fish S2 Pro on Tenstorrent. Served by tt-model's `tt-dit-server` kind:
    python -m uvicorn --host 0.0.0.0 --port <port> --lifespan on models.autoports.fishaudio_s2_pro.server.app:app
The model loads in the lifespan; uvicorn's "Application startup complete" is the readiness line.

Routes: GET/POST /v1/health · POST /v1/tts (fish-speech compatible; JSON / msgpack / multipart; streaming WAV)
        POST /v1/references/add · GET /v1/references/list · DELETE /v1/references/delete · POST /v1/references/update
        POST /v1/audio/speech (OpenAI-compatible) · GET /v1/models
"""
from __future__ import annotations

import logging
import os
import queue
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.concurrency import run_in_threadpool

from models.autoports.fishaudio_s2_pro.config import SAMPLE_RATE
from models.autoports.fishaudio_s2_pro.server import audio_io
from models.autoports.fishaudio_s2_pro.server.engine import S2Engine
from models.autoports.fishaudio_s2_pro.server.negotiation import pack_response, parse_body
from models.autoports.fishaudio_s2_pro.server.schemas import (
    AddReferenceResponse,
    DeleteReferenceResponse,
    ListReferencesResponse,
    OpenAISpeechRequest,
    ServeTTSRequest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
log = logging.getLogger("fish_s2_pro")
engine = S2Engine(log=log.info)
API_KEY = os.environ.get("FISH_S2_API_KEY")


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.start()
    await run_in_threadpool(engine.wait_ready)
    yield


app = FastAPI(title="Fish Audio S2 Pro on Tenstorrent", version="0.1.0", lifespan=lifespan)


async def auth(request: Request):
    if API_KEY and request.headers.get("authorization") != f"Bearer {API_KEY}":
        raise HTTPException(401, "invalid or missing API key")


def _check_streaming_format(req: ServeTTSRequest):
    if req.streaming and req.format != "wav":
        raise HTTPException(400, "streaming only supports the wav format")
    if not audio_io.available_formats().get(req.format):
        raise HTTPException(400, f"format {req.format!r} not available on this server")


def _submit(req: ServeTTSRequest):
    try:
        return engine.submit(req)
    except queue.Full:
        raise HTTPException(429, "server busy; try again")
    except FileNotFoundError as e:
        raise HTTPException(404, f"reference {e} not found")
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))


async def _collect(job):
    """Wait for the final result of a non-streaming job."""
    while True:
        kind, payload = await run_in_threadpool(job.out.get)
        if kind == "final":
            return payload
        if kind == "error":
            raise HTTPException(500, payload)
        if kind == "cancelled":
            raise HTTPException(499, "cancelled")


async def _stream(job, request: Request):
    yield audio_io.wav_chunk_header()
    while True:
        if await request.is_disconnected():
            job.cancel.set()
            return
        try:
            kind, payload = await run_in_threadpool(lambda: job.out.get(timeout=1.0))
        except queue.Empty:
            continue
        if kind == "segment":
            yield audio_io.pcm16(payload)
        elif kind == "final":
            return
        elif kind in ("error", "cancelled"):
            log.error(f"streaming job ended with {kind}: {payload}")
            return


@app.api_route("/v1/health", methods=["GET", "POST"])
async def health():
    h = engine.health()
    return JSONResponse(h, status_code=200 if h["status"] == "ok" else 503)


@app.get("/v1/models")
async def models():
    mid = os.environ.get("HF_MODEL", "fishaudio/s2-pro")
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": int(engine.started),
                "owned_by": "fishaudio",
                "served_on": "tenstorrent",
                "mesh": engine.health()["mesh"],
            }
        ],
    }


@app.post("/v1/tts", dependencies=[Depends(auth)])
async def tts(request: Request):
    req: ServeTTSRequest = await parse_body(request, ServeTTSRequest)
    _check_streaming_format(req)
    if req.reference_id and not engine.refs.exists(req.reference_id):
        raise HTTPException(404, f"reference {req.reference_id!r} not found")
    job = _submit(req)
    if req.streaming:
        return StreamingResponse(
            _stream(job, request),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=audio.wav"},
        )
    wav, stats = await _collect(job)
    data, ctype = audio_io.encode(wav, req.format)
    return Response(
        data,
        media_type=ctype,
        headers={
            "Content-Disposition": f"attachment; filename=audio.{req.format}",
            "X-Fish-Frames": str(stats["frames"]),
            "X-Fish-Audio-Seconds": f"{stats['audio_s']:.3f}",
            "X-Fish-Generation-Seconds": f"{stats['seconds']:.3f}",
        },
    )


@app.post("/v1/audio/speech", dependencies=[Depends(auth)])
async def openai_speech(request: Request):
    o: OpenAISpeechRequest = await parse_body(request, OpenAISpeechRequest)
    fmt = {"aac": "mp3"}.get(o.response_format, o.response_format)
    voice = (
        o.voice
        if o.voice not in ("default", "alloy", "echo", "fable", "onyx", "nova", "shimmer", "")
        else os.environ.get("FISH_S2_DEFAULT_VOICE", "")
    )
    req = ServeTTSRequest(
        text=o.input,
        format=fmt,
        reference_id=voice or None,
        seed=o.seed,
        temperature=o.temperature if o.temperature is not None else 0.8,
        top_p=o.top_p if o.top_p is not None else 0.8,
    )
    _check_streaming_format(req)
    if req.reference_id and not engine.refs.exists(req.reference_id):
        raise HTTPException(404, f"voice {req.reference_id!r} not found; add it via /v1/references/add")
    job = _submit(req)
    wav, stats = await _collect(job)
    data, ctype = audio_io.encode(wav, fmt)
    headers = {"X-Fish-Frames": str(stats["frames"])}
    if abs(o.speed - 1.0) > 1e-6:
        headers["X-Fish-Warning"] = "speed is not supported and was ignored"
    return Response(data, media_type=ctype, headers=headers)


@app.post("/v1/references/add", dependencies=[Depends(auth)])
async def references_add(request: Request, id: str = Form(...), text: str = Form(...), audio: UploadFile = File(...)):
    if not engine.refs.valid_id(id):
        raise HTTPException(400, "invalid reference id")
    if engine.refs.exists(id):
        raise HTTPException(409, f"reference {id!r} already exists")
    data = await audio.read()
    try:
        wav = audio_io.decode_audio(data)
    except Exception as e:
        raise HTTPException(400, f"could not decode audio: {e}")
    if len(wav) < SAMPLE_RATE // 2:
        raise HTTPException(400, "reference audio shorter than 0.5 s")
    codes = await run_in_threadpool(engine.refs.add, id, wav, text)
    return pack_response(
        request, AddReferenceResponse(success=True, message=f"added {codes.shape[1]} frames", reference_id=id)
    )


@app.get("/v1/references/list", dependencies=[Depends(auth)])
async def references_list(request: Request):
    return pack_response(request, ListReferencesResponse(success=True, reference_ids=engine.refs.list()))


@app.delete("/v1/references/delete", dependencies=[Depends(auth)])
async def references_delete(request: Request):
    body = await request.json() if (await request.body()) else {}
    ref_id = body.get("reference_id") or request.query_params.get("reference_id")
    if not ref_id or not engine.refs.valid_id(ref_id):
        raise HTTPException(400, "reference_id required")
    try:
        engine.refs.delete(ref_id)
    except FileNotFoundError:
        raise HTTPException(404, f"reference {ref_id!r} not found")
    return pack_response(request, DeleteReferenceResponse(success=True, message="deleted", reference_id=ref_id))


@app.post("/v1/references/update", dependencies=[Depends(auth)])
async def references_update(request: Request):
    body = await request.json()
    old, new = body.get("old_reference_id"), body.get("new_reference_id")
    if not (old and new and engine.refs.valid_id(old) and engine.refs.valid_id(new)):
        raise HTTPException(400, "old_reference_id and new_reference_id required")
    if not engine.refs.exists(old):
        raise HTTPException(404, f"reference {old!r} not found")
    if engine.refs.exists(new):
        raise HTTPException(409, f"reference {new!r} already exists")
    engine.refs.rename(old, new)
    return {"success": True, "message": "renamed", "old_reference_id": old, "new_reference_id": new}
