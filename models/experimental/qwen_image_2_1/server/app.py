# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""FastAPI server for the tt-model container: Qwen-Image-2.1 text-to-image on one Blackhole p150.

POST /predict  {"prompt": str, "seed": int = 42, "num_steps": int = 40, "return_rgba": bool = false,
                "images": [base64 PNG/JPEG, ...] | null   (condition images for editing)}
               -> {"image": <base64 PNG (RGB over white)>, "image_rgba": <base64 PNG> | null, "timing_ms": {...}}
GET  /health   -> {"status": "ok" | "loading"}
GET  /info     -> model / device / config facts
GET  /v1/models  (stub so tt-model's readiness probe does not 404)

The device is opened and the models are loaded at startup (a warm-up generation captures the metal traces),
requests are serialised on the chip. Importing this module has no side effects.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

STATE = {"ready": False, "error": None, "pipe": None, "dev": None, "info": {}}
STOP = threading.Event()

# Request queue with prompt affinity: one worker owns the device; when several requests wait, it serves those whose
# prompt state is already resident on the device (K/V slot + captured loop trace) before switching prompts, so a
# prompt switch (~1.5 s for text-to-image, ~7 s for a two-image edit: prefix pass + trace capture) is paid once per
# distinct prompt in the backlog instead of once per request. Requests older than QWEN_QUEUE_MAX_WAIT_S are served
# first regardless, so affinity never starves anyone. Single client: plain FIFO.
QUEUE: list = []  # (enqueued_at, seq, cache_key, work: callable, result: threading.Event holder)
QUEUE_COND = threading.Condition()
QUEUE_MAX_WAIT_S = float(os.environ.get("QWEN_QUEUE_MAX_WAIT_S", "120"))
QUEUE_MAX_PENDING = int(os.environ.get("QWEN_QUEUE_MAX_PENDING", "8"))
if QUEUE_MAX_PENDING < 1:
    raise ValueError("QWEN_QUEUE_MAX_PENDING must be positive")
_SEQ = [0]


def _queue_worker():
    while True:
        with QUEUE_COND:
            while not QUEUE:
                if STOP.is_set():
                    return
                QUEUE_COND.wait()
            pipe = STATE["pipe"]
            oldest = min(QUEUE, key=lambda e: e[1])
            pick = oldest
            if time.time() - oldest[0] < QUEUE_MAX_WAIT_S and pipe is not None:
                resident = [e for e in QUEUE if pipe.is_resident(e[2])]
                if resident:
                    pick = min(resident, key=lambda e: e[1])
            QUEUE.remove(pick)
        _, _, _, work, holder = pick
        try:
            holder["result"] = work()
        except BaseException as e:  # delivered to the waiting request handler
            holder["error"] = e
        holder["done"].set()


def _run_queued(cache_key, work):
    holder = {"done": threading.Event(), "result": None, "error": None}
    with QUEUE_COND:
        if STOP.is_set():
            raise HTTPException(status_code=503, detail="server shutting down")
        if len(QUEUE) >= QUEUE_MAX_PENDING:
            raise HTTPException(status_code=503, detail="request queue full")
        _SEQ[0] += 1
        QUEUE.append((time.time(), _SEQ[0], cache_key, work, holder))
        QUEUE_COND.notify()
    holder["done"].wait()
    if holder["error"] is not None:
        raise holder["error"]
    return holder["result"]


def _env_flag(name: str, default: str) -> bool:
    value = os.environ.get(name, default).lower()
    if value not in ("0", "1", "false", "true"):
        raise ValueError(f"{name} must be 0, 1, false or true; got {value!r}")
    return value in ("1", "true")


def load_config():
    from ..common.config import HF_REVISION

    cfg = {
        "weights_revision": os.environ.get("TT_WEIGHTS_REVISION", HF_REVISION),
        "editing": _env_flag("QWEN_IMAGE_EDITING", "1"),
        "eth_dispatch": _env_flag("QWEN_IMAGE_ETH_DISPATCH", "0"),
        "dit_weight_dtype": os.environ.get("QWEN_IMAGE_DIT_DTYPE", "bf16"),
        "te_weight_dtype": os.environ.get("QWEN_IMAGE_TE_DTYPE", "bfp8"),
        "default_steps": int(os.environ.get("QWEN_IMAGE_STEPS", "40")),
        "size": int(os.environ.get("QWEN_IMAGE_SIZE", "1024")),
        "warmup": _env_flag("QWEN_IMAGE_WARMUP", "1"),
        "warmup_prompt": os.environ.get(
            "QWEN_IMAGE_WARMUP_PROMPT", "White furry llama with black sunglasses, smiling and happy, jumping"
        ),
    }
    if cfg["weights_revision"] != HF_REVISION:
        raise ValueError(f"TT_WEIGHTS_REVISION must be {HF_REVISION}")
    for key in ("dit_weight_dtype", "te_weight_dtype"):
        if cfg[key] not in ("bf16", "bfp8"):
            raise ValueError(f"{key} must be bf16 or bfp8; got {cfg[key]!r}")
    if not 2 <= cfg["default_steps"] <= 100:
        raise ValueError("QWEN_IMAGE_STEPS must be between 2 and 100")
    if cfg["eth_dispatch"] and cfg["editing"]:
        raise ValueError("Image editing requires Tensix dispatch; set QWEN_IMAGE_ETH_DISPATCH=0")
    if cfg["size"] != 1024:
        raise ValueError("QWEN_IMAGE_SIZE must be 1024 for this implementation")
    return cfg


def _load_models():
    import ttnn
    from models.experimental.qwen_image_2_1.common.device import open_device
    from models.experimental.qwen_image_2_1.tt.dit import DiTPrecision
    from models.experimental.qwen_image_2_1.tt.pipeline import QwenImage21Pipeline
    from models.experimental.qwen_image_2_1.tt.text_encoder import TEPrecision

    cfg = load_config()
    t0 = time.time()
    dev = open_device(eth_dispatch=cfg["eth_dispatch"])
    STATE["dev"] = dev
    grid = dev.compute_with_storage_grid_size()
    dit_prec = DiTPrecision()
    if cfg["dit_weight_dtype"] == "bfp8":
        dit_prec.weight_dtype = ttnn.bfloat8_b
    te_prec = TEPrecision()
    if cfg["te_weight_dtype"] == "bf16":
        te_prec.weight_dtype = ttnn.bfloat16
    pipe = QwenImage21Pipeline(
        dev, dit_prec=dit_prec, te_prec=te_prec, height=cfg["size"], width=cfg["size"], load_editing=cfg["editing"]
    )
    load_s = time.time() - t0
    STATE["pipe"] = pipe
    warm = None
    if cfg["warmup"]:
        t0 = time.time()
        pipe.generate(cfg["warmup_prompt"], seed=42, num_steps=cfg["default_steps"])
        warm = time.time() - t0
    STATE.update(
        pipe=pipe,
        dev=dev,
        info={
            "model": "Qwen/Qwen-Image-2.1",
            "device": "Blackhole P150",
            "worker_grid": f"{grid.x}x{grid.y}",
            "load_s": round(load_s, 1),
            "warmup_s": None if warm is None else round(warm, 1),
            **cfg,
            "editing_loaded": pipe.te_vl is not None,
            "prompt_slots": len(pipe._slots),
            "queue": "prompt-affinity" if pipe._slots else "fifo",
        },
        ready=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    STOP.clear()
    STATE.update(ready=False, error=None, pipe=None, dev=None, info={})

    def worker():
        try:
            _load_models()
        except Exception as e:  # surfaced through /health
            STATE["error"] = repr(e)
            raise

    th = threading.Thread(target=worker, daemon=True)
    th.start()
    queue_thread = threading.Thread(target=_queue_worker, daemon=True, name="predict-queue")
    queue_thread.start()
    try:
        yield
    finally:
        # Finish startup and queued device work before releasing buffers or closing the mesh.
        STATE["ready"] = False
        with QUEUE_COND:
            STOP.set()
            QUEUE_COND.notify_all()
        th.join()
        queue_thread.join()
        STATE["ready"] = False
        if STATE["dev"] is not None:
            from models.experimental.qwen_image_2_1.common.device import close_device

            try:
                if STATE["pipe"] is not None:
                    STATE["pipe"].release_traces()
            finally:
                close_device(STATE["dev"])


app = FastAPI(title="qwen-image-2.1-p150", lifespan=lifespan)


class PredictRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    seed: int = 42
    num_steps: int | None = Field(None, ge=2, le=100)
    return_rgba: bool = False
    images: list[str] | None = Field(
        None, description="optional condition images (base64 PNG/JPEG) for editing", max_length=4
    )


def _decode_images(b64_list):
    from PIL import Image
    from ..common.images import condition_size

    out = []
    for b in b64_list or []:
        if "," in b[:64] and b.lstrip().startswith("data:"):
            b = b.split(",", 1)[1]
        image = Image.open(io.BytesIO(base64.b64decode(b, validate=True))).convert("RGBA")
        condition_size(image, STATE["info"]["size"])
        out.append(image)
    return out or None


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@app.get("/health")
def health():
    if STATE["error"]:
        raise HTTPException(status_code=500, detail=STATE["error"])
    return {"status": "ok" if STATE["ready"] else "loading"}


@app.get("/info")
def info():
    return STATE["info"] | {"ready": STATE["ready"]}


@app.get("/v1/models")
def v1_models():
    return {
        "object": "list",
        "data": [{"id": "changh95/qwen-image-2.1-p150", "object": "model", "owned_by": "changh95"}],
    }


@app.post("/predict")
def predict(req: PredictRequest):
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="model loading")
    pipe = STATE["pipe"]
    num_steps = STATE["info"]["default_steps"] if req.num_steps is None else req.num_steps
    try:
        images = _decode_images(req.images)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"bad image: {e}")
    if images and pipe.te_vl is None:
        raise HTTPException(status_code=400, detail="condition images are not supported by this server build")

    def work():
        t0 = time.time()
        rgb, rgba, _, tm = pipe.generate(req.prompt, seed=req.seed, num_steps=num_steps, images=images)
        return rgb, rgba, tm, time.time() - t0

    rgb, rgba, tm, total = _run_queued(pipe.cache_key_for(req.prompt, images), work)
    return {
        "image": _png_b64(rgb),
        "image_rgba": _png_b64(rgba) if req.return_rgba else None,
        "width": rgb.width,
        "height": rgb.height,
        "seed": req.seed,
        "num_steps": num_steps,
        "timing_ms": {k: round(v * 1000, 1) for k, v in tm.as_dict().items() if k.endswith("_s")}
        | {"total_ms": round(total * 1000, 1)},
    }
