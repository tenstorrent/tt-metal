# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A one-page web UI for d1-3B on a single Wormhole chip.

One process owns the N150 for its whole lifetime (the device can only be held by one process), loads the model
and captures the bucket traces once, then serves a page where you type a state (and/or drop an image), add typed
questions, and get every option's probability back.

    MESH_DEVICE=N150 python3 models/demos/multimodal/d1_3b/ui/server.py

Then open http://<host>:8000 . Host/port come from D1_UI_HOST (default 0.0.0.0) and D1_UI_PORT (default 8000).
"""

import base64
import io
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]  # .../tt-metal
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import ttnn  # noqa: E402
from models.demos.multimodal.d1_3b.tt.d1 import TtD1  # noqa: E402
from models.demos.multimodal.d1_3b.tt.model import L1_SMALL_SIZE, TRACE_REGION_SIZE  # noqa: E402

# The device is single-tenant: serialise every call that touches it.
LOCK = threading.Lock()
STATE: Dict[str, Any] = {}


def _load():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE)
    logger.info("loading d1-3B and capturing bucket traces...")
    d1 = TtD1.from_pretrained(mesh)
    STATE["mesh"], STATE["d1"] = mesh, d1
    logger.info("d1-3B ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    with LOCK:
        STATE["d1"].release()
        ttnn.close_mesh_device(STATE["mesh"])


app = FastAPI(lifespan=lifespan, title="d1-3B playground")


class AskBody(BaseModel):
    state: Optional[str] = None
    image: Optional[str] = None  # a data: URL
    questions: Dict[str, Dict[str, Any]]


def _decode_image(data_url: str):
    from PIL import Image

    raw = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def _image_markup(proc):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "\x00"}]}]
    tmpl = proc.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
    head = "<|im_start|>user\n"
    return tmpl[tmpl.index(head) + len(head) : tmpl.index("\x00")]


def _processor():
    """The HF image processor + the image markup the chat template emits, loaded once."""
    if "proc" not in STATE:
        from transformers import AutoProcessor

        d1 = STATE["d1"]
        proc = AutoProcessor.from_pretrained(str(d1.prompt.ckpt_dir), trust_remote_code=True)
        STATE["proc"], STATE["markup"] = proc, _image_markup(proc)
    return STATE["proc"], STATE["markup"]


def _ask_image(d1, image, questions):
    """Same path as the image demo: encode the image once on device, reuse it across every question."""
    proc, markup = _processor()
    model, prompt = d1.model, d1.prompt

    def encode(q):
        text = prompt.render(None, q, image_markup=markup)
        return proc(text=[text], images=[[image]], return_tensors="pt", add_special_tokens=False)

    qs = list(questions.values())
    inputs = encode(qs[0])
    if "pixel_attention_mask" in inputs:  # the runner wants the real patches only
        n = int(inputs["pixel_attention_mask"].sum(1).max())
        inputs["pixel_values"] = inputs["pixel_values"][:, :n]
        inputs["pixel_attention_mask"] = inputs["pixel_attention_mask"][:, :n]
    vision = model.vision_embeddings(
        inputs["pixel_values"],
        spatial_shapes=inputs.get("spatial_shapes"),
        pixel_attention_mask=inputs.get("pixel_attention_mask"),
    )

    answers, tokens = {}, 0
    for name, q in questions.items():
        ids = encode(q)["input_ids"][0].tolist()
        tokens = max(tokens, len(ids))
        hidden = model.hidden(ids, vision_embeddings=vision)
        probs = prompt.probabilities(q, model.option_logits(hidden, prompt.option_token_ids(q)))
        answers[name] = prompt.answer(q, probs)
    return {"answers": answers, "usage": {"input_tokens": tokens, "output_tokens": 0, "image_tokens": vision.shape[0]}}


@app.get("/")
def index():
    return HTMLResponse((HERE / "index.html").read_text())


@app.get("/health")
def health():
    return {"ok": "d1" in STATE}


@app.post("/ask")
def ask(body: AskBody):
    if not body.questions:
        return JSONResponse({"error": "add at least one question"}, status_code=400)
    d1 = STATE["d1"]
    try:
        with LOCK:
            t0 = time.perf_counter()
            if body.image:
                out = _ask_image(d1, _decode_image(body.image), body.questions)
            else:
                out = d1.system_one(body.state or "", body.questions)
            out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out
    except Exception as exc:  # surface the message to the page instead of a bare 500
        logger.exception("ask failed")
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=400)


if __name__ == "__main__":
    import os

    import uvicorn

    host, port = os.environ.get("D1_UI_HOST", "0.0.0.0"), int(os.environ.get("D1_UI_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
