# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Rerank server for llama-nemotron-rerank-1b-v2 on Blackhole P150 (FastAPI + uvicorn).

Same layout as the embedding servers (``pplx_embed_0_6b/demo/serve.py``): one resident
traced model per chip in its own spawned worker process, an async pool that hands work to
idle chips, and a collector thread resolving per-request futures.

  * The server tokenizes ``question:{query} \\n \\n passage:{document}`` (truncated to
    ``--max-length`` tokens) and ships token ids to the workers.
  * Each worker runs ``TracedReranker``: traces per (batch in {1, 8, 32}) x (length bucket
    in {128, 256, 512, ...}), captured at startup; a request's pairs are grouped by length
    bucket and packed into those batches.
  * A request's pairs are split into chunks of ``--chunk`` pairs that go to idle chips in
    parallel (one chip on a single P150).

Run
---
    python models/demos/blackhole/llama_nemotron_rerank_1b/demo/serve.py --port 8000

    curl http://localhost:8000/v1/rerank -H 'Content-Type: application/json' -d '{
        "model": "nvidia/llama-nemotron-rerank-1b-v2",
        "query": "how much protein should a female eat?",
        "documents": ["As a general guideline, ... 46 grams per day.", "Definition of summit ..."],
        "top_n": 2}'

Endpoints: ``POST /v1/rerank`` (and ``/rerank``; vLLM / Cohere / Jina request shape),
``POST /v1/score`` (vLLM score shape: ``text_1`` query, ``text_2`` documents),
``GET /health``, ``GET /v1/models``, ``GET /metrics``.

``relevance_score`` is sigmoid(logit) (``--score raw`` returns the logit); every result
also carries the raw ``logit``.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import multiprocessing as mp
import os
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

from loguru import logger

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, prompt_template  # noqa: E402


# --------------------------------------------------------------------------- #
# Chip worker
# --------------------------------------------------------------------------- #
def _worker(chip_id, max_length, task_q, result_q, ready_q):
    """One chip: build the traced reranker, then score ``(req_id, ids_list)`` tasks.

    Puts ``(chip_id, build_s)`` on ``ready_q`` when ready (``(chip_id, None, err)`` on
    failure) and ``(req_id, chip_id, scores, device_ms, err)`` on ``result_q`` per task.
    A ``None`` task stops the worker.
    """
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)
    device = None
    try:
        import ttnn
        from models.demos.blackhole.llama_nemotron_rerank_1b.demo.generator_vllm import NemotronRerankForScoring

        t0 = time.perf_counter()
        device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=400_000_000)
        model = NemotronRerankForScoring(device=device, max_seq_len=max_length)
        ready_q.put((chip_id, time.perf_counter() - t0))
        while True:
            task = task_q.get()
            if task is None:
                break
            req_id, ids_list = task
            t = time.perf_counter()
            try:
                scores = model.score_ids(ids_list)
                result_q.put((req_id, chip_id, scores, (time.perf_counter() - t) * 1e3, None))
            except Exception as exc:  # report per-task errors, keep serving
                result_q.put((req_id, chip_id, None, 0.0, f"{type(exc).__name__}: {exc}"))
    except Exception as exc:
        ready_q.put((chip_id, None, f"{type(exc).__name__}: {exc}"))
    finally:
        if device is not None:
            try:
                ttnn.close_device(device)  # noqa: F821
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# Async chip pool
# --------------------------------------------------------------------------- #
class AsyncRerankPool:
    def __init__(self, dp: int, max_length: int, chunk: int):
        from transformers import AutoTokenizer

        self.n = dp
        self.max_length = max_length
        self.chunk = chunk
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        ctx = mp.get_context("spawn")
        self.task_qs = [ctx.Queue() for _ in range(dp)]
        self.result_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.procs = []
        for i in range(dp):
            p = ctx.Process(
                target=_worker,
                args=(i, max_length, self.task_qs[i], self.result_q, ready_q),
                name=f"chip-{i}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)
        logger.info(f"Building {dp} resident reranker(s) and capturing traces...")
        self.ready, self.failed = set(), {}
        for _ in range(dp):
            msg = ready_q.get()
            if len(msg) == 3 and msg[1] is None:
                self.failed[msg[0]] = msg[2]
                logger.error(f"chip {msg[0]} failed: {msg[2][:300]}")
            else:
                self.ready.add(msg[0])
                logger.info(f"chip {msg[0]} ready ({msg[1]:.0f} s)")
        if not self.ready:
            raise RuntimeError("no chip came up")
        self.loop = None
        self.idle = None
        self.pending = {}
        self._req = 0
        self.metrics = deque(maxlen=100_000)

    def start(self, loop):
        self.loop = loop
        self.idle = asyncio.Queue()
        for c in sorted(self.ready):
            self.idle.put_nowait(c)
        threading.Thread(target=self._collect, name="collector", daemon=True).start()

    def _collect(self):
        while True:
            item = self.result_q.get()
            if item is None:
                break
            req_id, chip, scores, dev_ms, err = item
            fut = self.pending.pop(req_id, None)
            if fut is not None:
                self.loop.call_soon_threadsafe(self._resolve, fut, scores, dev_ms, err)
            self.loop.call_soon_threadsafe(self.idle.put_nowait, chip)

    @staticmethod
    def _resolve(fut, scores, dev_ms, err):
        if fut.done():
            return
        if err is not None:
            fut.set_exception(RuntimeError(err))
        else:
            fut.set_result((scores, dev_ms))

    def tokenize(self, query: str, documents: List[str]):
        texts = [prompt_template(query, d) for d in documents]
        enc = self.tokenizer(texts, truncation=True, max_length=self.max_length)
        return enc["input_ids"]

    async def _score_chunk(self, ids_list):
        chip = await self.idle.get()
        req_id = self._req
        self._req += 1
        fut = self.loop.create_future()
        self.pending[req_id] = fut
        self.task_qs[chip].put((req_id, ids_list))
        return await fut

    async def score(self, query: str, documents: List[str]):
        t0 = time.perf_counter()
        ids = await self.loop.run_in_executor(None, self.tokenize, query, documents)
        t_tok = time.perf_counter()
        chunks = [ids[i : i + self.chunk] for i in range(0, len(ids), self.chunk)]
        parts = await asyncio.gather(*(self._score_chunk(c) for c in chunks))
        scores = [s for part, _ in parts for s in part]
        self.metrics.append(
            {
                "tokenize_ms": (t_tok - t0) * 1e3,
                "device_ms": sum(d for _, d in parts),
                "wall_ms": (time.perf_counter() - t0) * 1e3,
                "pairs": len(documents),
            }
        )
        return scores, sum(len(x) for x in ids)

    def stats(self, reset=False):
        samples = list(self.metrics)
        if reset:
            self.metrics.clear()
        out = {"count": len(samples)}
        for key in ("tokenize_ms", "device_ms", "wall_ms", "pairs"):
            vals = sorted(s[key] for s in samples)
            if vals:
                pick = lambda p: vals[min(len(vals) - 1, int(p / 100 * (len(vals) - 1) + 0.5))]
                out[key] = {"p50": round(pick(50), 3), "p90": round(pick(90), 3), "p99": round(pick(99), 3)}
        return out

    def close(self):
        for c in sorted(self.ready):
            try:
                self.task_qs[c].put(None)
            except Exception:
                pass
        try:
            self.result_q.put(None)
        except Exception:
            pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402


class RerankRequest(BaseModel):
    query: str
    documents: List[Union[str, dict]]
    model: str = MODEL_ID
    top_n: Optional[int] = None
    return_documents: bool = True


class ScoreRequest(BaseModel):
    text_1: str
    text_2: Union[str, List[str]]
    model: str = MODEL_ID


_CONFIG = SimpleNamespace(dp=1, max_length=512, chunk=32, score="sigmoid")
_POOL: Optional[AsyncRerankPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    loop = asyncio.get_running_loop()
    _POOL = await loop.run_in_executor(None, lambda: AsyncRerankPool(_CONFIG.dp, _CONFIG.max_length, _CONFIG.chunk))
    _POOL.start(loop)
    try:
        yield
    finally:
        if _POOL is not None:
            _POOL.close()
            _POOL = None


app = FastAPI(title="llama-nemotron-rerank-1b-v2 server", lifespan=lifespan)


def _doc_text(d):
    if isinstance(d, str):
        return d
    if isinstance(d, dict) and isinstance(d.get("text"), str):
        return d["text"]
    raise HTTPException(status_code=400, detail="each document must be a string or {'text': str}")


def _relevance(logit):
    return 1.0 / (1.0 + math.exp(-logit)) if _CONFIG.score == "sigmoid" else logit


async def _scores(query, docs):
    if _POOL is None:
        raise HTTPException(status_code=503, detail="pool not ready")
    if not docs:
        raise HTTPException(status_code=400, detail="documents must not be empty")
    try:
        return await _POOL.score(query, docs)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/rerank")
@app.post("/rerank")
async def rerank(req: RerankRequest):
    docs = [_doc_text(d) for d in req.documents]
    logits, tokens = await _scores(req.query, docs)
    order = sorted(range(len(docs)), key=lambda i: -logits[i])
    if req.top_n is not None:
        order = order[: max(0, req.top_n)]
    results = []
    for i in order:
        r = {"index": i, "relevance_score": _relevance(logits[i]), "logit": logits[i]}
        if req.return_documents:
            r["document"] = {"text": docs[i]}
        results.append(r)
    return {
        "id": f"rerank-{uuid.uuid4().hex}",
        "model": req.model,
        "usage": {"total_tokens": tokens},
        "results": results,
    }


@app.post("/v1/score")
@app.post("/score")
async def score(req: ScoreRequest):
    docs = [req.text_2] if isinstance(req.text_2, str) else list(req.text_2)
    logits, tokens = await _scores(req.text_1, docs)
    return {
        "id": f"score-{uuid.uuid4().hex}",
        "object": "list",
        "model": req.model,
        "data": [{"index": i, "object": "score", "score": _relevance(s)} for i, s in enumerate(logits)],
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


@app.get("/health")
async def health():
    ready = len(_POOL.ready) if _POOL is not None else 0
    return {"status": "ok" if ready else "starting", "chips_ready": ready, "chips_total": _CONFIG.dp}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}


@app.get("/metrics")
async def metrics(reset: bool = False):
    """Per-request latency (ms): tokenize on the server, device time in the workers, total wall time."""
    if _POOL is None:
        raise HTTPException(status_code=503, detail="pool not ready")
    return _POOL.stats(reset=reset)


def main():
    ap = argparse.ArgumentParser(description="llama-nemotron-rerank-1b-v2 rerank server (FastAPI).")
    ap.add_argument("--dp", type=int, default=1, help="chips / workers (default 1: one P150)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-length", type=int, default=512, help="max tokens per query+document pair (<= 8192)")
    ap.add_argument("--chunk", type=int, default=32, help="pairs per chip task")
    ap.add_argument(
        "--score", choices=("sigmoid", "raw"), default="sigmoid", help="relevance_score = sigmoid(logit) or logit"
    )
    args = ap.parse_args()
    _CONFIG.dp, _CONFIG.max_length, _CONFIG.chunk, _CONFIG.score = args.dp, args.max_length, args.chunk, args.score
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, workers=1, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
