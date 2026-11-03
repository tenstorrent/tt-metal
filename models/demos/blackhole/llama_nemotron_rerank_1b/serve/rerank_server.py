# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
FastAPI rerank server for nvidia/llama-nemotron-rerank-1b-v2 (Jerry Wang's NemotronRerankForScoring, tt-metal PR
#63253), v2 of rerank_serve_fastapi.py. The runtime follows Aniruddha Tupe's strands-decider server (tt-metal
atupe/strands-decider-fastapi @c83659abfbe, models/demos/strands_agent/serve/), audited in DESIGN.md D03:

  * The server accepts connections at once; /health and the endpoints return 503 "starting" until a worker is ready.
  * A watchdog (1 s) fails a worker whose process exited, and one with a task older than --hang-timeout-s (a hung
    device keeps the process alive); its pending requests get 503 and /health reports it.
  * Each worker batches queued request chunks up to --max-batch-pairs pairs per score_ids call (rerank_batcher.py,
    device-free, unit-tested); requests go to the live worker with the fewest outstanding pairs.
  * Pydantic validation (422) runs before tokenizing; tokenization uses one tokenizer per pool thread, all loaded
    at startup behind a barrier (a shared fast tokenizer raises "Already borrowed" when calls differ in truncation).
  * Warmup captures every (batch, row length) trace; a capture during serving is counted in /metrics.
  * Shutdown: None to each worker, join 30 s, then terminate / kill; binds 127.0.0.1 by default.

Request / response shape and 422 rules are tt-media-server's (tt-inference-server PR #5490). Workers are pinned by
PCI address (--chips), one per chip.

Run:  python rerank_server.py --chips 0000:86:00.0 --port 8111
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import math
import multiprocessing as mp
import os
import signal
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from models.demos.blackhole.llama_nemotron_rerank_1b.tt.model_config import MODEL_ID, SEQ_BUCKETS, prompt_template

logger = logging.getLogger("uvicorn.error")

SHORT_NAME = MODEL_ID.split("/")[-1]
REVISION = os.environ.get("NEMOTRON_RERANK_REVISION", "828765652b05bd439c9789d2a6d093db1caa1443")
MAX_DOCUMENTS = 1000  # tt-media-server MAX_RERANK_DOCUMENTS
TRACE_REGION_SIZE = int(os.environ.get("RR_TRACE_REGION_SIZE", 400_000_000))  # 18 traces fit (PR #5490)
SAMPLES = 100_000

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "chips": ["0"],
    "max_length": 512,
    "chunk": 32,
    "max_batch_pairs": 32,
    "batch_wait_ms": 0.0,
    "hang_timeout_s": 60.0,
    "tokenizer_threads": 4,
    "score": "sigmoid",
    "keep_alive_s": 30.0,
    "verbose": False,
}


def traced_max_length(max_length: int) -> int:
    """Longest sequence bucket not above ``max_length``: the runner traces only whole buckets, so 500 serves 256."""
    fits = [b for b in SEQ_BUCKETS if b <= max_length]
    if not fits:
        raise ValueError(f"max_length {max_length} is below the smallest bucket {SEQ_BUCKETS[0]}")
    return fits[-1]


def relevance_score(logit: float) -> float:
    """sigmoid(logit), written per sign so math.exp never overflows (tt-media-server open_ai_api/rerank.py)."""
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    z = math.exp(logit)
    return z / (1.0 + z)


def percentiles(values):
    if not values:
        return {}
    v = sorted(values)

    def pct(p):
        k = (len(v) - 1) * p / 100
        lo = int(k)
        hi = min(lo + 1, len(v) - 1)
        return v[lo] + (v[hi] - v[lo]) * (k - lo)

    return {
        "p50": round(pct(50), 3),
        "p90": round(pct(90), 3),
        "p99": round(pct(99), 3),
        "max": round(v[-1], 3),
        "mean": round(sum(v) / len(v), 3),
    }


def _histogram(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# Device worker process
# --------------------------------------------------------------------------- #
def worker_main(chip_idx, bdf, cfg, task_q, result_q):
    """Spawned device process for one chip: open it by PCI address, build and warm the traced reranker, then answer
    batches. Messages: {"type": "ready" | "failed" | "result", "chip": chip_idx, ...}."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the server owns shutdown
    os.environ["TT_VISIBLE_DEVICES"] = bdf
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def log(msg):
        print(f"[worker {chip_idx} {bdf}] {msg}", flush=True)

    device = model = None
    try:
        from rerank_batcher import collect_batch, run_batch

        import ttnn
        from models.demos.blackhole.llama_nemotron_rerank_1b.demo.generator_vllm import NemotronRerankForScoring

        t_start = time.perf_counter()
        device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=TRACE_REGION_SIZE)
        # warmup=True captures every (batch 1/8/32) x (row length bucket) trace, masked and unmasked
        model = NemotronRerankForScoring(device=device, max_batch_size=32, max_seq_len=cfg["max_length"], warmup=True)
        model.score_ids([[1, 2, 3, 4]])
        traces = getattr(model.runner, "_traces", {})
        n_traces = len(traces)
        warmup_s = time.perf_counter() - t_start
        result_q.put(
            {
                "type": "ready",
                "chip": chip_idx,
                "warmup_s": round(warmup_s, 1),
                "traces": n_traces,
                "max_length": model.runner.max_seq_len,
            }
        )
        log(f"ready after {warmup_s:.1f} s, {n_traces} traces")

        def get(timeout):
            if timeout is None:
                return task_q.get()
            if timeout == 0:
                return task_q.get_nowait()
            return task_q.get(timeout=timeout)

        batch_id, carry = 0, None
        while True:
            items, stop, carry = collect_batch(get, cfg["batch_wait_ms"] / 1000, cfg["max_batch_pairs"], carry)
            if items:
                batch_id += 1
                results = run_batch(model.score_ids, items, batch_id)
                new_traces = len(traces) - n_traces
                if new_traces:
                    log(f"WARNING: {new_traces} trace(s) captured while serving")
                    n_traces = len(traces)
                for r in results:
                    result_q.put({"type": "result", "chip": chip_idx, "new_traces": new_traces, **r})
                    new_traces = 0
                if cfg["verbose"]:
                    ok = [r for r in results if r["ok"]]
                    log(f"batch {batch_id} tasks={len(items)} ok={len(ok)} {ok[0]['timing'] if ok else {}}")
            if stop:
                break
    except BaseException:
        result_q.put({"type": "failed", "chip": chip_idx, "error": traceback.format_exc()})
    finally:
        try:
            if model is not None:
                model.release()
        finally:
            if device is not None:
                import ttnn

                ttnn.close_device(device)
            result_q.close()
            result_q.join_thread()


# --------------------------------------------------------------------------- #
# Server process: tokenizers, workers, futures, metrics
# --------------------------------------------------------------------------- #
class Encoder:
    """Tokenizer pool: one tokenizer per pool thread (a fast tokenizer instance is not thread safe)."""

    def __init__(self, threads, max_length):
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="tok")
        self.threads = threads
        self.max_length = max_length
        self.local = threading.local()
        self.auto_tokenizer = None

    def _tokenizer(self):
        tok = getattr(self.local, "tokenizer", None)
        if tok is None:
            tok = self.local.tokenizer = self.auto_tokenizer.from_pretrained(
                MODEL_ID, revision=REVISION, trust_remote_code=True
            )
        return tok

    def load(self):
        """Blocking: load a tokenizer in every pool thread."""
        from transformers import AutoTokenizer  # imported here: a first import from several threads races

        self.auto_tokenizer = AutoTokenizer
        barrier = threading.Barrier(self.threads)

        def warm():
            self._tokenizer()
            barrier.wait(120)

        for fut in [self.pool.submit(warm) for _ in range(self.threads)]:
            fut.result()

    def encode(self, query, documents):
        texts = [prompt_template(query, d) for d in documents]
        return self._tokenizer()(texts, truncation=True, max_length=self.max_length)["input_ids"]

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Worker:
    def __init__(self, idx, bdf):
        self.idx, self.bdf = idx, bdf
        self.state = "starting"  # starting | ok | failed
        self.error = None
        self.proc = self.task_q = None
        self.outstanding = 0  # pairs submitted and not answered
        self.info = {}
        self.last_batch = None


class Runtime:
    """Workers, request futures and metrics; all state changes happen on the event loop."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_length = traced_max_length(cfg["max_length"])
        self.tok_state = "starting"
        self.error = None
        self.encoder = Encoder(cfg["tokenizer_threads"], self.max_length)
        self.workers = [Worker(i, bdf) for i, bdf in enumerate(cfg["chips"])]
        self.result_q = self.loop = self.collector = self.watchdog = None
        self.pending = {}  # id -> (future, worker idx, t_submit monotonic, pairs)
        self.next_id = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.counts = {"ok": 0, "422": 0, "5xx": 0}
        self.mid_serving_captures = 0
        self.e2e = deque(maxlen=SAMPLES)
        self.encode_ms = deque(maxlen=SAMPLES)
        self.queue_wait = deque(maxlen=SAMPLES)
        self.device = deque(maxlen=SAMPLES)
        self.batch_tasks = deque(maxlen=SAMPLES)
        self.batch_pairs = deque(maxlen=SAMPLES)

    @property
    def status(self):
        if self.tok_state == "failed" or all(w.state == "failed" for w in self.workers):
            return "failed"
        if self.tok_state == "ok" and any(w.state == "ok" for w in self.workers):
            return "ok"
        return "starting"

    def load_tokenizer(self):
        try:
            self.encoder.load()
            self.tok_state = "ok"
            logger.info(f"tokenizers ready ({self.encoder.threads} threads), pairs truncated to {self.max_length}")
        except Exception as e:
            self.error = f"tokenizer failed: {e!r}"
            self.tok_state = "failed"
            logger.error(self.error)

    def start(self, loop):
        self.loop = loop
        ctx = mp.get_context("spawn")
        self.result_q = ctx.Queue()
        cfg = {k: self.cfg[k] for k in ("batch_wait_ms", "max_batch_pairs", "verbose")} | {
            "max_length": self.max_length
        }
        for w in self.workers:
            w.task_q = ctx.Queue()
            w.proc = ctx.Process(
                target=worker_main,
                args=(w.idx, w.bdf, cfg, w.task_q, self.result_q),
                name=f"rerank-{w.idx}",
                daemon=True,
            )
            w.proc.start()
        self.collector = threading.Thread(target=self._collect, name="result-collector", daemon=True)
        self.collector.start()
        self.watchdog = loop.create_task(self._watch())

    def _collect(self):
        while True:
            msg = self.result_q.get()
            if msg is None:
                break
            self.loop.call_soon_threadsafe(self._on_message, msg)

    def _on_message(self, msg):
        w = self.workers[msg["chip"]]
        kind = msg["type"]
        if kind == "result":
            entry = self.pending.pop(msg["id"], None)
            if entry is None:  # failed already (watchdog / worker failure)
                return
            fut, _, _, pairs = entry
            w.outstanding -= pairs
            self.mid_serving_captures += msg.get("new_traces", 0)
            if msg["ok"] and msg["timing"]["batch_id"] != w.last_batch:
                # one count per batch: the first result of each batch (no per-batch set to grow)
                w.last_batch = msg["timing"]["batch_id"]
                self.batch_tasks.append(msg["timing"]["batch_tasks"])
                self.batch_pairs.append(msg["timing"]["batch_pairs"])
            if not fut.done():
                fut.set_result(msg)
        elif kind == "ready":
            if w.state == "starting":
                w.state, w.info = "ok", msg
                if msg["max_length"] != self.max_length:
                    logger.warning(f"worker {w.idx} traced max_length {msg['max_length']} != server {self.max_length}")
                logger.info(f"worker {w.idx} ({w.bdf}) ready after {msg['warmup_s']} s, {msg['traces']} traces")
        elif kind == "failed":
            last = next((l for l in reversed(msg["error"].splitlines()) if l.strip()), "unknown error")
            logger.error(msg["error"])
            self._fail_worker(w, f"worker {w.idx} ({w.bdf}) failed: {last.strip()}")

    def _fail_worker(self, w, error):
        if w.state == "failed":
            return
        w.state, w.error = "failed", error
        logger.error(error)
        for rid in [rid for rid, (_, idx, _, _) in self.pending.items() if idx == w.idx]:
            fut = self.pending.pop(rid)[0]
            if not fut.done():
                fut.set_exception(HTTPException(503, error.splitlines()[0]))
        w.outstanding = 0
        if self.status == "failed":
            self.error = error.splitlines()[0]

    async def _watch(self):
        while True:
            await asyncio.sleep(1.0)
            now = time.monotonic()
            for w in self.workers:
                if w.state == "failed":
                    continue
                if w.proc is not None and not w.proc.is_alive():
                    self._fail_worker(w, f"worker {w.idx} ({w.bdf}) process exited with code {w.proc.exitcode}")
                    continue
                if w.state != "ok":
                    continue
                oldest = min((t for _, idx, t, _ in self.pending.values() if idx == w.idx), default=None)
                if oldest is not None and now - oldest > self.cfg["hang_timeout_s"]:
                    # the process is alive but answers nothing: a device hang; the chip may need a reset
                    self._fail_worker(
                        w, f"worker {w.idx} ({w.bdf}) gave no result for {now - oldest:.0f} s (device hang?)"
                    )

    async def submit(self, ids):
        """Runs on the event loop after tokenizing, so a worker that failed meanwhile is seen here (no lost
        future): pick the live worker with the fewest outstanding pairs, or 503."""
        live = [w for w in self.workers if w.state == "ok"]
        if not live:
            raise HTTPException(503, self.error or "no worker ready")
        w = min(live, key=lambda w: w.outstanding)
        fut = self.loop.create_future()
        self.next_id += 1
        rid = self.next_id
        self.pending[rid] = (fut, w.idx, time.monotonic(), len(ids))
        w.outstanding += len(ids)
        t_submit = time.time()
        w.task_q.put({"id": rid, "ids": ids, "t_submit": t_submit})
        msg = await fut
        if not msg["ok"]:
            raise HTTPException(msg["status"], msg["detail"])
        self.queue_wait.append((msg["timing"]["t_batch_start"] - t_submit) * 1000)
        self.device.append(msg["timing"]["device_ms"])
        return msg["logits"]

    def stop(self):
        self.encoder.shutdown()
        if self.watchdog is not None:
            self.watchdog.cancel()
        for w in self.workers:
            if w.proc is not None:
                w.task_q.put(None)
        for w in self.workers:
            if w.proc is None:
                continue
            w.proc.join(30)
            if w.proc.is_alive():
                w.proc.terminate()  # a hung device close would block the exit forever
                w.proc.join(3)
            if w.proc.is_alive():
                w.proc.kill()
                w.proc.join()
        if self.result_q is not None:
            self.result_q.put(None)
            self.collector.join(5)
        for q in [w.task_q for w in self.workers] + [self.result_q]:
            if q is not None:
                q.cancel_join_thread()
                q.close()
        for w in self.workers:
            w.task_q = None
        self.result_q = None
        gc.collect()

    def stats(self):
        return {
            "requests": dict(self.counts),
            "latency_ms": percentiles(self.e2e),
            "encode_ms": percentiles(self.encode_ms),
            "queue_wait_ms": percentiles(self.queue_wait),
            "device_ms": percentiles(self.device),
            "batch_tasks": _histogram(self.batch_tasks),
            "batch_pairs": _histogram(self.batch_pairs),
            "mid_serving_captures": self.mid_serving_captures,
        }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
def _check_model(model: Optional[str]) -> Optional[str]:
    """None (vLLM makes ``model`` optional) means the served model; any other name is a 422."""
    if model is not None and model not in (MODEL_ID, SHORT_NAME):
        raise ValueError(f"model {model!r} is not served here; use one of {sorted((MODEL_ID, SHORT_NAME))}")
    return model


class RerankDocument(BaseModel):
    text: str


class RerankRequest(BaseModel):
    """POST /v1/rerank (vLLM / Cohere shape). ``top_n`` 0 or omitted returns every document."""

    model: Optional[str] = None
    query: str
    documents: List[Union[str, RerankDocument]] = Field(min_length=1, max_length=MAX_DOCUMENTS)
    top_n: Optional[int] = Field(default=None, ge=0)
    return_documents: bool = False

    _validate_model = field_validator("model")(_check_model)


class ScoreRequest(BaseModel):
    """POST /v1/score (vLLM shape): text_1 is the query, text_2 one document or a list of 1 to 1000."""

    model: Optional[str] = None
    text_1: str
    text_2: Union[str, List[str]]

    _validate_model = field_validator("model")(_check_model)

    @field_validator("text_2")
    @classmethod
    def _document_count(cls, v):
        if isinstance(v, list) and not 1 <= len(v) <= MAX_DOCUMENTS:
            raise ValueError(f"text_2 must hold 1 to {MAX_DOCUMENTS} documents, got {len(v)}")
        return v


def create_app(config=None, start_worker=True) -> FastAPI:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rt = Runtime(cfg)

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        loading = loop.run_in_executor(None, rt.load_tokenizer)  # the server accepts connections meanwhile
        if start_worker:
            rt.start(loop)
        try:
            yield
        finally:
            await loading
            for rid in list(rt.pending):
                fut = rt.pending.pop(rid)[0]
                if not fut.done():
                    fut.set_exception(HTTPException(503, "server shutting down"))
            await loop.run_in_executor(None, rt.stop)

    app = FastAPI(title=f"{MODEL_ID} rerank server", lifespan=lifespan)
    app.state.runtime = rt

    @app.exception_handler(RequestValidationError)
    async def count_422(request: Request, exc: RequestValidationError):
        rt.counts["422"] += 1
        return await request_validation_exception_handler(request, exc)

    def relevance(logit):
        return relevance_score(logit) if cfg["score"] == "sigmoid" else logit

    async def logits_for(query, docs):
        """Tokenize in the pool, split into --chunk pairs per task, score; 503 while starting or after failure."""
        if rt.status != "ok":
            rt.counts["5xx"] += 1
            raise HTTPException(503, "model is starting" if rt.status == "starting" else rt.error)
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        ids = await loop.run_in_executor(rt.encoder.pool, rt.encoder.encode, query, docs)
        rt.encode_ms.append((time.perf_counter() - t0) * 1000)
        chunks = [ids[i : i + cfg["chunk"]] for i in range(0, len(ids), cfg["chunk"])]
        try:
            parts = await asyncio.gather(*(rt.submit(c) for c in chunks))
        except HTTPException:
            rt.counts["5xx"] += 1
            raise
        rt.counts["ok"] += 1
        rt.e2e.append((time.perf_counter() - t0) * 1000)
        return [s for p in parts for s in p], sum(len(x) for x in ids)

    @app.post("/v1/rerank")
    @app.post("/rerank")
    async def rerank(req: RerankRequest):
        docs = [d if isinstance(d, str) else d.text for d in req.documents]
        logits, tokens = await logits_for(req.query, docs)
        order = sorted(range(len(docs)), key=lambda i: -logits[i])
        if req.top_n:
            order = order[: req.top_n]
        results = []
        for i in order:
            r = {"index": i, "relevance_score": relevance(logits[i]), "logit": logits[i]}
            if req.return_documents:
                r["document"] = {"text": docs[i]}
            results.append(r)
        return {
            "id": f"rerank-{uuid.uuid4().hex}",
            "model": req.model or MODEL_ID,
            "usage": {"total_tokens": tokens},
            "results": results,
        }

    @app.post("/v1/score")
    @app.post("/score")
    async def score(req: ScoreRequest):
        docs = [req.text_2] if isinstance(req.text_2, str) else list(req.text_2)
        logits, tokens = await logits_for(req.text_1, docs)
        return {
            "id": f"score-{uuid.uuid4().hex}",
            "object": "list",
            "model": req.model or MODEL_ID,
            "data": [{"index": i, "object": "score", "score": relevance(s)} for i, s in enumerate(logits)],
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    @app.get("/health")
    async def health():
        workers = []
        for w in rt.workers:
            row = {"chip": w.bdf, "state": w.state, "outstanding_pairs": w.outstanding}
            if w.error:
                row["error"] = w.error.splitlines()[0]
            workers.append(row)
        body = {"status": rt.status, "tokenizer": rt.tok_state, "workers": workers}
        if rt.status != "ok":
            return JSONResponse(body, status_code=503)
        ready = next(w.info for w in rt.workers if w.state == "ok")
        return {
            **body,
            "model": MODEL_ID,
            "revision": REVISION,
            "max_length": rt.max_length,
            "traces": ready["traces"],
            "warmup_s": ready["warmup_s"],
            "max_batch_pairs": cfg["max_batch_pairs"],
            "batch_wait_ms": cfg["batch_wait_ms"],
            "chunk": cfg["chunk"],
            "hang_timeout_s": cfg["hang_timeout_s"],
        }

    @app.get("/v1/models")
    async def list_models():
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}

    @app.get("/metrics")
    async def metrics(reset: bool = False):
        out = rt.stats()
        if reset:
            rt.reset_metrics()
        return out

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=f"Rerank server for {MODEL_ID}, one worker per chip.")
    p.add_argument("--host", default=DEFAULT_CONFIG["host"], help="bind address (default 127.0.0.1)")
    p.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"])
    p.add_argument(
        "--chips", default=os.environ.get("RR_CHIPS", "0"), help="comma list of chip PCI addresses, one worker each"
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_CONFIG["max_length"],
        help="tokens per pair, rounded down to a traced bucket",
    )
    p.add_argument(
        "--chunk",
        type=int,
        default=DEFAULT_CONFIG["chunk"],
        help="pairs per task; a request's chunks spread over workers",
    )
    p.add_argument(
        "--max-batch-pairs",
        type=int,
        default=DEFAULT_CONFIG["max_batch_pairs"],
        help="pairs per score_ids call (hard cap; a single larger chunk runs alone)",
    )
    p.add_argument(
        "--batch-wait-ms",
        type=float,
        default=DEFAULT_CONFIG["batch_wait_ms"],
        help="wait for more tasks after the first (0 = take only what is queued)",
    )
    p.add_argument(
        "--hang-timeout-s",
        type=float,
        default=DEFAULT_CONFIG["hang_timeout_s"],
        help="fail a worker whose oldest task is older than this",
    )
    p.add_argument("--tokenizer-threads", type=int, default=DEFAULT_CONFIG["tokenizer_threads"])
    p.add_argument(
        "--score",
        choices=("sigmoid", "raw"),
        default=DEFAULT_CONFIG["score"],
        help="relevance_score = sigmoid(logit) or the logit",
    )
    p.add_argument("--keep-alive-s", type=float, default=DEFAULT_CONFIG["keep_alive_s"], help="HTTP keep-alive timeout")
    p.add_argument("--verbose", action="store_true", help="log one line per batch")
    args = p.parse_args(argv)
    try:
        traced_max_length(args.max_length)
    except ValueError as e:
        p.error(str(e))
    return args


def main(argv=None):
    import uvicorn

    args = parse_args(argv)
    cfg = {
        "host": args.host,
        "port": args.port,
        "chips": args.chips.split(","),
        "max_length": args.max_length,
        "chunk": args.chunk,
        "max_batch_pairs": args.max_batch_pairs,
        "batch_wait_ms": args.batch_wait_ms,
        "hang_timeout_s": args.hang_timeout_s,
        "tokenizer_threads": args.tokenizer_threads,
        "score": args.score,
        "keep_alive_s": args.keep_alive_s,
        "verbose": args.verbose,
    }
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    uvicorn.run(
        create_app(cfg),
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
        access_log=False,
        timeout_keep_alive=args.keep_alive_s,
    )


if __name__ == "__main__":
    main()
