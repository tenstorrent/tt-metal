# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""System One HTTP server for the strands decider: one device worker process, cross-request batching.

Run from the repo root: python -m models.demos.strands_agent.serve.server [flags]"""

import argparse
import asyncio
import gc
import logging
import multiprocessing as mp
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from models.demos.strands_agent.serve.schema import SystemOneRequest
from models.demos.strands_agent.serve.worker import worker_main
from models.demos.strands_agent.tt.prompting import encode_request, validate_questions

logger = logging.getLogger("uvicorn.error")

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8000,
    "max_seq_len": 4096,
    "batch_wait_ms": 0.0,
    "max_batch_questions": 32,
    "trace": True,
    "warmup_eager": False,
    "traced_buckets_only": True,
    "keep_alive_s": 30.0,
    "model_name": "strands-decider-2B-qwen3.5-v1-2610",
    "verbose": False,
}
SAMPLES = 100_000
TOKENIZER_THREADS = 4


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


class Encoder:
    """Tokenizer pool of the server process: one tokenizer per pool thread (fast tokenizers are not thread safe)."""

    def __init__(self, threads=TOKENIZER_THREADS):
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="tok")
        self.threads = threads
        self.local = threading.local()
        self.checkpoint = self.auto_tokenizer = self.max_length = None

    def _tokenizer(self):
        tok = getattr(self.local, "tokenizer", None)
        if tok is None:
            tok = self.local.tokenizer = self.auto_tokenizer.from_pretrained(self.checkpoint)
        return tok

    def load(self, max_seq_len):
        """Blocking: resolve the checkpoint, read the config and load a tokenizer in every pool thread."""
        from transformers import AutoTokenizer  # imported here: a first import from several threads races

        from models.demos.strands_agent.tt.head import load_config
        from models.demos.strands_agent.tt.weight_mapping import resolve_checkpoint

        self.auto_tokenizer = AutoTokenizer
        self.checkpoint = resolve_checkpoint()
        self.max_length = min(load_config(self.checkpoint).max_length, max_seq_len)
        barrier = threading.Barrier(self.threads)

        def warm():
            self._tokenizer()
            barrier.wait(60)

        for fut in [self.pool.submit(warm) for _ in range(self.threads)]:
            fut.result()

    def encode(self, state, questions):
        """Runs in a pool thread; ValueError for a request the prompt cannot hold."""
        return encode_request(self._tokenizer(), state, questions, self.max_length)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class Runtime:
    """Worker process, request futures and metrics; all state changes happen on the event loop."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "starting"  # worker: starting | ok | failed
        self.tok_state = "starting"  # tokenizer and config: starting | ok | failed
        self.encoder = Encoder()
        self.info = {}
        self.error = None
        self.proc = self.task_q = self.result_q = self.loop = self.collector = None
        self.pending = {}
        self.next_id = 0
        self.reset_metrics()

    def reset_metrics(self):
        self.counts = {"ok": 0, "422": 0, "5xx": 0}
        self.e2e = deque(maxlen=SAMPLES)
        self.encode = deque(maxlen=SAMPLES)
        self.seen_batches = set()
        self.queue_wait = deque(maxlen=SAMPLES)
        self.device = deque(maxlen=SAMPLES)
        self.batch_requests = deque(maxlen=SAMPLES)
        self.batch_questions = deque(maxlen=SAMPLES)

    @property
    def status(self):
        if "failed" in (self.state, self.tok_state):
            return "failed"
        return "ok" if self.state == self.tok_state == "ok" else "starting"

    def load_tokenizer(self):
        try:
            self.encoder.load(self.cfg["max_seq_len"])
            self.tok_state = "ok"
            logger.info(f"tokenizer ready, max_length {self.encoder.max_length}")
        except Exception as e:
            self.error = f"tokenizer failed: {e!r}"
            self.tok_state = "failed"
            logger.error(self.error)

    def start(self, loop):
        self.loop = loop
        ctx = mp.get_context("spawn")
        self.task_q, self.result_q = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(
            target=worker_main, args=(self.cfg, self.task_q, self.result_q), name="decider-worker", daemon=True
        )
        self.proc.start()
        self.collector = threading.Thread(
            target=self._collect, args=(self.result_q,), name="result-collector", daemon=True
        )
        self.collector.start()
        self.watchdog = loop.create_task(self._watch())

    def _collect(self, result_q):
        while True:
            msg = result_q.get()
            if msg is None:
                break
            self.loop.call_soon_threadsafe(self._on_message, msg)

    def _on_message(self, msg):
        kind = msg["type"]
        if kind == "result":
            fut = self.pending.pop(msg["id"], None)
            if fut is not None and not fut.done():
                fut.set_result(msg)
        elif kind == "ready":
            self.info = msg
            self.state = "ok"
            logger.info(f"model ready after {msg['warmup_s']} s")
        elif kind == "failed":
            self._fail(f"model failed: {msg['error']}")

    def _fail(self, error):
        self.state, self.error = "failed", error
        logger.error(error)
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(HTTPException(503, error))
        self.pending.clear()

    async def _watch(self):
        while True:
            await asyncio.sleep(1.0)
            if self.proc is not None and not self.proc.is_alive() and self.state != "failed":
                self._fail(f"model failed: worker process exited with code {self.proc.exitcode}")

    def stop(self):
        self.encoder.shutdown()
        if self.proc is None:
            return
        self.watchdog.cancel()
        self.task_q.put(None)
        self.proc.join(30)
        if self.proc.is_alive():
            self.proc.terminate()  # a hung device close would block the exit forever
            self.proc.join(3)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join()
        self.result_q.put(None)
        self.collector.join(5)  # let it read the sentinel before the queue is closed
        self.task_q.cancel_join_thread()
        self.result_q.cancel_join_thread()
        self.task_q.close()
        self.result_q.close()
        # drop the queues so the SemLock finalizers run before the interpreter exits (no resource_tracker warning)
        self.task_q = self.result_q = None
        gc.collect()

    async def submit(self, qs):
        fut = self.loop.create_future()
        self.next_id += 1
        self.pending[self.next_id] = fut
        t_submit = time.time()
        self.task_q.put({"id": self.next_id, "qs": qs, "t_submit": t_submit})
        msg = await fut
        return {**msg, "t_submit": t_submit}

    def stats(self):
        return {
            "requests": dict(self.counts),
            "latency_ms": percentiles(self.e2e),
            "encode_ms": percentiles(self.encode),
            "queue_wait_ms": percentiles(self.queue_wait),
            "device_ms": percentiles(self.device),
            "batch_requests": dict(sorted(_histogram(self.batch_requests).items())),
            "batch_questions": dict(sorted(_histogram(self.batch_questions).items())),
        }


def _histogram(values):
    out = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def create_app(config=None, start_worker=True) -> FastAPI:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rt = Runtime(cfg)

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        loading = loop.run_in_executor(None, rt.load_tokenizer)
        if start_worker:
            rt.start(loop)
        try:
            yield
        finally:
            await loading
            await loop.run_in_executor(None, rt.stop)

    app = FastAPI(title="strands-decider System One", lifespan=lifespan)
    app.state.runtime = rt

    def reject(status, detail):
        rt.counts["422" if status == 422 else "5xx"] += 1
        raise HTTPException(status, detail)

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest):
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        if rt.status == "starting":
            reject(503, "model is starting")
        if rt.status == "failed":
            reject(503, rt.error)
        if req.images:
            reject(422, "this server has no vision tower")
        try:
            validate_questions(req.questions)
        except ValueError as e:
            reject(422, str(e))
        t_enc = time.perf_counter()
        try:
            qs = await loop.run_in_executor(rt.encoder.pool, rt.encoder.encode, req.state, req.questions)
        except ValueError as e:
            reject(422, str(e))
        encode_ms = (time.perf_counter() - t_enc) * 1000
        try:
            msg = await rt.submit(qs)
        except HTTPException as e:
            rt.counts["5xx"] += 1
            raise e
        if not msg["ok"]:
            reject(msg["status"], msg["detail"])
        ms = (time.perf_counter() - t0) * 1000
        timing = msg["timing"]
        rt.counts["ok"] += 1
        rt.e2e.append(ms)
        rt.encode.append(encode_ms)
        rt.queue_wait.append((timing["t_batch_start"] - msg["t_submit"]) * 1000)
        rt.device.append(timing["device_ms"])
        if timing["batch_id"] not in rt.seen_batches:
            rt.seen_batches.add(timing["batch_id"])
            rt.batch_requests.append(timing["batch_requests"])
            rt.batch_questions.append(timing["batch_questions"])
        return JSONResponse({"model": cfg["model_name"], **msg["result"], "latency_ms": round(ms, 2)})

    @app.get("/health")
    async def health():
        if rt.status == "starting":
            return JSONResponse({"status": "starting"}, status_code=503)
        if rt.status == "failed":
            return JSONResponse({"status": "failed", "error": rt.error}, status_code=503)
        return {
            "status": "ok",
            "model": cfg["model_name"],
            "max_length": rt.info["max_length"],
            "max_seq_len": rt.info["max_seq_len"],
            "device": "blackhole 1x1",
            "trace": cfg["trace"],
            "traced_buckets_only": cfg["traced_buckets_only"],
            "batch_wait_ms": cfg["batch_wait_ms"],
            "max_batch_questions": cfg["max_batch_questions"],
            "warmup_s": rt.info["warmup_s"],
            "trace_region_use": rt.info["trace_region_use"],
        }

    @app.get("/metrics")
    async def metrics(reset: bool = False):
        out = rt.stats()
        if reset:
            rt.reset_metrics()
        return out

    return app


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="System One server for the strands decider on one Blackhole chip.")
    p.add_argument("--host", default=DEFAULT_CONFIG["host"])
    p.add_argument("--port", type=int, default=DEFAULT_CONFIG["port"])
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_CONFIG["max_seq_len"])
    p.add_argument("--batch-wait-ms", type=float, default=DEFAULT_CONFIG["batch_wait_ms"])
    p.add_argument("--max-batch-questions", type=int, default=DEFAULT_CONFIG["max_batch_questions"])
    p.add_argument("--no-trace", action="store_true", help="run every forward eagerly")
    p.add_argument("--eager-buckets", action="store_true", help="also serve prompts above the traced buckets, eagerly")
    p.add_argument("--warmup-eager", action="store_true", help="also compile the eager buckets (needs --eager-buckets)")
    p.add_argument("--model-name", default=DEFAULT_CONFIG["model_name"])
    p.add_argument("--keep-alive-s", type=float, default=DEFAULT_CONFIG["keep_alive_s"], help="HTTP keep-alive timeout")
    p.add_argument("--verbose", action="store_true", help="log one line per batch")
    args = p.parse_args(argv)
    if args.warmup_eager and not args.eager_buckets:
        p.error("--warmup-eager needs --eager-buckets")
    return args


def main(argv=None):
    import uvicorn

    args = parse_args(argv)
    cfg = {
        "host": args.host,
        "port": args.port,
        "max_seq_len": args.max_seq_len,
        "batch_wait_ms": args.batch_wait_ms,
        "max_batch_questions": args.max_batch_questions,
        "trace": not args.no_trace,
        "traced_buckets_only": not args.eager_buckets,
        "warmup_eager": args.warmup_eager,
        "keep_alive_s": args.keep_alive_s,
        "model_name": args.model_name,
        "verbose": args.verbose,
    }
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
