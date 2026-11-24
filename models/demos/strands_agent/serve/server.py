# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""System One HTTP server for the strands decider: cross-request batching, one device worker process per chip.

With --chips the server is data parallel: every worker pins one chip and takes batches from one shared task queue,
so an idle chip takes the next batch. The first worker starts alone and fills the weight and kernel caches; the
others start when it is ready.

With --frontends K the HTTP side runs in K processes that share the port (SO_REUSEPORT): one front end's event loop
saturates one core at several hundred requests per second, well below what 32 chips answer. Front end 0 owns the
workers and starts the other front ends; all of them put tasks on the one task queue and get their results on their
own result queue.

Run from the repo root: python -m models.demos.strands_agent.serve.server [flags]"""

import argparse
import asyncio
import gc
import logging
import multiprocessing as mp
import os
import socket
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
    "chips": None,  # None: one worker on the devices the process sees; else one worker per chip
    "tokenizer_threads": None,  # None: 4 for one worker, 16 for several
    "worker_threads": None,  # torch threads per device worker; None: torch's default for one worker, 1 for several
    "frontends": 1,  # HTTP front-end processes sharing the port; front end 0 owns the workers
}
SAMPLES = 100_000
TOKENIZER_THREADS = 4


def resolve_chips(spec):
    """--chips: "" or None (one worker, no pinning), "all" (every /dev/tenstorrent/N, as device indices), or a comma
    list of device indices or PCI addresses."""
    if not spec:
        return None
    if spec == "all":
        n = len([d for d in os.listdir("/dev/tenstorrent") if d.isdigit()])
        if n == 0:
            raise ValueError("--chips all: no /dev/tenstorrent/N devices")
        return [str(i) for i in range(n)]
    chips = [c.strip() for c in spec.split(",") if c.strip()]
    if len(set(chips)) != len(chips):
        raise ValueError(f"--chips lists a chip twice: {spec}")
    return chips


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
    """Worker processes, request futures and metrics of one front end; all state changes happen on the event loop.

    Front end 0 owns the task queue, one result queue per front end, the workers and the other front ends; front end
    k > 0 gets its two queues (queues=(task_q, result_q)) and learns worker state from the messages on its result
    queue."""

    def __init__(self, cfg, frontend=0, queues=None):
        self.cfg = cfg
        self.chips = cfg["chips"] or [None]
        self.frontend = frontend
        self.owner = frontend == 0
        self.state = "starting"  # workers: starting (none ready) | ok (at least one ready) | failed
        self.tok_state = "starting"  # tokenizer and config: starting | ok | failed
        several = len(self.chips) > 1 and cfg["frontends"] == 1
        threads = cfg["tokenizer_threads"] or (16 if several else TOKENIZER_THREADS)
        self.encoder = Encoder(threads)
        self.info = {}
        self.error = None
        self.procs = [None] * len(self.chips)
        self.ready = [False] * len(self.chips)
        self.frontend_procs = []
        self.result_qs = []
        self.task_q, self.result_q = queues or (None, None)
        self.loop = self.collector = None
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
        if self.owner:
            self.ctx = mp.get_context("spawn")
            self.task_q = self.ctx.Queue()
            self.result_qs = [self.ctx.Queue() for _ in range(self.cfg["frontends"])]
            self.result_q = self.result_qs[0]
            self._spawn(0)  # the others start when worker 0 is ready (it fills the weight and kernel caches)
            for fe in range(1, self.cfg["frontends"]):
                proc = self.ctx.Process(
                    target=frontend_main,
                    args=(self.cfg, fe, self.task_q, self.result_qs[fe]),
                    name=f"decider-frontend-{fe}",
                    daemon=True,
                )
                proc.start()
                self.frontend_procs.append(proc)
        self.collector = threading.Thread(
            target=self._collect, args=(self.result_q,), name="result-collector", daemon=True
        )
        self.collector.start()
        if self.owner:
            self.watchdog = loop.create_task(self._watch())

    def _spawn(self, worker):
        proc = self.ctx.Process(
            target=worker_main,
            args=(self.cfg, self.task_q, self.result_qs, worker, self.chips[worker]),
            name=f"decider-worker-{worker}",
            daemon=True,
        )
        proc.start()
        self.procs[worker] = proc

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
            worker = msg["worker"]
            self.ready[worker] = True
            self.info = self.info or msg
            if self.state != "failed":
                self.state = "ok"
            logger.info(f"worker {worker} ready after {msg['warmup_s']} s ({sum(self.ready)}/{len(self.ready)})")
            if self.owner and worker == 0 and self.state != "failed":
                for w in range(1, len(self.chips)):
                    self._spawn(w)
        elif kind == "failed":
            self._fail(f"model failed on worker {msg['worker']}: {msg['error']}")
        elif kind == "error":  # front end 0 found a dead worker or front end
            self._fail(msg["error"])

    def _fail(self, error):
        if self.owner:  # workers tell every front end about their own failures; process deaths only front end 0 sees
            for result_q in self.result_qs[1:]:
                result_q.put({"type": "error", "error": error})
        self.state, self.error = "failed", error
        logger.error(error)
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(HTTPException(503, error))
        self.pending.clear()

    async def _watch(self):
        while True:
            await asyncio.sleep(1.0)
            for worker, proc in enumerate(self.procs):
                if proc is not None and not proc.is_alive() and self.state != "failed":
                    self._fail(f"model failed: worker {worker} process exited with code {proc.exitcode}")
            for fe, proc in enumerate(self.frontend_procs, start=1):
                if not proc.is_alive() and self.state != "failed":
                    self._fail(f"front end {fe} process exited with code {proc.exitcode}")

    def stop(self):
        self.encoder.shutdown()
        if not self.owner:
            self.result_q.put(None)
            self.collector.join(5)
            self.task_q.cancel_join_thread()  # front end 0 closes the shared queues
            return
        for proc in self.frontend_procs:
            proc.terminate()  # SIGTERM: uvicorn shuts the front end down
        for proc in self.frontend_procs:
            proc.join(15)
            if proc.is_alive():
                proc.kill()
                proc.join()
        procs = [proc for proc in self.procs if proc is not None]
        if not procs:
            return
        self.watchdog.cancel()
        for _ in procs:
            self.task_q.put(None)  # one stop sentinel per worker
        deadline = time.monotonic() + 30
        for proc in procs:
            proc.join(max(0.1, deadline - time.monotonic()))
        for proc in procs:
            if proc.is_alive():
                proc.terminate()  # a hung device close would block the exit forever
                proc.join(3)
            if proc.is_alive():
                proc.kill()
                proc.join()
        self.result_q.put(None)
        self.collector.join(5)  # let it read the sentinel before the queue is closed
        for q in [self.task_q, *self.result_qs]:
            q.cancel_join_thread()
            q.close()
        # drop the queues so the SemLock finalizers run before the interpreter exits (no resource_tracker warning)
        self.task_q = self.result_q = None
        self.result_qs = []
        gc.collect()

    async def submit(self, qs):
        fut = self.loop.create_future()
        self.next_id += 1
        self.pending[self.next_id] = fut
        t_submit = time.time()
        self.task_q.put({"id": self.next_id, "qs": qs, "t_submit": t_submit, "fe": self.frontend})
        msg = await fut
        return {**msg, "t_submit": t_submit}

    def stats(self):
        return {
            "frontend": self.frontend,
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


def create_app(config=None, start_worker=True, frontend=0, queues=None) -> FastAPI:
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    rt = Runtime(cfg, frontend, queues)

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
            "device": f"blackhole 1x1 x {len(rt.chips)}",
            "workers": len(rt.chips),
            "workers_ready": sum(rt.ready),
            "frontends": cfg["frontends"],
            "chips": rt.chips if rt.cfg["chips"] else None,
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
    p = argparse.ArgumentParser(description="System One server for the strands decider on Blackhole chips.")
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
    p.add_argument(
        "--chips",
        default=os.environ.get("DECIDER_CHIPS", ""),
        help="data parallel: 'all' or a comma list of device indices / PCI addresses, one worker each "
        "(default: one worker on the devices the process sees; env DECIDER_CHIPS)",
    )
    p.add_argument(
        "--tokenizer-threads", type=int, default=None, help="tokenizer pool size (default 4, 16 with --chips)"
    )
    p.add_argument(
        "--worker-threads",
        type=int,
        default=None,
        help="torch threads per device worker (default: torch's default for one worker, 1 with --chips: the workers "
        "share the host's cores, and torch's default of one thread per core in every worker oversubscribes them)",
    )
    p.add_argument(
        "--frontends",
        type=int,
        default=1,
        help="HTTP front-end processes sharing the port (SO_REUSEPORT), for request rates one event loop cannot "
        "keep up with; they share all the workers (needs --chips)",
    )
    args = p.parse_args(argv)
    if args.warmup_eager and not args.eager_buckets:
        p.error("--warmup-eager needs --eager-buckets")
    if args.frontends < 1:
        p.error("--frontends must be >= 1")
    if args.frontends > 1 and not args.chips:
        p.error("--frontends needs --chips")
    return args


def main(argv=None):
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
        "chips": resolve_chips(args.chips),
        "tokenizer_threads": args.tokenizer_threads,
        "worker_threads": args.worker_threads,
        "frontends": args.frontends,
    }
    serve(cfg)


def serve(cfg, frontend=0, queues=None):
    """Run one front end; with several, every front end binds the port with SO_REUSEPORT."""
    import uvicorn

    config = uvicorn.Config(
        create_app(cfg, frontend=frontend, queues=queues),
        host=cfg["host"],
        port=cfg["port"],
        log_level="info",
        access_log=False,
        timeout_keep_alive=cfg["keep_alive_s"],
    )
    server = uvicorn.Server(config)
    if cfg["frontends"] == 1:
        server.run()
        return
    sock = socket.socket(socket.AF_INET6 if ":" in cfg["host"] else socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((cfg["host"], cfg["port"]))
    server.run(sockets=[sock])


def frontend_main(cfg, frontend, task_q, result_q):
    """Entry of front-end process k > 0 (spawned by front end 0)."""
    serve(cfg, frontend, (task_q, result_q))


if __name__ == "__main__":
    main()
