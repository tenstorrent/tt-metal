# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight OpenAI-compatible embedding server for pplx-embed-v1-0.6B on a
Blackhole Galaxy (DP=32).

This puts a thin async **FastAPI + uvicorn** layer on top of the resident chip
workers from ``live_demo.py``.  Each of the 32 chips runs one resident encoder
pinned to its own CPU core (cores ``0..dp-1``); the HTTP server's event loop and
thread-pool run on the *remaining* cores (``dp..N``), so request handling never
competes with the device workers.

Worker configuration is **fixed** for lowest per-request latency at batch 1:

  * ISL = 512 (``max_length=512``) — bucketed traces at 128 / 256 / 512.
  * ``BucketedEncoder`` ON — every request replays the smallest length tier that
    fits (short text ~5.5ms@128 vs ~7.8ms@512).
  * masked-attn ON (real-token mean-pool + SDPA padding mask) — near-reference
    accuracy at any length.
  * batch 1 per worker, 1 in-flight request per chip, 1 command queue.

Concurrency comes from the 32 chips running in parallel: the scheduler hands each
input to the next idle chip, so up to 32 requests run at once and a batched
request (``input: [...]``) fans out across all free chips.

Run
---
    python models/demos/blackhole/pplx_embed_0_6b/demo/serve.py --dp 32

Then (OpenAI-compatible)::

    curl http://localhost:8000/v1/embeddings \\
      -H 'Content-Type: application/json' \\
      -d '{"model": "pplx-embed-v1-0.6b", "input": "hello world"}'
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
from loguru import logger

# Make ``models...`` importable when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.demos.blackhole.pplx_embed_0_6b.demo.live_demo import (  # noqa: E402
    _dp_worker,
    _fast_encode,
    _smt_cores,
    compute_worker_cores,
    maybe_patch_fastokens,
)

MODEL_ID = os.environ.get("HF_MODEL", "pplx-embed-v1-0.6b")
EMBED_DIM = 1024


# --------------------------------------------------------------------------- #
# Async chip pool
# --------------------------------------------------------------------------- #
class AsyncEmbeddingPool:
    """Spawns N resident chip workers and schedules requests across idle chips.

    Each worker is the same ``_dp_worker`` used by ``live_demo.py`` (ISL 512,
    BucketedEncoder, masked-attn, bs1, 1-CQ), pinned to a single core.  A
    background collector thread drains the shared mp result queue and resolves
    per-request asyncio futures, returning the chip to the idle pool.
    """

    def __init__(
        self,
        dp: int,
        normalize: bool = True,
        cores_per_worker: int = 0,
        server_phys: int = 0,
        server_tokenize: bool = True,
    ):
        self.n = dp
        self.normalize = normalize
        self.isl = 512
        args = SimpleNamespace(
            dp=dp,
            max_length=self.isl,
            mask=True,
            no_bucket=False,
            cq2=False,
        )
        # Server-side tokenization: tokenize on the HTTP layer's dedicated cores
        # (in a GIL-releasing thread pool) and ship token ids to workers, instead
        # of tokenizing inside each device worker where it contends with the 32
        # workers' tt-metal dispatch threads. Falls back to worker-side tokenize.
        self.server_tokenize = server_tokenize
        self.tokenizer = None
        self._tok_pool: Optional[ThreadPoolExecutor] = None
        if self.server_tokenize:
            maybe_patch_fastokens()
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            # Threads run on the server's pinned cores (inherited affinity); the HF
            # Rust tokenizer releases the GIL during ``encode`` so they parallelize.
            n_tok_threads = max(4, 2 * server_phys) if server_phys > 0 else 8
            self._tok_pool = ThreadPoolExecutor(max_workers=n_tok_threads, thread_name_prefix="tok")
            logger.info(f"Server-side tokenization ON ({n_tok_threads} tokenizer threads).")
        ctx = mp.get_context("spawn")
        self.task_qs = [ctx.Queue() for _ in range(self.n)]
        self.result_q = ctx.Queue()
        ready_q = ctx.Queue()
        self.procs = []
        for i in range(self.n):
            worker_cores = compute_worker_cores(i, self.n, cores_per_worker, server_phys=server_phys)
            p = ctx.Process(
                target=_dp_worker,
                args=(i, args, normalize, self.task_qs[i], self.result_q, ready_q, worker_cores),
                name=f"chip-{i}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        logger.info(f"Building {self.n} resident encoders (one per chip) — this takes ~1-2 min...")
        self.ready, self.failed = set(), {}
        for _ in range(self.n):
            msg = ready_q.get()
            if len(msg) == 3 and msg[1] is None:
                self.failed[msg[0]] = msg[2]
            else:
                self.ready.add(msg[0])
                logger.info(f"  chip {msg[0]:>2} ready (build {msg[1]:.0f}s) [{len(self.ready)}/{self.n}]")
        if self.failed:
            logger.error(f"{len(self.failed)} chip(s) failed to build:")
            for cid, err in sorted(self.failed.items()):
                logger.error(f"  chip {cid}: {err[:200]}")
        if not self.ready:
            raise RuntimeError("All DP workers failed to build.")
        logger.info(f"Embedding pool ready: {len(self.ready)}/{self.n} chips live.")

        # Async scheduling state — bound to the event loop in ``start()``.
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.idle_chips: Optional[asyncio.Queue] = None
        self.pending: dict = {}
        self._req_id = 0
        self._collector: Optional[threading.Thread] = None
        # Per-request timing samples (written only from the event loop).
        self.metrics: deque = deque(maxlen=200_000)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the pool to the running event loop and start the collector."""
        self.loop = loop
        self.idle_chips = asyncio.Queue()
        for chip in sorted(self.ready):
            self.idle_chips.put_nowait(chip)
        self._collector = threading.Thread(target=self._collect, name="result-collector", daemon=True)
        self._collector.start()

    def _collect(self) -> None:
        while True:
            item = self.result_q.get()
            if item is None:  # shutdown sentinel
                break
            req_id, chip, emb, steps, n = item
            fut = self.pending.pop(req_id, None)
            if fut is not None:
                self.loop.call_soon_threadsafe(self._resolve, fut, emb, n, steps)
            self.loop.call_soon_threadsafe(self.idle_chips.put_nowait, chip)

    @staticmethod
    def _resolve(fut: asyncio.Future, emb, n: int, steps) -> None:
        if fut.done():
            return
        if emb is None:
            fut.set_exception(ValueError("empty input"))
        else:
            fut.set_result((emb, n, steps))

    async def embed_one(self, text: str):
        """Embed one text on the next idle chip.

        Returns ``(emb_np_f32, n_tokens, steps)`` where ``steps`` is the worker's
        per-step device timing.  Records a timing sample so the server can report
        device-vs-overhead latency under load via ``/metrics``.
        """
        t0 = time.perf_counter()
        # Tokenize on the server's dedicated cores (off the contended worker
        # cores) before reserving a chip, so the chip is held only for device work.
        if self.server_tokenize:
            payload = await self.loop.run_in_executor(self._tok_pool, _fast_encode, self.tokenizer, text, self.isl)
            if not payload:
                raise ValueError("empty input")
        else:
            payload = text
        t_tok = time.perf_counter()

        chip = await self.idle_chips.get()
        req_id = self._req_id
        self._req_id += 1
        fut = self.loop.create_future()
        self.pending[req_id] = fut
        t_disp = time.perf_counter()
        self.task_qs[chip].put((req_id, payload))
        emb, n, steps = await fut
        t_end = time.perf_counter()
        wall = (t_end - t0) * 1000.0
        if steps is not None:
            worker_total = float(steps.get("total", 0.0))
            sample = {
                "device": float(steps.get("device", 0.0)),
                "worker_total": worker_total,
                "server_wall": wall,
                "overhead": wall - worker_total,
                "bucket": int(steps.get("bucket", 0)),
                # Server-side tokenization hop (0.0 when tokenizing in the worker).
                "server_tok": (t_tok - t0) * 1000.0,
                "chip_wait": (t_disp - t_tok) * 1000.0,
            }
            # Per-hop breakdown of the server_wall (all in ms), if the worker
            # reported cross-process timestamps:
            #   server_tok : tokenize on the server threadpool (server path only)
            #   chip_wait  : wait for an idle chip (queueing under load)
            #   task_q     : dispatch -> worker dequeues the task (mp.Queue in)
            #   prepare    : (tokenize, if worker-side) + build host input tensors
            #   replay     : on-device + D2H + host finalize (== worker_total)
            #   result     : worker enqueues result -> future resolved on event loop
            t_recv, t_prep, t_done = steps.get("t_recv"), steps.get("t_prep"), steps.get("t_done")
            if t_recv and t_prep and t_done:
                sample["task_q"] = (t_recv - t_disp) * 1000.0
                sample["prepare"] = (t_prep - t_recv) * 1000.0
                sample["replay"] = (t_done - t_prep) * 1000.0
                sample["result"] = (t_end - t_done) * 1000.0
            if "tok" in steps:
                sample["tok"] = float(steps["tok"])
                sample["build"] = float(steps["build"])
            self.metrics.append(sample)
        return emb, n, steps

    async def embed_batch(self, texts: List[str]):
        """Embed a list of texts, auto load-balanced across free chips."""
        return await asyncio.gather(*(self.embed_one(t) for t in texts))

    def stats(self, reset: bool = False) -> dict:
        """Percentile summary of recent per-request timings (ms)."""
        samples = list(self.metrics)
        if reset:
            self.metrics.clear()
        out: dict = {"count": len(samples)}
        if not samples:
            return out

        def pct(vals, p):
            vals = sorted(vals)
            if not vals:
                return None
            k = (len(vals) - 1) * (p / 100.0)
            lo = int(k)
            hi = min(lo + 1, len(vals) - 1)
            return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

        keys = (
            "device",
            "worker_total",
            "server_wall",
            "overhead",
            "server_tok",
            "chip_wait",
            "task_q",
            "prepare",
            "tok",
            "build",
            "replay",
            "result",
        )
        for key in keys:
            vals = [s[key] for s in samples if key in s]
            if not vals:
                continue
            out[key] = {
                "p50": round(pct(vals, 50), 3),
                "p90": round(pct(vals, 90), 3),
                "p99": round(pct(vals, 99), 3),
                "max": round(max(vals), 3),
                "mean": round(sum(vals) / len(vals), 3),
            }
        buckets: dict = {}
        for s in samples:
            buckets[s["bucket"]] = buckets.get(s["bucket"], 0) + 1
        out["buckets"] = buckets
        return out

    def close(self) -> None:
        if self._tok_pool is not None:
            self._tok_pool.shutdown(wait=False, cancel_futures=True)
        # Ask workers to stop, then force-kill promptly. Device teardown
        # (ttnn.close_device) can hang/SIGABRT under load, so we don't wait on
        # graceful joins — SIGKILL frees the chips cleanly for the next run.
        for c in sorted(self.ready):
            try:
                self.task_qs[c].put(None)
            except Exception:
                pass
        try:
            self.result_q.put(None)  # stop collector
        except Exception:
            pass
        deadline = time.time() + 3.0
        for p in self.procs:
            p.join(timeout=max(0.0, deadline - time.time()))
        for p in self.procs:
            if p.is_alive():
                p.terminate()
        time.sleep(0.5)
        for p in self.procs:
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

try:
    from fastapi.responses import ORJSONResponse as _JSONResponse  # uses orjson

    _DEFAULT_RESPONSE_CLASS = _JSONResponse
except Exception:  # pragma: no cover - orjson optional
    from fastapi.responses import JSONResponse as _DEFAULT_RESPONSE_CLASS


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID
    encoding_format: str = Field("float", description='"float" or "base64"')


# Module-level config, set by ``main`` before uvicorn imports the app.
_CONFIG = SimpleNamespace(dp=32, cores_per_worker=0, server_phys=4, server_tokenize=False)
_POOL: Optional[AsyncEmbeddingPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    loop = asyncio.get_running_loop()
    # Build the (blocking) pool off the event loop so startup stays responsive.
    _POOL = await loop.run_in_executor(
        None,
        lambda: AsyncEmbeddingPool(
            dp=_CONFIG.dp,
            normalize=True,
            cores_per_worker=_CONFIG.cores_per_worker,
            server_phys=_CONFIG.server_phys,
            server_tokenize=_CONFIG.server_tokenize,
        ),
    )
    _POOL.start(loop)
    try:
        yield
    finally:
        if _POOL is not None:
            _POOL.close()
            _POOL = None


app = FastAPI(title="pplx-embed-v1-0.6b server", default_response_class=_DEFAULT_RESPONSE_CLASS, lifespan=lifespan)


def _encode_vector(emb: np.ndarray, fmt: str):
    if fmt == "base64":
        return base64.b64encode(np.ascontiguousarray(emb, dtype="<f4").tobytes()).decode("ascii")
    return np.asarray(emb, dtype=np.float32).tolist()


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if _POOL is None:
        raise HTTPException(status_code=503, detail="pool not ready")
    fmt = req.encoding_format
    if fmt not in ("float", "base64"):
        raise HTTPException(status_code=400, detail='encoding_format must be "float" or "base64"')
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(status_code=400, detail="input must not be empty")

    try:
        results = await _POOL.embed_batch(texts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    data = []
    total_tokens = 0
    for i, (emb, n, _steps) in enumerate(results):
        total_tokens += int(n)
        data.append({"object": "embedding", "index": i, "embedding": _encode_vector(emb, fmt)})
    return {
        "object": "list",
        "data": data,
        "model": req.model or MODEL_ID,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    }


@app.get("/health")
async def health():
    ready = len(_POOL.ready) if _POOL is not None else 0
    total = _CONFIG.dp
    return {"status": "ok" if ready else "starting", "chips_ready": ready, "chips_total": total}


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}],
    }


@app.get("/metrics")
async def metrics(reset: bool = False):
    """Per-request latency breakdown (ms): on-device replay vs. server overhead.

    ``device`` is the per-chip prefill+norm+pool replay; ``worker_total`` adds
    H2D/D2H/host-finalize inside the worker; ``server_wall`` is dispatch->result
    measured on the event loop; ``overhead`` = server_wall - worker_total
    (mp.Queue transit + scheduling). Under full load ``device`` should stay flat.
    """
    if _POOL is None:
        raise HTTPException(status_code=503, detail="pool not ready")
    return _POOL.stats(reset=reset)


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
def _pin_server_cores(server_phys: int) -> None:
    """Reserve the top ``server_phys`` physical cores (both SMT siblings) for the
    server's event loop / collector / HTTP threads.

    ``server_phys <= 0`` leaves the server unpinned (workers OS-float across all
    cores) — lowest tail latency for the worker-bound default.
    """
    if not server_phys or server_phys <= 0:
        logger.info("Server unpinned (OS-scheduled).")
        return
    try:
        total = os.cpu_count() or 64
        n_phys = max(1, total // 2)
        cores = _smt_cores(range(n_phys - server_phys, n_phys), n_phys)
        os.sched_setaffinity(0, cores)
        logger.info(
            f"Server pinned to {server_phys} physical core(s): logical {sorted(cores)} "
            f"(workers use physical 0-{n_phys - server_phys - 1})."
        )
    except (OSError, AttributeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible embedding server (DP=N) for pplx-embed-v1-0.6b.")
    parser.add_argument("--dp", type=int, default=32, help="Number of chips / workers (default 32).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    parser.add_argument(
        "--cores-per-worker",
        type=int,
        default=0,
        help="Physical cores dedicated per worker (SMT-aware; both hyperthreads). "
        "0 (default) = OS-scheduled across the worker pool (recommended).",
    )
    parser.add_argument(
        "--server-phys",
        type=int,
        default=4,
        help="Physical cores (top of the range) reserved for the server's event "
        "loop / result collector / HTTP threads; workers OS-float on the rest. "
        "Default 4 — isolating the single server process from the 32 workers is "
        "the biggest scaling win (~+45%% throughput at high concurrency). 0 = off.",
    )
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--server-tokenize",
        dest="server_tokenize",
        action="store_true",
        default=None,
        help="Tokenize on the server's dedicated cores (one process) and ship token "
        "ids to workers. This is the only safe home for --fastokens (its internal "
        "thread pool oversubscribes if replicated across 32 workers). Default: ON "
        "when --fastokens is set, else OFF (worker-side).",
    )
    tok_group.add_argument(
        "--worker-tokenize",
        dest="server_tokenize",
        action="store_false",
        help="Tokenize inside each device worker (the original path).",
    )
    parser.add_argument(
        "--fastokens",
        action="store_true",
        help="EXPERIMENTAL, NOT recommended for DP=32. Patch HF AutoTokenizer with the "
        "crusoe/atero 'fastokens' Rust engine. It is faster in isolation but measured to "
        "REGRESS server throughput here (server-side loads the single-process bottleneck; "
        "per-worker oversubscribes the host). The default worker-side HF path is best. "
        "Implies --server-tokenize unless --worker-tokenize is given.",
    )
    args = parser.parse_args()

    # Default: server-side tokenization iff fastokens (keeps fastokens single-process;
    # plain HF on the small server pool can worsen the tail under heavy concurrency).
    if args.server_tokenize is None:
        args.server_tokenize = args.fastokens

    if args.fastokens and not args.server_tokenize:
        logger.warning(
            "--fastokens with --worker-tokenize: fastokens' internal thread pool runs "
            "in all 32 workers and oversubscribes the host (large tail under load). "
            "Prefer --server-tokenize (the default with --fastokens)."
        )
    if args.fastokens:
        os.environ["PPLX_FASTOKENS"] = "1"  # propagated to spawned workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    total = os.cpu_count() or 64
    n_phys = max(1, total // 2)
    server_phys = max(0, min(args.server_phys, n_phys - 1))

    _CONFIG.dp = args.dp
    _CONFIG.cores_per_worker = args.cores_per_worker
    _CONFIG.server_phys = server_phys
    _CONFIG.server_tokenize = args.server_tokenize
    _pin_server_cores(server_phys)

    import uvicorn

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        loop="uvloop",
        http="httptools",
        workers=1,
        log_level="info",
        # Per-request access logging runs on the single event-loop thread for every
        # request; disabling it removes that fixed cost from the HTTP critical path.
        access_log=False,
    )


if __name__ == "__main__":
    main()
