# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Ashai's FastAPI embedding-server template (via Gio's BGE-M3 copy, gtobarTT/bge_m3_optimized_demo
models/demos/wormhole/bge_m3/demo/serve.py) with the device worker swapped to google/embeddinggemma-2
(EmbeddingGemma2ForEmbedding, 1x1 mesh like tt-media-server). Pool, scheduler, tokenization and HTTP layer are the
template's, with two changes: --max-batch N (texts per forward, emb_batcher.collect_batch) and, from Aniruddha's
strands-decider server (atupe/strands-decider-fastapi @ c83659abfbe, 2026-11-02), non-blocking start-up (503
"starting"), worker-failure and device-deadline watchdog (503 to pending requests, /health failed), a thread-safe
tokenizer pool (shared backend encode, tested), validation before tokenizing (422), 127.0.0.1 by default and a
join/terminate/kill shutdown. CPU tests with a stub worker: test_emb_serve.py.

Lightweight OpenAI-compatible embedding server for BGE-M3 (single chip / DP=N).

A thin async **FastAPI + uvicorn** layer on top of resident chip workers, each
running one resident ``BgeM3ForEmbeddingOptimized`` encoder (B1, ISL 512, CLS
pooling, bfloat8_b, trace replay) pinned to its own CPU core(s). The HTTP
server's event loop and thread-pool run on the *remaining* cores, so request
handling never competes with the device workers.

This avoids vLLM entirely (and its multiprocess IPC tax): concurrency comes from
the chips running in parallel — the scheduler hands each input to the next idle
chip, so a batched request (``input: [...]``) fans out across all free chips.

Architecture adapted from
``models/demos/blackhole/pplx_embed_0_6b/demo/serve.py``; the device-worker guts
are replaced with the BGE-M3 encoder path (no kv-cache / page-table; a single
``model.forward(input_ids) -> [B, HIDDEN]``).

Run
---
    python models/demos/wormhole/bge_m3/demo/serve.py --dp 1

Then (OpenAI-compatible)::

    curl http://localhost:8000/v1/embeddings \\
      -H 'Content-Type: application/json' \\
      -d '{"model": "BAAI/bge-m3", "input": "hello world"}'
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import multiprocessing as mp
import os
import queue
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

MODEL_ID = "google/embeddinggemma-2"
REVISION = os.environ.get("EMBEDDING_GEMMA2_REVISION", "914f7f89142e33e77833254d9c9b90c3cef7303b")
EMBED_DIM = 768
MAX_SEQ_LEN = 8192
# --chips: PCI addresses, one per worker (worker i opens chips[i]); default "0".
CHIPS = os.environ.get("EMB_CHIPS", "0").split(",")


# --------------------------------------------------------------------------- #
# CPU-core affinity helpers (SMT-aware) — model-agnostic, from pplx serve.py.
# --------------------------------------------------------------------------- #
def _smt_cores(phys_ids, n_phys):
    """Expand physical core ids to all their SMT-sibling logical cpu ids.

    On a 2-threads/core host, physical core ``p`` owns logical cpus ``p`` and
    ``p + n_phys``.
    """
    cores = set()
    for p in phys_ids:
        cores.add(int(p))
        cores.add(int(p) + n_phys)
    return cores


def compute_worker_cores(chip_id, n_workers, cpw, server_phys=0, total_logical=None):
    """Logical-cpu affinity set for worker ``chip_id`` (or ``None`` => OS-float)."""
    total_logical = total_logical or os.cpu_count() or 64
    n_phys = max(1, total_logical // 2)
    pool_phys = max(1, n_phys - max(0, int(server_phys)))
    if cpw and cpw > 0:
        phys = [(chip_id * cpw + i) % pool_phys for i in range(cpw)]
        return _smt_cores(phys, n_phys)
    if server_phys and server_phys > 0:
        return _smt_cores(range(pool_phys), n_phys)
    return None  # everything floats (OS-scheduled)


def _fast_encode(tokenizer, text, max_len):
    """Tokenize ``text`` to a list of ids, capped at ``max_len``.

    Prefers the fast Rust backend's ``encode(text).ids`` (skips the slower HF
    Python ``__call__``/``encode`` BatchEncoding bookkeeping); falls back to
    ``tokenizer.encode``.
    """
    if not text:
        return []
    raw = getattr(tokenizer, "backend_tokenizer", None)
    if raw is not None:
        ids = raw.encode(text).ids
        if len(ids) > max_len:
            ids = ids[:max_len]
        return ids
    return tokenizer.encode(text, truncation=True, max_length=max_len)


# --------------------------------------------------------------------------- #
# Device worker: one resident encoder per chip.
# --------------------------------------------------------------------------- #
def _stub_model(spec):
    """CPU stand-in for the chip (EMB_STUB_WORKER, tests only): 'ok', 'slow_start:S', 'die_after:N', 'hang_after:N'.
    Row r of the output is a 768-vector whose element 0 is the row's token count, so callers can check ordering."""
    mode, _, arg = spec.partition(":")
    if mode == "slow_start":
        time.sleep(float(arg))
    calls = {"n": -1}  # the warmup forward does not count

    def forward(id_lists):
        calls["n"] += 1
        if mode == "die_after" and calls["n"] > int(arg):
            os._exit(3)
        if mode == "hang_after" and calls["n"] > int(arg):
            while True:  # a device that never finishes
                time.sleep(3600)
        time.sleep(0.002)
        out = np.zeros((len(id_lists), EMBED_DIM), dtype=np.float32)
        for r, ids in enumerate(id_lists):
            out[r, 0] = len(ids)
        return out

    return forward


def _bge_worker(chip_id, args, normalize, task_q, result_q, ctrl_q, worker_cores):
    """Single chip: build a resident EmbeddingGemma2ForEmbedding, serve from ``task_q``.

    Protocol:
      * ``ctrl_q``: ``(chip_id, "ready", build_s)`` after warmup; ``(chip_id, "failed", error)`` on any exception,
        during warmup or while serving (the server fails its pending requests on it).
      * For each ``(idx, payload)`` task, puts ``(idx, chip_id, emb_or_None, steps_or_None, n_tokens)`` on
        ``result_q``; ``payload`` is a token-id list (server-side tokenize) or raw text (worker-side tokenize).
      * A ``None`` task terminates the worker.
    """
    os.environ["TT_VISIBLE_DEVICES"] = CHIPS[chip_id]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if worker_cores:
        try:
            os.sched_setaffinity(0, set(worker_cores))
        except (OSError, AttributeError):
            pass
    from emb_batcher import collect_batch, right_pad

    device = None
    stub = os.environ.get("EMB_STUB_WORKER")
    try:
        t_build0 = time.perf_counter()
        if stub:
            run_rows = _stub_model(stub)
        else:
            import torch

            import ttnn
            from models.demos.embedding_gemma2.demo.generator_vllm import EmbeddingGemma2ForEmbedding

            device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=512 * 1024 * 1024)
            # The constructor compiles and captures a trace for every served (batch, bucket) shape; forward() pads
            # every request to one of them, so serving never compiles or captures (checked in test_emb_serve.py).
            model = EmbeddingGemma2ForEmbedding(device=device, max_batch_size=32, max_seq_len=MAX_SEQ_LEN, model_name=MODEL_ID)

            def run_rows(id_lists):
                input_ids, mask = right_pad(id_lists)
                out = model.forward(torch.tensor(input_ids), attention_mask=torch.tensor(mask))  # [B, HIDDEN], unit L2
                ttnn.synchronize_device(device)
                return out.float().numpy().astype(np.float32)

        tokenizer = None
        if not args.server_tokenize:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

        run_rows([[2, 3, 4, 5]])  # first replay outside the request path
        ctrl_q.put((chip_id, "ready", time.perf_counter() - t_build0))

        def get(timeout):
            if timeout is None:
                return task_q.get()
            if timeout == 0:
                return task_q.get_nowait()
            return task_q.get(timeout=timeout)

        max_texts = max(1, int(args.max_batch))
        wait_s = max(0.0, float(args.batch_wait_ms)) / 1000.0
        batch_id = 0
        while True:
            tasks, stop = collect_batch(get, wait_s, max_texts)
            t_recv = time.perf_counter()
            rows = []
            for idx, payload in tasks:
                ids = _fast_encode(tokenizer, payload, MAX_SEQ_LEN) if isinstance(payload, str) else payload
                if ids:
                    rows.append((idx, ids))
                else:
                    result_q.put((idx, chip_id, None, None, 0))
            if rows:
                batch_id += 1
                if os.environ.get("EMB_BATCH_LOG"):
                    print(f"[batch] n={len(rows)} L={max(len(ids) for _, ids in rows)} t={time.time():.3f}", file=sys.stderr, flush=True)
                t_prep = time.perf_counter()
                embs = run_rows([ids for _, ids in rows])
                t_done = time.perf_counter()
                for r, (idx, ids) in enumerate(rows):
                    steps = {
                        "device": (t_done - t_prep) * 1000.0,
                        "total": (t_done - t_recv) * 1000.0,
                        "t_recv": t_recv,
                        "t_prep": t_prep,
                        "t_done": t_done,
                        "batch": len(rows),
                        "batch_id": batch_id,
                        "first_in_batch": r == 0,
                    }
                    result_q.put((idx, chip_id, embs[r], steps, len(ids)))
            if stop:
                break
    except BaseException as exc:  # surface build and serve failures to the server
        try:
            ctrl_q.put((chip_id, "failed", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            if device is not None:
                ttnn.close_mesh_device(device)  # noqa: F821
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Async chip pool (from pplx serve.py), with the strands-decider server's start-up, failure and shutdown handling.
# --------------------------------------------------------------------------- #
class _TokenizerPool:
    """Thread pool that tokenizes on the server. One tokenizer shared by the threads: the server only calls the Rust
    backend's encode() (_fast_encode), which test_emb_serve.py checks is safe and identical under 16 threads; HF's
    __call__ (which raises "Already borrowed" when shared) is never used. Loaded once, in the background, so the
    event loop keeps answering /health during start-up (per-thread copies held the GIL for over 30 s)."""

    def __init__(self, threads: int):
        self.threads = threads
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="tok")
        self.tokenizer = None

    def load(self) -> None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        if getattr(tok, "backend_tokenizer", None) is None:
            raise RuntimeError("server-side tokenization needs a fast tokenizer (backend encode is the thread-safe path)")
        self.tokenizer = tok

    def encode(self, text: str, max_len: int):
        return _fast_encode(self.tokenizer, text, max_len)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


class AsyncEmbeddingPool:
    """Spawns N resident chip workers and schedules requests across them.

    State (event-loop owned): ``starting`` until every worker and the tokenizers are ready, then ``ok``; ``failed``
    for good when a worker reports an error, exits, or holds a batch longer than ``device_timeout_s`` (a device hang
    keeps the process alive, so liveness alone does not catch it). On failure every pending request gets 503."""

    def __init__(
        self,
        dp: int,
        normalize: bool = True,
        cores_per_worker: int = 0,
        server_phys: int = 0,
        server_tokenize: bool = True,
        max_batch: int = 1,
        batch_wait_ms: float = 0.0,
        device_timeout_s: float = 60.0,
    ):
        self.n = dp
        self.max_batch = max(1, max_batch)
        self.normalize = normalize
        self.isl = MAX_SEQ_LEN
        self.device_timeout_s = device_timeout_s
        self.cores_per_worker, self.server_phys = cores_per_worker, server_phys
        self.args = SimpleNamespace(
            dp=dp, max_length=self.isl, cq2=False, server_tokenize=server_tokenize, max_batch=self.max_batch, batch_wait_ms=batch_wait_ms
        )
        self.server_tokenize = server_tokenize
        n_tok_threads = max(4, 2 * server_phys) if server_phys > 0 else 8
        self.tokenizers = _TokenizerPool(n_tok_threads) if server_tokenize else None
        self.state, self.tok_state, self.error = "starting", ("starting" if server_tokenize else "ok"), None
        self.ready, self.failed = set(), {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.idle_chips: Optional[asyncio.Queue] = None
        self.failed_event: Optional[asyncio.Event] = None
        self.pending: dict = {}  # req_id -> future
        self.inflight: dict = {}  # req_id -> (chip, t_dispatch) for the device deadline
        self._req_id = 0
        self.metrics: deque = deque(maxlen=200_000)
        self.batch_sizes: deque = deque(maxlen=200_000)
        self.procs, self.task_qs = [], []
        self.result_q = self.ctrl_q = None

    @property
    def status(self) -> str:
        if "failed" in (self.state, self.tok_state):
            return "failed"
        return "ok" if self.state == self.tok_state == "ok" else "starting"

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Non-blocking: spawn the workers and the tokenizer load; the server accepts requests (503) meanwhile."""
        self.loop = loop
        self.idle_chips = asyncio.Queue()
        self.failed_event = asyncio.Event()
        ctx = mp.get_context("spawn")
        self.task_qs = [ctx.Queue() for _ in range(self.n)]
        self.result_q, self.ctrl_q = ctx.Queue(), ctx.Queue()
        for i in range(self.n):
            cores = compute_worker_cores(i, self.n, self.cores_per_worker, server_phys=self.server_phys)
            p = ctx.Process(
                target=_bge_worker, args=(i, self.args, self.normalize, self.task_qs[i], self.result_q, self.ctrl_q, cores), name=f"chip-{i}", daemon=True
            )
            p.start()
            self.procs.append(p)
        logger.info(f"Building {self.n} resident encoder(s) (one per chip); serving 503 until ready.")
        threading.Thread(target=self._collect, name="result-collector", daemon=True).start()
        threading.Thread(target=self._control, name="worker-control", daemon=True).start()
        if self.tokenizers is not None:
            threading.Thread(target=self._load_tokenizers, name="tok-load", daemon=True).start()
        self.watchdog = loop.create_task(self._watch())

    def _load_tokenizers(self) -> None:
        try:
            self.tokenizers.load()
            self.loop.call_soon_threadsafe(setattr, self, "tok_state", "ok")
            logger.info(f"Server-side tokenization ON ({self.tokenizers.threads} tokenizer threads, shared fast tokenizer).")
        except Exception as exc:
            self.loop.call_soon_threadsafe(self._fail, f"tokenizer load failed: {exc!r}")

    def _control(self) -> None:
        while True:
            msg = self.ctrl_q.get()
            if msg is None:
                break
            self.loop.call_soon_threadsafe(self._on_control, msg)

    def _on_control(self, msg) -> None:
        chip, kind, value = msg
        if kind == "ready":
            self.ready.add(chip)
            logger.info(f"  chip {chip:>2} ready (build {value:.0f}s) [{len(self.ready)}/{self.n}]")
            for _ in range(self.max_batch):  # one slot per request a chip may hold (1 = the template)
                self.idle_chips.put_nowait(chip)
            if len(self.ready) == self.n and self.state == "starting":
                self.state = "ok"
                logger.info(f"Embedding pool ready: {self.n}/{self.n} chips live.")
        else:
            self.failed[chip] = value
            self._fail(f"chip {chip} worker failed: {value[:300]}")

    def _fail(self, error: str) -> None:
        if self.state == "failed":
            return
        self.state, self.error = "failed", error
        logger.error(error)
        self.failed_event.set()
        for fut in self.pending.values():
            if not fut.done():
                fut.set_exception(HTTPException(status_code=503, detail=error))
        self.pending.clear()
        self.inflight.clear()

    async def _watch(self) -> None:
        """Every second: a worker that exited, or a batch held past the device deadline, fails the server. A hung
        worker is killed so the chip is released (killing the process frees it; no reset needed)."""
        while True:
            await asyncio.sleep(1.0)
            if self.state == "failed":
                continue
            for i, p in enumerate(self.procs):
                if not p.is_alive():
                    self._fail(f"chip {i} worker exited with code {p.exitcode}")
                    break
            if self.state != "failed" and self.inflight:
                chip, t_disp = min(self.inflight.values(), key=lambda v: v[1])
                held = time.perf_counter() - t_disp
                if held > self.device_timeout_s:
                    self._fail(f"chip {chip} device timeout: a batch has not finished after {held:.0f} s (hang)")
                    p = self.procs[chip]
                    p.terminate()
                    await asyncio.sleep(3)
                    if p.is_alive():
                        p.kill()

    def _collect(self) -> None:
        while True:
            item = self.result_q.get()
            if item is None:
                break
            self.loop.call_soon_threadsafe(self._on_result, item)

    def _on_result(self, item) -> None:
        req_id, chip, emb, steps, n = item
        self.inflight.pop(req_id, None)
        fut = self.pending.pop(req_id, None)
        if fut is not None and not fut.done():
            if emb is None:
                fut.set_exception(ValueError("empty input"))
            else:
                fut.set_result((emb, n, steps))
        if steps and steps.get("first_in_batch"):
            self.batch_sizes.append(steps["batch"])
        if self.state != "failed":
            self.idle_chips.put_nowait(chip)

    def _check(self) -> None:
        if self.status == "starting":
            raise HTTPException(status_code=503, detail="model is starting")
        if self.status == "failed":
            raise HTTPException(status_code=503, detail=self.error)

    async def _slot(self) -> int:
        get = asyncio.ensure_future(self.idle_chips.get())
        failed = asyncio.ensure_future(self.failed_event.wait())
        done, _ = await asyncio.wait({get, failed}, return_when=asyncio.FIRST_COMPLETED)
        if get in done:
            failed.cancel()
            if self.state == "failed":  # a slot freed by a dying chip must not be used
                raise HTTPException(status_code=503, detail=self.error)
            return get.result()
        get.cancel()
        raise HTTPException(status_code=503, detail=self.error)

    async def embed_one(self, text: str):
        t0 = time.perf_counter()
        if self.server_tokenize:
            payload = await self.loop.run_in_executor(self.tokenizers.pool, self.tokenizers.encode, text, self.isl)
            if not payload:
                raise ValueError("empty input")
        else:
            payload = text
        t_tok = time.perf_counter()
        self._check()  # the server may have failed while this request was in the tokenizer
        chip = await self._slot()
        req_id = self._req_id
        self._req_id += 1
        fut = self.loop.create_future()
        self.pending[req_id] = fut
        t_disp = time.perf_counter()
        self.inflight[req_id] = (chip, t_disp)
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
                "server_tok": (t_tok - t0) * 1000.0,
                "chip_wait": (t_disp - t_tok) * 1000.0,
            }
            t_recv, t_prep, t_done = steps.get("t_recv"), steps.get("t_prep"), steps.get("t_done")
            if t_recv and t_prep and t_done:
                sample["task_q"] = (t_recv - t_disp) * 1000.0
                sample["prepare"] = (t_prep - t_recv) * 1000.0
                sample["replay"] = (t_done - t_prep) * 1000.0
                sample["result"] = (t_end - t_done) * 1000.0
            self.metrics.append(sample)
        return emb, n, steps

    async def embed_batch(self, texts: List[str]):
        return await asyncio.gather(*(self.embed_one(t) for t in texts))

    def stats(self, reset: bool = False) -> dict:
        samples, sizes = list(self.metrics), list(self.batch_sizes)
        if reset:
            self.metrics.clear()
            self.batch_sizes.clear()
        out: dict = {"count": len(samples), "status": self.status}
        hist: dict = {}
        for s in sizes:
            hist[s] = hist.get(s, 0) + 1
        out["texts_per_forward"] = dict(sorted(hist.items()))
        if not samples:
            return out

        def pct(vals, p):
            vals = sorted(vals)
            k = (len(vals) - 1) * (p / 100.0)
            lo = int(k)
            hi = min(lo + 1, len(vals) - 1)
            return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

        for key in ("device", "worker_total", "server_wall", "overhead", "server_tok", "chip_wait", "task_q", "prepare", "replay", "result"):
            vals = [s[key] for s in samples if key in s]
            if vals:
                out[key] = {
                    "p50": round(pct(vals, 50), 3),
                    "p90": round(pct(vals, 90), 3),
                    "p99": round(pct(vals, 99), 3),
                    "max": round(max(vals), 3),
                    "mean": round(sum(vals) / len(vals), 3),
                }
        return out

    def close(self) -> None:
        """Stop sentinel, join 30 s, then terminate / kill a worker stuck in a hung device close."""
        if getattr(self, "watchdog", None) is not None:
            self.watchdog.cancel()
        if self.tokenizers is not None:
            self.tokenizers.shutdown()
        for q in self.task_qs:
            try:
                q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(30)
            if p.is_alive():
                p.terminate()
                p.join(3)
            if p.is_alive():
                p.kill()
                p.join()
        for q in (self.result_q, self.ctrl_q):
            if q is not None:
                try:
                    q.put(None)
                except Exception:
                    pass
        time.sleep(0.5)  # let the collector threads read their sentinels before the queues close
        for q in [*self.task_qs, self.result_q, self.ctrl_q]:
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self.task_qs, self.result_q, self.ctrl_q = [], None, None  # drop the SemLocks before interpreter exit
        import gc

        gc.collect()  # their finalizers run now, so the resource tracker reports no leaked semaphores


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

try:
    from fastapi.responses import ORJSONResponse as _DEFAULT_RESPONSE_CLASS  # uses orjson
except Exception:  # pragma: no cover - orjson optional
    _DEFAULT_RESPONSE_CLASS = JSONResponse


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID
    encoding_format: str = Field("float", description='"float" or "base64"')


_CONFIG = SimpleNamespace(
    dp=1, cores_per_worker=0, server_phys=4, server_tokenize=True, max_batch=1, batch_wait_ms=0.0, device_timeout_s=60.0, max_inputs=2048
)
_POOL: Optional[AsyncEmbeddingPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    loop = asyncio.get_running_loop()
    _POOL = AsyncEmbeddingPool(
        dp=_CONFIG.dp,
        normalize=True,
        cores_per_worker=_CONFIG.cores_per_worker,
        server_phys=_CONFIG.server_phys,
        server_tokenize=_CONFIG.server_tokenize,
        max_batch=_CONFIG.max_batch,
        batch_wait_ms=_CONFIG.batch_wait_ms,
        device_timeout_s=_CONFIG.device_timeout_s,
    )
    _POOL.start(loop)  # non-blocking: requests get 503 "starting" until the workers report ready
    try:
        yield
    finally:
        pool, _POOL = _POOL, None
        await loop.run_in_executor(None, pool.close)


app = FastAPI(title="google/embeddinggemma-2 embedding server (FastAPI template)", default_response_class=_DEFAULT_RESPONSE_CLASS, lifespan=lifespan)


def _encode_vector(emb: np.ndarray, fmt: str):
    if fmt == "base64":
        return base64.b64encode(np.ascontiguousarray(emb, dtype="<f4").tobytes()).decode("ascii")
    return np.asarray(emb, dtype=np.float32).tolist()


@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest):
    if _POOL is None:
        raise HTTPException(status_code=503, detail="model is starting")
    _POOL._check()
    # Validate on the event loop before any tokenizing, so a bad request costs no tokenizer or chip time.
    fmt = req.encoding_format
    if fmt not in ("float", "base64"):
        raise HTTPException(status_code=422, detail='encoding_format must be "float" or "base64"')
    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        raise HTTPException(status_code=422, detail="input must not be empty")
    if len(texts) > _CONFIG.max_inputs:
        raise HTTPException(status_code=422, detail=f"at most {_CONFIG.max_inputs} inputs per request")
    if any(not t for t in texts):
        raise HTTPException(status_code=422, detail="every input must be a non-empty string")

    try:
        results = await _POOL.embed_batch(texts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

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
    if _POOL is None or _POOL.status == "starting":
        return JSONResponse({"status": "starting", "chips_ready": len(_POOL.ready) if _POOL else 0, "chips_total": _CONFIG.dp}, status_code=503)
    if _POOL.status == "failed":
        return JSONResponse({"status": "failed", "error": _POOL.error}, status_code=503)
    return {"status": "ok", "chips_ready": len(_POOL.ready), "chips_total": _CONFIG.dp, "max_batch": _CONFIG.max_batch, "device_timeout_s": _CONFIG.device_timeout_s}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}


@app.get("/metrics")
async def metrics(reset: bool = False):
    """Per-request latency breakdown (ms): on-device replay vs. server overhead; texts-per-forward histogram."""
    if _POOL is None:
        raise HTTPException(status_code=503, detail="model is starting")
    return _POOL.stats(reset=reset)


# --------------------------------------------------------------------------- #
# Launch
# --------------------------------------------------------------------------- #
def _pin_server_cores(server_phys: int) -> None:
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
    parser = argparse.ArgumentParser(description="OpenAI-compatible embedding server (DP=N) for google/embeddinggemma-2.")
    parser.add_argument("--dp", type=int, default=1, help="Number of chips / workers (default 1).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1; 0.0.0.0 exposes it).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000).")
    parser.add_argument(
        "--cores-per-worker",
        type=int,
        default=0,
        help="Physical cores dedicated per worker (SMT-aware). 0 = OS-scheduled (recommended).",
    )
    parser.add_argument(
        "--server-phys",
        type=int,
        default=4,
        help="Physical cores reserved for the server's event loop / collector / HTTP threads. "
        "0 = off (single-chip default leaves the server unpinned via 0).",
    )
    tok_group = parser.add_mutually_exclusive_group()
    tok_group.add_argument(
        "--server-tokenize",
        dest="server_tokenize",
        action="store_true",
        default=True,
        help="Tokenize on the server's dedicated cores and ship token ids to workers (default).",
    )
    tok_group.add_argument(
        "--worker-tokenize",
        dest="server_tokenize",
        action="store_false",
        help="Tokenize inside each device worker.",
    )
    parser.add_argument("--chips", default=os.environ.get("EMB_CHIPS", "0"), help="Comma list of chip PCI ids, one per worker.")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=1,
        help="Texts per forward per chip. 1 (default) = the template: one in-flight request per chip. N > 1 lets "
        "up to N requests queue at a worker and runs them as one batched forward (model ladder 1/8/32).",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=float,
        default=0.0,
        help="With --max-batch > 1: how long a worker waits for more requests after the first (default 0 = "
        "take only what is already queued).",
    )
    parser.add_argument(
        "--device-timeout-s",
        type=float,
        default=60.0,
        help="A batch still running after this long marks the chip hung: pending requests get 503, /health fails "
        "and the worker is killed to release the chip (default 60).",
    )
    parser.add_argument("--max-inputs", type=int, default=2048, help="Most texts accepted in one request (422 above).")
    args = parser.parse_args()
    os.environ["EMB_CHIPS"] = args.chips  # spawned workers re-import this module and read it
    CHIPS[:] = args.chips.split(",")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    total = os.cpu_count() or 64
    n_phys = max(1, total // 2)
    server_phys = max(0, min(args.server_phys, n_phys - 1))

    _CONFIG.dp = args.dp
    _CONFIG.cores_per_worker = args.cores_per_worker
    _CONFIG.server_phys = server_phys
    _CONFIG.server_tokenize = args.server_tokenize
    _CONFIG.max_batch = args.max_batch
    _CONFIG.batch_wait_ms = args.batch_wait_ms
    _CONFIG.device_timeout_s = args.device_timeout_s
    _CONFIG.max_inputs = args.max_inputs
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
        access_log=False,
    )


if __name__ == "__main__":
    main()
