# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
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

MODEL_ID = os.environ.get("HF_MODEL", "BAAI/bge-m3")
EMBED_DIM = 1024
MAX_SEQ_LEN = 512


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
# BGE-M3 device worker — one resident encoder per chip.
# --------------------------------------------------------------------------- #
def _bge_worker(chip_id, args, normalize, task_q, result_q, ready_q, worker_cores):
    """Single chip: build a resident BgeM3ForEmbeddingOptimized, serve from ``task_q``.

    Protocol:
      * On ready (after warmup) puts ``(chip_id, build_sec)`` on ``ready_q``.
      * For each ``(idx, payload)`` task, puts
        ``(idx, chip_id, emb_or_None, steps_or_None, n_tokens)`` on ``result_q``
        (emb is a float32 numpy array). ``payload`` is token-id list (server-side
        tokenize) or raw text (worker-side tokenize).
      * A ``None`` task terminates the worker.
    """
    os.environ["TT_VISIBLE_DEVICES"] = str(chip_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if worker_cores:
        try:
            os.sched_setaffinity(0, set(worker_cores))
        except (OSError, AttributeError):
            pass

    device = None
    try:
        import torch

        import ttnn
        from models.demos.wormhole.bge_m3.demo.generator_vllm_optimized import BgeM3ForEmbeddingOptimized

        t_build0 = time.perf_counter()
        device = ttnn.open_device(
            device_id=0,
            trace_region_size=200_000_000,
            num_command_queues=2 if getattr(args, "cq2", False) else 1,
        )
        model = BgeM3ForEmbeddingOptimized(
            device=device,
            max_batch_size=1,
            max_seq_len=MAX_SEQ_LEN,
            dtype=ttnn.bfloat8_b,
            model_name=MODEL_ID,
        )

        # Worker-side tokenizer (used only if server didn't pre-tokenize).
        tokenizer = None
        if not args.server_tokenize:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        def _run(ids):
            """ids: list[int] -> (emb_np_f32 [HIDDEN], n_tokens, steps)."""
            n = len(ids)
            input_ids = torch.tensor([ids], dtype=torch.int64)  # [1, n]
            t_prep = time.perf_counter()
            out = model.forward(input_ids)  # [1, HIDDEN] (already L2-normalized)
            ttnn.synchronize_device(device)
            t_done = time.perf_counter()
            emb = out[0]
            if not normalize:
                # The wrapper always L2-normalizes; only used if caller opts out.
                emb = emb  # no-op placeholder; normalization is intrinsic here
            return emb.float().numpy().astype(np.float32), n, t_prep, t_done

        # Warmup: first call compiles + captures the trace.
        _ = _run(_fast_encode(tokenizer, "warm up the model", MAX_SEQ_LEN)) if tokenizer else _run([0, 1, 2, 3])
        ready_q.put((chip_id, time.perf_counter() - t_build0))

        while True:
            task = task_q.get()
            t_recv = time.perf_counter()
            if task is None:
                break
            idx, payload = task
            if isinstance(payload, str):
                ids = _fast_encode(tokenizer, payload, MAX_SEQ_LEN)
            else:
                ids = payload
            if not ids:
                result_q.put((idx, chip_id, None, None, 0))
                continue
            emb, n, t_prep, t_done = _run(ids)
            steps = {
                "device": (t_done - t_prep) * 1000.0,
                "total": (t_done - t_recv) * 1000.0,
                "t_recv": t_recv,
                "t_prep": t_prep,
                "t_done": t_done,
            }
            result_q.put((idx, chip_id, emb, steps, n))
    except Exception as exc:  # surface build/serve failures to the parent
        try:
            ready_q.put((chip_id, None, f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        try:
            if device is not None:
                ttnn.close_device(device)  # noqa: F821
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Async chip pool — model-agnostic, from pplx serve.py.
# --------------------------------------------------------------------------- #
class AsyncEmbeddingPool:
    """Spawns N resident chip workers and schedules requests across idle chips."""

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
        self.isl = MAX_SEQ_LEN
        args = SimpleNamespace(
            dp=dp,
            max_length=self.isl,
            cq2=False,
            server_tokenize=server_tokenize,
        )
        self.server_tokenize = server_tokenize
        self.tokenizer = None
        self._tok_pool: Optional[ThreadPoolExecutor] = None
        if self.server_tokenize:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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
                target=_bge_worker,
                args=(i, args, normalize, self.task_qs[i], self.result_q, ready_q, worker_cores),
                name=f"chip-{i}",
                daemon=True,
            )
            p.start()
            self.procs.append(p)

        logger.info(f"Building {self.n} resident encoder(s) (one per chip) — this takes ~1-2 min...")
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
                logger.error(f"  chip {cid}: {err[:300]}")
        if not self.ready:
            raise RuntimeError("All DP workers failed to build.")
        logger.info(f"Embedding pool ready: {len(self.ready)}/{self.n} chips live.")

        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.idle_chips: Optional[asyncio.Queue] = None
        self.pending: dict = {}
        self._req_id = 0
        self._collector: Optional[threading.Thread] = None
        self.metrics: deque = deque(maxlen=200_000)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.idle_chips = asyncio.Queue()
        for chip in sorted(self.ready):
            self.idle_chips.put_nowait(chip)
        self._collector = threading.Thread(target=self._collect, name="result-collector", daemon=True)
        self._collector.start()

    def _collect(self) -> None:
        while True:
            item = self.result_q.get()
            if item is None:
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
        t0 = time.perf_counter()
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
        return out

    def close(self) -> None:
        for q in self.task_qs:
            try:
                q.put(None)
            except Exception:
                pass
        try:
            self.result_q.put(None)
        except Exception:
            pass
        for p in self.procs:
            try:
                p.join(timeout=5)
            except Exception:
                pass


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

try:
    from fastapi.responses import ORJSONResponse as _DEFAULT_RESPONSE_CLASS  # uses orjson
except Exception:  # pragma: no cover - orjson optional
    from fastapi.responses import JSONResponse as _DEFAULT_RESPONSE_CLASS


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = MODEL_ID
    encoding_format: str = Field("float", description='"float" or "base64"')


_CONFIG = SimpleNamespace(dp=1, cores_per_worker=0, server_phys=4, server_tokenize=True)
_POOL: Optional[AsyncEmbeddingPool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _POOL
    loop = asyncio.get_running_loop()
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


app = FastAPI(title="BAAI/bge-m3 embedding server", default_response_class=_DEFAULT_RESPONSE_CLASS, lifespan=lifespan)


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
    return {"status": "ok" if ready else "starting", "chips_ready": ready, "chips_total": _CONFIG.dp}


@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "tenstorrent"}]}


@app.get("/metrics")
async def metrics(reset: bool = False):
    """Per-request latency breakdown (ms): on-device replay vs. server overhead."""
    if _POOL is None:
        raise HTTPException(status_code=503, detail="pool not ready")
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
    parser = argparse.ArgumentParser(description="OpenAI-compatible embedding server (DP=N) for BAAI/bge-m3.")
    parser.add_argument("--dp", type=int, default=1, help="Number of chips / workers (default 1).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default 0.0.0.0).")
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
    args = parser.parse_args()

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
        access_log=False,
    )


if __name__ == "__main__":
    main()
