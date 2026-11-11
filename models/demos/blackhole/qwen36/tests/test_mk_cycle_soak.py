"""Megakernel engine-ON cycle soak (rc1b OR-replay hang, 11-11 14:29Z: a u1 verify on the 8-row bucket with 7 idle rows, right
after every row was released and a long prompt was prefilled).

One driver (blocks recycled across cycles, so idle rows keep stale page-table rows / kvpos of freed blocks that get
re-allocated) loops:
    release all rows -> admit k users (k = 1, every CYC_MULTI_EVERY-th cycle k = CYC_K) with long prompts that share a
    long prefix with earlier ones (ctx from CYC_CTX, chunked traced prefill) -> CYC_STEPS verify steps -> release.
Logs "[cycsoak] cycle c k ctx steps wall" per cycle and, per verify read, "[cycsoak] h <hash>" whenever the host view of
(kvpos aux page, page-table rows) changes (cheap: host tensors only). The report-only watchdog arms on "[cycsoak]".

  CYC_N=200 CYC_CTX=32768,49152,69000 CYC_STEPS=120 CYC_MULTI_EVERY=10 CYC_K=3 CYC_SEED=1 CYC_HASH=1
  QWEN36_R16_MEGAKERNEL=1 + rc.env, full model (NL 64), pytest models/demos/blackhole/qwen36/tests/test_mk_cycle_soak.py -s -x
"""
import hashlib
import math
import os
import time

import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import _marks

N = int(os.environ.get("CYC_N", "200"))
CTX = [int(c) for c in os.environ.get("CYC_CTX", "32768,49152,69000").split(",")]
STEPS = int(os.environ.get("CYC_STEPS", "120"))
MULTI_EVERY = int(os.environ.get("CYC_MULTI_EVERY", "10"))
KM = int(os.environ.get("CYC_K", "3"))
SEED = int(os.environ.get("CYC_SEED", "1"))
HASH = os.environ.get("CYC_HASH", "1") == "1"
B = 8


def _prompt(tokenizer, n, salt):
    """A long prompt; every prompt starts with the same base text (shared long prefix), the salt shifts the tail."""
    base = tc._prompt_pool(tokenizer, 4096)
    rep = base * (n // len(base) + 2)
    head = rep[: n - 512]
    tail = rep[(salt * 131) % len(base):][:512]
    return (head + tail)[:n]


def _hash_tap(model):
    last = [None]
    orig = model.verify_after_external_sync

    def wrapped(handle, *a, **k):
        r = orig(handle, *a, **k)
        aux = getattr(model, "_vfy_aux_staged", None)
        pt = getattr(model, "_vfy_pt_host", None)
        h = hashlib.md5()
        for t in (aux, pt):
            if t is not None:
                h.update(torch.as_tensor(t).contiguous().view(torch.uint8).numpy().tobytes())
        d = h.hexdigest()[:12]
        if d != last[0]:
            last[0] = d
            logger.info(f"[cycsoak] h {d} bucket {getattr(handle, 'bucket', None)}")
        return r

    model.verify_after_external_sync = wrapped
    return lambda: setattr(model, "verify_after_external_sync", orig)


@_marks
def test_mk_cycle_soak(mesh_device, monkeypatch):
    assert os.environ.get("QWEN36_R16_MEGAKERNEL", "").strip() == "1", "run with QWEN36_R16_MEGAKERNEL=1"
    T = tc.K + 1
    nb = max(math.ceil((max(CTX) + 64 * T + STEPS * T) / 64), 8)
    monkeypatch.setattr(tc, "B", B)
    monkeypatch.setattr(tc, "NB", nb)
    monkeypatch.setattr(tc, "NUM_BLOCKS", B * nb + 40)
    model, eng, tokenizer, _ = tc._setup(mesh_device)
    mk1 = model._vfy_mk1
    assert mk1 is not None and mk1.ready and B in mk1.programs, f"engine bucket {B} not built: {getattr(mk1, 'reason', None)}"
    untap = _hash_tap(model) if HASH else (lambda: None)
    g = torch.Generator().manual_seed(SEED)
    drv = AsyncDriver(eng, seed=SEED, sampled=lambda s: False)
    # AsyncDriver is built with recycle=False (its pipeline may _finish a row that the pending step still carries). Rows
    # here are only released after pipeline() drained (max_new = STEPS * T is never reached inside it), so recycling is
    # safe, and needed: without it the free list ran dry at cycle 9 (E25a "pop from empty list").
    drv.recycle = True
    rid = 0
    try:
        for c in range(N):
            k = KM if MULTI_EVERY and c % MULTI_EVERY == MULTI_EVERY - 1 else 1
            slots = torch.randperm(B, generator=g)[:k].tolist()   # varying rows: idle rows hold other cycles' page tables
            ctx = [CTX[int(torch.randint(len(CTX), (1,), generator=g))] for _ in slots]
            t0 = time.perf_counter()
            for s, n in zip(slots, ctx):
                drv.admit(s, rid, _prompt(tokenizer, n, rid), max_new=STEPS * T)
                rid += 1
            t1 = time.perf_counter()
            drv.pipeline(STEPS)
            for r in drv.live():
                drv._finish(r)   # release every row (blocks recycled -> re-allocated by the next cycle)
            logger.info(f"[cycsoak] cycle {c} k {k} slots {slots} ctx {ctx} prefill {t1 - t0:.1f}s "
                        f"verify {time.perf_counter() - t1:.1f}s steps {drv.steps} free {len(drv.free)}")
    finally:
        untap()
    logger.info(f"[cycsoak] DONE {N} cycles, {drv.steps} steps, no hang")
