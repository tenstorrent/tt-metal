# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Measurement harness: does the device wait on the host in the production spec loop?

Production serves with QWEN36_SPEC_FUSED=1 QWEN36_SPEC_CHAIN=1 QWEN36_SPEC_CHAIN_SAMPLED=1: every verify is a deferred
fused step (decode_forward(defer_read=True): chain trace, verify trace, accept trace, drafter trace and non-blocking
reads, no synchronize), and vLLM's async scheduler enqueues step n+1, chained on step n, BEFORE it reads step n (depth
1). This test drives that order (test_spec_async_engine.AsyncDriver) at a long KV position without the B x 32K prefill:
the engine's _prefill_row is replaced by its host bookkeeping (as the specprof harness does; kernel time depends on
positions and shapes, not on values, so acceptance here is meaningless). Per steady-state step it records

  wall  time between consecutive step completions in the host loop
  wait  time the host blocks in event_synchronize on the step it is about to read
  host  wall - wait: enqueue of the next step + parse / commit / propose of the read one

A step whose wait is > 0 was read while the device still ran it, so the next step (enqueued before that read) was
already queued: the device did not wait on the host at that boundary, and wall is device time. Right after the pipeline
(its rows still live) it replays the traces of one step alone (blocking) for the trace-only device time; wall minus that
is what the in-order H2D stages, D2H reads and launches add per step. Before the pipeline it times the synchronous fused
step (decode_forward without defer_read), which reads every step before the next is enqueued, as the demo loop does.

Env: VD_CTX (first verify position, default 32768), VD_RUNS (comma list of <users>:<greedy|sampled>, default
"8:greedy,4:greedy"; 4 users = the 16-row bucket), VD_STEPS (pipeline steps per run, default 60), VD_SUSTAIN (1 = also
replay the step's traces back to back, and the verify 10 vs 150 times, to separate per-step H2D / D2H / launch cost from
a slowdown under sustained load).
"""

import math
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests import test_spec_continuous as tc
from models.demos.blackhole.qwen36.tests.test_spec_async_engine import AsyncDriver
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine import ChainDriver
from models.demos.blackhole.qwen36.tt import spec_engine as se
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

B, K = 8, 3
T = K + 1
CTX = int(os.environ.get("VD_CTX", "32768"))
RUNS = [
    (int(u), m.strip() == "sampled")
    for u, m in (r.split(":") for r in os.environ.get("VD_RUNS", "8:greedy,4:greedy").split(",") if r.strip())
]
STEPS = int(os.environ.get("VD_STEPS", "60"))
SUSTAIN = os.environ.get("VD_SUSTAIN", "0") == "1"  # also time the step's traces back to back and the verify under load
WARM = 6  # steady state starts after this many pipeline steps
NEW = 3 * T * STEPS * len(RUNS) + 64  # positions every run's pipeline and sync comparison may consume


def _fake_prefill_row(
    self, slot, prompt_ids, page_table_row, _cap_blocks=None, start_pos=None, vision_tokens=None, final=True
):
    """MTPSpecEngine._prefill_row minus all device work: register the row as a fresh prompt of len(prompt_ids)."""
    assert final and vision_tokens is None and start_pos is None
    se.reset_draft_skip(self, slot)
    self._parked.pop(slot, None)
    se._chunk_prog_of(self).pop(slot, None)
    self.__dict__.setdefault("_finishing", set()).discard(slot)
    self._fresh_dirty.discard(slot)
    prompt_list = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
    self._prompts[slot] = (prompt_list, torch.as_tensor(page_table_row).clone())
    se._row_vision_of(self).pop(slot, None)
    self._fresh[slot] = len(prompt_list)
    self._active.add(slot)
    se._record_row_delta(self, slot, None)
    lt = torch.zeros(self.vocab)
    lt[1000] = 1.0
    return lt


class TimedAsyncDriver(AsyncDriver):
    """AsyncDriver whose pipeline splits each step into host time and time blocked on the device."""

    def __init__(self, eng, seed, **kw):
        super().__init__(eng, seed, **kw)
        self.rec = []  # (wall_ms, wait_ms, enq_ms) per pipeline iteration that read a step

    def timed_pipeline(self, n):
        t_prev = time.perf_counter()
        for _ in range(n):
            if not self.live():
                break
            t0 = time.perf_counter()
            self.enq()
            t1 = time.perf_counter()
            if len(self.pend) == 2:
                ev = self.pend[0][0].event
                tw = time.perf_counter()
                if ev is not None:
                    ttnn.event_synchronize(ev)
                tw1 = time.perf_counter()
                self.resolve()
                t2 = time.perf_counter()
                self.rec.append((1e3 * (t2 - t_prev), 1e3 * (tw1 - tw), 1e3 * (t1 - t0)))
                t_prev = t2
        self.drain()


def _stats(xs):
    xs = sorted(xs)
    return f"median {statistics.median(xs):.2f} p10 {xs[len(xs) // 10]:.2f} p90 {xs[(9 * len(xs)) // 10]:.2f}"


def _trace_alone_ms(mesh, tid, reps=10):
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t0) / reps


@run_for_blackhole()
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_deferred_device_idle(mesh_device, monkeypatch):
    for k, v in (
        ("QWEN36_SPEC_FUSED", "1"),
        ("QWEN36_SPEC_CHAIN", "1"),
        ("QWEN36_SPEC_CHAIN_SAMPLED", "1"),
        ("QWEN36_SPEC_FUSED_AUDIT_EVERY", os.environ.get("QWEN36_SPEC_FUSED_AUDIT_EVERY", "8")),
        ("QWEN36_SPEC_CHAIN_AUDIT_EVERY", os.environ.get("QWEN36_SPEC_CHAIN_AUDIT_EVERY", "8")),
    ):
        monkeypatch.setenv(k, v)
    nb = math.ceil((CTX + NEW + 2 * T) / BLOCK_SIZE) + 1
    monkeypatch.setattr(tc, "B", B)
    monkeypatch.setattr(tc, "NB", nb)
    monkeypatch.setattr(tc, "NUM_BLOCKS", B * nb + 40)
    monkeypatch.setattr(se.MTPSpecEngine, "_prefill_row", _fake_prefill_row)
    mesh = mesh_device
    mesh.enable_program_cache()
    t0 = time.time()
    model = Qwen36Model.from_pretrained(mesh, max_batch_size=B, max_seq_len=nb * BLOCK_SIZE, enable_mtp=None)
    assert model.mtp is not None, "no MTP head"
    monkeypatch.setattr(model, "warmup_prefill_masked_buckets", lambda *a, **k: None)  # the fake prefill runs none
    model.allocate_kv_caches(
        [tc.NUM_BLOCKS, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim], ttnn.bfloat16, batch_size=B
    )
    model.set_gdn_fused_decode(True)
    eng = se.get_spec_engine(model, B, K)
    eng.prepare(torch.stack([torch.arange(1 + u * nb, 1 + (u + 1) * nb, dtype=torch.int32) for u in range(B)]))
    eng.capture()
    assert eng._ch_trace_g is not None and eng._fa_trace is not None, "chain / fused accept traces missing"
    logger.info(f"[devidle] model + engine ready in {time.time() - t0:.0f}s: ctx={CTX} runs={RUNS} nb={nb}")
    prompts = [[1000 + (7919 * (i + 13 * u)) % 50000 for i in range(CTX)] for u in range(B)]
    for users, sampled in RUNS:
        _run(eng, model, mesh, monkeypatch, prompts, users, sampled)


def _run(eng, model, mesh, monkeypatch, prompts, users, sampled_on):
    sampled = (lambda s: True) if sampled_on else (lambda s: False)
    tag = f"ctx={CTX} users={users} {'sampled' if sampled_on else 'greedy'}"
    # the synchronous fused step for comparison (each step read before the next is enqueued)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "0")
    sd = ChainDriver(eng, seed=42, sampled=sampled)
    for u in range(users):
        sd.admit(u, u, prompts[u], max_new=10**6)
    for _ in range(min(STEPS, 40)):
        sd.step()
    sw = sd.wall_ms[WARM:]
    for r in sd.live():
        sd._finish(r)
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_AUDIT", "1")
    drv = TimedAsyncDriver(eng, seed=41, sampled=sampled)
    for u in range(users):
        drv.admit(u, u, prompts[u], max_new=10**6)
    drv.timed_pipeline(STEPS)
    rec = drv.rec[WARM:]
    walls, waits, enqs = [r[0] for r in rec], [r[1] for r in rec], [r[2] for r in rec]
    hosts = [w - x for w, x in zip(walls, waits)]
    starved = sum(1 for x in waits if x < 0.05)
    vb = eng._vb if eng._bucketed() else B
    med = statistics.median(walls)
    logger.info(
        f"[devidle] DEFERRED {tag} bucket={vb} steps={len(rec)}: wall {_stats(walls)} | wait {_stats(waits)} | "
        f"host {_stats(hosts)} | enqueue {_stats(enqs)} | steps with wait<0.05ms (device may idle): {starved} | "
        f"chain_steps={eng.chain_steps} chain_breaks={dict(eng.chain_breaks)} "
        f"fused_fallbacks={dict(getattr(eng, 'fused_fallbacks', {}))}"
    )
    logger.info(
        f"[devidle] SYNC fused {tag}: wall {_stats(sw)} ms/step vs deferred {med:.2f} "
        f"(sync - deferred = {statistics.median(sw) - med:.2f} ms/step that the deferred order hides)"
    )
    # Trace-only device time of one step: chain + verify (+ topk when sampled) + accept + drafter, each replayed alone
    # and blocking, right after the pipeline drained while this run's rows are still live (the replays read the
    # positions and live rows this run left on the device). The replays leave the device state inconsistent with the
    # engine's host bookkeeping, so the rows are released right after and the next run starts from fresh admissions.
    _trace_times(eng, model, mesh, tag, vb, sampled_on, med)
    for r in drv.live():
        drv._finish(r)


def _trace_times(eng, model, mesh, tag, vb, sampled, med):
    vt = model._vfy_trace_id
    bk = (getattr(model, "_vfy_bk", None) or {}).get(int(vb)) if int(vb) != B else None
    if bk is not None:
        vt = bk.trace_id
    parts = {
        "chain": eng._ch_trace_s if sampled else eng._ch_trace_g,
        "verify": vt,
        "accept": eng._fs_trace if sampled else eng._fa_trace,
        "drafter": eng._dr_trace_for(vb),
    }
    if sampled:
        parts["topk"] = bk.topk_trace_id if bk is not None else getattr(model, "_vfy_topk_trace_id", None)
    alone = {k: _trace_alone_ms(mesh, tid) for k, tid in parts.items() if tid is not None}
    tsum = sum(alone.values())
    logger.info(
        f"[devidle] TRACES ALONE {tag} bucket={vb}: "
        + " ".join(f"{k}={v:.2f}" for k, v in alone.items())
        + f" sum={tsum:.2f} ms | deferred wall median {med:.2f} ms -> in-order H2D/D2H/launch + any host stall "
        f"{med - tsum:.2f} ms/step ({100 * (med - tsum) / med:.1f}%)"
    )
    if SUSTAIN:
        # The step's traces back to back with no host staging or reads in between (what the deferred order would cost if
        # H2D / D2H / launches were free), then the verify alone 10 times after 2 s idle vs 150 times (does the device
        # slow down under sustained load?).
        seq = [tid for tid in parts.values() if tid is not None]
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(30):
            for tid in seq:
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        seq_ms = 1e3 * (time.perf_counter() - t0) / 30
        time.sleep(2.0)
        v10 = _trace_alone_ms(mesh, parts["verify"], reps=10)
        v150 = _trace_alone_ms(mesh, parts["verify"], reps=150)
        logger.info(
            f"[devidle] SUSTAIN {tag}: step traces back to back {seq_ms:.2f} ms/step (deferred wall {med:.2f}, "
            f"traces-alone sum {tsum:.2f}) | verify alone x10 after 2 s idle {v10:.2f} ms, x150 {v150:.2f} ms"
        )
