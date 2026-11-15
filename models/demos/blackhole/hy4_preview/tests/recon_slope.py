# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""reconcile task: N-layer slope / intercept of the traced pool decode (B=4, 4x8 mesh, real weights for layers 0..NLMAX-1, no MTP).
One process: build NLMAX layers + head, real prefill of 4 short prompts (real MoE routing), hand-off, then for every prefix length n in NLS capture a trace of
embedding + layers[0:n] + head (ch.layers is temporarily sliced) and time trace replay only (device wall, no host loop). Optional stage B: synthetic fill + topk mode at CTXS.
Env: NLMAX (16), NLS ('0,1,2,3,4,6,8,12,16'), CTXS ('' ), NLS_B (subset for stage B), REPS (5), HY4_OPT etc. as the integ runs. Output: RES json lines."""
import json
import os
import statistics
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh
from models.demos.blackhole.hy4_preview.tt.hy4_chain import CHUNK, Hy4Chain, dram_view
from models.demos.blackhole.hy4_preview.tt.hy4_pool_model import Hy4PoolDecoder
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

CKPT = os.environ.get("HY4_CKPT", "/ckpt")
OUT = os.environ.get("OUT", "/out")
NLMAX = int(os.environ.get("NLMAX", "16"))
NLS = [int(x) for x in os.environ.get("NLS", "0,1,2,3,4,6,8,12,16").split(",") if x]
NLS_B = [int(x) for x in os.environ.get("NLS_B", "1,4,8,16").split(",") if x]
CTXS = [int(x) for x in os.environ.get("CTXS", "").split(",") if x]
REPS = int(os.environ.get("REPS", "5"))
PAD = 120002


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def trace_ms(md, dec, n=20):
    ttnn.synchronize_device(md)
    v = []
    for _ in range(REPS):
        t1 = time.perf_counter()
        for _ in range(n):
            ttnn.execute_trace(md, dec.trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        v.append((time.perf_counter() - t1) / n * 1e3)
    return round(statistics.median(v), 3), round(min(v), 3), round(max(v), 3)


def cap_and_time(md, ch, dec, full, n, tag, **kw):
    ch.layers = full[:n]
    dec.release()
    t0 = time.time()
    dec.capture()
    cs = time.time() - t0
    med, lo, hi = trace_ms(md, dec)
    res(event="slope", tag=tag, n=n, trace_ms_med=med, min=lo, max=hi, capture_s=round(cs, 1), **kw)
    if os.environ.get("HOSTLOOP", "0") == "1":  # product loop: set_inputs + replay + full-vocab read-back + host argmax
        tl, ts_, tr_, tk_ = [], [], [], []
        pos = [int(p) for p in dec.ctx.step_pos]
        for s_ in range(12):
            t0 = time.perf_counter()
            dec.set_inputs([1000 + s_, 2000, 3000, 4000], pos)
            t1 = time.perf_counter()
            out = dec.run()
            t2 = time.perf_counter()
            dec.read_argmax(out)
            t3 = time.perf_counter()
            ts_.append(t1 - t0), tr_.append(t2 - t1), tk_.append(t3 - t2), tl.append(t3 - t0)
        m = lambda v: round(statistics.median(v[2:]) * 1e3, 2)
        res(
            event="hostloop",
            tag=tag,
            n=n,
            total_ms=m(tl),
            set_inputs_ms=m(ts_),
            run_ms=m(tr_),
            readback_argmax_ms=m(tk_),
        )
    ch.layers = full


def main():
    prompts = torch.load(os.environ.get("PROMPTS", f"{OUT}/decode_prompts.pt"))
    lens = [len(p) for p in prompts]
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "700e6"))))
    try:
        from models.demos.blackhole.hy4_preview.tt import hy4_pool_decode as _pd
        from models.demos.blackhole.hy4_preview.tt.hy4_ccl_safe import hy4_fabric_selftest

        links = hy4_fabric_selftest(md)
        _pd._POOL_AG_LINKS = links.pool_ag
        res(event="fabric_selftest", num_links=links.num_links, pool_ag=links.pool_ag, forced=links.forced)
        mc, ccl = mesh_4x8(), CCLManager(md, num_links=links.num_links, topology=ttnn.Topology.Ring)
        ch = Hy4Chain(md, mc, ccl, ckpt=CKPT)
        t0 = time.time()
        ch.add_layers(0, NLMAX, log=lambda s: None)
        ch.build_head()
        res(event="built", sec=round(time.time() - t0, 1), nl=NLMAX, **dram_view(md))
        pages = max([c // 512 for c in CTXS] + [8])
        dec = Hy4PoolDecoder(ch, pages, mode="dense")
        full = list(ch.layers)
        pos = [100, 100, 100, 100]
        if os.environ.get("NOPREFILL", "0") != "1":
            ids = torch.full((4, CHUNK), PAD, dtype=torch.long)
            for r in range(4):
                ids[r, : lens[r]] = torch.tensor(prompts[r])
            t0 = time.time()
            x4, _ = ch.prefill(ids)
            ttnn.synchronize_device(md)
            res(event="prefill", sec=round(time.time() - t0, 1))
            ttnn.deallocate(x4)
            dec.handoff(CHUNK)
            ch.reset()
            pos = list(lens)
        dec.set_inputs([1000, 2000, 3000, 4000], pos)
        for n in NLS:
            cap_and_time(md, ch, dec, full, n, "dense_short")
        for C in CTXS:
            if C > 2048 and not getattr(dec, "_filled", False):
                dec.release()
                dec.fill_synthetic(max(CTXS) // 512 * 512, torch.Generator().manual_seed(11))
                dec._filled = True
            mode = "dense" if C <= 2048 else "topk"
            dec.mode = mode
            dec.ctx.set_bucket(C, mode)
            dec.set_inputs([1000, 2000, 3000, 4000], [C - 1] * 4)
            for n in NLS_B:
                cap_and_time(md, ch, dec, full, n, f"ctx{C}_{mode}")
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
