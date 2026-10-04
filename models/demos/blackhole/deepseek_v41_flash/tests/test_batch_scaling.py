# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Batch-scaling study of decode (40 layers, real GSM8K prompts, ISL ~100, no indexer), ONE process per batch:
  phase 1: eager decode steps with the REAL routed expert ids of every MoE layer copied to the host (DSV41_ROUTE_CAPTURE=1)
  phase 2: traced decode steps -> ms/token (device loop, as the demo)
  phase 3: per-op table (recorder + device profiler, as test_layer_op_table.py) of layers DSV41_BS_LAYERS (default 5,0) fed with the real
           per-layer inputs / paged step state of the model.
Env: DSV41_BS_BATCH (16/32/64/128), DSV41_BS_OUT, DSV41_BS_STEPS (eager steps, default 6), DSV41_BS_TSTEPS (traced steps, default 24),
     DSV41_BS_LAYERS, DSV41_BS_PROFILE=0 skips phase 3.  Run phase 3 with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1.
"""
import gc
import json
import os
import shutil
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.demo import text_demo as TD
from models.demos.blackhole.deepseek_v41_flash.tests.op_table_recorder import OpRecorder, dev_id
from models.demos.blackhole.deepseek_v41_flash.tests.test_layer_op_table import MarkDict
from models.demos.blackhole.deepseek_v41_flash.tt import moe_block as MB
from models.demos.blackhole.deepseek_v41_flash.tt.dsv41_model import Model

B = int(os.environ.get("DSV41_BS_BATCH", "16"))
OUT = os.environ.get("DSV41_BS_OUT", f"/mnt/tt-data/ssinghal/dsv4-logs/bs_b{B}")
REC = OpRecorder()
LAYERS = [int(x) for x in os.environ.get("DSV41_BS_LAYERS", "5,0").split(",")]


def profile_layer(md, L, layer, x, pre, st, out):
    os.makedirs(out, exist_ok=True)
    prof = bool(os.environ.get("TT_METAL_DEVICE_PROFILER"))

    def fwd(profile=None):
        o, n = layer.forward(x, pre, st, profile=profile)
        ttnn.synchronize_device(md)
        ttnn.deallocate(o)
        ttnn.deallocate(n)

    for _ in range(3):
        fwd()
        gc.collect()
    result = dict(layer=L, B=B, profiler=prof)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    layer.forward(x, pre, st)
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    n = 50
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    result["traced_ms"] = (time.perf_counter() - t) / n * 1e3
    ttnn.release_trace(md, tid)
    print(f"BS layer {L} B={B} traced {result['traced_ms']:.4f} ms", flush=True)
    rec = REC
    rec.install()
    rec.rows, rec.zero, rec.marks = [], {}, []
    ttnn.synchronize_device(md)
    if prof:
        ttnn.ReadDeviceProfiler(md)
    rec.enabled = True
    op0 = dev_id()
    fwd(MarkDict(rec))
    op1 = dev_id()
    rec.enabled = False
    result.update(op0=op0, op1=op1, eager_rows=rec.rows, eager_marks=rec.marks, eager_zero=rec.zero)
    if prof:
        ttnn.ReadDeviceProfiler(md)
    rec.rows, rec.zero, rec.marks = [], {}, []
    rec.enabled = True
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    c0 = dev_id()
    layer.forward(x, pre, st)
    c1 = dev_id()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    rec.enabled = False
    ttnn.synchronize_device(md)
    result.update(cap0=c0, cap1=c1, trace_rows=rec.rows)
    if prof:
        ttnn.ReadDeviceProfiler(md)
        for _ in range(3):
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ttnn.ReadDeviceProfiler(md)
    ttnn.release_trace(md, tid)
    json.dump(result, open(os.path.join(out, "raw.json"), "w"), indent=1)


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 1_600_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_batch_scaling(mesh_device, device_params):
    md = mesh_device
    os.makedirs(OUT, exist_ok=True)
    cache, steps, routes = {}, [], []
    orig = Model.decode_forward

    def wrapped(self, tokens, current_pos, enable_trace=True, reload_inputs=True):
        if MB.ROUTE_LOG is not None:
            MB.ROUTE_LOG.clear()
            MB.ROUTE_ON[0] = not enable_trace
        t = time.perf_counter()
        o = orig(self, tokens, current_pos, enable_trace=enable_trace, reload_inputs=reload_inputs)
        dt = time.perf_counter() - t
        MB.ROUTE_ON[0] = False
        steps.append((enable_trace, dt, current_pos.clone()))
        if not enable_trace and MB.ROUTE_LOG:
            routes.append([r.clone() for r in MB.ROUTE_LOG])
        return o

    Model.decode_forward = wrapped
    gsm = TD.GSM
    common = (TD.GREEDY,)
    # phase 1 + 2 share one model through ``cache``
    for tr, ngen in (
        (False, int(os.environ.get("DSV41_BS_STEPS", "6")) + 1),
        (True, int(os.environ.get("DSV41_BS_TSTEPS", "24")) + 1),
    ):
        TD._run_demo(md, gsm, B, 1, 512, ngen, None, TD.GREEDY, tr, True, None, True, True, False, cache=cache)
        cache_model = list(cache.values())[0][1]
        cache_model.release_trace()
    torch.save(routes, os.path.join(OUT, "routes.pt"))
    tr_steps = [dt for (e, dt, _) in steps if e][2:]
    summ = dict(
        B=B,
        traced_step_ms=[1e3 * x for x in tr_steps],
        traced_mean_ms=1e3 * sum(tr_steps) / max(len(tr_steps), 1),
        n_route_steps=len(routes),
    )
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"))
    print(
        f"BS B={B} traced decode mean {summ['traced_mean_ms']:.2f} ms/token over {len(tr_steps)} steps; route steps {len(routes)}",
        flush=True,
    )
    if os.environ.get("DSV41_BS_PROFILE", "1") == "0":
        return
    model = list(cache.values())[0][1]
    dec = model.dec
    grab = {}
    for lid, layer, _ in dec.layers:
        if lid in LAYERS:
            f0 = layer.forward

            def mk(lid, layer, f0):
                def f(x, pre, st, **kw):
                    grab[lid] = (x, pre, st)
                    return f0(x, pre, st, **kw)

                return f

            layer.forward = mk(lid, layer, f0)
    dec.forward()
    ttnn.synchronize_device(md)
    for lid, layer, _ in dec.layers:
        if lid in LAYERS:
            layer.forward = layer.__class__.forward.__get__(layer)
    byid = {lid: layer for lid, layer, _ in dec.layers}
    for L in LAYERS:
        x, pre, st = grab[L]
        profile_layer(md, L, byid[L], x, pre, st, os.path.join(OUT, f"layer_{L}"))
    prof = os.environ.get("TT_METAL_PROFILER_DIR")
    if prof:
        src = os.path.join(prof, ".logs", "cpp_device_perf_report.csv")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, "cpp.csv"))
