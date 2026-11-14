# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.60 diagnostic (iterating only, not a gate): where the combine's extra host time with MOE_REPLICAS=prev goes.

One model (PROBE_LAYERS layers, default 4), the novel prompt chunks 0..3 back to back, a few passes per variant; per
variant the median host us a call of routed_experts_combine (the C++ op launch, inclusive) and of the planner's
dispatch, and the host ms a chunk (first call to the last enqueue, no sync inside). Variants:
  off      prev switched off (the stock combine, no fill)
  legacy   MOE_REPLICAS=prev, MOE_PREV_HOST=legacy (P.56's host path)
  device   MOE_REPLICAS=prev, MOE_PREV_HOST=device (P.60)
  fixring  legacy, but every call is handed ring[0] (timing only: the copies are wrong; bounded waits, never a hang)
  noset    legacy, but set_combine_fill_exact only on a pass's first call (timing only)
Outputs are not checked here (test_moe_prev_replicas.py does that).

    PROBE_VARIANTS=off,prev,fixring,noset scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_host_probe.py
"""

from __future__ import annotations

import os
import statistics
import time

from loguru import logger

import ttnn
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
SEQ = 56320
CHUNKS = 4
PASSES = 3
NL = int(os.environ.get("PROBE_LAYERS", "4"))
VARIANTS = os.environ.get("PROBE_VARIANTS", "off,legacy,device").split(",")


@mesh_parametrize(timeout_s=3 * 3600)
def test_moe_prev_host_probe(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _ids, _prevs, _side
    from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    settings.thaw()
    layers = S.layers()[:NL]
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mb = ttnn.bringup.moe_block_main
    rec = {"combine": [], "after": [], "set": []}
    orig_comb, orig_set, orig_after = mb.routed_experts_combine, mb.set_combine_fill_exact, XP.PrevReplicas.after
    mode = {"v": "prev", "first": True}

    def comb(*a, **k):
        t0 = time.perf_counter()
        out = orig_comb(*a, **k)
        rec["combine"].append(time.perf_counter() - t0)
        return out

    def setf(res, en, *a, **k):
        if en and mode["v"] == "fixring":
            a = list(a)
            pr = _prevs(mesh_device)[0]
            a[3] = pr.rings()[0]
        if en and mode["v"] == "noset" and not mode["first"] and k.get("ring", 0) == 0:
            return None
        mode["first"] = False
        t0 = time.perf_counter()
        out = orig_set(res, en, *a, **k)
        rec["set"].append(time.perf_counter() - t0)
        return out

    def after(pr, layer, counts):
        t0 = time.perf_counter()
        mode["in_after"] = True
        orig_after(pr, layer, counts)
        mode["in_after"] = False
        rec["after"].append(time.perf_counter() - t0)

    orig_rp = ttnn.bringup.run_prepared

    def run_prepared(*a, **k):
        t0 = time.perf_counter()
        out = orig_rp(*a, **k)
        if mode.get("in_after"):
            rec["after_run_prepared"].append(time.perf_counter() - t0)
        return out

    rec["after_run_prepared"] = []
    monkeypatch.setattr(ttnn.bringup, "run_prepared", run_prepared)

    monkeypatch.setattr(mb, "routed_experts_combine", comb)
    monkeypatch.setattr(mb, "set_combine_fill_exact", setf)
    monkeypatch.setattr(XP.PrevReplicas, "after", after)
    tokens = _ids("novel")
    chunk = int(S.get("target.chunk"))
    out = {}
    for v in VARIANTS:
        mode["v"] = v
        if v == "off":
            _side(mesh_device, "off")
        else:
            _side(mesh_device, "prev", "device" if v == "device" else ("legacy" if _prevs(mesh_device) else None))
        hosts, walls = [], []
        for p in range(PASSES + 1):
            for k in rec:
                rec[k].clear()
            mode["first"] = True
            state = model.new_state(SEQ)
            try:
                for c in range(CHUNKS):
                    st = c * chunk
                    t = tokens[st : st + chunk]
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    begin_chunk(model, t, st)
                    h = model.embed(t)
                    for i in layers:
                        h2 = model.layer(i, h, st, state)
                        model.free(h)
                        h = h2
                    model.free(h)
                    t1 = time.perf_counter()
                    ttnn.synchronize_device(mesh_device)
                    t2 = time.perf_counter()
                    if p and c:
                        hosts.append((t1 - t0) * 1e3)
                        walls.append((t2 - t0) * 1e3)
            finally:
                _free(state)
        med = {k: 1e6 * statistics.median(x) for k, x in rec.items() if x}
        out[v] = {"host_ms": statistics.mean(hosts), "wall_ms": statistics.mean(walls), "us": med}
        logger.info(f"P.60 probe {v}: {out[v]}")
    if os.environ.get("PROBE_PLANNER") == "1":  # (last: the extra planner runs advance the device ordinal)
        from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _main

        pr = _prevs(mesh_device)[0]
        pp = pr._plan_pp[(pr.mode, pr.lean())]
        mains = list(_main().values())
        ios = []
        for m in mains:
            counts = (getattr(m, "_last_head", None) or m.block().last_head).counts
            ios.append([counts, pr.landing, pr.G, m._prev_state, m.head_w.router_bias, m.w_gu, m.w_d, pr.consts])
        res = {}
        for name, seq in (("same", [ios[0]] * 40), ("alternate", [ios[i % len(ios)] for i in range(40)])):
            for rep in range(3):
                ttnn.synchronize_device(mesh_device)
                ts = []
                for io in seq:
                    t0 = time.perf_counter()
                    ttnn.bringup.run_prepared(pp, io)
                    ts.append(time.perf_counter() - t0)
                ttnn.synchronize_device(mesh_device)
            res[name] = 1e6 * statistics.median(ts)
        out["planner_run_prepared_us"] = res
        print(f"P.60 probe planner run_prepared us a call: {res}")
    for v, r in out.items():
        if v.startswith("planner"):
            continue
        print(
            f"P.60 probe {v}: host {r['host_ms']:.2f} ms wall {r['wall_ms']:.2f} ms a chunk ({NL} layers), us {r['us']}"
        )
