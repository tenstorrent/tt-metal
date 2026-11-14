# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.63 diagnostic (iterating only, not a gate): the host cost a call of the fused head and the combine, and its parts.

One model (PROBE_LAYERS layers, default 4; shipped settings, MOE_REPLICAS=prev), the novel prompt chunks 0..3 back to
back, PASSES passes; prints the median host us a call of mb.fused_head and mb.routed_experts_combine (Python wrapper
inclusive) and the host ms a chunk. With DS41_DIAG_TIMERS=1 and a build carrying the fork's diagnostic timers the C++
side prints its own split (entry / prim / hash / override) to stderr; run with -s.

    DS41_DIAG_TIMERS=1 scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_mesh_program_probe.py -s
"""

from __future__ import annotations

import os
import statistics
import time

import ttnn
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
SEQ = 56320
CHUNKS = 4
PASSES = int(os.environ.get("PROBE_PASSES", "4"))
NL = int(os.environ.get("PROBE_LAYERS", "4"))
SIDES = os.environ.get("PROBE_SIDES", "legacy,on,legacy,on").split(",")


@mesh_parametrize(timeout_s=3 * 3600)
def test_mesh_program_probe(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_prev_replicas import _ids
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    settings.thaw()
    layers = S.layers()[:NL]
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mb = ttnn.bringup.moe_block_main
    rec = {"fused_head": [], "routed_experts_combine": [], "sparse_dispatch_2d": []}

    def timed(name):
        orig = getattr(mb, name)

        def f(*a, **k):
            t0 = time.perf_counter()
            out = orig(*a, **k)
            rec[name].append(time.perf_counter() - t0)
            return out

        monkeypatch.setattr(mb, name, f)

    for n in rec:
        timed(n)
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    tokens = _ids("novel")
    chunk = int(S.get("target.chunk"))
    res = {}
    for side in SIDES:
        hosts = []
        for p in range(PASSES + 1):
            for m in A._MAIN_LAYERS:
                m._cached_slots = side == "on"
            for k in rec:
                rec[k].clear()
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
                    if p and c:
                        hosts.append((t1 - t0) * 1e3)
            finally:
                _free(state)
            if p:
                med = {k: round(1e6 * statistics.median(x), 1) for k, x in rec.items() if x}
                print(f"P.63 probe {side} pass {p}: us a call (median) {med}", flush=True)
                for k, x in med.items():
                    res.setdefault((side, k), []).append(x)
        res.setdefault((side, "host_ms"), []).append(statistics.mean(hosts))
        print(f"P.63 probe {side}: host {statistics.mean(hosts):.2f} ms a chunk ({NL} layers)", flush=True)
    for (side, k), x in sorted(res.items()):
        print(f"P.63 probe summary {side} {k}: median {statistics.median(x):.1f} over {len(x)}", flush=True)
