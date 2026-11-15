# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.64 diagnostic (iterating only, not a gate): op launches a layer by type, section and call site, and the host
us a call, for the ATTN_FUSE sides.

One model (CENSUS_LAYERS, default one layer of each block type: 0,1,2,3,14,20,21,24; shipped settings,
ENGRAM_TABLE=resident), the novel prompt chunks 0..CENSUS_CHUNKS-1 back to back per side; the last chunk of the
second pass is counted. Terms: launch = one outermost ttnn op call (ttnn.decorators.FastOperation) or one
ttnn.bringup.run_prepared; us a call = its host time (Python wrapper inclusive); section = run_block's step
(signpost); site = the first frame in the model's tt/ code. Prints, per side and layer, launches by section and
op, and the top call sites; then per side the host ms a chunk (CENSUS_LAYERS layers) and the launch total.
With CENSUS_COMPARE=1 (default) the sides' layer outputs are compared byte for byte.

    CENSUS_SIDES=legacy,on scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_attn_launch_census.py -s
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from collections import Counter, defaultdict

import torch

import ttnn
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
SEQ = 56320
CHUNKS = int(os.environ.get("CENSUS_CHUNKS", "4"))
LAYERS = [int(x) for x in os.environ.get("CENSUS_LAYERS", "0,1,2,3,14,20,21,24").split(",")]
SIDES = os.environ.get("CENSUS_SIDES", "legacy,on").split(",")
PASSES = int(os.environ.get("CENSUS_PASSES", "2"))
COMPARE = os.environ.get("CENSUS_COMPARE", "1") == "1"
TT = os.sep + os.path.join("deepseek_v41_flash_d_p", "tt") + os.sep


class Census:
    def __init__(self):
        self.on = False
        self.layer = None
        self.section = None
        self.depth = 0
        self.rows = []  # (layer, section, op, site, seconds)

    def site(self):
        f = sys._getframe(2)
        while f is not None:
            fn = f.f_code.co_filename
            if TT in fn:
                return f"{os.path.basename(fn)}:{f.f_lineno}:{f.f_code.co_name}"
            f = f.f_back
        return "?"

    def wrap(self, name_of, fn):
        c = self

        def run(*a, **k):
            if not c.on or c.depth:
                return fn(*a, **k)
            site = c.site()
            c.depth += 1
            t0 = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                c.depth -= 1
                c.rows.append((c.layer, c.section, name_of(a), site, time.perf_counter() - t0))

        return run


def _install(monkeypatch, census):
    import ttnn.decorators as D

    orig = D.FastOperation.__call__

    def opname(a):
        n = a[0].python_fully_qualified_name
        return n[5:] if n.startswith("ttnn.") else n

    w = census.wrap(opname, orig)
    monkeypatch.setattr(D.FastOperation, "__call__", lambda self, *a, **k: w(self, *a, **k))
    rp = census.wrap(lambda a: "run_prepared", ttnn.bringup.run_prepared)
    monkeypatch.setattr(ttnn.bringup, "run_prepared", rp)
    orig_sp = profiler.signpost

    def signpost(name):
        census.section = name
        orig_sp(name)

    monkeypatch.setattr(profiler, "signpost", signpost)


def _report(side, rows, n_chunks_host):
    by_layer = defaultdict(list)
    for r in rows:
        by_layer[r[0]].append(r)
    total = 0
    for layer in sorted(by_layer, key=lambda x: (x is None, x)):
        rs = by_layer[layer]
        total += len(rs)
        sec = Counter(r[1] for r in rs)
        ops = Counter(r[2] for r in rs)
        print(f"P.64 census {side} L{layer}: {len(rs)} launches; by section {dict(sec.most_common())}", flush=True)
        print(f"P.64 census {side} L{layer}: by op {dict(ops.most_common())}", flush=True)
        sites = defaultdict(list)
        for r in rs:
            sites[(r[1], r[2], r[3])].append(r[4])
        for (sec_, op, site), ts in sorted(sites.items(), key=lambda kv: (str(kv[0][0]), kv[0][2])):
            print(
                f"P.64 census {side} L{layer}:   {sec_} {op} @ {site} x{len(ts)} {1e6 * statistics.median(ts):.1f} us",
                flush=True,
            )
    per_op = defaultdict(list)
    for r in rows:
        per_op[r[2]].append(r[4])
    print(
        f"P.64 census {side}: {total} launches a chunk ({len(LAYERS)} layers); us a call (median) "
        + str({k: round(1e6 * statistics.median(v), 1) for k, v in sorted(per_op.items(), key=lambda kv: -len(kv[1]))}),
        flush=True,
    )


def _host(t) -> torch.Tensor:
    """Every chip's copy of a device tensor (chip-linear), stacked on the host."""
    return torch.stack([ttnn.to_torch(x) for x in ttnn.get_device_tensors(t)])


@mesh_parametrize(timeout_s=3 * 3600)
def test_attn_launch_census(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tests.workloads import prompt_ids as _ids
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    settings.thaw()
    census = Census()
    model = batch.device_model(S, mesh_device, LAYERS, lm_head=False)
    _install(monkeypatch, census)
    tokens = _ids("novel")
    chunk = int(S.get("target.chunk"))
    outs = {}
    for side in SIDES:
        monkeypatch.setenv("DS41_ATTN_FUSE", side)
        settings.thaw()
        hosts = []
        for p in range(PASSES):
            state = model.new_state(SEQ)
            try:
                for c in range(CHUNKS):
                    st = c * chunk
                    t = tokens[st : st + chunk]
                    last = p == PASSES - 1 and c == CHUNKS - 1
                    census.rows.clear()
                    census.on = last
                    ttnn.synchronize_device(mesh_device)
                    t0 = time.perf_counter()
                    census.layer = None
                    begin_chunk(model, t, st)
                    h = model.embed(t)
                    for i in LAYERS:
                        census.layer = i
                        h2 = model.layer(i, h, st, state)
                        if COMPARE and p == PASSES - 1 and not census.on:
                            outs.setdefault(side, {})[(c, i)] = _host(h2.streams)
                        model.free(h)
                        h = h2
                    model.free(h)
                    t1 = time.perf_counter()
                    census.on = False
                    ttnn.synchronize_device(mesh_device)
                    if p and c and not last and not COMPARE:  # (the compare's reads are host time)
                        hosts.append((t1 - t0) * 1e3)
                    if COMPARE and p == PASSES - 1:
                        for i in LAYERS:  # (the served snapshot reads the ring: its pieces' bytes)
                            w = state.window[i]
                            if getattr(w, "snap", None) is not None:
                                outs.setdefault(side, {})[(c, i, "snap")] = torch.stack(w.snapshot_host())
                            if getattr(w, "rtail", None) is not None:
                                outs.setdefault(side, {})[(c, i, "rtail")] = _host(w.rtail)
                    if last:
                        _report(side, list(census.rows), CHUNKS)
            finally:
                _free(state)
        if hosts:
            print(
                f"P.64 census {side}: host {statistics.mean(hosts):.2f} ms a chunk ({len(LAYERS)} layers)", flush=True
            )
    if COMPARE and len(outs) > 1:
        a, b = list(outs)[:2]
        bad = 0
        for key, x in outs[a].items():
            y = outs[b].get(key)
            if y is None:
                continue
            if not torch.equal(x, y):
                bad += 1
                d = (x.float() - y.float()).abs().max().item()
                print(f"P.64 census compare {a} vs {b} chunk/layer {key}: max abs diff {d}", flush=True)
        n = sum(1 for k in outs[a] if outs[b].get(k) is not None)
        print(
            f"P.64 census compare {a} vs {b}: {bad} of {n} differ (every chip: layer streams of chunks 0..{CHUNKS - 2}, "
            "snapshots and kept tails of every chunk)",
            flush=True,
        )
        assert bad == 0
