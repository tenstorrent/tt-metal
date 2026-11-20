# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.87 (attempt 2): the device split of what a fused engram lookup + KV-history gather could overlap, on the held
model at P87_POS (default 145408): the engram lookup program alone, each history source's pack (slice + tilize +
ckv_bfp4) and its record all-gather (high_bw_all_gather) separately, the index-key prefetch, and lookup + the whole
history prefetch back to back. rt profiler (program start / end on the chips), P87_REPS repetitions; per-chip max of
each section. A measurement only (no model change): the fusion's ceiling is min(lookup, the gathers' fabric time).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "DS41_DECODER_SKIP=decode
        BRINGUP_POS_TOKENS=agentic scripts/run_safe_pytest.sh --run-all --no-precompile
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_engram_split_probe.py"
"""

from __future__ import annotations

import importlib
import os

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden
from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("P87_POS",)
BATCH_POS_VARS = ("P87_POS",)
BATCH_LAYERS = "executed"


def _side(monkeypatch, v: str) -> None:
    monkeypatch.setenv(settings.env_name("ENGRAM_DEVICE"), v)
    if settings.snap is not None:
        settings.snap["ENGRAM_DEVICE"] = v


@mesh_parametrize(timeout_s=3600)
def test_engram_split_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn

    hg = importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.history_gather")
    ccl = importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.ccl")
    ed = importlib.import_module("models.demos.deepseek_v41_flash_d_p.tt.engram_device")
    chunk = int(S.get("target.chunk"))
    p = int(os.environ.get("P87_POS", "145408"))
    reps = int(os.environ.get("P87_REPS", "3"))
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    monkeypatch.setitem(S.data["target"], "seq", max(p + chunk, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    assert model.engram_dev is not None, "the device tables did not load"
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    _side(monkeypatch, "on")
    g = PositionGolden(S, p, chunk)
    ids = g.tokens()
    state = model.new_state(p + chunk)
    try:
        # the real previous chunk (its caches and the slot's device id history), then this chunk's begin only
        start = p - chunk
        model.begin_chunk(ids[start:p], start, last=False)
        h = model.embed(ids[start:p])
        for i in layers:
            h2 = model.layer(i, h, start, state)
            model.free(h)
            h = h2
        model.free(h)
        model.begin_chunk(ids[p : p + chunk], p, last=False)
        model._pf_start = model._kpf_start = None  # (the probe enqueues the prefetches itself)
        st = model.state
        srcs = [st.compress[i].ckv for i in sorted(model.history_sources()) if i in st.compress]
        nat = ids[p : p + chunk].reshape(-1).long()
        dev_ids = ed.engine_ids(mesh_device, nat, p, chunk)
        hist, _ = st.device_engram(p)
        info = []
        for c in srcs:
            owners, axis = hg.owners_and_axis(c.mesh, c.tp_axis is not None)
            hrows = (p // chunk) * (c.stripe // c.ratio)
            info.append((c, owners, axis, hrows))
        print(
            "P87 probe sources: " + str([(c.history_key, owners, hrows, c.slab.rows) for c, owners, _, hrows in info]),
            flush=True,
        )

        def lookup():
            model.engram_dev(dev_ids, hist, p, chunk, None)

        def pack(k):
            c, owners, axis, hrows = info[k]
            return hg._pack(c, 0, hrows)

        def gather(k, loc):
            c, owners, axis, hrows = info[k]
            buf = hg._buffer(c.mesh, c.history_key)
            s0, s1 = ccl._sems(buf.mesh, axis)
            ttnn.experimental.high_bw_all_gather(
                loc,
                2,
                buf.tensor,
                cluster_axis=axis,
                num_links=ccl.AG_LINKS,
                gathered_dim_size=owners * hrows,
                ready_semaphore=s0,
                data_valid_semaphore=s1,
            )

        # warm (program caches)
        for _ in range(2):
            lookup()
            for k in range(len(info)):
                gather(k, pack(k))
        ttnn.synchronize_device(mesh_device)
        import time

        walls = {}

        def timed(name, fn, n=10):
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            ttnn.synchronize_device(mesh_device)
            walls[name] = round((time.perf_counter() - t0) / n * 1e3, 3)

        timed("lookup", lookup)
        for k in range(len(info)):
            timed(f"pack{k}", lambda k=k: pack(k))
            timed(f"pack_gather{k}", lambda k=k: gather(k, pack(k)))
        timed("prefetch_all", lambda: [gather(k, pack(k)) for k in range(len(info))])
        timed("lookup_prefetch_all", lambda: [lookup()] + [gather(k, pack(k)) for k in range(len(info))])
        print(f"P87 probe synced walls ms a call: {walls}", flush=True)
        for k, v in walls.items():
            metrics.record(f"p87_probe_wall_{k}_ms_p{p}", v)
        profiler.enable(mesh_device, rt=True)
        try:
            for _ in range(reps):
                profiler.signpost("probe.lookup")
                lookup()
                for k in range(len(info)):
                    profiler.signpost(f"probe.pack{k}")
                    loc = pack(k)
                    profiler.signpost(f"probe.gather{k}")
                    gather(k, loc)
                profiler.signpost("probe.both")
                lookup()
                for k in range(len(info)):
                    gather(k, pack(k))
                profiler.signpost("probe.idle")
            profiler.signpost("end")
            profiler.finish_rt()
            res = profiler.result()
        finally:
            profiler.disable()
        dev_ids.deallocate()
    finally:
        _side(monkeypatch, str(settings.table["ENGRAM_DEVICE"].default))
        state.free()
    out = {}
    for sec, per_chip in sorted(res["kernel_ns_dev"].items()):
        if not per_chip:
            print(f"P87 probe {sec}: no records", flush=True)
            continue
        ms = max(per_chip.values()) / 1e6 / reps
        out[sec] = round(ms, 3)
        progs = res["programs"].get(sec, 0) // reps
        print(f"P87 probe {sec}: {ms:.3f} ms a call (slowest chip; {progs} programs a rep)", flush=True)
        metrics.record(f"p87_probe_{sec.split('.', 1)[-1]}_ms_p{p}", round(ms, 3))
    packs = sum(v for k, v in out.items() if k.startswith("probe.pack"))
    gathers = sum(v for k, v in out.items() if k.startswith("probe.gather"))
    print(
        f"P87 probe p{p}: lookup {out.get('probe.lookup')} ms, packs {packs:.3f}, gathers {gathers:.3f}, "
        f"both {out.get('probe.both')} ms; fusion ceiling min(lookup, gathers) = "
        f"{min(out.get('probe.lookup', 0), gathers):.3f} ms",
        flush=True,
    )
