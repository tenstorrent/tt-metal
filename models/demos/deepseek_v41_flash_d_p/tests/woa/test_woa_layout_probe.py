# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.3 (OV-79, OWNER 2026-11-27: "why scatter the cores into 8 clusters at all?"), op level on the held mesh, no model
load: the bank-local reader on TODAY's core set (bank.plan shape "rect": one 8 x c rectangle from (0, 0), grid row g =
group g = bank g, its x sender at column 0 as today's) vs the band clusters (WA.2's best: band, 10 cores a bank), device
AND host us a call at 256 and 160 rows, each exact vs today's call (same block_w: today's K order).

Host (OV-78): the split of a call's host time, model-like (output buffer passed, as attention._out_wo does):
woa_lp_host_us_<cfg>_<rows> (the whole call), woa_lp_host_py_us_<cfg>_<rows> (the Python wrapper alone: checks and the
prepared-program key, run_prepared stubbed), woa_lp_host_rp_us_<cfg>_<rows> (run_prepared alone, the enqueue), and the
cost of the per-call str() of each weight's memory config the key carried before WA.3 (woa_lp_host_str_us_<kind>).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_layout_probe.py"
"""

from __future__ import annotations

import os
import time

import pytest
from ttnn.bringup.grouped_matmul_ttnn import bank

import ttnn
from models.demos.common.bringup.core.metrics import record
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tests.woa.test_woa_probe import ITERS, Case, G, K, N, _maxdiff
from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

BATCH_SHARED = True  # (a mesh-only op probe: the resident runner's held mesh, 0 model loads)
BATCH_ITEM_VARS = ("WOA_LP_FULL_GRID", "WOA_LP_SWEEP")
SWEEP = [(d, wi, xa) for d in (2, 3, 4) for wi in range(1, d + 1) for xa in range(0, min(3, d - 1) + 1)]
ROWS = (256, 160)
HOST_ITERS = 50
BEST = {256: (4, 1, 1), 160: (4, 2, 1)}  # WA.2's best band-10 knobs (depth, w_inflight, x_ahead) at today's block_w
# today's core set: its knob sweep (run 1 of this probe: rect11 best 147.9 us at 256 rows, 121.5 at 160; rect10 worse)
RECT_KNOBS = {256: [(3, 2, 1)], 160: [(3, 1, 1)]}
LAYOUTS = [  # (name, shape, cores a bank, order, contiguous bands)
    ("band10", "band", 10, "row", False),
    ("band10c", "band", 10, "col", False),
    ("band10cc", "band", 10, "col", True),
    ("rect11", "rect", 11, "row", False),
]
HOST_ROUNDS = 5


def GRID(mesh):
    """The layouts' grid: the model's (WOA_BANK, tt/attention.woa_layout): the matmul-capped compute grid (11 x 10 on a
    Galaxy; a gather-hide region's compute sub-device, DS41-382), or the full grid when WOA_LP_FULL_GRID=1 (WA.2's)."""
    return None if os.environ.get("WOA_LP_FULL_GRID") == "1" else max_grid(mesh)  # diagnostic


def _host(mesh, fn, iters=HOST_ITERS) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    return (t1 - t0) / iters * 1e6


@mesh_parametrize(timeout_s=1800)
def test_woa_layout_probe(mesh_device):
    mesh = mesh_device
    if mesh is None:
        pytest.skip("device probe")
    cap = max_grid(mesh)
    cap = 10**9 if cap is None else cap.x * cap.y
    summary = []
    for rows in ROWS:
        c = Case(mesh, rows)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, G, rows, N]), c.ydt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
        )
        lays = {
            name: bank.plan(mesh, G, K, N, cap, n, shape=sh, order=o, contiguous=cg, grid=GRID(mesh))
            for name, sh, n, o, cg in LAYOUTS
        }
        assert all(v is not None for v in lays.values()), {k: v is None for k, v in lays.items()}
        wbs = {k: bank.upload(c.wt, lay, mesh) for k, lay in lays.items()}

        def today(c=c, out=out):
            return ttnn.bringup.grouped_matmul(
                c.x, c.w, dtype=c.ydt, compute_kernel_config=c.ckc, output_tensor=out, max_grid=c.grid,
                block_w=c.bw, depth=c.depth, rt=c.rt,
            )  # fmt: skip

        def bank_call(name, d, wi, xa, c=c, out=out, o=None):
            lay_name, og = (name[:-3], True) if name.endswith("_1g") else (name, False)
            return ttnn.bringup.grouped_matmul(
                c.x, wbs[lay_name], w_layout=lays[lay_name], dtype=c.ydt, compute_kernel_config=c.ckc,
                output_tensor=out if o is None else o, max_grid=c.grid, block_w=c.bw, depth=d, rt=c.rt,
                w_inflight=wi, x_ahead=xa, one_group=og,
            )  # fmt: skip

        # (a "_1g" name: the same layout with one kernel group, the column tiles a runtime arg)
        names = ("band10", "band10c", "band10c_1g", "band10_1g", "band10cc", "band10cc_1g")
        cfgs = [(b, BEST[rows]) for b in names] + [("rect11", k) for k in RECT_KNOBS[rows]]
        if os.environ.get("WOA_LP_SWEEP") == "1":  # diagnostic: band10 one group's knobs on the model's grid
            cfgs += [("band10_1g", k) for k in SWEEP if k != BEST[rows]]
        # exactness: each config vs today's call (fresh outputs), and a repeat
        ref = ttnn.bringup.grouped_matmul(
            c.x, c.w, dtype=c.ydt, compute_kernel_config=c.ckc, max_grid=c.grid, block_w=c.bw, depth=c.depth, rt=c.rt
        )
        exact = {}
        for name, (d, wi, xa) in cfgs:
            o1 = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, G, rows, N]), c.ydt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            o2 = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, G, rows, N]), c.ydt, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
            )
            bank_call(name, d, wi, xa, o=o1)
            bank_call(name, d, wi, xa, o=o2)
            exact[(name, d, wi, xa)] = (_maxdiff(o1, ref), _maxdiff(o2, o1))
            ttnn.deallocate(o1)
            ttnn.deallocate(o2)
        ttnn.deallocate(ref)
        cands = [("today", today)] + [
            (f"{n}_d{d}_wi{wi}_xa{xa}", lambda n=n, d=d, wi=wi, xa=xa: bank_call(n, d, wi, xa))
            for n, (d, wi, xa) in cfgs
        ]
        t = {}
        for _ in range(2):  # (twice, alternating order inside time_us's one section a config; the min of the two)
            got = _dev_us(mesh, cands)
            for tag, us in got.items():
                t[tag] = min(t.get(tag, float("inf")), us)
        # host split: the whole call, the Python wrapper alone (run_prepared stubbed), run_prepared alone; rounds
        # alternating between the configs, the median a config (the box's host time drifts between runs)
        import importlib
        import statistics

        gm = importlib.import_module(
            "ttnn.bringup.grouped_matmul_ttnn.generic.grouped_matmul"
        )  # (NC.1: pre-port route)
        mg = (c.grid.x, c.grid.y) if hasattr(c.grid, "x") else c.grid
        rp = ttnn.bringup.run_prepared
        pps = {"today": (gm._prepared(c.x, c.w, out, c.ckc, c.grid, c.bw, c.depth, c.rt), c.w)}
        for name, (d, wi, xa) in cfgs[: len(names) + 1]:
            ln, og = (name[:-3], True) if name.endswith("_1g") else (name, False)
            pps[name] = (
                gm._prepared(c.x, wbs[ln], out, c.ckc, c.grid, c.bw, d, c.rt, (), (lays[ln], wi, xa, og)),
                wbs[ln],
            )
        calls = {"today": cands[0][1]} | {cfgs[i][0]: cands[i + 1][1] for i in range(len(names) + 1)}
        samples = {k: {"whole": [], "py": [], "rp": []} for k in calls}
        for _ in range(HOST_ROUNDS):
            for k, fn in calls.items():
                samples[k]["whole"].append(_host(mesh, fn))
                pp, w = pps[k]
                samples[k]["rp"].append(_host(mesh, lambda pp=pp, w=w: rp(pp, [c.x, w, out])))
                ttnn.bringup.run_prepared = lambda *a, **k_: None
                try:
                    samples[k]["py"].append(_host(mesh, fn, 4 * HOST_ITERS))
                finally:
                    ttnn.bringup.run_prepared = rp
        host = {k: {m: round(statistics.median(v), 1) for m, v in d_.items()} for k, d_ in samples.items()}
        rect_best = min((k for k in t if k.startswith("rect")), key=lambda k: t[k])
        for tag, us in t.items():
            record(f"woa_lp_us_{tag}_{rows}", round(us, 2))
        for tag, h in host.items():
            for m, v in h.items():
                record(f"woa_lp_host_{m}_us_{tag}_{rows}", v)
        bad = {k: v for k, v in exact.items() if v != (0.0, 0.0)}
        record(f"woa_lp_mismatch_{rows}", max(max(v) for v in exact.values()), inexact=len(bad))
        print(f"[wa3-lp] rows {rows} mg {mg}: device us {({k: round(v, 2) for k, v in t.items()})}")
        print(
            f"[wa3-lp] rows {rows}: rect best {rect_best} {t[rect_best]:.2f} us vs band10 {t[cands[1][0]]:.2f} vs today {t['today']:.2f}"
        )
        print(f"[wa3-lp] rows {rows}: host us (median of {HOST_ROUNDS} rounds) {host}")
        print(f"[wa3-lp] rows {rows}: inexact {bad}")
        summary.append((rows, t, host, bad))
        for w_ in wbs.values():
            ttnn.deallocate(w_)
        ttnn.deallocate(out)
    # the per-call str() of the memory configs the key carried before WA.3
    c = Case(mesh, 160)
    lay = bank.plan(mesh, G, K, N, cap, 10, shape="band", order="col", grid=GRID(mesh))
    wb = bank.upload(c.wt, lay, mesh)
    for kind, w in (("interleaved", c.w), ("bank", wb), ("x", c.x)):
        t0 = time.perf_counter()
        for _ in range(200):
            str(w.memory_config())
        us = round((time.perf_counter() - t0) / 200 * 1e6, 2)
        record(f"woa_lp_host_str_us_{kind}", us)
        print(f"[wa3-lp] str(memory_config()) of {kind}: {us} us")
    t0 = time.perf_counter()
    for _ in range(200):
        tuple(wb.memory_config().shard_spec.shape)
    us = round((time.perf_counter() - t0) / 200 * 1e6, 2)
    record("woa_lp_host_shardcheck_us", us)
    print(f"[wa3-lp] the bank weight's shard-shape check: {us} us")
    ttnn.deallocate(wb)
    for rows, *_r, bad in summary:
        assert not bad, f"rows {rows}: inexact configs {bad}"


def _dev_us(mesh, cands: list) -> dict:
    """Device us a call of each (tag, fn) (rt profiler, slowest chip, ITERS calls a section; fn writes a
    preallocated output)."""
    for _, fn in cands:
        fn()
    ttnn.synchronize_device(mesh)
    profiler.enable(mesh, rt=True)
    try:
        for tag, fn in cands:
            profiler.signpost(tag)
            for _ in range(ITERS):
                fn()
            profiler.signpost("gap")
            ttnn.synchronize_device(mesh)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    return {tag: prof["kernel_ns"].get(tag, float("nan")) / ITERS / 1e3 for tag, _ in cands}
