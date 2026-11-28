# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.2 (op-level probe on the mesh, no model load): wo_a's ttnn.bringup.grouped_matmul as the model calls it today
(8 groups x [rows, 4096] @ [4096, 1024], bfp8 x and weights, the model's output dtype, ATTN_FIDELITY / fp32 DEST, the
tuned knobs mm.gmm_knobs) at 256 and 160 rows a chip.

Step 0 (WOA_BANK_DESIGN.md section 7, OV-76): today's op with the diagnostic stubs (no weight reads, no compute, no x
read / multicast): if no-weight-reads runs >= 0.9 x today at 256 rows, wo_a is not weight-read bound and the bank layout
cannot pay (KILL). Records woa_us_today_<rows>, woa_repeat_diff_today_<rows> (today's output, two runs on fresh
buffers, max abs diff), woa_host_us_today_<rows> and woa_stub_us_<stub>_<rows>.

Step 1 (the bank path, ttnn.bringup.grouped_matmul(w_layout=bank.plan(..)), bank.py): at both row counts, the bank
path vs today's call (max abs diff on every chip, must be 0), a repeat of each on fresh buffers (must be 0), device and
host us a call. Step 2, the knob sweep IN PARALLEL ACROSS CHIPS: one mesh program maps a different bank-path knob set
(block_w, depth, w_inflight, x_ahead) to each of the 32 chips (each LAYOUTS entry, bank.plan's shape and cores a
bank: one weight tensor and its own launches); every chip's output is compared with today's op at the same block_w (the same K order) on that
chip, its device us normalised by that chip's own time of today's call (the chips differ by up to ~6 %); the best
exact config is then confirmed on the whole mesh (slowest chip) against today's call. Records woa_mismatch_<case>,
woa_repeat_diff_<case>, woa_us_today_<rows>, woa_us_bank_<rows> (best config at today's block_w: exact vs today's
shipped output), woa_gain_ms_c8192 / _c5120 (us a call x CALLS layers); the sweep table is printed ([wa2-sweep]).

Device us a call: the real-time profiler (slowest chip, ITERS calls a section); host us: ITERS warm calls back to
back, perf_counter, no sync between them.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_probe.py"
"""

from __future__ import annotations

import time

import pytest
import torch
from ttnn.bringup.grouped_matmul_ttnn import bank
from ttnn.bringup.grouped_matmul_ttnn.generic.grouped_matmul import (  # (NC.1: the pre-port Python route, test only)
    bank_program,
)

import ttnn
from models.demos.common.bringup.core.metrics import record
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize
from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
from models.demos.deepseek_v41_flash_d_p.tt.common import attn_compute_config, bf16_activations, gemm_dtype
from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

BATCH_SHARED = True  # (a mesh-only op probe: the resident runner's held mesh, 0 model loads)
G, K, N = 8, 4096, 1024
ROWS = (256, 160)
ITERS = 20
STUBS = ("no_w", "no_compute", "no_x")
CALLS = 21  # wo_a calls a chunk (the brief's "x 21 layers"; WA.1's bound used 20: layers 0-20 under DECODER_SKIP)
CHUNK = {256: 8192, 160: 5120}
LAYOUTS = [("band", 12), ("band", 10), ("row", 12), ("row", 11), ("row", 10)]  # (bank.plan shape, cores a bank)
SWEEP = [  # (block_w, depth, w_inflight, x_ahead)
    (bw, d, wi, xa) for bw in (2, 4, 8) for d in (2, 3, 4) for wi in range(1, d + 1) for xa in range(min(3, d - 1) + 1)
]
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _rep(mesh, t, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def _maxdiff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(_chips(a), _chips(b)))


class Case:
    """One row count: fixed-seed x and w on the mesh, the model's dtypes, config and knobs."""

    def __init__(self, mesh, rows: int):
        self.mesh, self.rows = mesh, rows
        g = torch.Generator().manual_seed(rows)
        self.xt = torch.randn(1, G, rows, K, generator=g)
        self.wt = torch.randn(1, G, K, N, generator=g) / K**0.5
        self.x = _rep(mesh, self.xt, gemm_dtype())  # (GEMM_INPUTS: wo_a's input)
        self.w = _rep(mesh, self.wt, ttnn.bfloat8_b)  # (DENSE_WEIGHTS: the fp8 checkpoint's bfp8)
        self.ydt = gemm_dtype() if bf16_activations() else ttnn.float32  # (attention._out_wo, glue on)
        self.ckc = attn_compute_config(mesh)
        self.grid = max_grid(mesh)
        self.bw, self.depth, self.rt = mm.gmm_knobs(tuple(self.x.shape), tuple(self.w.shape))

    def today(self, diag=(), **kw):
        a = dict(block_w=self.bw, depth=self.depth, rt=self.rt)
        a.update(kw)
        return ttnn.bringup.grouped_matmul(
            self.x, self.w, dtype=self.ydt, compute_kernel_config=self.ckc, max_grid=self.grid, diag=diag, **a
        )


def time_us(mesh, cands: list) -> dict:
    """cands: [(tag, fn)], fn() -> an output tensor (freed here). Device us a call of each (rt profiler, slowest chip,
    ITERS calls in one section) and the per-chip us ({tag: (us, {chip: us})})."""
    for _, fn in cands:  # (warm: every program built before the profiler starts)
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    profiler.enable(mesh, rt=True)
    try:
        for tag, fn in cands:
            profiler.signpost(tag)
            outs = [fn() for _ in range(ITERS)]
            profiler.signpost("gap")
            ttnn.synchronize_device(mesh)
            for o in outs:
                ttnn.deallocate(o)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    res = {}
    for tag, _ in cands:
        per = {c: ns / ITERS / 1e3 for c, ns in prof["kernel_ns_dev"].get(tag, {}).items()}
        res[tag] = (prof["kernel_ns"].get(tag, float("nan")) / ITERS / 1e3, per)
    return res


def host_us(mesh, fn) -> float:
    ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(ITERS)]
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    for o in outs:
        ttnn.deallocate(o)
    return (t1 - t0) / ITERS * 1e6


_CASES = {}


def cases(mesh) -> dict:
    """The row counts' inputs, built once a mesh (the tests share them)."""
    if mesh is None:
        pytest.skip("device probe")
    if _CASES.get("mesh") is not mesh:
        _CASES.clear()
        _CASES.update(mesh=mesh, rows={r: Case(mesh, r) for r in ROWS})
    return _CASES["rows"]


@mesh_parametrize
def test_woa_today_and_stubs(mesh_device):
    """Step 0: today's call, its repeat, and the three diagnostic stubs at 256 and 160 rows."""
    mesh = mesh_device
    for rows, c in cases(mesh).items():
        a, b = c.today(), c.today()
        rep = _maxdiff(a, b)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        record(f"woa_repeat_diff_today_{rows}", rep)
        cands = [("today", c.today)] + [(s, lambda s=s, c=c: c.today(diag=(s,))) for s in STUBS]
        t = time_us(mesh, [(f"r{rows}/{tag}", fn) for tag, fn in cands])
        us_today = t[f"r{rows}/today"][0]
        record(f"woa_us_today_{rows}", round(us_today, 2), knobs=[c.bw, c.depth, c.rt])
        record(f"woa_host_us_today_{rows}", round(host_us(mesh, c.today), 1))
        print(f"[wa2] rows {rows} knobs bw{c.bw} d{c.depth} rt{c.rt}: today {us_today:.2f} us, repeat diff {rep}")
        for s in STUBS:
            us = t[f"r{rows}/{s}"][0]
            record(f"woa_stub_us_{s}_{rows}", round(us, 2))
            print(f"[wa2]   stub {s}: {us:.2f} us ({us / us_today:.3f} x today)")
        per = t[f"r{rows}/today"][1]
        if per:
            print(f"[wa2]   today per chip: min {min(per.values()):.2f} max {max(per.values()):.2f} us")
        assert rep == 0.0, f"today's grouped_matmul not repeatable at {rows} rows: {rep}"


def _hetero(mesh, descs: list, io: list):
    """One mesh program, chip i (row-major mesh order) running descs[i]; prepared once, run on io."""
    rows, cols = mesh.shape[0], mesh.shape[1]
    md = ttnn.MeshProgramDescriptor()
    for i, d in enumerate(descs):
        r, c = divmod(i, cols)
        md[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = d
    assert len(descs) == rows * cols
    return ttnn.bringup.prepare_program(md, io)


def _per_chip_diff(a, b) -> list:
    return [float((x - y).abs().max()) for x, y in zip(_chips(a), _chips(b))]


def _sweep(mesh, c, lay_name, lay, wb, base_chip_us: dict) -> list:
    """Every SWEEP knob set on one chip each (32 at a time). Returns [{knobs, mismatch, repeat, us, norm}]."""
    nchips = mesh.get_num_devices()
    refs = {}
    for bw in sorted({k[0] for k in SWEEP}):
        t = c.today(block_w=bw)
        refs[bw] = _chips(t)
        ttnn.deallocate(t)
    out_shape = ttnn.Shape([1, G, c.rows, N])

    def alloc():
        return ttnn.allocate_tensor_on_device(out_shape, c.ydt, ttnn.TILE_LAYOUT, mesh, DRAM)

    rows = []
    for at in range(0, len(SWEEP), nchips):
        group = SWEEP[at : at + nchips]
        knobs = group + [group[0]] * (nchips - len(group))  # (chips past the list repeat the first set)
        y0 = alloc()
        descs = [bank_program(c.x, wb, y0, c.ckc, lay, bw, d, c.rt, wi, xa) for bw, d, wi, xa in knobs]
        pp = _hetero(mesh, descs, [c.x, wb, y0])
        ttnn.bringup.run_prepared(pp, [c.x, wb, y0])
        y1 = alloc()
        ttnn.bringup.run_prepared(pp, [c.x, wb, y1])
        ya, yb = _chips(y0), _chips(y1)
        ttnn.synchronize_device(mesh)

        def run():
            y = alloc()
            ttnn.bringup.run_prepared(pp, [c.x, wb, y])
            return y

        t = time_us(mesh, [(f"sweep{at}", run)])[f"sweep{at}"][1]
        for i, (bw, d, wi, xa) in enumerate(group):
            ref = refs[bw][i]
            us = t.get(i, float("nan"))
            rows.append(
                dict(
                    lay=lay_name,
                    bw=bw,
                    depth=d,
                    wi=wi,
                    xa=xa,
                    chip=i,
                    mismatch=float((ya[i] - ref).abs().max()),
                    repeat=float((yb[i] - ya[i]).abs().max()),
                    us=round(us, 2),
                    norm=round(us / base_chip_us[i], 4),
                )
            )
        for y in (y0, y1):
            ttnn.deallocate(y)
    return rows


@mesh_parametrize(timeout_s=3000)
def test_woa_bank(mesh_device):
    """Steps 1 and 2: the bank path vs today at 256 and 160 rows, its knob sweep across chips, the best confirmed."""
    mesh = mesh_device
    cap = max_grid(mesh)
    cap = 10**9 if cap is None else cap.x * cap.y  # (MAX_WORKER_CORES: 110 on a Galaxy, agent rule 18)
    table = {}
    for rows, c in cases(mesh).items():
        lays = {f"{sh}{cpb}": bank.plan(mesh, G, K, N, cap, cpb, shape=sh) for sh, cpb in LAYOUTS}
        assert lays["band12"] is not None, "no bank layout for wo_a"
        lays = {k: v for k, v in lays.items() if v is not None}
        wbs = {k: bank.upload(c.wt, lay, mesh) for k, lay in lays.items()}
        lay, wb = lays["band12"], wbs["band12"]

        def bank_call(lay=lay, wb=wb, c=c, **kw):
            a = dict(block_w=c.bw, depth=c.depth, rt=c.rt, w_inflight=1, x_ahead=1)
            a.update(kw)
            return ttnn.bringup.grouped_matmul(
                c.x, wb, w_layout=lay, dtype=c.ydt, compute_kernel_config=c.ckc, max_grid=c.grid, **a
            )

        # step 1: the default bank call (band12, today's knobs, w_inflight 1, x_ahead 1) vs today's call
        t0, t1, b0, b1 = c.today(), c.today(), bank_call(), bank_call()
        mm0, rep0, rep_today = _maxdiff(b0, t0), _maxdiff(b1, b0), _maxdiff(t1, t0)
        for t in (t0, t1, b0, b1):
            ttnn.deallocate(t)
        record(f"woa_mismatch_r{rows}_default", mm0, cores=len(lay.slots))
        record(f"woa_repeat_diff_r{rows}_default", rep0)
        record(f"woa_repeat_diff_today_{rows}", rep_today)
        base = time_us(mesh, [("today", c.today), ("bank_default", bank_call)])
        base_chip = base["today"][1]
        print(
            f"[wa2] rows {rows}: today {base['today'][0]:.2f} us, bank default (bw{c.bw} d{c.depth} wi1 xa1, "
            f"{len(lay.slots)} cores) {base['bank_default'][0]:.2f} us; mismatch {mm0} repeat {rep0}"
        )
        # step 2: the sweep, one knob set a chip
        sweep = []
        for name, lay_c in lays.items():
            t_s = time.perf_counter()
            sweep += _sweep(mesh, c, name, lay_c, wbs[name], base_chip)
            print(f"[wa2]   sweep {name}: {time.perf_counter() - t_s:.1f} s")
        bad = [r for r in sweep if r["mismatch"] or r["repeat"]]
        record(f"woa_mismatch_r{rows}_sweep", max(r["mismatch"] for r in sweep), configs=len(sweep), inexact=len(bad))
        record(f"woa_repeat_diff_r{rows}_sweep", max(r["repeat"] for r in sweep))
        exact = sorted((r for r in sweep if not r["mismatch"] and not r["repeat"]), key=lambda r: r["norm"])
        for r in exact[:8]:
            print(f"[wa2]   {r}")
        # the best at today's block_w (today's K order: exact vs the shipped output) and the best overall
        same_bw = [r for r in exact if r["bw"] == c.bw]
        picks = {"best": same_bw[0]}
        if exact and exact[0]["bw"] != c.bw:
            picks["best_any_bw"] = exact[0]
        conf = [("today", c.today)]
        for name, r in picks.items():
            kw = dict(block_w=r["bw"], depth=r["depth"], w_inflight=r["wi"], x_ahead=r["xa"])
            conf.append((name, lambda kw=kw, r=r: bank_call(lay=lays[r["lay"]], wb=wbs[r["lay"]], **kw)))
        # exactness of the picks on the whole mesh, vs today at their block_w
        res = {}
        for name, fn in conf[1:]:
            r = picks[name]
            ref = c.today(block_w=r["bw"])
            y0, y1 = fn(), fn()
            res[name] = (_maxdiff(y0, ref), _maxdiff(y1, y0))
            for t in (ref, y0, y1):
                ttnn.deallocate(t)
        t = time_us(mesh, conf)
        us_today = t["today"][0]
        record(f"woa_us_today_{rows}", round(us_today, 2), knobs=[c.bw, c.depth, c.rt])
        record(f"woa_host_us_today_{rows}", round(host_us(mesh, c.today), 1))
        for name, (mmx, rep) in res.items():
            r = picks[name]
            record(f"woa_mismatch_r{rows}_{name}", mmx, knobs=[r["lay"], r["bw"], r["depth"], r["wi"], r["xa"]])
            record(f"woa_repeat_diff_r{rows}_{name}", rep)
            print(f"[wa2]   {name} {r}: {t[name][0]:.2f} us (whole mesh), mismatch {mmx} repeat {rep}")
        b = picks["best"]
        us_bank = t["best"][0]
        record(f"woa_us_bank_{rows}", round(us_bank, 2), knobs=[b["lay"], b["bw"], b["depth"], b["wi"], b["xa"]])
        if "best_any_bw" in picks:
            record(f"woa_us_bank_any_bw_{rows}", round(t["best_any_bw"][0], 2))
        kw = dict(block_w=b["bw"], depth=b["depth"], w_inflight=b["wi"], x_ahead=b["xa"])
        record(
            f"woa_host_us_bank_{rows}",
            round(host_us(mesh, lambda: bank_call(lay=lays[b["lay"]], wb=wbs[b["lay"]], **kw)), 1),
        )
        gain = (us_today - us_bank) * CALLS / 1e3
        record(f"woa_gain_ms_c{CHUNK[rows]}", round(gain, 3), calls=CALLS)
        print(f"[wa2] rows {rows}: today {us_today:.2f} us, bank best {us_bank:.2f} us, gain {gain:.3f} ms a chunk")
        table[rows] = dict(
            today_us=us_today,
            bank_us=us_bank,
            default_us=base["bank_default"][0],
            picks=picks,
            confirm={k: v[0] for k, v in t.items()},
            sweep=sweep,
        )
        for w_ in wbs.values():
            ttnn.deallocate(w_)
    for rows, t in table.items():  # (the sweep table, for results/WA.2_probe.md)
        for r in sorted(t["sweep"], key=lambda r: r["norm"]):
            print(f"[wa2-sweep] rows {rows} {r}")
    for rows in table:
        assert table[rows]["picks"]["best"], rows
