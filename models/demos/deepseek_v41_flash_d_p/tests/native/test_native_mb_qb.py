# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.QB (OWNER 2026-11-28): the batching branch's wq_b bank layout (FIX.42 / QB.1) and the tiny-op fusions that apply
to single requests (NC.9) on model_bringup, op level on the held mesh (no model load), at the model's own calls:

    wqb<rows>        tt/attention.TtWindowAttention._wqb_mm (the shipped _q_proj call: x [1, 1, rows, 1280] bfp8 @
                     wq_b [1280, 32768] bfp8 -> bf16 into a preallocated output, ATTN_FIDELITY), WQB_BANK=on (the G = 1
                     bank call: tt/attention.wqb_layout, mm_configs.WQB_BANK_KNOBS[rows]) vs off (today's mm.linear with
                     the model's program config, MM_TUNED160 at 160 rows). Not bit-identical by design (another K order,
                     DS41-390): judged by the error vs fp64 of the device's own bfp8 inputs, the repeat, the cache.
    emb5120          (NC.9 (1)) tt/model.embed of model.reorder_ids' ids: on = IdsReorder.embed (the SP all-gather,
                     then ttnn.bringup.embedding with server_start / server_valid / clamp_max); off = IdsReorder then
                     ttnn.bringup.embedding of its [1, W] ids (TILE bf16 out, DRAM, max_grid). Two (start, valid) sets,
                     ids past the vocabulary (clamped).
    rot_<what>5120   (NC.9 (3)) tt/compressor.TtCompressor._write_from's replay rotation under POS_DYNAMIC: two page
                     copies into one rot buffer -> page_copy_many (one ttnn.chunk_program PAGE_MULTI): the compressed KV
                     (512 wide) and index keys (128 wide) at ratio 4 (n = 1280), the compressed KV at ratio 128 (n = 40).
    (NC.9 (2) history tail and (4) window carry: no such page-copy sequence on model_bringup's single path; BREADCRUMBS.)

Inputs: fixed-seed random values (fusions: each chip its own bytes; wq_b: replicated, as the model's). Per case:
    native_mismatch_<case> (fusions, gated == 0): fused vs today, max abs diff of the raw bits over every io / output
        of every chip, two value sets.
    native_repeat_<case> (gated == 0): the switched-on call again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by switched-on calls after warm-up, on fresh
        buffers (an allocation shifted in between) with new data (and, fusions, the other values).
    wqb_rel_<route>_<case>: |y - fp64| / |fp64| of chip 0 (route bank = on, today = off); wqb_maxdiff_vs_today_<case>
        (diagnostic: bank vs today, bf16 bits as values).
    native_host_us_<route>_<case>, native_device_us_<route>_<case> (route on / today), native_launches_<route>_<case>.

Writes results/MB.QB_layer.md.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_mb_qb.py"
"""

from __future__ import annotations

import os
import statistics
import time
import types
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.core.metrics import record
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True  # (a mesh-only op test: the resident runner's held mesh, 0 model loads)
BATCH_LAYERS = "none"  # (F309: no model)
ITERS = 20
HOST_ITERS = 40
HOST_ROUNDS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG
CHUNK = 5120  # model_bringup's chunk
VOCAB = 4096  # (the embedding rows: the program does not depend on their count; ids up to VOCAB + 300 clamp)
DIM = 4096  # the model's hidden size (a chip: DIM / TP columns)
WQB_K, WQB_N = 1280, 32768  # wq_b under ATTN_SP32_PROJ: q_lora x heads * 512, replicated
ROUTES = ("on", "today")
REPORT = Path(__file__).resolve().parents[2] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results" / "MB.QB_layer.md"
INFO = {  # family -> (what, programs a call today -> on, calls a 5120 chunk)
    "wqb": ("wq_b: stock 1D matmul -> G = 1 bank grouped_matmul (WQB_BANK)", "1 -> 1", "21 (layers 0-20)"),
    "emb": ("ids reorder + embedding -> embedding (reorder as its reader; NATIVE_FUSIONS)", "3 -> 2", "1"),
    "rot": ("replay rotation: 2 page copies -> 1 (NATIVE_FUSIONS)", "2 -> 1", "2 a replay layer (ckv + index keys)"),
}


def _switch(env: str, on: bool) -> None:
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    os.environ[env] = "on" if on else "off"
    settings.refresh()


def _rand(gen, shape, dtype):
    if dtype == ttnn.uint32:
        return torch.randint(0, VOCAB + 300, shape, generator=gen, dtype=torch.int64).to(torch.int32)
    return (torch.randn(*shape, generator=gen) * 2).to(torch.bfloat16)


def _up(mesh, host, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mapper=None):
    R, C = tuple(mesh.shape)
    mapper = mapper or ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(0, 1))
    return ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh, memory_config=DRAM, mesh_mapper=mapper)


def _read(t) -> list:
    out = []
    for d in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(d)
        out.append(x.view(torch.int16) if x.dtype == torch.bfloat16 else x)
    return out


def _maxdiff(a: list, b: list) -> float:
    m = 0.0
    for xa, xb in zip(a, b):
        assert len(xa) == len(xb), (len(xa), len(xb))
        for ca, cb in zip(xa, xb):
            assert ca.shape == cb.shape, (ca.shape, cb.shape)
            m = max(m, float((ca.to(torch.int64) - cb.to(torch.int64)).abs().max()))
    return m


class Io:
    """A case's io tensors: (dtype, per-chip shape [1, 1, ...]) each, the host tensor [R, C, ...] (a chip its own)."""

    def __init__(self, mesh, specs: list, hosts: list):
        self.mesh, self.specs, self.hosts = mesh, specs, hosts
        self.t = [_up(mesh, h, dt) for (dt, _), h in zip(specs, hosts)]

    @staticmethod
    def new(mesh, specs: list, gen) -> "Io":
        R, C = tuple(mesh.shape)
        return Io(mesh, specs, [_rand(gen, [R, C] + list(shape[2:]), dt) for dt, shape in specs])

    def copy(self) -> "Io":
        return Io(self.mesh, self.specs, self.hosts)

    def read(self) -> list:
        return [_read(t) for t in self.t]

    def free(self):
        for t in self.t:
            if t.is_allocated():
                ttnn.deallocate(t)


class Case:
    """``call(route, io, v)`` runs the model's call with value set ``v`` (0 / 1) under the route's switch and returns
    the tensors it made (read and compared besides the io)."""

    def __init__(self, name, family, specs, call, launches, env):
        self.name, self.family, self.specs, self.call, self.launches, self.env = (
            name,
            family,
            specs,
            call,
            launches,
            env,
        )


def _fusion_cases(mesh, weight) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
    from models.demos.deepseek_v41_flash_d_p.tt.page_copy import IdsReorder, page_copy_many

    R, C = tuple(mesh.shape)
    cases = []
    env = "DS41_NATIVE_FUSIONS"

    # (1) the engine ids -> the embedding (tt/model.embed of reorder_ids)
    w = CHUNK // R
    reorder = IdsReorder(mesh, CHUNK, VOCAB)
    kw = dict(layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM, max_grid=max_grid(mesh))
    sets = [(139264 + 3 * w + 96, CHUNK), (51200 + 64, CHUNK - 1000)]

    def emb_call(route, io, v):
        start, valid = sets[v]
        if route == "on":
            return [reorder.embed(io.t[0], start, valid, weight, **kw)]
        ids = reorder(io.t[0], start, valid)
        e = ttnn.bringup.embedding(ids, weight, **kw)
        ttnn.deallocate(ids)
        return [e]

    launches = {"on": 1 + 1, "today": 1 + 2}  # (the SP all-gather + the programs)
    cases.append(Case(f"emb{CHUNK}", "emb", [(ttnn.uint32, [1, 1, w])], emb_call, launches, env))

    # (3) the replay rotation: rot[0, n - off) = rm[off, n), rot[n - off, n) = rm[0, off); off = (lo - start) / r
    def rot_case(name, n, width, offs):
        def call(route, io, v):
            rm = io.t[0]
            off = offs[v]
            rot = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n, width]), rm.dtype, rm.layout, mesh, DRAM)
            page_copy_many([(rm, rot, off, 0, n - off), (rm, rot, 0, n - off, off)])
            return [rot]

        return Case(name, "rot", [(ttnn.bfloat16, [1, 1, n, width])], call, {"on": 1, "today": 2}, env)

    cases.append(rot_case(f"rot_ckv{CHUNK}", CHUNK // 4, 512, (1024 // 4, 4096 // 4)))
    cases.append(rot_case(f"rot_idx{CHUNK}", CHUNK // 4, 128, (32 // 4 * 3, 4096 // 4 + 8)))
    cases.append(rot_case(f"rot_ckv{CHUNK}_r128", CHUNK // 128, 512, (2048 // 128, 128 // 128)))
    return cases


def _routed(case: Case, route: str, io, v: int) -> list:
    _switch(case.env, route == "on")
    try:
        return case.call(route, io, v)
    finally:
        _switch(case.env, True)


def _host_us(mesh, fn, iters=HOST_ITERS) -> float:
    for t in fn():
        ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        for t in fn():
            ttnn.deallocate(t)
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    return (t1 - t0) / iters * 1e6


def _dev_us(mesh, cands: list) -> dict:
    for _, fn in cands:
        for t in fn():
            ttnn.deallocate(t)
    ttnn.synchronize_device(mesh)
    profiler.enable(mesh, rt=True)
    try:
        for tag, fn in cands:
            profiler.signpost(tag)
            for _ in range(ITERS):
                for t in fn():
                    ttnn.deallocate(t)
            profiler.signpost("gap")
            ttnn.synchronize_device(mesh)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    return {tag: prof["kernel_ns"].get(tag, float("nan")) / ITERS / 1e3 for tag, _ in cands}


def _timings(mesh, case: Case, calls: dict) -> tuple:
    samples = {r: [] for r in ROUTES}
    for _ in range(HOST_ROUNDS):
        for r, fn in calls.items():
            samples[r].append(_host_us(mesh, fn))
    host = {r: round(statistics.median(s), 1) for r, s in samples.items()}
    dev = _dev_us(mesh, [(f"{case.name}/{r}", fn) for r, fn in calls.items()])
    return host, {r: round(dev[f"{case.name}/{r}"], 2) for r in ROUTES}


def _check_fusion(mesh, case: Case) -> dict:
    gen = torch.Generator().manual_seed(sum(map(ord, case.name)))
    mismatch = 0.0
    a = Io.new(mesh, case.specs, gen)
    b = a.copy()
    for v in (0, 1):  # two value sets, the same calls on both sides
        ma = _routed(case, "on", a, v)
        mb = _routed(case, "today", b, v)
        ttnn.synchronize_device(mesh)
        got = a.read() + [_read(t) for t in ma]
        mismatch = max(mismatch, _maxdiff(got, b.read() + [_read(t) for t in mb]))
        if v == 0:
            ref = got
        for t in ma + mb:
            ttnn.deallocate(t)
    # repeat: the first fused call again on fresh buffers from the same bytes
    c = Io(mesh, case.specs, a.hosts)
    mc = _routed(case, "on", c, 0)
    ttnn.synchronize_device(mesh)
    repeat = _maxdiff(ref, c.read() + [_read(t) for t in mc])
    for t in mc:
        ttnn.deallocate(t)
    for io in (a, b, c):
        io.free()
    # cache growth: warm (both value sets ran), then fresh buffers shifted by a pad, new data, the other values
    pad = Io.new(mesh, [(ttnn.bfloat16, [1, 1, 64, 512])], gen)
    d = Io.new(mesh, case.specs, gen)
    ttnn.synchronize_device(mesh)
    entries = mesh.num_program_cache_entries()
    made = _routed(case, "on", d, 1) + _routed(case, "on", d, 0)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - entries
    for t in made:
        ttnn.deallocate(t)
    host, dev = _timings(mesh, case, {r: (lambda r=r: _routed(case, r, d, 0)) for r in ROUTES})
    d.free()
    pad.free()
    return dict(mismatch=mismatch, repeat=repeat, growth=growth, host=host, dev=dev)


# ------------------------------------------------------------------------------------------------ wq_b (WQB_BANK)
class Wqb:
    """The attention's wq_b at ``rows`` rows a chip, both copies (WQB_BANK_COPIES=both), called through the model's own
    ``TtWindowAttention._wqb_mm`` on a stand-in holding the attributes it reads."""

    def __init__(self, mesh, rows: int):
        from models.demos.deepseek_v41_flash_d_p.tt import attention as A
        from models.demos.deepseek_v41_flash_d_p.tt.common import attn_compute_config, gemm_dtype

        self.A, self.mesh, self.rows = A, mesh, rows
        g = torch.Generator().manual_seed(rows + WQB_K)
        self.xt = torch.randn(1, 1, rows, WQB_K, generator=g)
        wt = torch.randn(1, 1, WQB_K, WQB_N, generator=g) / WQB_K**0.5
        rep = ttnn.ReplicateTensorToMesh(mesh)
        self.xdt = gemm_dtype()
        self.w = ttnn.from_torch(
            wt, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM, mesh_mapper=rep
        )
        shape = (1, 1, WQB_K, WQB_N)
        ok = A.wqb_bank_ok(mesh, shape, rows, True)
        assert ok, f"no wq_b bank layout at {rows} rows a chip on this mesh"
        lay = A.wqb_layout(mesh, shape, rows)
        self.wb = A.gmm_bank.upload(wt, lay, mesh, ttnn.bfloat8_b)
        self.att = types.SimpleNamespace(
            wq_b=self.w,
            wq_b_bank=self.wb,
            wqb_layout=lay,
            wqb_bank_ok=True,
            geo=types.SimpleNamespace(q_rows=rows),
            ckc=attn_compute_config(mesh),
        )
        self.cores = len(lay.slots)
        self.truth = None

    def x(self, xt=None):
        return ttnn.from_torch(
            self.xt if xt is None else xt, dtype=self.xdt, layout=ttnn.TILE_LAYOUT, device=self.mesh,
            memory_config=DRAM, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )  # fmt: skip

    def call(self, route: str, x) -> ttnn.Tensor:
        """The model's call (_q_proj): bf16 out into a preallocated [1, 1, rows, N] output (HOST_ALLOC's out)."""
        _switch("DS41_WQB_BANK", route == "on")
        try:
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, self.rows, WQB_N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, self.mesh, DRAM
            )
            return self.A.TtWindowAttention._wqb_mm(self.att, x, ttnn.bfloat16, DRAM, out=out)
        finally:
            _switch("DS41_WQB_BANK", True)

    def fp64(self, x) -> torch.Tensor:
        """The fp64 product of the device's own bfp8 inputs (chip 0)."""
        xd = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).double()
        wd = ttnn.to_torch(ttnn.get_device_tensors(self.w)[0]).double()
        return xd @ wd

    def free(self):
        for t in (self.w, self.wb):
            ttnn.deallocate(t)


def _chip0(t) -> torch.Tensor:
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).double()


def _rel(t, truth) -> float:
    return float((_chip0(t) - truth).norm() / truth.norm())


def _check_wqb(mesh, rows: int) -> dict:
    p = Wqb(mesh, rows)
    case = Case(f"wqb{rows}", "wqb", None, None, {"on": 1, "today": 1}, "DS41_WQB_BANK")
    x = p.x()
    truth = p.fp64(x)
    y_on, y_today = p.call("on", x), p.call("today", x)
    ttnn.synchronize_device(mesh)
    rel = {"on": _rel(y_on, truth), "today": _rel(y_today, truth)}
    vs_today = float((_chip0(y_on) - _chip0(y_today)).abs().max())
    ref = _read(y_on)
    # every chip the same bytes (replicated inputs)
    chips_equal = all(torch.equal(ref[0], c) for c in ref[1:])
    # repeat: on, fresh input / output buffers from the same values (a pad shifts the allocations)
    pad = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
    x2 = p.x()
    y2 = p.call("on", x2)
    ttnn.synchronize_device(mesh)
    repeat = _maxdiff([ref], [_read(y2)])
    for t in (y_on, y_today, y2, x2):
        ttnn.deallocate(t)
    # cache growth: warm, then fresh buffers (shifted again) with new data
    pad2 = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 96, 512]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
    x3 = p.x(torch.randn(1, 1, rows, WQB_K, generator=torch.Generator().manual_seed(7)))
    ttnn.synchronize_device(mesh)
    entries = mesh.num_program_cache_entries()
    y3 = p.call("on", x3)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - entries
    ttnn.deallocate(y3)
    host, dev = _timings(mesh, case, {r: (lambda r=r: [p.call(r, x3)]) for r in ROUTES})
    for t in (x, x3, pad, pad2):
        ttnn.deallocate(t)
    cores = p.cores
    p.free()
    return dict(
        repeat=repeat, growth=growth, host=host, dev=dev, rel=rel, vs_today=vs_today, chips_equal=chips_equal,
        cores=cores,
    )  # fmt: skip


def _report(res: dict, fam: dict) -> str:
    lines = [
        "# MB.QB layer check: wq_b bank layout (WQB_BANK) and the single-request fusions (NATIVE_FUSIONS), 5120 chunk",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_mb_qb.py` (op level on the held mesh, "
        "no model load; fixed-seed values in the model's io shapes, through the model's own calls). on = the switch on "
        "(the shipped default), today = off (model_bringup before MB.QB).",
        "",
        "| case | change | mismatch (on vs today) | repeat | cache growth (on) | host us today | host us on "
        "| device us today | device us on | programs a call | calls a chunk |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        h, d = r["host"], r["dev"]
        what, progs, calls = INFO[fam[name]]
        if "mismatch" in r:
            mm = r["mismatch"]
        else:  # (wq_b: not gated on bytes, DS41-390; its max |on - today| and rel vs fp64)
            mm = f"{r['vs_today']} (rel vs fp64 {r['rel']['today']:.4e} -> {r['rel']['on']:.4e})"
        lines.append(
            f"| {name} | {what} | {mm} | {r['repeat']} | {r['growth']} | {h['today']} | {h['on']} | {d['today']} "
            f"| {d['on']} | {progs} | {calls} |"
        )
    lines += [
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating the "
        f"routes. Device us: real-time profiler, slowest chip, {ITERS} calls a section. emb programs: the SP all-gather "
        "+ (today) the IdsReorder kernel and the embedding, (on) the embedding with the reorder as its reader.",
    ]
    for name, r in res.items():
        if fam[name] == "wqb":
            lines.append(
                f"{name}: rel vs fp64 of the device's bfp8 inputs (chip 0): today {r['rel']['today']:.6e}, bank "
                f"{r['rel']['on']:.6e}; max |bank - today| {r['vs_today']} (bf16 values); every chip equal "
                f"{r['chips_equal']}; bank cores a chip {r['cores']} (<= the 110-core matmul cap, rule 18)."
            )
    lines.append("")
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_mb_qb(mesh_device):
    mesh = mesh_device
    if mesh is None or mesh.get_num_devices() < 2:
        pytest.skip("device test on a mesh of at least two chips")
    from models.demos.deepseek_v41_flash_d_p.tt.cores import init_caps
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    init_caps(mesh)  # (the model's max_grid() / sub_cores() with no argument: this mesh)
    R, C = tuple(mesh.shape)
    olds = {k: os.environ.get(k) for k in ("DS41_NATIVE_FUSIONS", "DS41_WQB_BANK")}
    gen = torch.Generator().manual_seed(9)
    w_host = (torch.randn(VOCAB, DIM, generator=gen) * 2).to(torch.bfloat16)
    weight = ttnn.from_torch(
        w_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(None, 1)),
    )  # fmt: skip
    res, fam = {}, {}
    rows = CHUNK // (R * C)  # (ATTN_SP32_PROJ: the chip's own query rows)
    try:
        r = _check_wqb(mesh, rows)
        name = f"wqb{rows}"
        res[name], fam[name] = r, "wqb"
        record(f"native_repeat_{name}", r["repeat"])
        record(f"native_cache_growth_{name}", r["growth"])
        record(f"wqb_rel_bank_{name}", r["rel"]["on"])
        record(f"wqb_rel_today_{name}", r["rel"]["today"])
        record(f"wqb_maxdiff_vs_today_{name}", r["vs_today"])
        for route in ROUTES:
            record(f"native_host_us_{route}_{name}", r["host"][route])
            record(f"native_device_us_{route}_{name}", r["dev"][route])
            record(f"native_launches_{route}_{name}", 1)
        print(f"[mbqb] {name}: {r}")
        for case in _fusion_cases(mesh, weight):
            r = _check_fusion(mesh, case)
            res[case.name], fam[case.name] = r, case.family
            record(f"native_mismatch_{case.name}", r["mismatch"])
            record(f"native_repeat_{case.name}", r["repeat"])
            record(f"native_cache_growth_{case.name}", r["growth"])
            for route in ROUTES:
                record(f"native_host_us_{route}_{case.name}", r["host"][route])
                record(f"native_device_us_{route}_{case.name}", r["dev"][route])
                record(f"native_launches_{route}_{case.name}", case.launches[route])
            print(f"[mbqb] {case.name}: {r}")
    finally:
        for k, v in olds.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        settings.refresh()
        ttnn.deallocate(weight)
    REPORT.write_text(_report(res, fam))
    bad = {
        c: (r.get("mismatch", 0.0), r["repeat"], r["growth"])
        for c, r in res.items()
        if (r.get("mismatch", 0.0), r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) off: {bad}"
    w = res[f"wqb{rows}"]
    assert w["chips_equal"], "wq_b bank: the chips' outputs differ (replicated inputs)"
    # (equal or better vs fp64, within one bf16 rounding of the output: rule 14's floor, 2^-9 relative)
    assert w["rel"]["on"] <= w["rel"]["today"] + 2.0**-9, f"wq_b bank rel vs fp64 {w['rel']}"
