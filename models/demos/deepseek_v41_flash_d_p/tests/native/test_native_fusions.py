# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.9 (OWNER 2026-11-28, bringup/NATIVE_PORT_RULES.md): the remaining tiny-op fusions, each one native C++ program,
against today's sequence (NATIVE_FUSIONS=off, NC.7's calls). Op level on the resident runner's held mesh (no model
load), at the model's calls:

    emb<chunk>       (1) tt/model.embed of model.reorder_ids' ids: on = IdsReorder.embed (the SP all-gather, then
                     ttnn.bringup.embedding with server_start / server_valid / clamp_max: the reorder is the
                     embedding's reader); off = IdsReorder (gather + the ids kernel) then ttnn.bringup.embedding of its
                     [1, W] ids (the model's call: TILE bf16 out, DRAM, max_grid). Chunks 8192 and 5120, two
                     (start, valid) sets, ids past the vocabulary (clamped).
    rot_<what>       (3) tt/compressor.TtCompressor._write_from's replay rotation (dyn("replay")): two page copies
                     into one rot buffer -> page_copy_many, one ttnn.chunk_program PAGE_MULTI (bf16 ROW_MAJOR rows:
                     the compressed KV (512 wide) and index keys (128 wide) at ratio 4, n = chunk / 4; ratio 128).
    carry<n>         (4) tt/attention.TtWindowCache._ring_carry_pages' carry of a start < 128: zero_carry's rows and
                     the ring rows into one res buffer -> page_copy_many (bf16 ROW_MAJOR [1, 1, 128, 512]).
    (2) tt/history_gather.tails_finish's page copy + slice_write is not on the shipped path (BREADCRUMBS NC.9: the
    owned histories of PB.HG return before it) and has no case.

Inputs: fixed-seed random bytes (each chip its own); both routes run on the same bytes. Per case:
    native_mismatch_<case> (gated == 0): fused vs today, max abs diff (raw bits) over every io / output of every chip,
        two data / value sets.
    native_repeat_<case> (gated == 0): the fused call again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by fused calls after warm-up on fresh buffers
        (an allocation shifted in between) with new data and the other values.
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes), fused / today.
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), fused / today.
    native_launches_<route>_<case>: device programs a call (the call's ops, the all-gather included), fused / today.

Writes results/NC.9_layer.md.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_fusions.py"
"""

from __future__ import annotations

import os
import statistics
import time
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
VOCAB = 4096  # (the embedding rows: the program does not depend on their count; ids up to VOCAB + 300 clamp)
DIM_CHIP = 1024  # the model's embedding columns a chip (dim 4096 / TP 4 on the 8x4)
WINDOW = 128
SWITCH = "DS41_NATIVE_FUSIONS"
ROUTES = ("fused", "today")
REPORT = Path(__file__).resolve().parents[2] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results" / "NC.9_layer.md"
# (fusion, programs a call today -> fused, calls a chunk: single 8192 / packed 4 x 2k)
INFO = {
    "emb": ("(1) ids reorder + embedding -> embedding (reorder as its reader)", "3 -> 2", "1 / 0"),
    "rot": ("(3) replay rotation: 2 page copies -> 1", "2 -> 1", "2 a replay chunk (ckv + index keys, a layer)"),
    "carry": ("(4) window carry: zeros + rows -> 1", "2 -> 1", "a carry at 0 < start < 128, a window layer"),
}


def _switch(route: str) -> None:
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    os.environ[SWITCH] = "on" if route == "fused" else "off"
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

    def __init__(self, name, family, specs, call, launches):
        self.name, self.family, self.specs, self.call, self.launches = name, family, specs, call, launches


def _cases(mesh, weight) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
    from models.demos.deepseek_v41_flash_d_p.tt.page_copy import IdsReorder, page_copy_many

    R, C = tuple(mesh.shape)
    cases = []

    # (1) the engine ids -> the embedding (tt/model.embed of reorder_ids)
    for chunk in (8192, 5120):
        w = chunk // R
        reorder = IdsReorder(mesh, chunk, VOCAB)
        kw = dict(layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=DRAM, max_grid=max_grid(mesh))
        sets = [(139264 + 3 * w + 96, chunk), (51200 + 64, chunk - 1000)]

        def emb_call(route, io, v, reorder=reorder, kw=kw, sets=sets):
            start, valid = sets[v]
            if route == "fused":
                return [reorder.embed(io.t[0], start, valid, weight, **kw)]
            ids = reorder(io.t[0], start, valid)
            e = ttnn.bringup.embedding(ids, weight, **kw)
            ttnn.deallocate(ids)
            return [e]

        launches = {"fused": 1 + 1, "today": 1 + 2}  # (the SP all-gather + the programs)
        cases.append(Case(f"emb{chunk}", "emb", [(ttnn.uint32, [1, 1, w])], emb_call, launches))

    # (3) the replay rotation: rot[0, n - off) = rm[off, n), rot[n - off, n) = rm[0, off)
    def rot_case(name, n, width, offs):
        def call(route, io, v):
            rm = io.t[0]
            off = offs[v]
            rot = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, n, width]), rm.dtype, rm.layout, mesh, DRAM)
            page_copy_many([(rm, rot, off, 0, n - off), (rm, rot, 0, n - off, off)])
            return [rot]

        return Case(name, "rot", [(ttnn.bfloat16, [1, 1, n, width])], call, {"fused": 1, "today": 2})

    cases.append(rot_case("rot_ckv8192", 8192 // 4, 512, (8 * 32 // 4 * 24, 32 // 4 * 3)))
    cases.append(rot_case("rot_idx8192", 8192 // 4, 128, (8 * 32 // 4 * 24, 32 // 4 * 3)))
    cases.append(rot_case("rot_ckv5120", 5120 // 4, 512, (1024 // 4, 4096 // 4)))
    cases.append(rot_case("rot_ckv8192_r128", 8192 // 128, 512, (2048 // 128, 128 // 128)))

    # (4) the window carry of a start < 128: res[0, 128 - n) = zero_carry's rows, res[128 - n, 128) = the ring rows
    def carry_case(name, ns):
        def call(route, io, v):
            n = ns[v]
            zero, out = io.t[0], io.t[1]
            res = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, WINDOW, 512]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, DRAM
            )
            page_copy_many([(zero, res, 0, 0, WINDOW - n), (out, res, 0, WINDOW - n, n)])
            return [res]

        specs = [(ttnn.bfloat16, [1, 1, WINDOW, 512]), (ttnn.bfloat16, [1, 1, WINDOW, 512])]
        return Case(name, "carry", specs, call, {"fused": 1, "today": 2})

    cases.append(carry_case("carry", (96, 32)))
    return cases


def _routed(case: Case, route: str, io: Io, v: int) -> list:
    _switch(route)
    try:
        return case.call(route, io, v)
    finally:
        _switch("fused")


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


def _check(mesh, case: Case) -> dict:
    gen = torch.Generator().manual_seed(sum(map(ord, case.name)))
    mismatch = 0.0
    a = Io.new(mesh, case.specs, gen)
    b = a.copy()
    for v in (0, 1):  # two value sets, the same calls on both sides
        ma = _routed(case, "fused", a, v)
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
    mc = _routed(case, "fused", c, 0)
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
    made = _routed(case, "fused", d, 1) + _routed(case, "fused", d, 0)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - entries
    for t in made:
        ttnn.deallocate(t)
    # host us a warm call, rounds alternating the routes; device us
    calls = {r: (lambda r=r: _routed(case, r, d, 0)) for r in ROUTES}
    samples = {r: [] for r in ROUTES}
    for _ in range(HOST_ROUNDS):
        for r, fn in calls.items():
            samples[r].append(_host_us(mesh, fn))
    host = {r: round(statistics.median(s), 1) for r, s in samples.items()}
    dev = _dev_us(mesh, list(calls.items()))
    d.free()
    pad.free()
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, host=host, dev={k: round(x, 2) for k, x in dev.items()}
    )


def _report(res: dict, cases: dict) -> str:
    lines = [
        "# NC.9 layer check: the remaining tiny-op fusions (NATIVE_FUSIONS on = fused, off = today's sequence)",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_fusions.py` (op level on the held "
        "mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). (2) the history "
        "tail's page copy + slice_write is not on the shipped path (BREADCRUMBS NC.9) and was not fused.",
        "",
        "| case | fusion | mismatch | repeat | cache growth (fused) | host us today | host us fused | device us today "
        "| device us fused | programs a call (today -> fused) | calls a chunk |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in res.items():
        h, d = r["host"], r["dev"]
        what, progs, calls = INFO[cases[name].family]
        lines.append(
            f"| {name} | {what} | {r['mismatch']} | {r['repeat']} | {r['growth']} | {h['today']} | {h['fused']} "
            f"| {d['today']} | {d['fused']} | {progs} | {calls} |"
        )
    lines += [
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating the "
        f"routes. Device us: real-time profiler, slowest chip, {ITERS} calls a section. emb programs: the SP all-gather "
        "+ (today) the IdsReorder kernel and the embedding, (fused) the embedding with the reorder as its reader.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_fusions(mesh_device):
    mesh = mesh_device
    if mesh is None or mesh.get_num_devices() < 2:
        pytest.skip("device test on a mesh of at least two chips")
    R, C = tuple(mesh.shape)
    old = os.environ.get(SWITCH)
    gen = torch.Generator().manual_seed(9)
    w_host = (torch.randn(VOCAB, DIM_CHIP * C, generator=gen) * 2).to(torch.bfloat16)
    weight = ttnn.from_torch(
        w_host, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(None, 1)),
    )  # fmt: skip
    res, by = {}, {}
    try:
        for case in _cases(mesh, weight):
            r = _check(mesh, case)
            res[case.name], by[case.name] = r, case
            record(f"native_mismatch_{case.name}", r["mismatch"])
            record(f"native_repeat_{case.name}", r["repeat"])
            record(f"native_cache_growth_{case.name}", r["growth"])
            for route in ROUTES:
                record(f"native_host_us_{route}_{case.name}", r["host"][route])
                record(f"native_device_us_{route}_{case.name}", r["dev"][route])
                record(f"native_launches_{route}_{case.name}", case.launches[route])
            print(f"[nc9] {case.name}: {r}")
    finally:
        if old is None:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = old
        from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

        settings.refresh()
        ttnn.deallocate(weight)
    REPORT.write_text(_report(res, by))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) off: {bad}"
