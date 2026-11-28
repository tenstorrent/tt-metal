# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.7 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the per-chunk prepared programs as the native C++ op
ttnn.chunk_program (ttnn/cpp/ttnn/operations/chunk_program) against the Python prepared route each class keeps for this
test only (``route = "python"``). Op level on the resident runner's held mesh (no model load), at the model's calls:

    pc_<geo>             tt/page_copy.PageCopy (POS_DYNAMIC=on page copies): a history record slab (uint8 rows of
                         320 B), bf16 tiles, a uint32 index row, a served ND-sharded slab (bf16 [batches, 1, 128, 512])
                         -> interleaved rows; blocks / strides as values
    rope<chunk>          tt/rope.RopeTable._rows (POS_DYNAMIC=on, ROPE_IMPL=on: cos and sin bf16 TILE [T, 64]), each mesh
                         row its natural block of one start: python = two sp_stride page copies (model_bringup's MB.2
                         route), native = tt/page_copy.pages_of_pair, cos + sin in ONE ttnn.chunk_program GLUED_ROWS
                         program (the batching branch's NC.7 fusion, one piece a mesh row)
    crope<chunk>         compressor.CompressRope._rows (ratio 4: chunk / 4 rows of bf16 ROW_MAJOR [J, 64]), the same
    ids<chunk>           tt/page_copy.IdsReorder (the engine's uint32 [SP, 1, W] input, a start and a valid count)
    (MB.NC: model_bringup runs the 5120 chunk only and has no packed path: the batching branch's 8192, rope4x2k,
    GluedIdsReorder, Derive and OwnedSend cases are not here; those C++ kinds have no call site on this branch)

Inputs: fixed-seed random bytes in every tensor (each chip its own); both routes run the same kernels on the same
bytes. Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every io / output tensor of every chip, two
        data sets with their own values.
    native_repeat_<case> (gated == 0): the C++ op again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by C++ calls after warm-up on fresh buffers
        (an allocation shifted in between) with new data and the other values.
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes), native / python.
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native / python.

Writes its section of results/MB.NC_layer.md (tests/native/mb_nc_report.py; MB.NC, from the batching branch's NC.7
8e8a55a77fc).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py"
"""

from __future__ import annotations

import statistics
import time

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
VOCAB = 129280
ROPE_W = 64  # rope_head_dim (ROPE_IMPL=on pair tables: 2 x half)
TABLE_ROWS = 155648  # (a RoPE table long enough for the starts below; the program does not depend on its length)
# (program, calls a 5120 chunk: estimated from the call sites, layers 0-20 under DECODER_SKIP=decode; the page copies
# the batching branch's NC.7 count, ~40-80, at its middle)
INFO = {
    "pc": ("PageCopy", 60),
    "rope": ("PageCopy x 2 (sp_stride) -> GluedRows (cos + sin)", 1),
    "crope": ("PageCopy x 2 (sp_stride) -> GluedRows (cos + sin)", 2),
    "ids": ("IdsReorder", 1),
}


def _rand(gen, shape, dtype):
    if dtype == ttnn.uint32:
        return torch.randint(0, 1 << 31, shape, generator=gen, dtype=torch.int64).to(torch.int32)
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, generator=gen, dtype=torch.int64).to(torch.uint8)
    return (torch.randn(*shape, generator=gen) * 2).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)


def _up(mesh, host, dtype, layout, mc=DRAM, mapper=None):
    R, C = tuple(mesh.shape)
    mapper = mapper or ttnn.ShardTensor2dMesh(mesh, mesh_shape=(R, C), dims=(0, 1))
    return ttnn.from_torch(host, dtype=dtype, layout=layout, device=mesh, memory_config=mc, mesh_mapper=mapper)


def _read(t) -> list:
    out = []
    for d in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(d)
        out.append(
            x.view(torch.int16) if x.dtype == torch.bfloat16 else x.view(torch.int32) if x.dtype == torch.float32 else x
        )
    return out


def _maxdiff(a: list, b: list) -> float:
    m = 0.0
    for xa, xb in zip(a, b):
        for ca, cb in zip(xa, xb):
            m = max(m, float((ca.to(torch.int64) - cb.to(torch.int64)).abs().max()))
    return m


class Spec:
    """A tensor of a case: per-chip host shape ``shape`` (the host tensor is [R, C, *shape[2:]]), dtype, layout, or
    ``make(gen)`` -> a tensor (special layouts)."""

    def __init__(self, shape, dtype, layout=ttnn.ROW_MAJOR_LAYOUT, make=None, host=None):
        self.shape, self.dtype, self.layout, self.make, self.host = shape, dtype, layout, make, host


_UID = [0]


class Io:
    def __init__(self, mesh, specs: list, hosts: list):
        self.mesh, self.specs, self.hosts = mesh, specs, hosts
        _UID[0] += 1
        self.uid = _UID[0]
        self.t = []
        for s, h in zip(specs, hosts):
            self.t.append(s.make(h) if s.make is not None else _up(mesh, h, s.dtype, s.layout))

    @staticmethod
    def new(mesh, specs: list, gen) -> "Io":
        R, C = tuple(mesh.shape)
        hosts = []
        for s in specs:
            if s.host is not None:
                hosts.append(s.host(gen))
            else:
                hosts.append(_rand(gen, [R, C] + list(s.shape[2:]), s.dtype))
        return Io(mesh, specs, hosts)

    def copy(self) -> "Io":
        return Io(self.mesh, self.specs, self.hosts)

    def read(self) -> list:
        return [_read(t) for t in self.t]

    def free(self):
        for t in self.t:
            if t.is_allocated():
                ttnn.deallocate(t)


class Case:
    """``call(route, io, v)`` runs the model call with value set ``v`` (0 / 1) and returns the tensors it made (new
    outputs; read and compared besides the io)."""

    def __init__(self, name, family, specs, call, timed=None):
        self.name, self.family, self.specs, self.call = name, family, specs, call
        self.timed = timed  # (route, io) -> a callable for the host / device timing (default: call(route, io, 0))
        self.prepare = None  # (io) -> None: allocate what a call keeps (before the cache-growth count)


def _cases(mesh) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.page_copy import IdsReorder, PageCopy, pages_of_pair

    R, C = tuple(mesh.shape)
    n = R * C
    cases = []

    # --- page copies (the model's geometries; values: (s0, d0, n, nb, ss, ds))
    def pc_case(name, src, dst, vals):
        holder = {}

        def call(route, io, v):
            pc = holder.get(route)
            if pc is None:
                pc = holder[route] = PageCopy(mesh, io.t[0], io.t[1])
            pc.route = route
            pc(io.t[0], io.t[1], *vals[v])
            return []

        return Case(name, "pc", [src, dst], call)

    cases.append(
        pc_case(
            "pc_hist320",
            Spec([1, 1, 4096, 320], ttnn.uint8),
            Spec([1, 1, 2048, 320], ttnn.uint8),
            [(128, 0, 1536, 1, 0, 0), (37, 64, 300, 3, 1200, 600)],
        )
    )
    cases.append(
        pc_case(
            "pc_tile_bf16",
            Spec([1, 1, 1024, 512], ttnn.bfloat16, ttnn.TILE_LAYOUT),
            Spec([1, 1, 256, 512], ttnn.bfloat16, ttnn.TILE_LAYOUT),
            [(16, 0, 128, 1, 0, 0), (0, 32, 32, 2, 256, 48)],
        )
    )
    cases.append(
        pc_case(
            "pc_row_u32",
            Spec([1, 1, 8, 1024], ttnn.uint32),
            Spec([1, 1, 1, 1024], ttnn.uint32),
            [(5, 0, 1, 1, 0, 0), (2, 0, 1, 1, 0, 0)],
        )
    )

    def slab(h):
        from models.demos.deepseek_v41_flash_d_p.tt.served import alloc_served

        ref = alloc_served(mesh, 512, 128, 32, 6, dedup=False)
        mc = ref.memory_config()
        ttnn.deallocate(ref)
        return ttnn.from_torch(
            h.reshape(n * 6, 1, 128, 512), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh,
            memory_config=mc, mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )  # fmt: skip

    slab_spec = Spec(None, ttnn.bfloat16, make=slab, host=lambda g: _rand(g, [n * 6, 1, 128, 512], ttnn.bfloat16))
    cases.append(
        pc_case(
            "pc_slab",
            slab_spec,
            Spec([1, 1, 256, 512], ttnn.bfloat16),
            [(2 * 128, 0, 128, 1, 0, 0), (4 * 128 + 32, 64, 96, 1, 0, 0)],
        )
    )

    # --- RoPE rows: cos + sin of one start, each mesh row its natural block (pages_of_pair)
    def rope_case(name, family, rows, layout, n_rows, starts):
        unit = 32 if layout == ttnn.TILE_LAYOUT else 1  # (rows a source unit: a tile row's pages, or a row)
        ppu = ROPE_W // 32 if layout == ttnn.TILE_LAYOUT else 1  # (pages a unit)
        m = n_rows // R  # rows a mesh row

        def call(route, io, v):
            n = m // unit * ppu
            s0 = starts[v] // unit * ppu
            if route == "native":
                return list(pages_of_pair(io.t[0], io.t[1], s0, n, [1, 1, m, ROPE_W], n))
            PageCopy.route = "python"  # (pages_of_pair then takes model_bringup's two sp_stride page copies)
            try:
                return list(pages_of_pair(io.t[0], io.t[1], s0, n, [1, 1, m, ROPE_W], n))
            finally:
                PageCopy.route = "native"

        spec = Spec([1, 1, rows, ROPE_W], ttnn.bfloat16, layout)
        return Case(name, family, [spec, spec], call)

    for chunk in (5120,):
        cases.append(rope_case(f"rope{chunk}", "rope", TABLE_ROWS, ttnn.TILE_LAYOUT, chunk, [139264, 51200 + 2048]))
        cases.append(
            rope_case(
                f"crope{chunk}", "crope", TABLE_ROWS // 4, ttnn.ROW_MAJOR_LAYOUT, chunk // 4,
                [139264 // 4, 51200 // 4 + 512],
            )
        )  # fmt: skip

    # --- the engine ids
    for chunk in (5120,):
        w = chunk // R
        holder = {}

        def ids_call(route, io, v, chunk=chunk, holder=holder):
            r = holder.get(0)
            if r is None:
                r = holder[0] = IdsReorder(mesh, chunk, VOCAB)
            r.route = route
            try:
                start, valid = [(139264 + 3 * 1024 + 96, chunk), (51200 + 64, chunk - 1000)][v]
                return [r(io.t[0], start, valid)]
            finally:
                r.route = "native"

        ids_host = lambda g, w=w: torch.randint(0, VOCAB + 300, (R, C, w), generator=g).to(torch.int32)  # noqa: E731
        cases.append(Case(f"ids{chunk}", "ids", [Spec([1, 1, w], ttnn.uint32, host=ids_host)], ids_call))

    return cases


def _host_us(mesh, fn, iters=HOST_ITERS) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        r = fn()
        if isinstance(r, list):
            for t in r:
                ttnn.deallocate(t)
    t1 = time.perf_counter()
    ttnn.synchronize_device(mesh)
    return (t1 - t0) / iters * 1e6


def _dev_us(mesh, cands: list) -> dict:
    for _, fn in cands:
        fn()
    ttnn.synchronize_device(mesh)
    profiler.enable(mesh, rt=True)
    try:
        for tag, fn in cands:
            profiler.signpost(tag)
            for _ in range(ITERS):
                r = fn()
                if isinstance(r, list):
                    for t in r:
                        ttnn.deallocate(t)
            profiler.signpost("gap")
            ttnn.synchronize_device(mesh)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    return {tag: prof["kernel_ns"].get(tag, float("nan")) / ITERS / 1e3 for tag, _ in cands}


def _owned(made: list, io: Io) -> list:
    """The made tensors that are not the io's (a native op returns a new wrapper over a caller's buffer, DS41-403:
    match by buffer address)."""
    ioa = {int(t.buffer_address()) for t in io.t}
    return [t for t in made if int(t.buffer_address()) not in ioa]


def _check(mesh, case: Case) -> dict:
    gen = torch.Generator().manual_seed(sum(map(ord, case.name)))
    keep = case.family == "derive"  # (its outputs live in the instance: never freed here)

    def made(ts):
        return [_read(t) for t in ts]

    def drop(ts, io):
        if not keep:
            for t in _owned(ts, io):
                ttnn.deallocate(t)

    mismatch = 0.0
    a = Io.new(mesh, case.specs, gen)
    b = a.copy()
    initial = a.read()
    effect = True
    for v in (0, 1):  # two data sets / value sets, the same calls on both sides
        ma = case.call("native", a, v)
        mb = case.call("python", b, v)
        ttnn.synchronize_device(mesh)
        got = a.read() + made(ma)
        mismatch = max(mismatch, _maxdiff(got, b.read() + made(mb)))
        if v == 0:
            ref = got
            if case.family in ("pc", "owned"):  # (in place: the call must have written its destination)
                effect = any(not torch.equal(x, y) for xa, xb in zip(initial, a.read()) for x, y in zip(xa, xb))
        drop(ma, a)
        drop(mb, b)
    # repeat: the same first native call on fresh buffers from the same bytes
    c = Io(mesh, case.specs, a.hosts)
    mc = case.call("native", c, 0)
    ttnn.synchronize_device(mesh)
    repeat = _maxdiff(ref, c.read() + made(mc))
    drop(mc, c)
    for io in (a, b, c):
        io.free()
    # cache growth: warm (both value sets ran), then fresh buffers shifted by a pad, new data, the other values
    ttnn.synchronize_device(mesh)
    pad = Io.new(mesh, [Spec([1, 1, 64, 512], ttnn.bfloat16)], gen)
    d = Io.new(mesh, case.specs, gen)
    if case.prepare is not None:
        case.prepare(d)
    ttnn.synchronize_device(mesh)
    entries = mesh.num_program_cache_entries()
    m1 = case.call("native", d, 1)
    m0 = case.call("native", d, 0)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - entries
    drop(m1 + m0, d)
    # host us a warm call, rounds alternating the routes; device us
    calls = {}
    for r in ("native", "python"):
        if case.timed is not None:
            calls[r] = case.timed(r, d)
        else:
            calls[r] = lambda r=r: _owned(case.call(r, d, 0), d)
    samples = {k: [] for k in calls}
    for _ in range(HOST_ROUNDS):
        for k, fn in calls.items():
            samples[k].append(_host_us(mesh, fn))
    host = {k: round(statistics.median(s), 1) for k, s in samples.items()}
    dev = _dev_us(mesh, list(calls.items()))
    d.free()
    pad.free()
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, effect=effect, host=host,
        dev={k: round(x, 2) for k, x in dev.items()},
    )  # fmt: skip


def _report(res: dict, family: dict) -> str:
    lines = [
        "## ttnn.chunk_program (NC.7): the per-chunk programs, native C++ vs the Python prepared route",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py` (op level on the "
        "held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide "
        "verification: task MB.NCL.",
        "",
        "| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) "
        "| device us python | device us native | calls a 5120 chunk (est.) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, r in res.items():
        h, d = r["host"], r["dev"]
        prog, calls = INFO[family[case]]
        lines.append(
            f"| {case} | {prog} | {r['mismatch']} | {r['repeat']} | {r['growth']} | {h['python']} | {h['native']} "
            f"| {d['python']} | {d['native']} | {calls} |"
        )
    lines += [
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating the "
        "routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = "
        f"ttnn.chunk_program). Device us: real-time profiler, slowest chip, {ITERS} calls a section. rope / crope: a "
        "call = the cos and the sin rows of one start (python: two page copies; native: one GLUED_ROWS program).",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_chunk_programs(mesh_device):
    mesh = mesh_device
    if mesh is None or mesh.get_num_devices() < 2:
        pytest.skip("device test on a mesh of at least two chips")
    assert getattr(ttnn.chunk_program, "is_cpp_operation", False), "ttnn.chunk_program is not the C++ op"
    res, family = {}, {}
    for case in _cases(mesh):
        r = _check(mesh, case)
        res[case.name], family[case.name] = r, case.family
        record(f"native_mismatch_{case.name}", r["mismatch"])
        record(f"native_repeat_{case.name}", r["repeat"])
        record(f"native_cache_growth_{case.name}", r["growth"])
        for route, us in r["host"].items():
            record(f"native_host_us_{route}_{case.name}", us)
        for route, us in r["dev"].items():
            record(f"native_device_us_{route}_{case.name}", us)
        print(f"[nc7] {case.name}: {r}")
    from models.demos.deepseek_v41_flash_d_p.tests.native import mb_nc_report

    rows = [
        dict(
            program=f"ttnn.chunk_program ({INFO[family[c]][0]}, {c})",
            before_us=r["host"]["python"],
            after_us=r["host"]["native"],
            calls=INFO[family[c]][1],
            note="est. calls",
        )
        for c, r in res.items()
    ]
    if any(family[c] == "pc" for c in res):  # (one row's calls for the page copies: their median, not each geometry's)
        pcs = [c for c in res if family[c] == "pc"]
        rows = [x for x in rows if "PageCopy," not in x["program"]]
        rows.insert(
            0,
            dict(
                program=f"ttnn.chunk_program (PageCopy, {', '.join(pcs)}: median)",
                before_us=round(statistics.median(res[c]["host"]["python"] for c in pcs), 1),
                after_us=round(statistics.median(res[c]["host"]["native"] for c in pcs), 1),
                calls=INFO["pc"][1],
                note="est. calls (the batching branch's NC.7: ~40-80)",
            ),
        )
    mb_nc_report.write("chunk_programs", rows, _report(res, family))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"], r["effect"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"], r["effect"]) != (0.0, 0.0, 0, True)
    }
    assert not bad, f"(mismatch, repeat, cache growth, wrote its destination) off: {bad}"
