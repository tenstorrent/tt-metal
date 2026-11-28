# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.7 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the per-chunk prepared programs as the native C++ op
ttnn.chunk_program (ttnn/cpp/ttnn/operations/chunk_program) against the Python prepared route each class keeps for this
test only (``route = "python"``). Op level on the resident runner's held mesh (no model load), at the model's calls:

    pc_<geo>             tt/page_copy.PageCopy (POS_DYNAMIC / PACKED_GLUED_DYNAMIC page copies): a history record
                         slab (uint8 rows of 320 B), bf16 tiles, a uint32 index row, a served ND-sharded slab (bf16
                         [batches, 1, 128, 512]) -> interleaved rows; blocks / strides as values
    rope<chunk>          tt/lane_rows.GluedRows, the RoPE rows of one start (rope.RopeTable._rows: cos and sin bf16
                         TILE [T, 64], ROPE_IMPL=on): native = cos + sin in ONE program (glued_rows_many), python = two
    rope4x2k             the same, a packed glued chunk's 4 pieces (+ pad) at 4 starts
    crope<chunk>         compressor.CompressRope (ratio 4: chunk / 4 rows of bf16 ROW_MAJOR [J, 64])
    ids<chunk>           tt/page_copy.IdsReorder (the engine's uint32 [SP, 1, W] input, a start and a valid count)
    gids4x2k_<cap>       tt/page_copy.GluedIdsReorder (PACKED_CHIPS=on: align = 256 rows a chip; the header in the input)
    derive4x2k           tt/glued_derive.Derive (align 256): keep, ckeep (2048), idx plain / block-select / candidate,
                         region (a carry mask), srv (one segment, three segments); chunk 1 runs every section alone
                         (first use), chunk 2 the value-free ones in ONE begin program (native) and region / srv alone
    owned_hist / owned_keys   tt/owned_gather.OwnedSend (PB.HG history records: 320 B rows, axis 0 (TP-dedup owners);
                         PB.KR keys: 1088 B pages (a bfp8 tile), axis 1 (a column)): 4 chip-aligned segments

Inputs: fixed-seed random bytes in every tensor (each chip its own); both routes run the same kernels on the same
bytes. Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every io / output tensor of every chip, two
        data sets with their own values.
    native_repeat_<case> (gated == 0): the C++ op again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by C++ calls after warm-up on fresh buffers
        (an allocation shifted in between) with new data and the other values.
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes), native / python.
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native / python.

Writes results/NC.7_layer.md.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py"
"""

from __future__ import annotations

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
VOCAB = 129280
ROPE_W = 64  # rope_head_dim (ROPE_IMPL=on pair tables: 2 x half)
TABLE_ROWS = 155648  # (a RoPE table long enough for the starts below; the program does not depend on its length)
HDR = 32  # tt/glued_derive.HDR_WORDS
REPORT = Path(__file__).resolve().parents[2] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results" / "NC.7_layer.md"
# (program, calls a chunk: single 8192 / packed 4 x 2k; estimated from the call sites, layers 0-20 under DECODER_SKIP)
INFO = {
    "pc": ("PageCopy", "~40-80 / ~40-80"),
    "rope": ("GluedRows x 2 (cos + sin) -> 1", "1 pair / 1 pair"),
    "crope": ("GluedRows x 2 (cos + sin) -> 1", "1-2 pairs / 1-2 pairs"),
    "ids": ("IdsReorder", "1 / 0"),
    "gids": ("GluedIdsReorder", "0 / 1"),
    "derive": ("Derive (a chunk: every section)", "0 / 1 chunk"),
    "owned": ("OwnedSend", "0 / ~20"),
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
    from models.demos.deepseek_v41_flash_d_p.tt.lane_rows import GluedRows
    from models.demos.deepseek_v41_flash_d_p.tt.page_copy import GluedIdsReorder, IdsReorder, PageCopy
    from models.demos.deepseek_v41_flash_d_p.tt.rope import GluedStarts

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

    # --- RoPE rows (GluedRows): cos + sin of one start
    def rope_case(name, family, rows, layout, n_rows, starts):
        holder = {}

        def call(route, io, v):
            g = holder.get(0)
            if g is None:
                g = holder[0] = GluedRows(mesh, io.t[0], n_rows)
            st = GluedStarts(starts[v])
            if route == "native":
                return g.many([io.t[0], io.t[1]], st)
            g.route = "python"
            try:
                return [g(io.t[0], st), g(io.t[1], st)]
            finally:
                g.route = "native"

        spec = Spec([1, 1, rows, ROPE_W], ttnn.bfloat16, layout)
        return Case(name, family, [spec, spec], call)

    for chunk in (8192, 5120):
        cases.append(
            rope_case(
                f"rope{chunk}", "rope", TABLE_ROWS, ttnn.TILE_LAYOUT, chunk,
                [((139264, chunk),), ((51200 + 2048, chunk),)],
            )
        )  # fmt: skip
        cases.append(
            rope_case(
                f"crope{chunk}", "crope", TABLE_ROWS // 4, ttnn.ROW_MAJOR_LAYOUT, chunk // 4,
                [((139264 // 4, chunk // 4),), ((51200 // 4 + 512, chunk // 4),)],
            )
        )  # fmt: skip
    cases.append(
        rope_case(
            "rope4x2k", "rope", TABLE_ROWS, ttnn.TILE_LAYOUT, 8192,
            [((139264, 2048), (0, 2048), (51200, 1792), (0, 256), (4096, 2048)),
             ((2048, 1024), (0, 1024), (98304, 2048), (32, 1536), (0, 512), (12288, 2048))],
        )
    )  # fmt: skip

    # --- the engine ids
    for chunk in (8192, 5120):
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

    # --- the glued chunk's ids and header (PACKED_CHIPS=on: align = rows a chip)
    W = 8192 // R
    align = 8192 // n
    headers = [
        [4, 0, 139264, 141312, 139264, 1, 0, 2048, 0, 2, 51200, 52992, 51200, 3, 4096, 6144, 4096],
        [3, 5, 98304, 100352, 98304, 1, 2048, 3584, 2048, 4, 0, 1800, 0],
    ]

    def engine_input(g, v):
        ids = torch.randint(0, VOCAB + 300, (R, C, W), generator=g).to(torch.int32)
        h = torch.tensor(headers[v] + [0] * (HDR - len(headers[v])), dtype=torch.int32)
        return torch.cat([ids, h.view(1, 1, HDR).expand(R, C, HDR)], dim=-1)

    for cap in (True, False):
        holder = {}

        def gids_call(route, io, v, cap=cap, holder=holder):
            r = holder.get(0)
            if r is None:
                r = holder[0] = GluedIdsReorder(mesh, 8192, VOCAB, HDR, 64, align)
            r.route = route
            try:
                return [r(io.t[v], cap)]
            finally:
                r.route = "native"

        specs = [Spec([1, 1, W + HDR], ttnn.uint32, host=lambda g, v=v: engine_input(g, v)) for v in (0, 1)]
        cases.append(Case(f"gids4x2k_{'cap' if cap else 'end'}", "gids", specs, gids_call))

    # --- the glued chunk's derive (a whole chunk: every section)
    cases.append(_derive_case(mesh, engine_input, align))
    # --- the owned send
    cases += _owned_cases(mesh)
    return cases


def _derive_case(mesh, engine_input, align) -> Case:
    from models.demos.deepseek_v41_flash_d_p.tt.glued_derive import Derive

    seq = [
        ("keep", (), ()),
        ("ckeep", (2048,), ()),
        ("idx", (8192, 4, 262144, 1, 0, False), ()),
        ("idx", (8192, 4, 262144, 32, 0, False), ()),  # (PB.BM block pre-selection: block 32)
        ("idx", (8192, 4, 262144, 4, 512, True), ()),  # (a candidate indexer)
        ("region", (), (0b0010,)),
        ("srv", (1,), (2,)),
        ("srv", (3,), (0, 1, 3)),
    ]
    insts: dict = {}

    def inst(route, io):
        key = (route, io.uid)
        d = insts.get(key)
        if d is None:
            d = insts[key] = Derive(mesh, 8192, 64, align)
            d.route = route
        return d

    def chunk(d, inp):
        d.begin(None, inp)
        for kind, params, values in seq:
            d.run(kind, params, values)

    def call(route, io, v):
        d = inst(route, io)
        if v == 0:  # (a fresh instance: its first chunk runs every section alone; the native route's second its begin)
            chunk(d, io.t[0])
        chunk(d, io.t[v])
        return [t for k in sorted(d._outs, key=repr) for t in d._outs[k]]

    def timed(route, io):
        d = inst(route, io)
        return lambda: chunk(d, io.t[0])

    def prepare(io):  # (the fresh instance's persistent outputs, allocated once)
        d = inst("native", io)
        for kind, params, values in seq:
            d._outputs(kind, params if kind != "region" else (int(values[0]),))

    specs = [
        Spec([1, 1, 8192 // int(mesh.shape[0]) + HDR], ttnn.uint32, host=lambda g, v=v: engine_input(g, v))
        for v in (0, 1)
    ]
    case = Case("derive4x2k", "derive", specs, call, timed)
    case.prepare = prepare
    return case


def _owned_cases(mesh) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.owned_gather import OwnedSend

    R, C = tuple(mesh.shape)
    n = R * C
    snd = OwnedSend(mesh, 2)  # (HIST_OWNED_LINKS default 2)
    out = []
    for name, page, rows, axis in (("owned_hist", 320, 256, 0), ("owned_keys", 1088, 128, 1)):
        owners = n if axis == 0 else R
        ns = 4
        specs = [Spec([1, 1, owners * rows, page], ttnn.uint8)] + [Spec([1, 1, rows, page], ttnn.uint8)] * ns
        if axis == 0:
            dests = [[q * ns // n for q in range(n)], [None if q % 7 == 3 else (q * 3 // n) for q in range(n)]]
        else:
            dests = [[(q // C) // 2 for q in range(n)], [((q // C) + 1) % 4 for q in range(n)]]
        ranges = [[(0, rows), (0, rows - 64), (32, rows), (0, 96)], [(0, 64), (16, rows), (0, rows), (rows // 2, rows)]]

        def call(route, io, v, rows=rows, axis=axis, page=page, dests=dests, ranges=ranges):
            snd.route = route
            try:
                ents = [(io.t[1 + j], lo, hi) for j, (lo, hi) in enumerate(ranges[v])]
                snd(io.t[0], rows, axis, ents, dests[v], page=page if axis else None)
                return []
            finally:
                snd.route = "native"

        out.append(Case(name, "owned", specs, call))
    return out


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
        "# NC.7 layer check: the per-chunk programs as the native C++ ttnn.chunk_program vs the Python prepared route",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_chunk_programs.py` (op level on the "
        "held mesh, no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide "
        "verification is deferred to the end of the port series (owner, NC.8). LaneRows and ChipRows are not on the "
        "shipped path (BREADCRUMBS NC.7, OV-82) and were not ported.",
        "",
        "| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) "
        "| device us python | device us native | calls a chunk (single / packed) |",
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
        "call = the cos and the sin rows of one start (python: two programs; native: one). derive4x2k: a call = one "
        "chunk (begin + every section's run: python 8 programs; native 1 begin program + region + 2 srv).",
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
    REPORT.write_text(_report(res, family))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"], r["effect"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"], r["effect"]) != (0.0, 0.0, 0, True)
    }
    assert not bad, f"(mismatch, repeat, cache growth, wrote its destination) off: {bad}"
