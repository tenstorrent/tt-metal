# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.6 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the indexer block-select fork's programs as native C++ ops
(the tt_ops_code_gen native_migrate ports; the model's entries ttnn.bringup.block_max_bf16 / block_gather / block_keys
now call ttnn.<op>) against the Python prepared routes they replaced (the pre-port copy,
ttnn.bringup.block_select_ttnn.generic, kept for this test only). Op level on the resident runner's held mesh (no model
load), at the model's calls (OV-82: tt/indexer.py TtIndexer.block_topk, INDEXER_BLOCK_SELECT=on,
INDEXER_BMAX_COMPUTE=on, every op given the model's max_grid):

    bmax_s<chunk>     block_max_bf16(strip_major=True, n=kv_len // 32, output_tensor=bm): the score op's strip-major
                      maxes [1, 1, Sq / 32 x NB / 8, 256] bf16 -> bm [1, 1, Sq, NB] bf16 (the load-time buffer); single
                      chunk (n a value). 8192: Sq 256 (8192 / SP 8 / TP 4), NB 16384 (a 1,048,576-token slot, ratio-2
                      keys: T 524288); 5120: Sq 160, NB 16320 (T 522240, the deployment's 1,044,480 tokens)
    bmax_p8192        the same, packed glued (tt/packed_glued.py: n_tensor = each chip's [1, 1, 1, 1] uint32 length)
    bmax_fold         block_max_bf16(factor=4, n=) on the writer's fp32 8-key maxes (INDEXER_BMAX_COMPUTE=off: not the
                      shipped path, but the same entry, ported with it)
    gather<chunk>     block_gather(scores [1, 1, Sq, T] bf16, ids [1, 1, Sq, 512] uint32, block=32, output_tensor=gath
                      [1, 1, Sq, 16384]); ids past the valid blocks 0xFFFFFFFF (the top-k's none)
    keys<chunk>       block_keys(pos [1, 1, Sq, 512] uint32, ids, block=32, width=T) -> [1, 1, Sq, 512] uint32

Inputs: fixed-seed data; the scores (256 MB a chip) replicated, everything else each chip its own; both routes run the
same kernels on the same bytes. Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every output of every chip, two data sets.
    native_repeat_<case> (gated == 0): the C++ op again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by C++ calls after warm-up on fresh buffers
        (an allocation shifted in between) with new data and a new per-call value (n / the chips' lengths).
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes): native (the model's
        entry ttnn.bringup.<op>), native_direct (ttnn.<op>), generic (the Python prepared route).
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native and generic.

Writes results/NC.6_layer.md.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py"
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
SENT = -1  # 0xFFFFFFFF as int32
BLOCK, TOPK = 32, 512
REPORT = Path(__file__).resolve().parents[2] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results" / "NC.6_layer.md"
# calls a chunk on the block path (3 plain indexers a chunk under DECODER_SKIP=decode: layers 2 / 8 / 14; single 8192
# from 163840 keys, packed glued from 196608: INDEXER_BLOCK_SELECT_MIN / _MIN_PACKED auto)
CALLS = {
    "block_max_bf16": (3, "TtIndexer.block_topk, the strip transpose"),
    "block_gather": (3, "TtIndexer.block_topk"),
    "block_keys": (3, "TtIndexer.block_topk"),
}
# (Sq, T) a chunk: Sq = chunk / SP / TP query rows a chip, T = the ratio-2 key capacity of the slot
GEO = {8192: (256, 524288), 5120: (160, 522240)}


def _nb(t: int) -> int:
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import bsel_blocks

    return bsel_blocks(t)


def _grid(mesh):
    from models.demos.deepseek_v41_flash_d_p.tt.cores import init_caps, max_grid

    init_caps(mesh)
    return max_grid(mesh)


def _per_chip(mesh, parts: torch.Tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    """parts [rows, cols, R, W] -> each chip's [1, 1, R, W]."""
    rows, cols = tuple(mesh.shape)
    t = ttnn.from_torch(
        parts, device=mesh, dtype=dtype, layout=layout, memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )  # fmt: skip
    return ttnn.reshape(t, (1, 1, parts.shape[2], parts.shape[3]))


def _replicated(mesh, x: torch.Tensor, dtype):
    return ttnn.from_torch(
        x, device=mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )  # fmt: skip


def _filled(mesh, shape, dtype, value):
    """A fresh output buffer with known contents (columns past n are compared too)."""
    return ttnn.full(shape, value, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, memory_config=DRAM)


def _chips(t) -> list[torch.Tensor]:
    out = []
    for d in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(d)
        out.append(x.to(torch.int64) if not x.is_floating_point() else x.float())
    return out


def _maxdiff(a: list, b: list) -> float:
    """Max abs diff; -inf == -inf counts equal, any other non-finite mismatch is inf."""
    m = 0.0
    for x, y in zip(a, b):
        for p, q in zip(_chips(x), _chips(y)):
            if p.is_floating_point():
                same_inf = torch.isinf(p) & torch.isinf(q) & (torch.sign(p) == torch.sign(q))
                d = torch.where(same_inf, torch.zeros_like(p), (p - q).abs())
                if torch.isnan(d).any() or torch.isinf(d).any():
                    return float("inf")
            else:
                d = (p - q).abs().float()
            m = max(m, float(d.max()))
    return m


def _ids_host(g, n_chips, sq, nb, valid):
    """Block ids [chips, sq, 512] int32: distinct-ish ids < valid, a tail of none on some rows, one out-of-range."""
    ids = torch.randint(0, valid, (n_chips, sq, TOPK), generator=g, dtype=torch.int32)
    ids[:, 0, -37:] = SENT
    ids[:, 1, 5] = nb  # just out of range
    return ids


class BmaxStrip:
    op = "block_max_bf16"

    def __init__(self, mesh, chunk, packed=False):
        self.mesh, self.packed = mesh, packed
        self.sq, t = GEO[chunk]
        self.nb = _nb(t)
        self.name = f"bmax_{'p' if packed else 's'}{chunk}"
        self.shape = f"strip [{self.sq // 32 * self.nb // 8}, 256] bf16 -> [{self.sq}, {self.nb}] bf16" + (
            ", n_tensor" if packed else ", n"
        )
        full = self.nb - 256  # (a long history: kv_len // 32 below the capacity's blocks)
        self.values = [full, full - 4000 + 3]  # (the second: new length, not a multiple of 8)

    def build(self, seed, v):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        x = torch.randn(r, c, self.sq // 32 * self.nb // 8, 256, generator=g)
        x[..., ::5, ::3] = float("-inf")
        io = {
            "x": _per_chip(self.mesh, x.to(torch.bfloat16), ttnn.bfloat16),
            "out": _filled(self.mesh, [1, 1, self.sq, self.nb], ttnn.bfloat16, 7.0),
        }
        if self.packed:  # each chip its own length (the derived lens / BSEL_BLOCK)
            n_chips = r * c
            lens = torch.tensor([max(8, v - 64 * (i % 5)) for i in range(n_chips)], dtype=torch.int32)
            io["n"] = _per_chip(self.mesh, lens.view(r, c, 1, 1), ttnn.uint32)
        return io

    def call(self, fn, io, v):
        kw = dict(n_tensor=io["n"]) if self.packed else dict(n=v)
        return [fn(io["x"], strip_major=True, output_tensor=io["out"], max_grid=_grid(self.mesh), **kw)]


class BmaxFold:
    op = "block_max_bf16"

    def __init__(self, mesh, chunk=8192):
        self.mesh = mesh
        self.sq, t = GEO[chunk]
        self.nb = _nb(t)
        self.name = "bmax_fold"
        self.shape = f"[{self.sq}, {self.nb * 4}] fp32 (factor 4) -> [{self.sq}, {self.nb}] bf16, n"
        self.values = [self.nb - 256, self.nb - 4253]

    def build(self, seed, v):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        x = torch.randn(r, c, self.sq, self.nb * 4, generator=g).to(torch.bfloat16).float()
        x[..., 3, :] = float("-inf")
        return {
            "x": _per_chip(self.mesh, x, ttnn.float32),
            "out": _filled(self.mesh, [1, 1, self.sq, self.nb], ttnn.bfloat16, 7.0),
        }

    def call(self, fn, io, v):
        return [fn(io["x"], factor=4, n=v, output_tensor=io["out"], max_grid=_grid(self.mesh))]


class _Scores:
    """The [1, 1, Sq, T] bf16 scores (replicated, a causal -inf tail, ties), one upload a seed for the whole case."""

    def __init__(self, mesh, sq, t):
        self.mesh, self.sq, self.t, self.held = mesh, sq, t, {}

    def get(self, seed):
        if seed not in self.held:
            g = torch.Generator().manual_seed(seed)
            s = torch.randn(1, 1, self.sq, self.t, generator=g, dtype=torch.bfloat16)
            s[..., :, ::7] = 0.5
            s[..., :, self.t - 8192 :] = float("-inf")  # (columns past kv_len: -inf as the future columns)
            self.held[seed] = _replicated(self.mesh, s, ttnn.bfloat16)
        return self.held[seed]

    def free(self):
        for t in self.held.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        self.held.clear()


class Gather:
    op = "block_gather"

    def __init__(self, mesh, chunk):
        self.mesh = mesh
        self.sq, self.t = GEO[chunk]
        self.name = f"gather{chunk}"
        self.shape = f"scores [{self.sq}, {self.t}] bf16 + ids [{self.sq}, {TOPK}] -> [{self.sq}, {TOPK * BLOCK}] bf16"
        self.scores = _Scores(mesh, self.sq, self.t)
        self.values = [None, None]
        self.shared = ("scores",)  # (inputs held across builds: not freed per build)

    def build(self, seed, v):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        nbw = self.t // BLOCK
        ids = _ids_host(g, r * c, self.sq, nbw, nbw - 200).view(r, c, self.sq, TOPK)
        return {
            "scores": self.scores.get(seed % 2),
            "ids": _per_chip(self.mesh, ids, ttnn.uint32),
            "out": _filled(self.mesh, [1, 1, self.sq, TOPK * BLOCK], ttnn.bfloat16, 7.0),
        }

    def call(self, fn, io, v):
        return [fn(io["scores"], io["ids"], block=BLOCK, output_tensor=io["out"], max_grid=_grid(self.mesh))]

    def done(self):
        self.scores.free()


class Keys:
    op = "block_keys"

    def __init__(self, mesh, chunk):
        self.mesh = mesh
        self.sq, self.t = GEO[chunk]
        self.name = f"keys{chunk}"
        self.shape = f"pos [{self.sq}, {TOPK}] + ids [{self.sq}, {TOPK}] uint32 -> [{self.sq}, {TOPK}] uint32"
        self.values = [None, None]

    def build(self, seed, v):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        nbw = self.t // BLOCK
        ids = _ids_host(g, r * c, self.sq, nbw, nbw - 200).view(r, c, self.sq, TOPK)
        pos = torch.randint(0, TOPK * BLOCK, (r, c, self.sq, TOPK), generator=g, dtype=torch.int32)
        pos[:, :, 4, :7] = SENT
        return {"pos": _per_chip(self.mesh, pos, ttnn.uint32), "ids": _per_chip(self.mesh, ids, ttnn.uint32)}

    def call(self, fn, io, v):
        return [fn(io["pos"], io["ids"], block=BLOCK, width=self.t, max_grid=_grid(self.mesh))]


def _cases(mesh):
    return [
        BmaxStrip(mesh, 8192),
        BmaxStrip(mesh, 8192, packed=True),
        BmaxStrip(mesh, 5120),
        BmaxFold(mesh),
        Gather(mesh, 8192),
        Gather(mesh, 5120),
        Keys(mesh, 8192),
        Keys(mesh, 5120),
    ]


def _routes(op: str) -> dict:
    from ttnn.bringup.block_select_ttnn import generic

    gen = getattr(generic, f"{op}_generic")
    direct = getattr(ttnn, op)
    assert getattr(direct, "is_cpp_operation", False), f"ttnn.{op} is not the C++ op"
    return {"native": getattr(ttnn.bringup, op), "native_direct": direct, "generic": gen}


def _free(C, io: dict):
    for k, t in io.items():
        if k not in getattr(C, "shared", ()) and t.is_allocated():
            ttnn.deallocate(t)


def _owned(t, io) -> bool:
    """Whether ``t`` is a buffer of ``io`` (the C++ ops return a new wrapper over a caller's output_tensor)."""
    return t.is_allocated() and any(u.is_allocated() and u.buffer_address() == t.buffer_address() for u in io.values())


def _outs_free(outs, io):
    """Free the outputs a call allocated (not the io tensors written in place)."""
    for t in outs:
        if t.is_allocated() and not _owned(t, io):
            ttnn.deallocate(t)


def _host_us(mesh, fn, iters=HOST_ITERS) -> float:
    fn()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
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
                fn()
            profiler.signpost("gap")
            ttnn.synchronize_device(mesh)
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    return {tag: prof["kernel_ns"].get(tag, float("nan")) / ITERS / 1e3 for tag, _ in cands}


def _check_case(mesh, C, routes: dict) -> dict:
    nat, gen = routes["native"], routes["generic"]
    seed0 = sum(map(ord, C.name)) * 10
    v0, v1 = C.values
    # warm-up: both routes (the first call of a configuration builds its program)
    for f in (nat, gen):
        io = C.build(seed0, v0)
        _outs_free(C.call(f, io, v0), io)
        ttnn.synchronize_device(mesh)
        _free(C, io)
    # mismatch: two data sets, each side from the same bytes
    mismatch = 0.0
    for i in range(2):
        a_io, b_io = C.build(seed0 + i, v0), C.build(seed0 + i, v0)
        a, b = C.call(nat, a_io, v0), C.call(gen, b_io, v0)
        mismatch = max(mismatch, _maxdiff(a, b))
        _outs_free(a, a_io)
        _outs_free(b, b_io)
        _free(C, a_io)
        _free(C, b_io)
    # cache growth on fresh buffers (an allocation shifted in between), new data, new per-call values
    ttnn.synchronize_device(mesh)
    pad = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
    x_io = C.build(seed0 + 7, v1)
    g_io = C.build(seed0 + 7, v1)
    ttnn.synchronize_device(mesh)
    before = mesh.num_program_cache_entries()
    y = C.call(nat, x_io, v1)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - before
    g = C.call(gen, g_io, v1)
    mismatch = max(mismatch, _maxdiff(y, g))
    # repeat: the C++ op again on fresh buffers from the same bytes and values
    r_io = C.build(seed0 + 7, v1)
    y2 = C.call(nat, r_io, v1)
    repeat = _maxdiff(y, y2)
    for outs, io in ((y, x_io), (g, g_io), (y2, r_io)):
        _outs_free(outs, io)
        _free(C, io)
    ttnn.deallocate(pad)
    # host us a warm call, rounds alternating the routes (in-place outputs: the same io every call)
    io = C.build(seed0 + 9, v0)
    held = []

    def mk(f):
        def run():
            outs = C.call(f, io, v0)
            held.extend(t for t in outs if not _owned(t, io))

        return run

    calls = {name: mk(f) for name, f in routes.items()}
    samples = {k: [] for k in calls}
    for _ in range(HOST_ROUNDS):
        for k, fn in calls.items():
            samples[k].append(_host_us(mesh, fn))
            ttnn.synchronize_device(mesh)
            for t in held:
                if t.is_allocated():
                    ttnn.deallocate(t)
            held.clear()
    host = {k: round(statistics.median(v), 1) for k, v in samples.items()}
    dev = _dev_us(mesh, [(k, calls[k]) for k in ("native", "generic")])
    ttnn.synchronize_device(mesh)
    for t in held:
        if t.is_allocated():
            ttnn.deallocate(t)
    _free(C, io)
    if hasattr(C, "done"):
        C.done()
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, host=host, dev={k: round(v, 2) for k, v in dev.items()}
    )


def _report(res: dict, cases: list) -> str:
    lines = [
        "# NC.6 layer check: the indexer block-select fork's programs as native C++ ops vs the Python prepared routes",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_block_select.py` (op level on the "
        "held mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide "
        "verification is deferred to the end of the port series (owner, NC.8).",
        "",
        "| case | program | call | mismatch | repeat | cache growth (C++) | host us generic (before) | host us native "
        "(after) | host us native_direct | device us generic | device us native |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for C in cases:
        r = res[C.name]
        h, d = r["host"], r["dev"]
        lines.append(
            f"| {C.name} | ttnn.{C.op} | {C.shape} | {r['mismatch']} | {r['repeat']} | {r['growth']} | {h['generic']} "
            f"| {h['native']} | {h['native_direct']} | {d['generic']} | {d['native']} |"
        )
    lines += ["", "| program | calls a chunk on the block path (est.) | host us saved a call (median of its shipped "
              "cases) | host ms a chunk saved |", "|---|---|---|---|"]  # fmt: skip
    total = 0.0
    for op, (n, where) in CALLS.items():
        saved = [
            res[C.name]["host"]["generic"] - res[C.name]["host"]["native"]
            for C in cases
            if C.op == op and C.name != "bmax_fold"
        ]
        s = statistics.median(saved) if saved else float("nan")
        total += n * s / 1e3
        lines.append(f"| ttnn.{op} | {n} ({where}) | {s:.1f} | {n * s / 1e3:.3f} |")
    lines += [
        f"| total | | | {total:.3f} |",
        "",
        "Calls a chunk: the 3 plain indexers (layers 2 / 8 / 14 under DECODER_SKIP=decode) take the block path from "
        "163840 compressed keys (single) / 196608 (packed glued); below that 0. bmax_fold is INDEXER_BMAX_COMPUTE=off "
        "(not shipped): ported with the entry, left out of the saving.",
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating "
        "the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = "
        f"the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, {ITERS} calls a "
        "section.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_block_select(mesh_device):
    mesh = mesh_device
    if mesh is None:
        pytest.skip("device test")
    cases = _cases(mesh)
    res = {}
    for C in cases:
        r = _check_case(mesh, C, _routes(C.op))
        res[C.name] = r
        case = C.name
        record(f"native_mismatch_{case}", r["mismatch"])
        record(f"native_repeat_{case}", r["repeat"])
        record(f"native_cache_growth_{case}", r["growth"])
        for route, us in r["host"].items():
            record(f"native_host_us_{route}_{case}", us)
        for route, us in r["dev"].items():
            record(f"native_device_us_{route}_{case}", us)
        print(f"[nc6] {case}: {r}")
    REPORT.write_text(_report(res, cases))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) not 0: {bad}"
