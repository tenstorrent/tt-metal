# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.5 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the low-bit forks' programs as native C++ ops (the
tt_ops_code_gen native_migrate ports; the model's entries ttnn.bringup.<op> now call ttnn.<op>) against the Python
prepared routes they replaced (the pre-port copies, ttnn.bringup.<fork>.generic, kept for this test only). Op level on
the resident runner's held mesh (no model load), at the model's calls (OV-82: only the programs the shipped path runs):

    pair_pool<R>      tt/compressor.TtCompressor._pool (TINY_KERNELS=on, ratio-2 KV sources 2 / 8 / 14):
                      x [1, 1, R, 1024] fp32 TILE (ATTN_SP32 full rows: R = 5120 / SP = 640; MB.NC: model_bringup's
                      5120 chunk only, the batching branch's 8192 cases dropped) ->
                      out [1, 1, R / 2, 512] fp32 TILE (the load-time output site), max_grid the model's cap
    pack<n>           tt/compressor.fp8_rows_tile (INDEX_K_SERVED=fp8_row192): the stock per_token_cast_to_fp8 of
                      [1, 1, n, 128] bf16 keys (a few all-zero pad rows) -> pack_fp8_rows(e4m3, scale,
                      memory_config=DRAM, max_grid) [1, 1, n, 192] uint8; n = the chunk's keys a chip
                      (5120: 320 ratio 2 / 640 ratio 1)
    unpack_<form>     tt/index_keys._decode (INDEX_K_SOURCE=served, POS_DYNAMIC=on: whole_grid; model_bringup has no
                      COMMON_VALUE_SLOTS, so common_values stays off as its caller leaves it): the served index rows [B, 1, crows, 192] uint8 ND-sharded (tt/served.alloc_served,
                      shard 32 / ratio rows) -> the KeyBuffer [1, 1, crows, 128] bfloat8_b TILE, rows [lo, hi) of batch
                      b; hist = the history [0, c wr), chunk = the chunk's W blocks
    rows_<form>       tt/history_gather._pack_rows (HISTORY_FORMAT=bfp4, POS_DYNAMIC=on):
                      ckv_bfp4_rows(served compress_kv slab [B, 1, R, 512] bf16 ROW_MAJOR ND-sharded, the local record
                      buffer [1, 1, R, 320] uint8, b R + a, n, a, max_grid); hist = a whole history (a 0), tail = the
                      chunk's own rows (a > 0)

Inputs: fixed-seed data (each chip its own); both routes run the same kernels on the same bytes. Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every output of every chip, two data sets.
    native_repeat_<case> (gated == 0): the C++ op again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by C++ calls after warm-up on fresh buffers
        (an allocation shifted in between) with new data and new per-call values (rows / batch where the call has them).
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes): native (the model's
        entry ttnn.bringup.<op>), native_direct (ttnn.<op>), generic (the Python prepared route).
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native and generic.

Writes its section of results/MB.NC_layer.md (tests/native/mb_nc_report.py; MB.NC, from the batching branch's NC.5
d9d05d50b5c).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py"
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
# calls a single 5120 chunk (est., the batching branch's NC.5 count of the same call sites: layers 0-20 under
# DECODER_SKIP=decode; KV sources 2 / 8 / 14 ratio 2, 20 ratio 1)
CALLS = {
    "pair_pool": (3, "TtCompressor._pool, ratio-2 sources 2 / 8 / 14"),
    "pack_fp8_rows": (4, "fp8_rows_tile, index sources 2 / 8 / 14 / 20"),
    "unpack_fp8_rows": (8, "index_keys._decode, history + chunk a source"),
    "ckv_bfp4_rows": (8, "history_gather._pack_rows, history + tail a source"),
}


def _grid(mesh):
    from models.demos.deepseek_v41_flash_d_p.tt.cores import init_caps, max_grid

    init_caps(mesh)
    return max_grid(mesh)


def _per_chip(mesh, parts: torch.Tensor, dtype, layout, mc=DRAM):
    """parts [rows, cols, R, W] -> each chip's [1, 1, R, W]."""
    rows, cols = tuple(mesh.shape)
    t = ttnn.from_torch(
        parts, device=mesh, dtype=dtype, layout=layout, memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )  # fmt: skip
    return ttnn.reshape(t, (1, 1, parts.shape[2], parts.shape[3]))


def _chips(t) -> list[torch.Tensor]:
    out = []
    for d in ttnn.get_device_tensors(t):
        x = ttnn.to_torch(d)
        out.append(x.to(torch.int32) if x.dtype == torch.uint8 else x.float())
    return out


def _maxdiff(a: list, b: list) -> float:
    m = 0.0
    for x, y in zip(a, b):
        for p, q in zip(_chips(x), _chips(y)):
            d = (p - q).abs()
            if torch.isnan(d).any():
                return float("inf")
            m = max(m, float(d.max()))
    return m


def _nd_sharded(shard_rows: int, width: int, mesh):
    from models.demos.common.prefill.runners.migration import get_num_dram_banks

    banks = get_num_dram_banks(mesh)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def _upload_batches(mesh, host: torch.Tensor, dtype, mc):
    """host [chips, B, 1, R, W] -> [B, 1, R, W] a chip (each chip its own), ``mc`` (ND-sharded as tt/served)."""
    rows, cols = tuple(mesh.shape)
    n, B, _, R, W = host.shape
    # chip (r, c) = host[r cols + c]: dim 0 split into `rows` pieces of B, dim 1 into `cols` pieces of 1
    parts = host.reshape(rows, cols, B, R, W).permute(0, 2, 1, 3, 4).reshape(rows * B, cols, R, W).contiguous()
    return ttnn.from_torch(
        parts, device=mesh, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# the cases: build(seed) -> io (dict of tensors), call(fn, io, values) -> the outputs (in-place outputs included)


class PairPool:
    op = "pair_pool"

    def __init__(self, mesh, rows):
        self.mesh, self.rows, self.name = mesh, rows, f"pair_pool{rows}"
        self.shape = f"x [{rows}, 1024] fp32 -> [{rows // 2}, 512]"
        self.values = [None, None]

    def build(self, seed):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        kv = torch.randn(r, c, self.rows, 512, generator=g)
        gate = torch.randn(r, c, self.rows, 512, generator=g) * 3
        x = _per_chip(self.mesh, torch.cat([kv, gate], -1), ttnn.float32, ttnn.TILE_LAYOUT)
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, self.rows // 2, 512]), ttnn.float32, ttnn.TILE_LAYOUT, self.mesh, DRAM
        )
        return {"x": x, "out": out}

    def call(self, fn, io, v):
        r = fn(io["x"], output_tensor=io["out"], max_grid=_grid(self.mesh))
        return [r]


class Pack:
    op = "pack_fp8_rows"

    def __init__(self, mesh, n):
        self.mesh, self.n, self.name = mesh, n, f"pack{n}"
        self.shape = f"[{n}, 128] e4m3 + scale -> [{n}, 192] uint8"
        self.values = [None, None]

    def build(self, seed):
        g = torch.Generator().manual_seed(seed)
        r, c = tuple(self.mesh.shape)
        k = torch.randn(r, c, self.n, 128, generator=g) * 4
        k[..., -40:, :] = 0  # zeroed pad rows (written all zero)
        tile = _per_chip(self.mesh, k, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        e4m3, scale = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(
            tile, memory_config=DRAM, round_scale_to_power_of_two=False
        )
        ttnn.deallocate(tile)
        return {"e4m3": e4m3, "scale": scale}

    def call(self, fn, io, v):
        return [fn(io["e4m3"], io["scale"], memory_config=DRAM, max_grid=_grid(self.mesh))]


def _fp8_rows_host(g, n_chips, B, crows) -> torch.Tensor:
    """Valid 192-byte rows (e4m3 of key / scale | fp32 scale | zeros): [chips, B, 1, crows, 192] uint8."""
    vals = (torch.randn(n_chips, B, 1, crows, 128, generator=g) * 100).clamp(-448, 448)
    e = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    s = (torch.rand(n_chips, B, 1, crows, 1, generator=g) * 0.1 + 1e-3).float().view(torch.uint8)
    z = torch.zeros(n_chips, B, 1, crows, 60, dtype=torch.uint8)
    return torch.cat([e, s, z], -1)


class Unpack:
    op = "unpack_fp8_rows"
    B, CROWS, RATIO = 3, 4096, 2

    def __init__(self, mesh, form):
        self.mesh, self.form, self.name = mesh, form, f"unpack_{form}"
        self.shape = f"rows [{self.B}, 1, {self.CROWS}, 192] ND (shard {32 // self.RATIO}) -> [{self.CROWS}, 128] bfp8"
        # (batch, lo, hi) of the two calls (the second: new per-call values for the cache-growth check)
        self.values = [(1, 0, 2048), (2, 0, 3072)] if form == "hist" else [(1, 2048, 2560), (0, 3072, 3584)]

    def build(self, seed):
        g = torch.Generator().manual_seed(seed)
        n = self.mesh.get_num_devices()
        host = _fp8_rows_host(g, n, self.B, self.CROWS)
        rows = _upload_batches(self.mesh, host, ttnn.uint8, _nd_sharded(32 // self.RATIO, 192, self.mesh))
        r, c = tuple(self.mesh.shape)
        out = _per_chip(self.mesh, torch.zeros(r, c, self.CROWS, 128), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        return {"rows": rows, "out": out}

    def call(self, fn, io, v):
        b, lo, hi = v
        fn(io["rows"], io["out"], row_lo=lo, row_hi=hi, batch=b, max_grid=_grid(self.mesh), whole_grid=True)
        return [io["out"]]


class Rows:
    op = "ckv_bfp4_rows"
    B, R, RATIO = 3, 4096, 2

    def __init__(self, mesh, form):
        self.mesh, self.form, self.name = mesh, form, f"rows_{form}"
        self.shape = f"slab [{self.B}, 1, {self.R}, 512] bf16 ND (shard {32 // self.RATIO}) -> [{self.R}, 320] uint8"
        # (batch, a, n): hist = a whole history from row 0 (PACK_BUCKET 2048 rows), tail = the chunk's own rows
        self.values = [(1, 0, 2048), (2, 0, 4096)] if form == "hist" else [(1, 2048, 512), (0, 3072, 512)]

    def build(self, seed):
        g = torch.Generator().manual_seed(seed)
        n = self.mesh.get_num_devices()
        host = (torch.randn(n, self.B, 1, self.R, 512, generator=g) * 2).to(torch.bfloat16)
        src = _upload_batches(self.mesh, host, ttnn.bfloat16, _nd_sharded(32 // self.RATIO, 512, self.mesh))
        r, c = tuple(self.mesh.shape)
        loc = _per_chip(self.mesh, torch.zeros(r, c, self.R, 320, dtype=torch.uint8), ttnn.uint8,
                        ttnn.ROW_MAJOR_LAYOUT)  # fmt: skip
        return {"src": src, "loc": loc}

    def call(self, fn, io, v):
        b, a, n = v
        fn(io["src"], io["loc"], b * self.R + a, n, a, max_grid=_grid(self.mesh))
        return [io["loc"]]


def _cases(mesh):
    return [
        PairPool(mesh, 640),
        Pack(mesh, 320),
        Pack(mesh, 640),
        Unpack(mesh, "hist"),
        Unpack(mesh, "chunk"),
        Rows(mesh, "hist"),
        Rows(mesh, "tail"),
    ]


def _routes(op: str) -> dict:
    if op == "pair_pool":
        from ttnn.bringup.pair_pool_ttnn.generic import pair_pool_generic as gen
    elif op == "pack_fp8_rows":
        from ttnn.bringup.fp8_rows_ttnn.generic import pack_fp8_rows_generic as gen
    elif op == "unpack_fp8_rows":
        from ttnn.bringup.fp8_rows_ttnn.generic import unpack_fp8_rows_generic as gen
    else:
        from ttnn.bringup.ckv_bfp4_ttnn.generic import ckv_bfp4_rows_generic as gen
    direct = getattr(ttnn, op)
    assert getattr(direct, "is_cpp_operation", False), f"ttnn.{op} is not the C++ op"
    return {"native": getattr(ttnn.bringup, op), "native_direct": direct, "generic": gen}


def _free(io: dict, keep=()):
    for k, t in io.items():
        if k not in keep and t.is_allocated():
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


def _outs_free(outs, io):
    """Free the outputs a call allocated (not the io tensors written in place)."""
    ids = {id(t) for t in io.values()}
    for t in outs:
        if id(t) not in ids and t.is_allocated():
            ttnn.deallocate(t)


def _check_case(mesh, C, routes: dict) -> dict:
    nat, gen = routes["native"], routes["generic"]
    seed0 = sum(map(ord, C.name)) * 10
    v0, v1 = C.values
    # warm-up: both routes (the first call of a configuration builds its program)
    for f in (nat, gen):
        io = C.build(seed0)
        _outs_free(C.call(f, io, v0), io)
        ttnn.synchronize_device(mesh)
        _free(io)
    # mismatch: two data sets, each side from the same bytes
    mismatch = 0.0
    for i in range(2):
        a_io, b_io = C.build(seed0 + i), C.build(seed0 + i)
        a, b = C.call(nat, a_io, v0), C.call(gen, b_io, v0)
        mismatch = max(mismatch, _maxdiff(a, b))
        _outs_free(a, a_io)
        _outs_free(b, b_io)
        _free(a_io)
        _free(b_io)
    # cache growth on fresh buffers (an allocation shifted in between), new data, new per-call values
    ttnn.synchronize_device(mesh)
    pad = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 1, 64, 1024]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, DRAM)
    x_io = C.build(seed0 + 7)
    g_io = C.build(seed0 + 7)
    ttnn.synchronize_device(mesh)
    before = mesh.num_program_cache_entries()
    y = C.call(nat, x_io, v1)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - before
    g = C.call(gen, g_io, v1)
    mismatch = max(mismatch, _maxdiff(y, g))
    # repeat: the C++ op again on fresh buffers from the same bytes and values
    r_io = C.build(seed0 + 7)
    y2 = C.call(nat, r_io, v1)
    repeat = _maxdiff(y, y2)
    for outs, io in ((y, x_io), (g, g_io), (y2, r_io)):
        _outs_free(outs, io)
        _free(io)
    ttnn.deallocate(pad)
    # host us a warm call, rounds alternating the routes (in-place outputs: the same io every call)
    io = C.build(seed0 + 9)
    held = []

    def mk(f):
        def run():
            outs = C.call(f, io, v0)
            held.extend(t for t in outs if id(t) not in {id(u) for u in io.values()})

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
    _free(io)
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, host=host, dev={k: round(v, 2) for k, v in dev.items()}
    )


def _report(res: dict, cases: list) -> str:
    lines = [
        "## ttnn.pair_pool / pack_fp8_rows / unpack_fp8_rows / ckv_bfp4_rows (NC.5): native C++ vs the Python prepared "
        "routes",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_lowbit.py` (op level on the held "
        "mesh, no model load; fixed-seed data of the model's shapes and the callers' options). Model-wide verification "
        "is task MB.NCL.",
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
    lines += ["", "| program | calls a single 5120 chunk (est.) | host us saved a call (median of its cases) | "
              "host ms a chunk saved |", "|---|---|---|---|"]  # fmt: skip
    total = 0.0
    for op, (n, where) in CALLS.items():
        saved = [res[C.name]["host"]["generic"] - res[C.name]["host"]["native"] for C in cases if C.op == op]
        s = statistics.median(saved) if saved else float("nan")
        total += n * s / 1e3
        lines.append(f"| ttnn.{op} | {n} ({where}) | {s:.1f} | {n * s / 1e3:.3f} |")
    lines += [
        f"| total | | | {total:.3f} |",
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating "
        "the routes. native = ttnn.bringup.<op> (the model's entry), native_direct = ttnn.<op> (the C++ op), generic = "
        f"the pre-port Python route (prepared programs). Device us: real-time profiler, slowest chip, {ITERS} calls a "
        "section.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_lowbit(mesh_device):
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
        print(f"[nc5] {case}: {r}")
    from models.demos.deepseek_v41_flash_d_p.tests.native import mb_nc_report

    rows = []
    for op, (n, where) in CALLS.items():
        cs = [C for C in cases if C.op == op]
        rows.append(
            dict(
                program=f"ttnn.{op} ({', '.join(C.name for C in cs)}: median)",
                before_us=round(statistics.median(res[C.name]["host"]["generic"] for C in cs), 1),
                after_us=round(statistics.median(res[C.name]["host"]["native"] for C in cs), 1),
                calls=n,
                note=f"est. calls: {where}",
            )
        )
    mb_nc_report.write("lowbit", rows, _report(res, cases))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) not 0: {bad}"
