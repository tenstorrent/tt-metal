# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.1 (OWNER 2026-11-27): the native C++ ttnn.grouped_matmul (the tt_ops_code_gen native_migrate port; the model's
entry ttnn.bringup.grouped_matmul now calls it) against the Python prepared route it replaced (the pre-port copy,
ttnn.bringup.grouped_matmul_ttnn.generic, kept for this test only). Layer level, op level on the resident runner's held
mesh (no model load): the model's wo_a calls (tt/attention._out_wo) at 256 and 160 rows a chip, the interleaved
(WOA_BANK=off, ATTN_ROOFLINE) and the bank (WOA_BANK=on, tt/attention.woa_layout) path, today's tuned knobs
(mm.gmm_knobs, mm.woa_bank_knobs, WOA_BANK_ONE_GROUP), the model's dtypes (GEMM_INPUTS x, bfp8 weights, the output
dtype of _out_wo) and compute config (attn_compute_config), the model's matmul grid (MAX_WORKER_CORES).

Inputs: fixed-seed bfloat8_b of the real shapes (real wo_a inputs need a model run: not cheap at op level; the two
routes run the same kernels in the same K order, so random data exercises the same byte path).

Cases <path><rows> (interleaved256, interleaved160, bank256, bank160):
    gmm_native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every chip, two data sets (two layers'
        weights, two inputs), each route into its own fresh output.
    gmm_native_repeat_<case> (gated == 0): the C++ op twice on fresh buffers (a new copy of x and w, new outputs).
    gmm_native_cache_growth_<case> (gated == 0): the mesh's program-cache entries added by C++ calls after warm-up on
        fresh x / w / output buffers with new data (another allocation shifted in between). (The Python route's own
        prepared-program LRU growth: gmm_py_cache_growth_<case>, not gated.)
    gmm_native_host_us_<route>_<case>: host us a warm call, model-like (the output buffer passed), perf_counter over
        HOST_ITERS calls, median of HOST_ROUNDS rounds alternating the routes; routes: native (ttnn.bringup.grouped_matmul,
        the model's entry), native_direct (ttnn.grouped_matmul, the C++ op without the ttnn.bringup registration),
        generic (the Python prepared route).
    gmm_native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip, ITERS calls a section),
        native and generic.

Writes results/NC.1_layer.md (the gate deletes a results/<task>_*.md before it runs, F292: this test writes it).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_native_layer.py"
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
G, K, N = 8, 4096, 1024
ROWS = (256, 160)
PATHS = ("interleaved", "bank")
ITERS = 20
HOST_ITERS = 50
HOST_ROUNDS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG
REPORT = Path(__file__).resolve().parents[2] / "bringup" / "ledgers" / "galaxy_8x4-full" / "results" / "NC.1_layer.md"


def _rep(mesh, t, dtype):
    return ttnn.from_torch(
        t.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )  # fmt: skip


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def _maxdiff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(_chips(a), _chips(b)))


def _data(seed: int, rows: int):
    g = torch.Generator().manual_seed(seed)
    xt = torch.randn(1, G, rows, K, generator=g)
    wt = torch.randn(1, G, K, N, generator=g) / K**0.5
    return xt, wt


class Layer:
    """One row count and path as tt/attention._out_wo calls it: its knobs, dtypes, config, grid and layout."""

    def __init__(self, mesh, rows: int, path: str):
        from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
        from models.demos.deepseek_v41_flash_d_p.tt.attention import gmm_bank, woa_layout
        from models.demos.deepseek_v41_flash_d_p.tt.common import attn_compute_config, bf16_activations, gemm_dtype
        from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid
        from models.demos.deepseek_v41_flash_d_p.tt.rope import glue
        from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

        self.mesh, self.rows, self.path = mesh, rows, path
        self.xdt, self.wdt = gemm_dtype(), ttnn.bfloat8_b  # (GEMM_INPUTS: wo_a's input; the fp8 checkpoint's bfp8)
        self.ydt = gemm_dtype() if (glue() and bf16_activations()) else ttnn.float32  # (attention._out_wo)
        self.ckc = attn_compute_config(mesh)
        self.grid = max_grid(mesh)
        x_shape, w_shape = (1, G, rows, K), (1, G, K, N)
        bw, depth, rt = mm.gmm_knobs(x_shape, w_shape)
        self.kw = dict(dtype=self.ydt, compute_kernel_config=self.ckc, max_grid=self.grid, block_w=bw, rt=rt)
        self.lay = None
        if path == "bank":
            self.lay = woa_layout(mesh, w_shape)
            bd, wi, xa = mm.woa_bank_knobs(x_shape)
            self.kw.update(
                w_layout=self.lay,
                depth=bd,
                w_inflight=wi,
                x_ahead=xa,
                one_group=bool(settings.get("WOA_BANK_ONE_GROUP")),
            )
        else:
            self.kw.update(depth=depth)
        self._bank = gmm_bank

    def x(self, xt):
        return _rep(self.mesh, xt, self.xdt)

    def w(self, wt):
        if self.lay is None:
            return _rep(self.mesh, wt, self.wdt)
        return self._bank.upload(wt, self.lay, self.mesh, self.wdt)

    def out(self):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, G, self.rows, N]), self.ydt, ttnn.TILE_LAYOUT, self.mesh, DRAM
        )

    def knobs(self) -> str:
        k = {a: v for a, v in self.kw.items() if a not in ("compute_kernel_config", "w_layout", "dtype", "max_grid")}
        return ", ".join(f"{a} {v}" for a, v in k.items())


def _routes():
    from ttnn.bringup.grouped_matmul_ttnn.generic import grouped_matmul_generic

    native = ttnn.bringup.grouped_matmul
    assert getattr(ttnn.grouped_matmul, "is_cpp_operation", False), "ttnn.grouped_matmul is not the C++ op"
    return {"native": native, "native_direct": ttnn.grouped_matmul, "generic": grouped_matmul_generic}


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


def _check_case(mesh, rows: int, path: str, routes: dict) -> dict:
    from ttnn.bringup.grouped_matmul_ttnn.generic import grouped_matmul as gen_mod

    L = Layer(mesh, rows, path)
    nat, gen = routes["native"], routes["generic"]
    sets = [_data(rows * 10 + i, rows) for i in range(2)]  # two layers' weights, two inputs
    xs = [L.x(xt) for xt, _ in sets]
    ws = [L.w(wt) for _, wt in sets]
    # warm-up: both routes, data set 0 (the first call of a configuration builds its program)
    for f in (nat, gen):
        o = f(xs[0], ws[0], **L.kw)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(o)
    # mismatch: C++ vs Python on both data sets, each into a fresh output
    mismatch = 0.0
    for x, w in zip(xs, ws):
        a, b = nat(x, w, output_tensor=L.out(), **L.kw), gen(x, w, output_tensor=L.out(), **L.kw)
        mismatch = max(mismatch, _maxdiff(a, b))
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    # cache growth on fresh buffers / new data: shift the allocator, new x / w copies of a third data set, new outputs
    ttnn.synchronize_device(mesh)
    py_before = len(gen_mod._PREPARED)
    pad = L.out()
    xt3, wt3 = _data(rows * 10 + 7, rows)
    x3, w3 = L.x(xt3), L.w(wt3)
    ttnn.synchronize_device(mesh)
    before = mesh.num_program_cache_entries()  # (the uploads may add their own programs: count the op's only)
    y3 = nat(x3, w3, output_tensor=L.out(), **L.kw)
    y4 = nat(xs[1], ws[0], **L.kw)  # a mixed pair, an op-allocated output
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - before
    g3 = gen(x3, w3, output_tensor=L.out(), **L.kw)
    ttnn.synchronize_device(mesh)
    py_growth = len(gen_mod._PREPARED) - py_before
    mismatch = max(mismatch, _maxdiff(y3, g3))
    # repeat: the C++ op again on fresh copies of x3 / w3 into a fresh output
    x3b, w3b = L.x(xt3), L.w(wt3)
    y3b = nat(x3b, w3b, output_tensor=L.out(), **L.kw)
    repeat = _maxdiff(y3, y3b)
    for t in (pad, y3, y4, g3, y3b, x3b, w3b):
        ttnn.deallocate(t)
    # host us a warm call, model-like (output passed), rounds alternating the routes
    out = L.out()
    calls = {name: (lambda f=f: f(x3, w3, output_tensor=out, **L.kw)) for name, f in routes.items()}
    samples = {k: [] for k in calls}
    for _ in range(HOST_ROUNDS):
        for k, fn in calls.items():
            samples[k].append(_host_us(mesh, fn))
    host = {k: round(statistics.median(v), 1) for k, v in samples.items()}
    dev = _dev_us(mesh, [(k, calls[k]) for k in ("native", "generic")])
    for t in [out, x3, w3, *xs, *ws]:
        ttnn.deallocate(t)
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, py_growth=py_growth, host=host,
        dev={k: round(v, 2) for k, v in dev.items()}, knobs=L.knobs(), ydt=str(L.ydt), grid=L.grid,
    )  # fmt: skip


def _report(res: dict) -> str:
    lines = [
        "# NC.1 layer check: native C++ ttnn.grouped_matmul vs the Python prepared route (wo_a)",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/woa/test_gmm_native_layer.py` (op level on the held "
        "mesh, no model load; fixed-seed bfloat8_b inputs of the real shapes, the model's knobs, dtypes, config and "
        "grid). Model-wide verification is deferred to the end of the prepared-op ports (owner).",
        "",
        "| case | knobs | mismatch | repeat | cache growth (C++) | Python LRU growth | host us native "
        "| host us native_direct | host us generic | device us native | device us generic |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, r in res.items():
        h, d = r["host"], r["dev"]
        lines.append(
            f"| {case} | {r['knobs']} | {r['mismatch']} | {r['repeat']} | {r['growth']} | {r['py_growth']} "
            f"| {h['native']} | {h['native_direct']} | {h['generic']} | {d['native']} | {d['generic']} |"
        )
    r0 = next(iter(res.values()))
    lines += [
        "",
        f"Output dtype {r0['ydt']}, grid {r0['grid']}. Host us: a warm call with the output buffer passed (as "
        f"attention._out_wo), perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating the "
        "routes. native = ttnn.bringup.grouped_matmul (the model's entry, the C++ op under the ttnn.bringup "
        "registration), native_direct = ttnn.grouped_matmul. Device us: real-time profiler, slowest chip, "
        f"{ITERS} calls a section.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_gmm_native_layer(mesh_device):
    mesh = mesh_device
    if mesh is None:
        pytest.skip("device test")
    routes = _routes()
    res = {}
    for rows in ROWS:
        for path in PATHS:
            case = f"{path}{rows}"
            r = _check_case(mesh, rows, path, routes)
            res[case] = r
            record(f"gmm_native_mismatch_{case}", r["mismatch"])
            record(f"gmm_native_repeat_{case}", r["repeat"])
            record(f"gmm_native_cache_growth_{case}", r["growth"])
            record(f"gmm_py_cache_growth_{case}", r["py_growth"])
            for route, us in r["host"].items():
                record(f"gmm_native_host_us_{route}_{case}", us)
            for route, us in r["dev"].items():
                record(f"gmm_native_device_us_{route}_{case}", us)
            print(f"[nc1] {case}: {r}")
    REPORT.write_text(_report(res))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) not 0: {bad}"
