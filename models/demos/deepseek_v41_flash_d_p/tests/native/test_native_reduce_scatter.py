# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.2 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the native C++ ttnn.fabric_reduce_scatter (the tt_ops_code_gen
native_migrate port; the model's entry ttnn.bringup.fabric_reduce_scatter now calls it) against the Python prepared
route it replaced (the pre-port copy, ttnn.bringup.fabric_reduce_scatter_ttnn.generic, kept for this test only). Op
level on the resident runner's held mesh (no model load), at the model's calls:

    tt/ccl.py _rs_bf16 (CCL_IMPL=fabric_bf16, the default: bf16 partials, FPU add), cluster_axis TP_AXIS,
        num_links rs_links(mesh), Linear, cache_descriptors / program_hash (GLUE_FOLD=on) / prepared
        (GENERIC_OP_PREPARED=on) on the Python route:
        attn_out<R>  sum_cols [R, 5120] dim 3 (attention wo_b / the MoE output partials), R = 640 (MB.NC: the 5120
                     chunk over SP 8, model_bringup's only chunk; the batching branch's 1024-row 8192 cases dropped)
        q_a<R>       tp_sum [R, 1792] dim 3 (TP_SUM_BLOCKS=cols)
        comp<R>      tp_sum [R, 1024] dim 3 (the compressor's partial)
        idx<R>       sum_rows [R, 32] dim 2 (the indexer's weights_proj, INDEXER_GLUE)
    tt/ccl.py exact path (CCL_IMPL=fabric): exact<R> [R, 5120] fp32 dim 3, exact_fp32_add
    models/demos/moe_block/moe_block.py _tp_rs_rows: moe<R> [R, 5120] bf16 dim 2, the default options, topology Ring
        when the TP axis wraps (the spec's FABRIC_2D_TORUS_XY), num_links rs_links (the block's geometry.links)

Inputs: fixed-seed bf16 / fp32 partials (one per chip, real data needs a model run; the two routes run the same kernels
in the same add order, so random data exercises the same byte path).

Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every chip, two data sets, each into its own
        output.
    native_repeat_<case> (gated == 0): the C++ op twice on fresh buffers (a new copy of the input, new outputs).
    native_cache_growth_<case> (gated == 0): the mesh's program-cache entries added by C++ calls after warm-up on a
        fresh input / output (another allocation shifted in between) with new data.
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes); routes native
        (ttnn.bringup.fabric_reduce_scatter, the model's entry), native_direct (ttnn.fabric_reduce_scatter), generic
        (the Python prepared route).
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native and generic.

Writes its section of results/MB.NC_layer.md (tests/native/mb_nc_report.py; MB.NC, from the batching branch's NC.2
8c26d3280e0).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_reduce_scatter.py"
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
TP_AXIS = 1
H = 5120
# (name, rows a chip, width, dim, dtype, exact, form): form "ccl" = tt/ccl.py's call, "moe" = moe_block's
CASES = [
    ("attn_out640", 640, H, 3, "bf16", False, "ccl"),
    ("q_a640", 640, 1792, 3, "bf16", False, "ccl"),
    ("comp640", 640, 1024, 3, "bf16", False, "ccl"),
    ("idx640", 640, 32, 2, "bf16", False, "ccl"),
    ("exact640", 640, H, 3, "fp32", True, "ccl"),
    ("moe640", 640, H, 2, "bf16", False, "moe"),
]
# calls a 5120 chunk (est. from the call sites, layers 0-20): wo_b's sum_cols and the MoE block's TP reduce one a layer,
# q_a's tp_sum one a layer, the compressor's partial one a compressing layer (2-20), the indexer's weights 4 (2/8/14/20)
# (the norms' / mHC's / engram's tp_sums not counted: a lower bound); CCL_IMPL=fabric (exact) is not shipped, and neither
# is moe_block.py's _tp_rs_rows (MOE_BLOCK_SRC=block runs the C++ moe_head / moe_tail, DS41-401): its 7.7 ms "before" is
# the fork's default route without cache_descriptors, never a shipped cost
CALLS = {"attn_out640": 21, "q_a640": 21, "comp640": 19, "idx640": 4, "exact640": 0, "moe640": 0}
ITERS = 20
HOST_ITERS = 50
HOST_ROUNDS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _chips(t) -> list[torch.Tensor]:
    return [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]


def _maxdiff(a, b) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(_chips(a), _chips(b)))


def _dtype(name: str):
    return ttnn.bfloat16 if name == "bf16" else ttnn.float32


def _upload(mesh, parts: torch.Tensor, dtype):
    """parts [rows, cols, R, W] -> each chip's [1, 1, R, W] partial."""
    rows, cols = tuple(mesh.shape)
    t = ttnn.from_torch(
        parts, device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=DRAM,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )  # fmt: skip
    return ttnn.reshape(t, (1, 1, parts.shape[2], parts.shape[3]))


def _data(mesh, seed: int, rows: int, width: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r, c = tuple(mesh.shape)
    return torch.randn(r, c, rows, width, generator=g)


def _topology(mesh):
    """moe_block.MoEBlock._tp_topology: Ring when the TP axis wraps (Geometry.of)."""
    cfg = ttnn.get_fabric_config()
    torus = cfg in (ttnn.FabricConfig.FABRIC_2D_TORUS_X, ttnn.FabricConfig.FABRIC_2D_TORUS_XY)
    return ttnn.Topology.Ring if torus else ttnn.Topology.Linear


class Case:
    """One model call: its shape, dtype and the options its caller passes."""

    def __init__(self, mesh, name, rows, width, dim, dt, exact, form):
        from models.demos.deepseek_v41_flash_d_p.tt.ccl import rs_links

        self.mesh, self.name, self.rows, self.width, self.dim = mesh, name, rows, width, dim
        self.dtype, self.exact, self.form = _dtype(dt), exact, form
        links = rs_links(mesh)
        if form == "ccl":
            # tt/ccl.py: GLUE_FOLD=on (program_hash), GENERIC_OP_PREPARED=on (prepared), the defaults
            self.kw = dict(cluster_axis=TP_AXIS, num_links=links, dim=dim, cache_descriptors=True, program_hash=True,
                           prepared=True)  # fmt: skip
            if exact:
                self.kw["exact_fp32_add"] = True
        else:  # moe_block._tp_rs_rows: the fork's defaults otherwise
            self.kw = dict(cluster_axis=TP_AXIS, topology=_topology(mesh), num_links=links)
        tp = mesh.shape[TP_AXIS]
        self.out_shape = [1, 1, rows // tp, width] if dim == 2 else [1, 1, rows, width // tp]

    def x(self, parts):
        return _upload(self.mesh, parts, self.dtype)

    def out(self):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(self.out_shape), self.dtype, ttnn.TILE_LAYOUT, self.mesh, DRAM)

    def opts(self) -> str:
        return ", ".join(f"{k} {v}" for k, v in self.kw.items())


def _routes():
    from ttnn.bringup.fabric_reduce_scatter_ttnn.generic import fabric_reduce_scatter_generic as generic

    assert getattr(
        ttnn.fabric_reduce_scatter, "is_cpp_operation", False
    ), "ttnn.fabric_reduce_scatter is not the C++ op"
    return {
        "native": ttnn.bringup.fabric_reduce_scatter,
        "native_direct": ttnn.fabric_reduce_scatter,
        "generic": generic,
    }


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


def _check_case(mesh, spec, routes: dict) -> dict:
    C = Case(mesh, *spec)
    nat, gen = routes["native"], routes["generic"]
    seed0 = sum(map(ord, C.name)) * 10
    xs = [C.x(_data(mesh, seed0 + i, C.rows, C.width)) for i in range(2)]
    # warm-up: both routes (the first call of a configuration builds its program / plan / scratch)
    for f in (nat, gen):
        o = f(xs[0], output=C.out(), **C.kw)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(o)
    # mismatch: C++ vs Python on both data sets, each into a fresh output
    mismatch = 0.0
    for x in xs:
        a, b = nat(x, output=C.out(), **C.kw), gen(x, output=C.out(), **C.kw)
        mismatch = max(mismatch, _maxdiff(a, b))
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    # cache growth on fresh buffers / new data: shift the allocator, a new input of a third data set, new outputs
    ttnn.synchronize_device(mesh)
    pad = C.out()
    p3 = _data(mesh, seed0 + 7, C.rows, C.width)
    x3 = C.x(p3)
    ttnn.synchronize_device(mesh)
    before = mesh.num_program_cache_entries()
    y3 = nat(x3, output=C.out(), **C.kw)
    y4 = nat(xs[1], **C.kw)  # an op-allocated output
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - before
    g3 = gen(x3, output=C.out(), **C.kw)
    mismatch = max(mismatch, _maxdiff(y3, g3))
    # repeat: the C++ op again on a fresh copy of x3 into a fresh output
    x3b = C.x(p3)
    y3b = nat(x3b, output=C.out(), **C.kw)
    repeat = _maxdiff(y3, y3b)
    for t in (pad, y3, y4, g3, y3b, x3b):
        ttnn.deallocate(t)
    # host us a warm call (output passed), rounds alternating the routes
    out = C.out()
    calls = {name: (lambda f=f: f(x3, output=out, **C.kw)) for name, f in routes.items()}
    samples = {k: [] for k in calls}
    for _ in range(HOST_ROUNDS):
        for k, fn in calls.items():
            samples[k].append(_host_us(mesh, fn))
    host = {k: round(statistics.median(v), 1) for k, v in samples.items()}
    dev = _dev_us(mesh, [(k, calls[k]) for k in ("native", "generic")])
    for t in [out, x3, *xs]:
        ttnn.deallocate(t)
    return dict(
        mismatch=mismatch, repeat=repeat, growth=growth, host=host, dev={k: round(v, 2) for k, v in dev.items()},
        opts=C.opts(), shape=f"[{C.rows}, {C.width}] {spec[4]} dim {C.dim}",
    )  # fmt: skip


def _report(res: dict) -> str:
    lines = [
        "## ttnn.fabric_reduce_scatter (NC.2): native C++ vs the Python prepared route",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_reduce_scatter.py` (op level on the "
        "held mesh, no model load; fixed-seed partials of the model's shapes and the callers' options). Model-wide "
        "verification: task MB.NCL.",
        "",
        "| case | partial a chip | options | mismatch | repeat | cache growth (C++) | host us native "
        "| host us native_direct | host us generic | device us native | device us generic |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, r in res.items():
        h, d = r["host"], r["dev"]
        lines.append(
            f"| {case} | {r['shape']} | {r['opts']} | {r['mismatch']} | {r['repeat']} | {r['growth']} "
            f"| {h['native']} | {h['native_direct']} | {h['generic']} | {d['native']} | {d['generic']} |"
        )
    lines += [
        "",
        f"Host us: a warm call with the output buffer passed, perf_counter over {HOST_ITERS} calls, the median of "
        f"{HOST_ROUNDS} rounds alternating the routes. native = ttnn.bringup.fabric_reduce_scatter (the model's entry), "
        "native_direct = ttnn.fabric_reduce_scatter (the C++ op), generic = the pre-port Python route "
        f"(cached descriptors, prepared programs). Device us: real-time profiler, slowest chip, {ITERS} calls a section.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_reduce_scatter(mesh_device):
    mesh = mesh_device
    if mesh is None:
        pytest.skip("device test")
    from models.demos.deepseek_v41_flash_d_p.tt import ccl

    ccl.setup(mesh)  # (the model's load-time plan / semaphores; idempotent)
    routes = _routes()
    res = {}
    for spec in CASES:
        r = _check_case(mesh, spec, routes)
        case = spec[0]
        res[case] = r
        record(f"native_mismatch_{case}", r["mismatch"])
        record(f"native_repeat_{case}", r["repeat"])
        record(f"native_cache_growth_{case}", r["growth"])
        for route, us in r["host"].items():
            record(f"native_host_us_{route}_{case}", us)
        for route, us in r["dev"].items():
            record(f"native_device_us_{route}_{case}", us)
        print(f"[nc2] {case}: {r}")
    from models.demos.deepseek_v41_flash_d_p.tests.native import mb_nc_report

    rows = [
        dict(
            program=f"ttnn.fabric_reduce_scatter ({case}: {r['shape']})",
            before_us=r["host"]["generic"],
            after_us=r["host"]["native"],
            calls=CALLS[case],
            note="est. calls" if CALLS[case] else "not shipped",
        )
        for case, r in res.items()
    ]
    mb_nc_report.write("reduce_scatter", rows, _report(res))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"]) != (0.0, 0.0, 0)
    }
    assert not bad, f"(mismatch, repeat, cache growth) not 0: {bad}"
