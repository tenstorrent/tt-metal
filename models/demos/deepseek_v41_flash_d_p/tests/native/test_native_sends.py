# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""NC.3 (OWNER 2026-11-27, bringup/NATIVE_PORT_RULES.md): the per-layer fabric sends as the native C++ op
ttnn.layer_send (ttnn/cpp/ttnn/operations/layer_send) against the Python prepared route each class keeps for this test
only (``route = "python"``). Op level on the resident runner's held mesh (no model load), at the model's calls:

    window_fused<chunk>  tt/window_send.WindowSend.fused (ATTN_FUSE=on, bf16 regions; attention.py window write):
                         region [1, 1, 128 + q_rows, 512], tail [1, 1, 128, 512], rows [1, 1, q_rows, 512], ring
                         [1, 1, 192, 512] bf16 ROW_MAJOR DRAM, ring pieces as WindowRing.plan makes them;
                         q_rows = chunk / chips (5120: 160; MB.NC: model_bringup's only chunk)
    window<chunk>        WindowSend.__call__ (ATTN_FUSE=legacy, not shipped): region, tail
    snap_dyn_<pat>       tt/snapshot_send.SnapshotSend._call_dyn (POS_DYNAMIC=on, shipped): ring
                         [1, 1, 192, 512] interleaved, slab = the served snapshot tensor (tt/served.alloc_served:
                         [batches, 1, 128, 512] ND-sharded), page0 a batch, 4 blocks (kind, ring row)
    snap_static_<pat>    SnapshotSend.__call__'s static path (POS_DYNAMIC=off; one program a block pattern)
    (MB.NC: the batching branch's packed-only snap_multi4 / glued4x2k_* cases are not here: model_bringup has no packed
    path, and the C++ op's multi / glued kinds have no call site on it)
    pat: "own" = the steady state (the last chip owns the 4 blocks), "mixed" = zero / local / two owners

Inputs: fixed-seed random bf16 bytes in every io tensor (each chip its own); both routes run the same kernels on the
same bytes. Per case:
    native_mismatch_<case> (gated == 0): C++ vs Python, max abs diff over every io tensor of every chip, after one call
        each on two sides started from the same bytes (and a second data set with other values).
    native_repeat_<case> (gated == 0): the C++ op again on fresh buffers from the same bytes.
    native_cache_growth_<case> (gated == 0): program-cache entries added by a C++ call after warm-up on fresh buffers
        (an allocation shifted in between) with new data and new per-call values (another page0 / ring rows / pieces).
    native_host_us_<route>_<case>: host us a warm call (median of rounds alternating the routes), native / python.
    native_device_us_<route>_<case>: device us a call (real-time profiler, slowest chip), native / python.

Writes its section of results/MB.NC_layer.md (tests/native/mb_nc_report.py; MB.NC, from the batching branch's NC.3
b947498334e).

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "scripts/run_safe_pytest.sh
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/native/test_native_sends.py"
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
W = 128  # window carry rows
RING = 192
WIDTH = 512
SNAP_BATCHES = 6  # served snapshot batches (slot, layer) of the test slab
ITERS = 20
HOST_ITERS = 40
HOST_ROUNDS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG
CALLS_A_CHUNK = 21  # layers 0-20: one call of the layer's window send and one of its snapshot send a layer
SHIPPED = {"window_fused", "snap_dyn"}  # (ATTN_FUSE=on, POS_DYNAMIC=on: the forms a 5120 chunk runs)
PROGRAM = {
    "window": "WindowSend.__call__",
    "window_fused": "WindowSend.fused",
    "snap_dyn": "SnapshotSend._call_dyn",
    "snap_static": "SnapshotSend static",
}


def _bf16(gen, *shape) -> torch.Tensor:
    return (torch.randn(*shape, generator=gen) * 2).to(torch.bfloat16)


class Io:
    """One side's io tensors, uploaded from host bytes (each chip its own): ("rm", rows) per-chip [1, 1, rows, 512]
    interleaved, ("slab",) the served snapshot tensor."""

    def __init__(self, mesh, kinds: list, host: list):
        self.mesh, self.kinds, self.host = mesh, kinds, host
        self.t = [self._up(k, h) for k, h in zip(kinds, host)]

    def _up(self, kind, h):
        R, C = tuple(self.mesh.shape)
        if kind[0] == "slab":
            from models.demos.deepseek_v41_flash_d_p.tt.served import alloc_served

            ref = alloc_served(self.mesh, WIDTH, W, 32, SNAP_BATCHES, dedup=False)
            mc = ref.memory_config()
            ttnn.deallocate(ref)
            return ttnn.from_torch(
                h.reshape(R * C * SNAP_BATCHES, 1, W, WIDTH), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh, memory_config=mc, mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            )  # fmt: skip
        t = ttnn.from_torch(
            h, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh, memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh, mesh_shape=(R, C), dims=(0, 1)),
        )  # fmt: skip
        return t

    @staticmethod
    def new(mesh, kinds: list, gen) -> "Io":
        R, C = tuple(mesh.shape)
        host = []
        for k in kinds:
            host.append(_bf16(gen, R * C, SNAP_BATCHES, W, WIDTH) if k[0] == "slab" else _bf16(gen, R, C, k[1], WIDTH))
        return Io(mesh, kinds, host)

    def copy(self) -> "Io":
        return Io(self.mesh, self.kinds, self.host)

    def read(self) -> list:
        return [[ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] for t in self.t]

    def free(self):
        for t in self.t:
            ttnn.deallocate(t)


def _maxdiff(a: list, b: list) -> float:
    m = 0.0
    for xa, xb in zip(a, b):
        for ca, cb in zip(xa, xb):
            m = max(m, float((ca.float() - cb.float()).abs().max()))
    return m


def _changed(before: list, after: list) -> bool:
    return any(not torch.equal(x, y) for xa, xb in zip(before, after) for x, y in zip(xa, xb))


class Case:
    """A model call: ``kinds`` its io tensors, ``call(route, io, v)`` runs it with values set ``v`` (0 / 1)."""

    def __init__(self, name, family, kinds, call):
        self.name, self.family, self.kinds, self.call = name, family, kinds, call


def _cases(mesh, monkeypatch) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt import page_copy
    from models.demos.deepseek_v41_flash_d_p.tt.snapshot_send import SnapshotSend
    from models.demos.deepseek_v41_flash_d_p.tt.window_send import WindowSend

    R, C = tuple(mesh.shape)
    n = R * C
    cases = []
    # --- window sends (WindowSend(mesh, region rows, 1 KiB a row, 128): attention.WindowTables, bf16_tiles)
    pieces = {
        256: [[(256 - 64, 160, 32), (256 - 32, 0, 32)], [(256 - 32, 64, 32), (0, 96, 96)]],
        160: [[(160 - 32, 64, 32), (0, 96, 96)], [(0, 128, 32), (32, 160, 32), (64, 0, 32), (96, 32, 32)]],
    }
    for chunk in (5120,):
        q = chunk // n
        ws = WindowSend(mesh, W + q, 2 * WIDTH, W)

        def fused(route, io, v, ws=ws, q=q):
            ws.route = route
            try:
                ws.fused(io.t[0], io.t[1], io.t[2], io.t[3], pieces[q][v])
            finally:
                ws.route = "native"

        cases.append(
            Case(f"window_fused{chunk}", "window_fused", [("rm", W + q), ("rm", W), ("rm", q), ("rm", RING)], fused)
        )

        def plain(route, io, v, ws=ws):
            ws.route = route
            try:
                ws(io.t[0], io.t[1])
            finally:
                ws.route = "native"

        cases.append(Case(f"window{chunk}", "window", [("rm", W + q), ("rm", W)], plain))
    # --- snapshot sends
    snd = SnapshotSend(mesh)
    pats = {
        "own": [[(n - 1, 0), (n - 1, 32), (n - 1, 64), (n - 1, 96)], [(n - 1, 64), (n - 1, 96), (n - 1, 128), (n - 1, 160)]],
        "mixed": [[("zero", 0), ("local", 64), (min(5, n - 1), 96), (n - 1, 128)],
                  [("zero", 32), ("local", 0), (min(5, n - 1), 128), (n - 1, 160)]],
    }  # fmt: skip
    for pat, bl in pats.items():

        def dyn(route, io, v, bl=bl):
            snd.route = route
            try:
                snd._call_dyn(io.t[0], io.t[1], (2 + v) * W, bl[v])
            finally:
                snd.route = "native"

        def static(route, io, v, bl=bl):
            snd.route = route
            monkeypatch.setattr(page_copy, "on", lambda: False)  # (POS_DYNAMIC=off: the static path)
            try:
                # (the same pattern for both value sets: a pattern is a program, by design)
                snd(
                    io.t[0],
                    io.t[1],
                    (2 + v) * W,
                    [(k, rr + 32 * v if rr + 32 * v + 32 <= RING else rr) for k, rr in bl[0]],
                )
            finally:
                monkeypatch.undo()
                snd.route = "native"

        cases.append(Case(f"snap_dyn_{pat}", "snap_dyn", [("rm", RING), ("slab",)], dyn))
        cases.append(Case(f"snap_static_{pat}", "snap_static", [("rm", RING), ("slab",)], static))
    return cases


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


def _check(mesh, case: Case) -> dict:
    gen = torch.Generator().manual_seed(sum(map(ord, case.name)))
    mismatch, effect = 0.0, True
    for v in (0, 1):  # two data sets, each with its own values
        a = Io.new(mesh, case.kinds, gen)
        b = a.copy()
        before = a.read()
        case.call("native", a, v)
        case.call("python", b, v)
        ttnn.synchronize_device(mesh)
        ra, rb = a.read(), b.read()
        mismatch = max(mismatch, _maxdiff(ra, rb))
        effect = effect and _changed(before, ra)
        if v == 0:  # repeat: the C++ op again on fresh buffers from the same bytes
            c = a.copy()
            case.call("native", c, v)
            ttnn.synchronize_device(mesh)
            repeat = _maxdiff(ra, c.read())
            c.free()
        a.free()
        b.free()
    # cache growth: warm (both value sets ran), then fresh buffers shifted by a pad, new data, the other values
    ttnn.synchronize_device(mesh)
    pad = Io.new(mesh, [("rm", 64)], gen)
    d = Io.new(mesh, case.kinds, gen)
    ttnn.synchronize_device(mesh)
    entries = mesh.num_program_cache_entries()
    case.call("native", d, 1)
    case.call("native", d, 0)
    ttnn.synchronize_device(mesh)
    growth = mesh.num_program_cache_entries() - entries
    # host us a warm call, rounds alternating the routes; device us
    calls = {r: (lambda r=r: case.call(r, d, 0)) for r in ("native", "python")}
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
        "## ttnn.layer_send (NC.3): the per-layer fabric sends, native C++ vs the Python prepared route",
        "",
        "Written by `models/demos/deepseek_v41_flash_d_p/tests/native/test_native_sends.py` (op level on the held mesh, "
        "no model load; fixed-seed bytes in the model's io shapes and the callers' arguments). Model-wide verification "
        "is task MB.NCL.",
        "",
        "| case | program | mismatch | repeat | cache growth (C++) | host us python (before) | host us native (after) "
        f"| device us python | device us native | calls a chunk | host ms a chunk saved |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for case, r in res.items():
        h, d = r["host"], r["dev"]
        calls = CALLS_A_CHUNK if family[case] in SHIPPED else 0
        saved = (h["python"] - h["native"]) * calls / 1e3
        lines.append(
            f"| {case} | {PROGRAM[family[case]]} | {r['mismatch']} | {r['repeat']} | {r['growth']} | {h['python']} "
            f"| {h['native']} | {d['python']} | {d['native']} | {calls} | {saved:.2f} |"
        )
    lines += [
        "",
        f"Host us: a warm call, perf_counter over {HOST_ITERS} calls, the median of {HOST_ROUNDS} rounds alternating the "
        "routes (python = the class's prepared route: ttnn.bringup.prepare_program / run_prepared; native = "
        f"ttnn.layer_send). Device us: real-time profiler, slowest chip, {ITERS} calls a section. Calls a chunk: one a "
        "layer for layers 0-20 of the shipped forms (ATTN_FUSE=on: the fused window write; POS_DYNAMIC=on: the dynamic "
        "snapshot send); the other forms 0.",
        "",
    ]
    return "\n".join(lines)


@mesh_parametrize(timeout_s=1800)
def test_native_sends(mesh_device, monkeypatch):
    mesh = mesh_device
    if mesh is None or mesh.get_num_devices() < 2:
        pytest.skip("device test on a mesh of at least two chips")
    assert getattr(ttnn.layer_send, "is_cpp_operation", False), "ttnn.layer_send is not the C++ op"
    res, family = {}, {}
    for case in _cases(mesh, monkeypatch):
        r = _check(mesh, case)
        res[case.name], family[case.name] = r, case.family
        record(f"native_mismatch_{case.name}", r["mismatch"])
        record(f"native_repeat_{case.name}", r["repeat"])
        record(f"native_cache_growth_{case.name}", r["growth"])
        for route, us in r["host"].items():
            record(f"native_host_us_{route}_{case.name}", us)
        for route, us in r["dev"].items():
            record(f"native_device_us_{route}_{case.name}", us)
        print(f"[nc3] {case.name}: {r}")
    from models.demos.deepseek_v41_flash_d_p.tests.native import mb_nc_report

    rows = [
        dict(
            program=f"ttnn.layer_send ({PROGRAM[family[c]]}, {c})",
            before_us=r["host"]["python"],
            after_us=r["host"]["native"],
            calls=CALLS_A_CHUNK if family[c] in SHIPPED else 0,
            note="shipped form" if family[c] in SHIPPED else "not shipped",
        )
        for c, r in res.items()
    ]
    mb_nc_report.write("sends", rows, _report(res, family))
    bad = {
        c: (r["mismatch"], r["repeat"], r["growth"], r["effect"])
        for c, r in res.items()
        if (r["mismatch"], r["repeat"], r["growth"], r["effect"]) != (0.0, 0.0, 0, True)
    }
    assert not bad, f"(mismatch, repeat, cache growth, changed anything) off: {bad}"
