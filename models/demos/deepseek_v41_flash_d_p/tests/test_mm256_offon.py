# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.MM256 (OWNER 2026-11-23): the 8192 chunk's re-swept matmul configs (MM_TUNED256=on: tt/mm_configs.GALAXY256 and
the grouped wo_a's GMM256 knobs), off vs on, a real 8192 chunk single AND a chip-aligned packed 4 x 2k chunk at a long
history. OWNER rule 3 (2026-11-23): bit-identity with today's config is NOT a criterion; accuracy is judged by the
gate's test_chunk8192_reference.py with the switch on. Gated here: the on side's own repeats are bit-identical.

The serving runtime over the held performance model (adapter.build_runtime through the gate's one load, layers 0-20,
DECODER_SKIP=decode), the engine's KV cache of 8 slots. Four requests of real text (agentic_cc; request r reads the text
from r x SHIFT) are prefilled from 0 to the history H (MM256_POS, default 139264: the AgentX p50) in TWO slot sets,
slots 0-3 and 4-7, the same way (single 8192 chunks, MM_TUNED256=off: the same bytes). Then off in slots 0-3 and on in
slots 4-7 (single, then 4x2k); slots 0-3 are re-prefilled to H and both cases run on AGAIN in them (the repeat, over
the same history: a slot past H keeps its engram tail and compressor state, so a chunk is never re-run in place):

    single   request 0's chunk [H, H + 8192) (runtime.prefill_chunk: today's single-request path)
    4x2k     request 0's [H + 8192, H + 10240) + requests 1-3's [H, H + 2048), 8 chips each (runtime.prefill_packed)

mm256_repeat_max_abs_diff_<case> (gated == 0): on vs on, bitwise (the digest of every chip's bytes) over every computed
layer's output (streams and the pre's used columns, all 8192 rows) and every served record of every slot of the case
over [0, prefill_end(end)) (window rows / the served 128-row snapshot, compressed KV, index keys; every layer) and its
host engram tails; 0 when all match, else the max abs diff (at least 1e-30). mm256_offon_rel_l2_<case>_L<i>
(diagnostic): off vs on, ||off - on|| / ||off|| over the layer output's 64 sampled rows.

Device ms (the program real-time profiler, slowest chip; per section: testing/profiler.phase_of) off vs on: the same
chunks again (slots 0 / 4 and 0-3 / 4-7 at the same positions), MM256_PROF_RUNS (2) profiled runs a side, alternating,
the median: mm256_device_ms_<case>_<off|on>, mm256_sec_ms_<case>_<off|on>_<phase>.

confirm (op level, the held mesh, every chip): each GALAXY256 / GMM256 entry's call off vs on (tt/mm_configs.linear /
matmul under the switch; the grouped wo_a with its two knob sets) on random inputs: each side's 2 repeats and every
chip bit-identical (mm256_repeat_max_abs_diff_confirm, gated), its device us (20 calls, the slowest chip):
mm256_confirm_us_<site>_<off|on>, and off vs on rel L2 (diagnostic).

capture (MM256_PARTS=capture, diagnostic: the sweep's inventory, tests/perf/test_matmul_sweep.py): the off chunks once
more with tt/mm_configs.RECORD on (every dense matmul call: site, shapes, dtypes, fidelity, config, L1 free for CBs)
and the grouped_matmul calls, and once under the op-mode real-time profiler (every op's device ns a chunk):
generated/ds41_mm256/calls_<case>.json, calls_mm256.json (both cases), ops_<case>.json.

    PYTHONPATH=$PWD BRINGUP_CHUNK=8192 DS41_PACKED=on ... DS41_MM_TUNED256=on scripts/run_safe_pytest.sh --run-all \\
        --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_mm256_offon.py
Diagnostics: MM256_PARTS (offon,confirm | capture | confirm), MM256_POS, MM256_PROF_RUNS.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import sys
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests.packed import isolation_common as _ic
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc

_pc = importlib.reload(_pc)  # (the resident runner keeps helper modules of an earlier request: take this file's)
_ic = importlib.reload(_ic)
CHUNK = _pc.CHUNK

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("MM256_PARTS", "MM256_POS", "MM256_PROF_RUNS", "MM256_REPLAY_LAYERS", "MM256_SIDES")
BATCH_LAYERS = "executed"  # F285: a performance measurement (layers 0-20 under DECODER_SKIP=decode)
CAP = 155648  # the runtime's max_seq: H + 8192 + 2048 rounded up to a chunk (19 x 8192)
BATCH_MAX_SEQ = CAP
SLOTS = 8
SHIFT = 4096  # request r reads the text from r x SHIFT
SEG = 2048
OUT = pathlib.Path("generated/ds41_mm256")
ITERS = 20
FID = {
    "MathFidelity.LoFi": "LoFi",
    "MathFidelity.HiFi2": "HiFi2",
    "MathFidelity.HiFi3": "HiFi3",
    "MathFidelity.HiFi4": "HiFi4",
}


def _parts() -> set:
    return set(os.environ.get("MM256_PARTS", "offon,confirm").split(","))  # diagnostic


def _pos() -> int:
    h = int(os.environ.get("MM256_POS", "139264"))  # diagnostic
    assert h % CHUNK == 0 and h + CHUNK + SEG <= CAP, h
    return h


# ------------------------------------------------------------------------------------------------- module handles
def _mm_globals() -> list:
    """Every live module dict of tt/mm_configs (the resident runner may hold the model's functions from an earlier
    module object than this import, DS41-311): RECORD must be set where the model's linear / matmul read it."""
    out = {}
    name = "models.demos.deepseek_v41_flash_d_p.tt.mm_configs"
    if name in sys.modules:
        out[id(vars(sys.modules[name]))] = vars(sys.modules[name])
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith("models.demos.deepseek_v41_flash_d_p.tt") or mod is None:
            continue
        m = getattr(mod, "mm", None)
        if m is not None and hasattr(m, "RECORD"):
            out[id(vars(m))] = vars(m)
        for f in ("linear", "matmul"):
            fn = getattr(mod, f, None)
            g = getattr(fn, "__globals__", None)
            if g is not None and "RECORD" in g and "GALAXY" in g:
                out[id(g)] = g
    return list(out.values())


class Record:
    """tt/mm_configs.RECORD on (every live module object) and ttnn.bringup.grouped_matmul recorded, for the calls
    inside."""

    def __init__(self, mesh):
        self.mesh = mesh
        self.calls = []

    def __enter__(self):
        import ttnn

        self.gl = _mm_globals()
        for g in self.gl:
            g["RECORD"] = self.calls
        self.gmm = ttnn.bringup.grouped_matmul
        mm = sys.modules["models.demos.deepseek_v41_flash_d_p.tt.mm_configs"]
        me, f = self, self.gmm

        def rec(x, w, *a, **k):
            ckc = k.get("compute_kernel_config")
            me.calls.append(
                {
                    "op": "grouped_matmul",
                    "site": "attn.wo_a.gmm",
                    "a": list(x.shape),
                    "b": list(w.shape),
                    "a_dtype": str(x.dtype),
                    "b_dtype": str(w.dtype),
                    "out_dtype": str(k.get("dtype", x.dtype)),
                    "fidelity": str(getattr(ckc, "math_fidelity", "")),
                    "fp32_dest_acc_en": bool(getattr(ckc, "fp32_dest_acc_en", False)),
                    "packer_l1_acc": bool(getattr(ckc, "packer_l1_acc", False)),
                    "math_approx_mode": bool(getattr(ckc, "math_approx_mode", False)),
                    "config": f"gmm: block_w={k.get('block_w')} depth={k.get('depth', 2)} rt={k.get('rt', 1)}",
                    "max_grid": str(k.get("max_grid")),
                    **mm._l1_view(x.device()),
                }
            )
            return f(x, w, *a, **k)

        ttnn.bringup.grouped_matmul = rec
        return self

    def __exit__(self, *exc):
        import ttnn

        for g in self.gl:
            g["RECORD"] = None
        ttnn.bringup.grouped_matmul = self.gmm
        return False


class Replay:
    """Diagnostic (MM256_PARTS=replay): every dense matmul / grouped_matmul call of the layers in ``layers`` keeps its
    real input (every chip's shard, read to the host at the call) and its config; ``check`` re-runs each call with
    today's config (twice) and with the MM_TUNED256=on one on the same inputs and compares every chip's bytes."""

    def __init__(self, mesh, model, layers):
        self.mesh, self.model, self.layers = mesh, model, set(layers)
        self.calls = []
        self.cur = [None]

    def __enter__(self):
        import ttnn

        self.gl = _mm_globals()
        self.rec = []
        me = self
        self.old = []
        for g in self.gl:
            g["RECORD"] = self.rec
            run0 = g["_run"]
            self.old.append((g, run0))

            def run(op, x, w, pc, dtype, ckc, memory_config, out=None, _run0=run0, _g=g):
                if me.cur[0] in me.layers:
                    site = _g["RECORD"][-1]["site"] if _g["RECORD"] else "?"
                    me.calls.append(
                        dict(
                            kind="mm",
                            layer=me.cur[0],
                            site=site,
                            op=op,
                            x=_chips_host(x),
                            xdt=x.dtype,
                            xshape=tuple(x.shape),
                            w=w,
                            pc=pc,
                            dtype=dtype,
                            ckc=ckc,
                            batched=False,
                        )
                    )
                return _run0(op, x, w, pc, dtype, ckc, memory_config, out)

            g["_run"] = run
        self.gmm = ttnn.bringup.grouped_matmul
        f = self.gmm

        def gmm(x, w, *a, **k):
            if me.cur[0] in me.layers:
                me.calls.append(
                    dict(
                        kind="gmm",
                        layer=me.cur[0],
                        site="attn.wo_a.gmm",
                        x=_chips_host(x),
                        xdt=x.dtype,
                        xshape=tuple(x.shape),
                        w=w,
                        kw={n: v for n, v in k.items() if n != "output_tensor"},
                    )
                )
            return f(x, w, *a, **k)

        ttnn.bringup.grouped_matmul = gmm
        layer = self.model.layer

        def wrapped(i, h, *a, **k):
            me.cur[0] = int(i)
            try:
                return layer(i, h, *a, **k)
            finally:
                me.cur[0] = None

        self.model.layer = wrapped
        return self

    def __exit__(self, *exc):
        import ttnn

        for g, run0 in self.old:
            g["_run"] = run0
            g["RECORD"] = None
        ttnn.bringup.grouped_matmul = self.gmm
        if "layer" in vars(self.model):
            del self.model.layer
        return False

    def check(self) -> list:
        import ttnn

        mm = sys.modules["models.demos.deepseek_v41_flash_d_p.tt.mm_configs"]
        DRAM = ttnn.DRAM_MEMORY_CONFIG
        out = []
        for c in self.calls:
            n = len(c["x"])
            x = ttnn.from_torch(
                torch.cat(c["x"], dim=0),
                dtype=c["xdt"],
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh, dim=0),
            )
            x = ttnn.reshape(x, c["xshape"]) if tuple(x.shape) != c["xshape"] else x
            back = _chips_host(x)
            same_in = all(_bits_equal(a, b) for a, b in zip(back, c["x"]))

            def run(side):
                with _pc.switch("MM_TUNED256", side):
                    if c["kind"] == "gmm":
                        kw = dict(c["kw"])
                        kw["block_w"], kw["depth"], kw["rt"] = mm.gmm_knobs(c["xshape"], tuple(c["w"].shape))
                        y = ttnn.bringup.grouped_matmul(x, c["w"], **kw)
                    else:
                        batched = len(c["w"].shape) >= 3 and c["w"].shape[1] > 1
                        pc = mm.program_config(x, c["w"], c["dtype"], c["ckc"], batched=batched)
                        if side == "off":
                            pc = c["pc"]  # (the config the model's call took)
                        y = mm._run(c["op"], x, c["w"], pc, c["dtype"], c["ckc"], DRAM)
                    h = _chips_host(y)
                    ttnn.deallocate(y)
                    return h

            a, a2, b = run("off"), run("off"), run("on")
            det = all(_bits_equal(p, q) for p, q in zip(a, a2))
            same = [k for k, (p, q) in enumerate(zip(a, b)) if not _bits_equal(p, q)]
            vals = all(torch.equal(p.float(), q.float()) for p, q in zip(a, b))  # (equal as numbers: +0 == -0)
            md = max(float((p.double() - q.double()).abs().max()) for p, q in zip(a, b))
            r = dict(
                layer=c["layer"],
                site=c["site"],
                shape=f"{c['xshape']}x{tuple(c['w'].shape)}",
                input_roundtrip=same_in,
                det_off=det,
                chips_differ=len(same),
                first_chips=same[:4],
                equal_as_numbers=vals,
                max_abs=md,
            )
            out.append(r)
            print(f"MM256 replay {r}", flush=True)
            ttnn.deallocate(x)
        return out


def _chips_host(t) -> list:
    import ttnn

    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype in (torch.bfloat16, torch.float16):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))
    if a.dtype == torch.float32:
        return torch.equal(a.view(torch.int32), b.view(torch.int32))
    return torch.equal(a, b)


# ------------------------------------------------------------------------------------------------- one chunk
class Outs:
    """Per computed layer: the digest of every chip's output rows (streams and the pre's used columns) and 64 sampled
    rows (float32)."""

    def __init__(self):
        self.cap = _pc.Capture()

    def wrap(self, model):
        from models.demos.deepseek_v41_flash_d_p.tt.layout import HC

        layer = model.layer
        me = self

        def wrapped(i, h, *a, **k):
            out = layer(i, h, *a, **k)
            if out is not h and getattr(out, "streams", None) is not None:
                st = torch.cat([p.reshape(-1, out.streams.shape[-1]) for p in _ic.chips(out.streams)], dim=0)
                pr = torch.cat([p.reshape(-1, out.pre.shape[-1]) for p in _ic.chips(out.pre)], dim=0)[:, :HC]
                me.cap.add(int(i), torch.cat([st.float(), pr.float()], dim=-1))
            return out

        model.layer = wrapped

    @staticmethod
    def unwrap(model):
        if "layer" in vars(model):
            del model.layer


def _served(runtime, kv, model, slot: int, end: int) -> dict:
    """{(layer, name): (tensor maker)} of the slot's served records over [0, prefill_end(end)) (isolation_common
    Bench.served's reads) and its host engram tails."""
    from models.demos.deepseek_v41_flash_d_p.tt.attention import prefill_end

    st = runtime.state(kv, int(slot))
    n = int(prefill_end(int(end)))
    out = {}
    for i in model.layers:
        c = st.window[i]
        out[(int(i), "0_window_kv")] = lambda c=c: _ic.window_rows(c, n)
        if getattr(c, "snap", None) is not None:
            out[(int(i), "snapshot")] = lambda c=c: _ic.slab_rows(c.snap)
        if i in st.compress:
            for name, cc in st.compress[i].by_config().items():
                out[(int(i), name)] = lambda cc=cc: _ic.compressed_rows(cc, n)
    for i, t in (st.tail or {}).items():
        if isinstance(t, torch.Tensor):
            out[(int(i), "engram_tail")] = lambda t=t: t.clone()
    return out


def _served_diff(runtime, kv, model, pairs) -> tuple[float, list]:
    """pairs [(slot off, slot on, end)]: max abs diff of every served record (0 = every digest equal) and the
    differing keys."""
    worst, bad = 0.0, []
    for a, b, end in pairs:
        ra, rb = _served(runtime, kv, model, a, end), _served(runtime, kv, model, b, end)
        assert set(ra) == set(rb), (sorted(set(ra) ^ set(rb))[:4], a, b)
        for k in sorted(ra):
            ta, tb = ra[k](), rb[k]()
            if _pc.digest(ta) == _pc.digest(tb):
                continue
            bad.append((a, b) + k)
            if ta.is_floating_point() and ta.shape == tb.shape:
                d = torch.nan_to_num((ta.double() - tb.double()).abs(), nan=float("inf"))
                worst = max(worst, float(d.max()) if d.numel() else 0.0, 1e-30)
            else:
                worst = max(worst, float((ta.long() != tb.long()).any()) if ta.shape == tb.shape else 1.0, 1e-30)
    return worst, bad


def _prof_run(mesh, fn) -> tuple[float, dict]:
    from models.demos.common.bringup.testing import profiler
    from models.demos.deepseek_v41_flash_d_p.tests.packed import ladder_common as L

    profiler.enable(mesh, rt=True)
    profiler.set_layer(None)
    try:
        fn()
        profiler.signpost("end")
        profiler.finish_rt()
        pr = profiler.result()
    finally:
        profiler.disable()
    return L._device(pr), L._phases(pr)


def _ops_run(mesh, fn) -> list:
    """Every op's device ns of one run (op-mode real-time profiler): [{section, op, shape, calls, ns}]."""
    from models.demos.common.bringup.testing import profiler

    profiler.enable(mesh, ops=True, rt=True)
    profiler.set_layer(None)
    try:
        fn()
        profiler.signpost("end")
        profiler.finish_rt()
        pr = profiler.result()
    finally:
        profiler.disable()
    rows = []
    for sec, ops in (pr.get("op_ns") or {}).items():
        for r in ops:
            rows.append(
                {
                    "section": sec,
                    "op": r.get("op"),
                    "shape": r.get("shape"),
                    "calls": r.get("calls"),
                    "ns": r.get("ns"),
                }
            )
    return rows


# ------------------------------------------------------------------------------------------------- confirm
def _ckc(mesh, call: dict | None, fp32: bool):
    import ttnn

    fid = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi3": ttnn.MathFidelity.HiFi3}
    f = FID.get((call or {}).get("fidelity", ""), "HiFi2")
    return ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=fid.get(f, ttnn.MathFidelity.HiFi4),
        math_approx_mode=bool((call or {}).get("math_approx_mode", False)),
        fp32_dest_acc_en=bool((call or {}).get("fp32_dest_acc_en", fp32)),
        packer_l1_acc=bool((call or {}).get("packer_l1_acc", False)),
    )


def _captured() -> list:
    p = OUT / "calls_mm256.json"
    return json.loads(p.read_text())["calls"] if p.exists() else []


def _confirm(mesh) -> dict:
    """Every GALAXY256 / GMM256 entry off vs on, on the whole mesh (module docstring)."""
    import ttnn
    from models.demos.common.bringup.testing import profiler
    from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm

    DT = {"BFLOAT16": ttnn.bfloat16, "FLOAT32": ttnn.float32, "BFLOAT8_B": ttnn.bfloat8_b, "BFLOAT4_B": ttnn.bfloat4_b}
    caps = _captured()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def up(shape, dt, scale=1.0):
        return ttnn.from_torch(
            (torch.randn(*shape) * scale).contiguous(),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=rep,
        )

    def match(key, op):
        for c in caps:
            if c["op"] != op:
                continue
            k = mm.table_key(c["a"], c["b"], c["a_dtype"], c["b_dtype"], c["out_dtype"], c["fp32_dest_acc_en"], 11, 10)
            k = (k[0], k[1], *[str(v).split(".")[-1] if isinstance(v, str) else v for v in k[2:]])
            if k[:6] == key[:6]:
                return c
        return None

    cases = []
    for key in mm.GALAXY256:
        shape, wshape, a, b, o, fp32 = key[:6]
        batched = len(wshape) >= 3 and wshape[1] > 1
        op = "matmul" if batched else "linear"
        c = match(key, op) or match(key, "linear" if batched else "matmul")
        cases.append(("mm", key, c, shape, wshape, DT[a], DT[b], DT[o], fp32, c["op"] if c else op))
    for (xs, ws), knobs in mm.GMM256.items():
        c = next((c for c in caps if c["op"] == "grouped_matmul" and tuple(c["a"]) == xs and tuple(c["b"]) == ws), None)
        o = DT[c["out_dtype"].split(".")[-1]] if c else ttnn.bfloat8_b
        cases.append(("gmm", (xs, ws), c, xs, ws, ttnn.bfloat8_b, ttnn.bfloat8_b, o, True, "grouped_matmul"))
    res = {}
    torch.manual_seed(256)
    for kind, key, c, shape, wshape, adt, bdt, odt, fp32, op in cases:
        site = (c or {}).get("site", "?").replace(".", "_") + f"_{shape[-2]}x{shape[-1]}x{wshape[-1]}"
        ckc = _ckc(mesh, c, fp32)
        x = up(shape, adt)
        w = up(wshape, bdt, 1.0 / shape[-1] ** 0.5)

        def run(side):
            with _pc.switch("MM_TUNED256", side):
                if kind == "gmm":
                    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

                    bw, depth, rt = mm.gmm_knobs(shape, wshape)
                    return ttnn.bringup.grouped_matmul(
                        x, w, dtype=odt, compute_kernel_config=ckc, max_grid=max_grid(), block_w=bw, depth=depth, rt=rt
                    )
                f = mm.matmul if op == "matmul" else mm.linear
                return f(x, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM, site="mm256.confirm")

        outs = {}
        for side in ("off", "on"):
            outs[side] = []
            for _ in range(2):
                y = run(side)
                outs[side].append([ttnn.to_torch(t) for t in ttnn.get_device_tensors(y)])
                ttnn.deallocate(y)
        diff = 0.0  # repeats of each side (and every chip: the same inputs) bit-identical
        for side in ("off", "on"):
            ref = outs[side][0]
            for rep_ in outs[side]:
                for p, q in zip(rep_, ref):
                    if not torch.equal(p, q):
                        diff = max(diff, float((p.double() - q.double()).abs().max()), 1e-30)
            for p in ref[1:]:
                if not torch.equal(p, ref[0]):
                    diff = max(diff, 1e-30)
        a, b = outs["off"][0][0].double(), outs["on"][0][0].double()
        offon_rel = float((a - b).norm() / a.norm().clamp_min(1e-30))  # (diagnostic: off vs on, chip 0)
        us = {}
        profiler.enable(mesh, rt=True)
        try:
            for side in ("off", "on", "off", "on"):
                profiler.signpost(f"{side}")
                for _ in range(ITERS):
                    ttnn.deallocate(run(side))
            profiler.signpost("end")
            profiler.finish_rt()
            pr = profiler.result()
        finally:
            profiler.disable()
        for side in ("off", "on"):
            us[side] = pr["kernel_ns"].get(side, float("nan")) / (2 * ITERS) / 1e3
        res[site] = {
            "diff": diff,
            "offon_rel_l2": offon_rel,
            "us_off": us["off"],
            "us_on": us["on"],
            "key": repr(key),
            "kind": kind,
        }
        print(
            f"MM256 confirm {site}: off {us['off']:.1f} us, on {us['on']:.1f} us, repeat diff {diff}, off vs on rel "
            f"L2 {offon_rel:.2e}",
            flush=True,
        )
        for t in (x, w):
            ttnn.deallocate(t)
    return res


# ------------------------------------------------------------------------------------------------- the test
@mesh_parametrize(timeout_s=3 * 3600)
def test_mm256_offon(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    from models.demos.common.bringup.reference import texts
    from models.demos.common.bringup.testing import batch
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    parts = _parts()
    mesh = mesh_device
    res = {}
    if parts & {"offon", "capture", "replay"}:
        monkeypatch.setenv(settings.env_name("DECODER_SKIP"), "decode")  # (call-time: the served mode, as the gate)
        full = list(S.layers())
        with _pc.switch("DECODER_SKIP", "decode"):  # (DS41-314: a held model reads its frozen snapshot)
            layers = batch.run_layers(S, full, last_chunk=False)
        assert len(layers) < len(full), f"DECODER_SKIP=decode runs layers 0-20 ({layers})"
        monkeypatch.setitem(S.data["target"], "seq", max(CAP, int(S.data["target"]["seq"])))
        monkeypatch.setenv("WALLS_FRESH_SLOTS", str(SLOTS))  # (the runtime's KV cache)
        runtime, adapter, hf, params = SP._runtime(mesh, layers, CHUNK, CAP)
        model = runtime.model
        loaded = model.skip_mode
        if loaded != "decode":
            model.use_decoder_skip("decode")
        kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
        H = _pos()
        text, info = texts.tokens(S, 3 * SHIFT + H + CHUNK + SEG, "agentic_cc")
        print(f"MM256: layers {layers[0]}-{layers[-1]}, H {H}, text {info}", flush=True)

        def ids(r, s, e):
            return text[r * SHIFT + s : r * SHIFT + e]

        def single(slot, r, s, e):
            inp = runtime.make_chunk_input(ids(r, s, e).tolist(), s)
            runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e)
            return inp

        def packed_chunk(base):
            segs = [Segment(base + 0, H + CHUNK, H + CHUNK + SEG)] + [Segment(base + r, H, H + SEG) for r in (1, 2, 3)]
            lane = [ids(0, H + CHUNK, H + CHUNK + SEG)] + [ids(r, H, H + SEG) for r in (1, 2, 3)]
            desc = ChunkDescriptor.packed(CHUNK, segs)
            with _pc.packed("on"):
                inp = runtime.make_packed_input(desc, [t.long() for t in lane])
                runtime.prefill_packed(inp, kv, desc)
            return inp

        import ttnn

        def run_single(side, base):
            with _pc.switch("MM_TUNED256", side):
                inp = single(base, 0, H, H + CHUNK)
                model.sync()
            ttnn.deallocate(inp)

        def run_packed(side, base):
            with _pc.switch("MM_TUNED256", side):
                inp = packed_chunk(base)
                model.sync()
            ttnn.deallocate(inp)

        try:
            t0 = time.time()
            with _pc.switch("MM_TUNED256", "off"):
                for base in (0, 4):
                    for r in range(4):
                        for s in range(0, H, CHUNK):
                            ttnn.deallocate(single(base + r, r, s, s + CHUNK))
                model.sync()
            print(f"MM256: histories [0, {H}) in slots 0-7 in {time.time() - t0:.0f} s", flush=True)
            if "replay" in parts:  # diagnostic: the real calls of the single chunk replayed off vs on
                lays = [int(v) for v in os.environ.get("MM256_REPLAY_LAYERS", "16,17").split(",")]  # diagnostic
                with Replay(mesh, model, lays) as rp:
                    run_single("off", 0)
                res["replay"] = rp.check()
                bad = [r for r in res["replay"] if r["chips_differ"] or not r["det_off"]]
                print(f"MM256 replay: {len(res['replay'])} calls, {len(bad)} differ: {bad[:6]}", flush=True)
            if "offon" in parts:
                # off in slots 0-3 then on in slots 4-7 (single, then 4x2k); slots 0-3 re-prefilled to H (off, the same
                # bytes as slots 4-7's history) and on AGAIN in them (the repeat: a slot past H keeps its engram tail
                # and compressor state, so a chunk is never re-run in place). The repeat (on vs on: every layer output
                # and every served record of both slot sets) must be bit-identical; off vs on is a diagnostic (rel L2 a
                # layer over the sampled rows), judged by the reference gate
                sides = os.environ.get("MM256_SIDES", "off,on,on").split(",")  # diagnostic: off,off,off = control
                cases = (("single", run_single), ("4x2k", run_packed))
                o = {}
                for case, fn in cases:
                    for name, base, real in (("off", 0, sides[0]), ("on", 4, sides[1])):
                        o[(case, name)] = Outs()
                        o[(case, name)].wrap(model)
                        try:
                            fn(real, base)
                        finally:
                            Outs.unwrap(model)
                t0 = time.time()
                with _pc.switch("MM_TUNED256", "off"):
                    for r in range(4):
                        for s_ in range(0, H, CHUNK):
                            ttnn.deallocate(single(r, r, s_, s_ + CHUNK))
                    model.sync()
                print(f"MM256: slots 0-3 re-prefilled to {H} in {time.time() - t0:.0f} s", flush=True)
                for case, fn in cases:
                    o[(case, "on2")] = Outs()
                    o[(case, "on2")].wrap(model)
                    try:
                        fn(sides[2], 0)
                    finally:
                        Outs.unwrap(model)
                pairs = [(0, 4, H + CHUNK + SEG)] + [(r, 4 + r, H + SEG) for r in (1, 2, 3)]
                srv_rep, bad_srv = _served_diff(runtime, kv, model, pairs)
                for case, _fn in cases:
                    on, on2, off = o[(case, "on")].cap, o[(case, "on2")].cap, o[(case, "off")].cap
                    keys = sorted(set(on.dig) | set(on2.dig))
                    assert len(keys) >= len(layers) - 1, (case, keys)
                    out_rep = _pc.Capture.diff(on, on2, keys)
                    rep_layers = [k for k in keys if on.dig.get(k) != on2.dig.get(k)]
                    d = max(out_rep, srv_rep)  # (the served records: both cases' slots, compared once)
                    rel = {}
                    for k in keys:
                        a, b = off.smp.get(k), on.smp.get(k)
                        if a is None or b is None:
                            continue
                        a, b = a.double(), b.double()
                        rel[k] = float((a - b).norm() / a.norm().clamp_min(1e-30))
                        metrics.record(f"mm256_offon_rel_l2_{case}_L{k}", rel[k])
                    offon_same = [k for k in keys if off.dig.get(k) == on.dig.get(k)]
                    res[case] = {
                        "repeat_max_abs_diff": d,
                        "repeat_out_max_abs_diff": out_rep,
                        "layers_compared": len(keys),
                        "repeat_bad_layers": rep_layers,
                        "served_bad": [list(map(str, b)) for b in bad_srv[:20]],
                        "served_bad_n": len(bad_srv),
                        "offon_rel_l2": {str(k): v for k, v in rel.items()},
                        "offon_bit_identical_layers": offon_same,
                    }
                    metrics.record(f"mm256_repeat_max_abs_diff_{case}", d)
                    print(
                        f"MM256 {case}: on vs on repeat max abs diff {d} (layers {len(keys)}, outputs differ "
                        f"{rep_layers}, served differ {len(bad_srv)}: {bad_srv[:4]}); off vs on rel L2 max "
                        f"{max(rel.values() or [0.0]):.3e} (bit-identical layers {len(offon_same)} of {len(keys)})",
                        flush=True,
                    )
                # device ms off vs on (the same chunks again, alternating; the median of MM256_PROF_RUNS a side)
                runs = int(os.environ.get("MM256_PROF_RUNS", "2"))  # diagnostic
                for case, fn in (("single", run_single), ("4x2k", run_packed)):
                    for side, base in (("off", 0), ("on", 4)):
                        fn(side, base)  # (warm after the outputs' reads)
                    got = {"off": [], "on": []}
                    for _ in range(runs):
                        for side, base in (("off", 0), ("on", 4)):
                            got[side].append(_prof_run(mesh, lambda: fn(side, base)))
                    for side in ("off", "on"):
                        rs = sorted(got[side], key=lambda t: t[0])
                        dev, ph = rs[len(rs) // 2]
                        res.setdefault(case, {})[f"device_ms_{side}"] = round(dev, 3)
                        res[case][f"sections_{side}"] = {k: round(v, 3) for k, v in ph.items()}
                        res[case][f"device_ms_runs_{side}"] = [round(t[0], 3) for t in got[side]]
                        metrics.record(f"mm256_device_ms_{case}_{side}", round(dev, 3))
                        for k, v in ph.items():
                            metrics.record(f"mm256_sec_ms_{case}_{side}_{k}", round(v, 3))
                    print(
                        f"MM256 {case}: device ms off {res[case]['device_ms_off']} on {res[case]['device_ms_on']} "
                        f"(runs {res[case]['device_ms_runs_off']} / {res[case]['device_ms_runs_on']})",
                        flush=True,
                    )
            if "capture" in parts:
                OUT.mkdir(parents=True, exist_ok=True)
                from models.demos.deepseek_v41_flash_d_p.tt.cores import worker_grid

                g = worker_grid(mesh)
                allc = []
                for case, fn in (("single", run_single), ("4x2k", run_packed)):
                    fn("off", 0)  # (warm)
                    with Record(mesh) as rec:
                        fn("off", 0)
                    for r in rec.calls:
                        r["case"] = case
                    allc += rec.calls
                    (OUT / f"calls_{case}.json").write_text(
                        json.dumps({"grid": [g.x, g.y], "case": case, "pos": H, "calls": rec.calls}, indent=1)
                    )
                    ops = _ops_run(mesh, lambda: fn("off", 0))
                    (OUT / f"ops_{case}.json").write_text(json.dumps(ops, indent=1))
                    print(f"MM256 capture {case}: {len(rec.calls)} matmul calls, {len(ops)} op rows", flush=True)
                (OUT / "calls_mm256.json").write_text(
                    json.dumps({"grid": [g.x, g.y], "pos": H, "calls": allc}, indent=1)
                )
        finally:
            for key in [k for k in runtime._states if k[0] == id(kv)]:
                try:
                    runtime._states.pop(key).free()
                except Exception as e:  # noqa: BLE001
                    print(f"state free: {e}")
            for t in kv.contract.by_config.values():
                if t.is_allocated():
                    ttnn.deallocate(t)
            if loaded != "decode":
                model.use_decoder_skip(loaded)
    if "confirm" in parts:
        conf = _confirm(mesh)
        res["confirm"] = conf
        d = max([v["diff"] for v in conf.values()] or [0.0])
        metrics.record("mm256_repeat_max_abs_diff_confirm", d)
        for site, v in conf.items():
            metrics.record(f"mm256_confirm_us_{site}_off", round(v["us_off"], 2))
            metrics.record(f"mm256_confirm_us_{site}_on", round(v["us_on"], 2))
            metrics.record(f"mm256_confirm_repeat_diff_{site}", v["diff"])
            metrics.record(f"mm256_confirm_offon_rel_l2_{site}", v["offon_rel_l2"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offon.json").write_text(json.dumps(res, indent=1, default=str))
    worst = max([v.get("repeat_max_abs_diff", 0.0) for k, v in res.items() if k not in ("confirm", "replay")] or [0.0])
    if "confirm" in res:
        worst = max([worst] + [v["diff"] for v in res["confirm"].values()])
    assert worst == 0, f"MM_TUNED256=on is not deterministic: {json.dumps(res, default=str)[:2000]}"
