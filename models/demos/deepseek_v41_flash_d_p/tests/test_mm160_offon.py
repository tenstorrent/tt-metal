# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MB.9 (OWNER 2026-11-23): the 5120 chunk's re-swept matmul configs (MM_TUNED160=on: tt/mm_configs.GALAXY160 and the
grouped wo_a's GMM160 knobs), off vs on, a real 5120 chunk at a long history on the single-request serving path
(model_bringup has no packed path). OWNER rule 3 (2026-11-23): bit-identity with today's config is NOT a criterion;
accuracy is judged by the gate's test_ladder (last rung) with the switch on. Gated here: the on side's own repeat is
bit-identical.

The serving runtime over the held performance model (adapter.build_runtime through the gate's one load, layers 0-20,
DECODER_SKIP=decode), the engine's KV cache of 3 slots. One request of real text (agentic_cc) is prefilled from 0 to
the history H (MM160_POS, default 138240 = 27 x 5120, the AgentX p50) in slots 0, 1 and 2 the same way (whole 5120
chunks, MM_TUNED160=off: the same bytes). Then its chunk [H, H + 5120) (runtime.prefill_chunk: the single-request
path) runs off in slot 0, on in slot 1 and on AGAIN in slot 2 (the repeat over the same history: a slot past H keeps
its engram tail and compressor state, so a chunk is never re-run in place for a comparison).

mm160_repeat_max_abs_diff_single (gated == 0): slot 1 vs slot 2, bitwise (the digest of every chip's bytes) over every
computed layer's output (streams and the pre's used columns, every chip's rows) and the slot's served records of every
layer over [0, H + 5120) but the window rows (compressed KV, index keys; the window's served 128-row snapshot where
WINDOW_KV_KEEP keeps one); 0 when all match, else the max abs diff (at least 1e-30). mm160_offon_rel_l2_single_L<i>
(diagnostic): off vs on, ||off - on|| / ||off|| over the layer output's 64 sampled rows.

Device ms (the program real-time profiler, slowest chip; per section: testing/profiler.phase_of) off vs on: the same
chunk again in slots 0 / 1, MM160_PROF_RUNS (2) profiled runs a side, alternating, the median:
mm160_device_ms_single_<off|on>, mm160_sec_ms_single_<off|on>_<phase>.

confirm (op level, the held mesh, every chip): each GALAXY160 / GMM160 entry's call off vs on (tt/mm_configs.linear /
matmul under the switch; the grouped wo_a with its two knob sets) on random inputs: each side's 2 repeats and every
chip bit-identical (mm160_repeat_max_abs_diff_confirm, gated), its device us (20 calls a side, twice, the slowest chip):
mm160_confirm_us_<site>_<off|on>, and off vs on rel L2 (diagnostic).

capture (MM160_PARTS=capture, diagnostic: the sweep's inventory, tests/perf/test_matmul_sweep.py): the off chunk once
more with tt/mm_configs.RECORD on (every dense matmul call: site, shapes, dtypes, fidelity, config, L1 free for CBs)
and the grouped_matmul calls, and once under the op-mode real-time profiler (every op's device ns a chunk):
generated/ds41_mm160/calls_mm160.json, ops_single.json.

    python -m models.demos.common.bringup.resident submit --spec $BRINGUP_SPEC --cmd "BRINGUP_CHUNK=5120
        DS41_ENGRAM_TABLE=resident BRINGUP_POS_TOKENS=agentic_cc DS41_DECODER_SKIP=decode DS41_MM_TUNED160=on
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/test_mm160_offon.py"
Diagnostics: MM160_PARTS (offon,confirm | capture | confirm), MM160_POS, MM160_PROF_RUNS.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
import sys
import time

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
CHUNK = 5120
BATCH_SHARED = True
BATCH_ITEM_VARS = ("MM160_PARTS", "MM160_POS", "MM160_PROF_RUNS")
BATCH_LAYERS = "executed"  # F285: a performance measurement (layers 0-20 under DECODER_SKIP=decode)
CAP = 143360  # the runtime's max_seq: H + 5120 (28 x 5120)
BATCH_MAX_SEQ = CAP
SLOTS = 3
OUT = pathlib.Path("generated/ds41_mm160")
ITERS = 20
FID = {
    "MathFidelity.LoFi": "LoFi",
    "MathFidelity.HiFi2": "HiFi2",
    "MathFidelity.HiFi3": "HiFi3",
    "MathFidelity.HiFi4": "HiFi4",
}
MM = "models.demos.deepseek_v41_flash_d_p.tt.mm_configs"


def _parts() -> set:
    return set(os.environ.get("MM160_PARTS", "offon,confirm").split(","))  # diagnostic


def _pos() -> int:
    h = int(os.environ.get("MM160_POS", "138240"))  # diagnostic
    assert h % CHUNK == 0 and h + CHUNK <= CAP, h
    return h


@contextlib.contextmanager
def switch(key: str, value: str):
    """A call-time switch (bringup/hooks.py batch_call_settings) = value for the calls inside: the frozen snapshot's
    entry when the model froze its settings (DS41-314), and the environment."""
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    name = settings.env_name(key)
    snap = settings.snap
    old_env = os.environ.get(name)
    old_snap = None if snap is None else snap.get(key)
    os.environ[name] = value
    if snap is not None:
        snap[key] = value
    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old_env
        if snap is not None and settings.snap is snap:
            snap[key] = old_snap


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def sample_idx(n: int, k: int = 64) -> torch.Tensor:
    return torch.unique(torch.linspace(0, max(0, n - 1), min(k, n)).long())


def chips(t) -> list:
    """Every chip's shard of device tensor ``t`` (get_device_tensors' order) as host tensors, in one read."""
    import ttnn

    mesh = t.device()
    n, b = int(mesh.get_num_devices()), int(t.shape[0])
    allc = ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    assert allc.shape[0] == n * b, (tuple(allc.shape), tuple(t.shape), n)
    return list(allc.reshape(n, b, *allc.shape[1:]).unbind(0))


class Capture:
    """Per key: the digest of the rows and a float32 copy of 64 sampled rows (the max abs diff where digests differ)."""

    def __init__(self):
        self.dig, self.smp = {}, {}

    def add(self, key, rows: torch.Tensor) -> None:
        self.dig[key] = digest(rows)
        self.smp[key] = rows[sample_idx(rows.shape[0])].float().clone()

    @staticmethod
    def diff(a: "Capture", b: "Capture", keys) -> float:
        worst = 0.0
        for k in keys:
            if a.dig.get(k) == b.dig.get(k) and k in a.dig:
                continue
            if k not in a.smp or k not in b.smp or a.smp[k].shape != b.smp[k].shape:
                return float("inf")
            d = torch.nan_to_num((a.smp[k] - b.smp[k]).abs(), nan=float("inf"))
            worst = max(worst, float(d.max()) if d.numel() else 0.0, 1e-30)
        return worst


# ------------------------------------------------------------------------------------------------- module handles
def _mm_globals() -> list:
    """Every live module dict of tt/mm_configs (the resident runner may hold the model's functions from an earlier
    module object than this import, DS41-311 / DS41-329): RECORD must be set where the model's linear / matmul read
    it."""
    out = {}
    if MM in sys.modules:
        out[id(vars(sys.modules[MM]))] = vars(sys.modules[MM])
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith("models.demos.deepseek_v41_flash_d_p.tt") or mod is None:
            continue
        m = getattr(mod, "mm", None)
        if m is not None and hasattr(m, "RECORD"):
            out[id(vars(m))] = vars(m)
        for f in ("linear", "matmul"):
            g = getattr(getattr(mod, f, None), "__globals__", None)
            if g is not None and "RECORD" in g and "GALAXY" in g:
                out[id(g)] = g
    return list(out.values())


class Record:
    """tt/mm_configs.RECORD on (every live module object) and ttnn.bringup.grouped_matmul recorded, for the calls
    inside."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        import ttnn

        self.gl = _mm_globals()
        for g in self.gl:
            g["RECORD"] = self.calls
        self.gmm = ttnn.bringup.grouped_matmul
        mm = sys.modules[MM]
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


class Hits:
    """OV-58: count the calls that take a MM_TUNED160 config, at call time, in every live tt/mm_configs module dict:
    a dense call whose program_config key is a GALAXY160 key while the switch reads on (program_config then returns that
    entry), and every gmm_knobs call answered from GMM160. {site: calls}."""

    def __init__(self):
        self.n = {}

    def __enter__(self):
        self.old = []
        me = self
        for g in _mm_globals():
            sc0, gk0 = g["_site_config"], g["gmm_knobs"]
            self.old.append((g, sc0, gk0))

            def sc(x, w, dtype, ckc, batched=False, rule=True, _f=sc0, _g=g):
                pc = _f(x, w, dtype, ckc, batched=batched, rule=rule)
                if pc is not None and _g["mode"]() == "galaxy" and _g["tuned160"]():
                    grid = _g["worker_grid"](x.device())
                    fp32 = _g["_ckc_fp32"](ckc)
                    tk = _g["table_key"](x.shape, w.shape, x.dtype, w.dtype, dtype, fp32, grid.x, grid.y)
                    if tk in _g["GALAXY160"]:
                        key = f"{tuple(x.shape)[-2]}x{tuple(x.shape)[-1]}x{tuple(w.shape)[-1]}"
                        me.n[key] = me.n.get(key, 0) + 1
                return pc

            def gk(x_shape, w_shape, _f=gk0, _g=g):
                out = _f(x_shape, w_shape)
                if _g["mode"]() == "galaxy" and _g["tuned160"]() and (tuple(x_shape), tuple(w_shape)) in _g["GMM160"]:
                    me.n["wo_a_gmm"] = me.n.get("wo_a_gmm", 0) + 1
                return out

            g["_site_config"], g["gmm_knobs"] = sc, gk
        return self

    def __exit__(self, *exc):
        for g, sc0, gk0 in self.old:
            g["_site_config"], g["gmm_knobs"] = sc0, gk0
        return False

    @property
    def total(self) -> int:
        return sum(self.n.values())


# ------------------------------------------------------------------------------------------------- one chunk
class Outs:
    """Per computed layer: the digest of every chip's output rows (streams and the pre's used columns) and 64 sampled
    rows (float32)."""

    def __init__(self):
        self.cap = Capture()

    def wrap(self, model):
        from models.demos.deepseek_v41_flash_d_p.tt.layout import HC

        layer = model.layer
        me = self

        def wrapped(i, h, *a, **k):
            out = layer(i, h, *a, **k)
            if out is not h and getattr(out, "streams", None) is not None:
                w = out.streams.shape[-1]
                st = torch.cat([p.reshape(-1, w) for p in chips(out.streams)], dim=0)
                parts = [st.float()]
                if getattr(out, "pre", None) is not None:
                    wp = out.pre.shape[-1]
                    parts.append(torch.cat([p.reshape(-1, wp) for p in chips(out.pre)], dim=0)[:, :HC].float())
                me.cap.add(int(i), torch.cat(parts, dim=-1))
            return out

        model.layer = wrapped

    @staticmethod
    def unwrap(model):
        if "layer" in vars(model):
            del model.layer


def _served(runtime, kv, model, slot: int, end: int) -> Capture:
    """The slot's served records of every layer over [0, end) but the window rows (module docstring)."""
    st = runtime.state(kv, int(slot))
    cap = Capture()
    for i in model.layers:
        for name, t in st.served(int(i), int(end)).items():
            if name == "0_window_kv":
                continue
            cap.add((int(i), name), t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(-1, 1))
        c = st.window[i]
        if getattr(c, "snap", None) is not None:  # (WINDOW_KV_KEEP=snapshot128: the served 128-row snapshot)
            rows = torch.cat([x.reshape(-1, x.shape[-1]) for x in c.snapshot_host()], dim=0)
            cap.add((int(i), "snapshot"), rows)
    return cap


def _prof(mesh, fn) -> tuple[float, dict]:
    """One profiled run: the slowest chip's device ms and the per-phase ms (test_skip_positions._device_split)."""
    from models.demos.common.bringup.testing import profiler
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP

    profiler.enable(mesh, rt=True)
    profiler.set_layer(None)
    try:
        fn()
        profiler.signpost("end")
        profiler.finish_rt()
        dev = SP._device_split(profiler.result())
    finally:
        profiler.disable()
    return dev["slowest"], dev["phases"]


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
                {"section": sec, "op": r.get("op"), "shape": r.get("shape"), "calls": r.get("calls"), "ns": r.get("ns")}
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
    p = OUT / "calls_mm160.json"
    return json.loads(p.read_text())["calls"] if p.exists() else []


def _confirm(mesh) -> dict:
    """Every GALAXY160 / GMM160 entry off vs on, on the whole mesh (module docstring)."""
    import ttnn
    from models.demos.common.bringup.testing import profiler

    mm = sys.modules[MM]
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
    for key in mm.GALAXY160:
        shape, wshape, a, b, o, fp32 = key[:6]
        batched = len(wshape) >= 3 and wshape[1] > 1
        op = "matmul" if batched else "linear"
        c = match(key, op) or match(key, "linear" if batched else "matmul")
        cases.append(("mm", key, c, shape, wshape, DT[a], DT[b], DT[o], fp32, c["op"] if c else op))
    for (xs, ws), _knobs in mm.GMM160.items():
        c = next((c for c in caps if c["op"] == "grouped_matmul" and tuple(c["a"]) == xs and tuple(c["b"]) == ws), None)
        o = DT[c["out_dtype"].split(".")[-1]] if c else ttnn.bfloat8_b
        cases.append(("gmm", (xs, ws), c, xs, ws, ttnn.bfloat8_b, ttnn.bfloat8_b, o, True, "grouped_matmul"))
    res = {}
    torch.manual_seed(160)
    for kind, key, c, shape, wshape, adt, bdt, odt, fp32, op in cases:
        site = (c or {}).get("site", "?").replace(".", "_") + f"_{shape[-2]}x{shape[-1]}x{wshape[-1]}"
        ckc = _ckc(mesh, c, fp32)
        x = up(shape, adt)
        w = up(wshape, bdt, 1.0 / shape[-1] ** 0.5)

        def run(side):
            with switch("MM_TUNED160", side):
                if kind == "gmm":
                    from models.demos.deepseek_v41_flash_d_p.tt.cores import max_grid

                    bw, depth, rt = mm.gmm_knobs(shape, wshape)
                    return ttnn.bringup.grouped_matmul(
                        x, w, dtype=odt, compute_kernel_config=ckc, max_grid=max_grid(), block_w=bw, depth=depth, rt=rt
                    )
                f = mm.matmul if op == "matmul" else mm.linear
                return f(x, w, dtype=odt, compute_kernel_config=ckc, memory_config=DRAM, site="mm160.confirm")

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
            f"MM160 confirm {site}: off {us['off']:.1f} us, on {us['on']:.1f} us, repeat diff {diff}, off vs on rel "
            f"L2 {offon_rel:.2e}",
            flush=True,
        )
        for t in (x, w):
            ttnn.deallocate(t)
    return res


# ------------------------------------------------------------------------------------------------- the test
@mesh_parametrize(timeout_s=3 * 3600)
def test_mm160_offon(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=5120 (the load's chunk)"
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.common.bringup.testing import batch
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    parts = _parts()
    mesh = mesh_device
    res = {}
    if parts & {"offon", "capture"}:
        monkeypatch.setenv(settings.env_name("DECODER_SKIP"), "decode")  # (call-time: the served mode, as the gate)
        full = list(S.layers())
        with switch("DECODER_SKIP", "decode"):  # (DS41-314: a held model reads its frozen snapshot)
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
        text, info = texts.tokens(S, H + CHUNK, "agentic_cc")
        print(f"MM160: layers {layers[0]}-{layers[-1]}, H {H}, text {info}", flush=True)

        def single(slot, s, e):
            inp = runtime.make_chunk_input(text[s:e].tolist(), s)
            runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e)
            return inp

        def run(side, slot):
            with switch("MM_TUNED160", side):
                inp = single(slot, H, H + CHUNK)
                model.sync()
            ttnn.deallocate(inp)

        try:
            t0 = time.time()
            with switch("MM_TUNED160", "off"):
                for slot in range(SLOTS):
                    for s in range(0, H, CHUNK):
                        ttnn.deallocate(single(slot, s, s + CHUNK))
                model.sync()
            print(f"MM160: histories [0, {H}) in slots 0-{SLOTS - 1} in {time.time() - t0:.0f} s", flush=True)
            if "offon" in parts:
                o, hits = {}, {}
                for name, slot, side in (("off", 0, "off"), ("on", 1, "on"), ("on2", 2, "on")):
                    o[name] = Outs()
                    o[name].wrap(model)
                    try:
                        with Hits() as hc:
                            run(side, slot)
                        hits[name] = hc
                    finally:
                        Outs.unwrap(model)
                # OV-58: the proof the switch reaches the model's calls (on: every tuned site, off: none)
                for name in ("off", "on"):
                    metrics.record(f"mm160_tuned_hits_single_{name}", hits[name].total)
                    for site, n in hits[name].n.items():
                        metrics.record(f"mm160_tuned_hits_single_{name}_{site}", n)
                print(f"MM160 tuned-config hits: off {hits['off'].n}, on {hits['on'].n}", flush=True)
                assert hits["off"].total == 0 and hits["on"].total > 0, (hits["off"].n, hits["on"].n)
                end = H + CHUNK
                s1, s2 = _served(runtime, kv, model, 1, end), _served(runtime, kv, model, 2, end)
                skeys = sorted(set(s1.dig) | set(s2.dig))
                srv_rep = Capture.diff(s1, s2, skeys)
                bad_srv = [k for k in skeys if s1.dig.get(k) != s2.dig.get(k)]
                on, on2, off = o["on"].cap, o["on2"].cap, o["off"].cap
                keys = sorted(set(on.dig) | set(on2.dig))
                assert len(keys) >= len(layers) - 1, keys
                out_rep = Capture.diff(on, on2, keys)
                rep_layers = [k for k in keys if on.dig.get(k) != on2.dig.get(k)]
                d = max(out_rep, srv_rep)
                rel = {}
                for k in keys:
                    a, b = off.smp.get(k), on.smp.get(k)
                    if a is None or b is None:
                        continue
                    a, b = a.double(), b.double()
                    rel[k] = float((a - b).norm() / a.norm().clamp_min(1e-30))
                    metrics.record(f"mm160_offon_rel_l2_single_L{k}", rel[k])
                offon_same = [k for k in keys if off.dig.get(k) == on.dig.get(k)]
                res["single"] = {
                    "repeat_max_abs_diff": d,
                    "repeat_out_max_abs_diff": out_rep,
                    "repeat_served_max_abs_diff": srv_rep,
                    "layers_compared": len(keys),
                    "served_records_compared": len(skeys),
                    "repeat_bad_layers": rep_layers,
                    "served_bad": [list(map(str, b)) for b in bad_srv[:20]],
                    "offon_rel_l2": {str(k): v for k, v in rel.items()},
                    "offon_bit_identical_layers": offon_same,
                    "tuned_hits_off": hits["off"].n,
                    "tuned_hits_on": hits["on"].n,
                }
                metrics.record("mm160_repeat_max_abs_diff_single", d)
                print(
                    f"MM160 single: on vs on repeat max abs diff {d} (layers {len(keys)}, outputs differ {rep_layers}, "
                    f"served records {len(skeys)}, differ {bad_srv[:4]}); off vs on rel L2 max "
                    f"{max(rel.values() or [0.0]):.3e} (bit-identical layers {len(offon_same)} of {len(keys)})",
                    flush=True,
                )
                # device ms off vs on (the same chunk again in slots 0 / 1, alternating; the median of the runs)
                runs = int(os.environ.get("MM160_PROF_RUNS", "2"))  # diagnostic
                for side, slot in (("off", 0), ("on", 1)):
                    run(side, slot)  # (warm after the outputs' reads)
                got = {"off": [], "on": []}
                for _ in range(runs):
                    for side, slot in (("off", 0), ("on", 1)):
                        got[side].append(_prof(mesh, lambda: run(side, slot)))
                for side in ("off", "on"):
                    rs = sorted(got[side], key=lambda t: t[0])
                    dev, ph = rs[len(rs) // 2]
                    res["single"][f"device_ms_{side}"] = round(dev, 3)
                    res["single"][f"sections_{side}"] = {k: round(v, 3) for k, v in ph.items()}
                    res["single"][f"device_ms_runs_{side}"] = [round(t[0], 3) for t in got[side]]
                    metrics.record(f"mm160_device_ms_single_{side}", round(dev, 3))
                    for k, v in ph.items():
                        metrics.record(f"mm160_sec_ms_single_{side}_{k}", round(v, 3))
                print(
                    f"MM160 single: device ms off {res['single']['device_ms_off']} on {res['single']['device_ms_on']} "
                    f"(runs {res['single']['device_ms_runs_off']} / {res['single']['device_ms_runs_on']})",
                    flush=True,
                )
            if "capture" in parts:
                OUT.mkdir(parents=True, exist_ok=True)
                from models.demos.deepseek_v41_flash_d_p.tt.cores import worker_grid

                g = worker_grid(mesh)
                run("off", 0)  # (warm)
                with Record() as rec:
                    run("off", 0)
                for r in rec.calls:
                    r["case"] = "single"
                (OUT / "calls_mm160.json").write_text(
                    json.dumps({"grid": [g.x, g.y], "pos": H, "calls": rec.calls}, indent=1)
                )
                ops = _ops_run(mesh, lambda: run("off", 0))
                (OUT / "ops_single.json").write_text(json.dumps(ops, indent=1))
                print(f"MM160 capture: {len(rec.calls)} matmul calls, {len(ops)} op rows", flush=True)
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
        metrics.record("mm160_repeat_max_abs_diff_confirm", d)
        for site, v in conf.items():
            metrics.record(f"mm160_confirm_us_{site}_off", round(v["us_off"], 2))
            metrics.record(f"mm160_confirm_us_{site}_on", round(v["us_on"], 2))
            metrics.record(f"mm160_confirm_repeat_diff_{site}", v["diff"])
            metrics.record(f"mm160_confirm_offon_rel_l2_{site}", v["offon_rel_l2"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "offon.json").write_text(json.dumps(res, indent=1, default=str))
    worst = max([v.get("repeat_max_abs_diff", 0.0) for k, v in res.items() if k != "confirm"] or [0.0])
    if "confirm" in res:
        worst = max([worst] + [v["diff"] for v in res["confirm"].values()])
    assert worst == 0, f"MM_TUNED160=on is not deterministic: {json.dumps(res, default=str)[:2000]}"
