# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""QB.1 (OWNER 2026-11-27, plan step 4): the model's wq_b with WQB_BANK off vs on in ONE load (WQB_BANK is call-time;
the held model gets both wq_b copies here, tt/attention.wqb_ensure), on real chunks at H = 139264 (the AgentX p50):

    single8192  request 0's chunk [H, H + 8192) (runtime.prefill_chunk: the single-request path)
    4x2k        request 0's [H + 8192, H + 10240) + requests 1-3's [H, H + 2048), 8 chips each (runtime.prefill_packed)

The serving runtime over the held performance model (adapter.build_runtime through the gate's one load, layers 0-20
under DECODER_SKIP=decode), the engine's KV cache of SLOTS slots. Five requests of real text (agentic_cc; request r reads
from r x SHIFT) are prefilled from 0 to H in two slot sets (A: 0-3 and 8, B: 4-7 and 9) with WQB_BANK=off. Then off in
set A and on in set B; set A is re-prefilled to H (off) and on runs AGAIN in it (the repeat).

The bank call is NOT bit-identical to the stock call (another K order, DS41-390): off vs on is a diagnostic only, the
accuracy is judged by the gate's reference test with the switch on.

    wqb_repeat_max_abs_diff_<case> (gated == 0): on vs on (set B vs the repeat in set A), bitwise (every chip's bytes)
        over every computed layer's output (streams and the pre's used columns, all rows) and every served record of
        every slot (window rows / snapshot, compressed KV, index keys, host engram tails); 0 when all match.
    wqb_offon_rel_l2_<case> (diagnostic): off vs on, the largest per-layer rel L2 over 64 sampled rows of every layer
        output (the per-layer values in generated/ds41_woa/wqb_offon.json and the record's ``layers``).
    wqb_device_ms_<case>_<off|on> (the whole chunk), wqb_attn_ms_<case>_<off|on> (the attention section: phase
        "attention" + wq_b's own "wqb" signposts), wqb_sec_ms_<case>_<off|on> (wq_b's own programs): the real-time
        profiler, slowest chip, the median of WQB_OFFON_PROF_RUNS runs a side, sides alternating.
    wqb_host_us_<off|on>_<case>: host us a wq_b call (perf_counter, warm, no profiler).
    wqb_load_s_bank: seconds the bank copies of the executed layers took (checkpoint re-read + host prep + upload).

    PYTHONPATH=$PWD BRINGUP_CHUNK=8192 DS41_PACKED=on ... DS41_DECODER_SKIP=decode DS41_WQB_BANK=on \\
        scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_wqb_offon.py
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
import statistics
import time

import pytest

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.deepseek_v41_flash_d_p.tests import test_mm256_offon as _mm
from models.demos.deepseek_v41_flash_d_p.tests.packed import packed_common as _pc


def _fresh(mod):
    """The helper module as this file sees it now (the resident runner keeps helper modules of an earlier request;
    WA.4: reload when present, else import again)."""
    import sys

    if sys.modules.get(mod.__name__) is mod:
        return importlib.reload(mod)
    return importlib.import_module(mod.__name__)


_pc = _fresh(_pc)
_mm = _fresh(_mm)
CHUNK = _pc.CHUNK

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("WQB_OFFON_PROF_RUNS",)
BATCH_LAYERS = "executed"  # F285: layers 0-20 under DECODER_SKIP=decode
H = 139264  # the AgentX p50 (17 x 8192; test_woa_offon's near layout)
CAP = 155648  # the runtime's max_seq: H + 8192 + 2048 rounded up to a chunk (19 x 8192)
BATCH_MAX_SEQ = CAP
SLOTS = 10
SHIFT = 4096  # request r reads the text from r x SHIFT
SEG = 2048
OUT = pathlib.Path("generated/ds41_woa")
CASES = ("single8192", "4x2k")


def _atts(model) -> list:
    return [b.attention for _, b in sorted(model.blocks.items()) if hasattr(b.attention, "wqb_ensure")]


class WqbCalls:
    """Every attention's _wqb_mm wrapped for the calls inside: host us a call (perf_counter), optionally a profiler
    section "wqb" around each call (the section before restored after it)."""

    def __init__(self, atts, sign: bool = False):
        self.atts, self.sign = atts, sign
        self.host_s, self.calls = 0.0, 0

    def __enter__(self):
        from models.demos.common.bringup.testing import profiler

        me = self
        for att in self.atts:
            f = type(att)._wqb_mm.__get__(att)

            def call(*a, f=f, **k):
                prev = profiler._P.get("current_full") if me.sign else None
                if me.sign:
                    profiler.signpost("wqb")
                t0 = time.perf_counter()
                y = f(*a, **k)
                me.host_s += time.perf_counter() - t0
                me.calls += 1
                if me.sign:
                    profiler.signpost(prev or "other")
                return y

            att._wqb_mm = call
        return self

    def __exit__(self, *exc):
        for att in self.atts:
            vars(att).pop("_wqb_mm", None)
        return False


def _prof(mesh, fn) -> tuple[float, float, float]:
    """(the slowest chip's device ms of the run, wq_b's ms: the "wqb" sections, the attention section with wq_b in it:
    phase "attention" + wq_b's ms) of one real-time profiled run."""
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
    wqb = sum(v for k, v in (pr.get("kernel_ns") or {}).items() if k == "wqb" or str(k).endswith(".wqb")) / 1e6
    att = L._phases(pr).get("attention", 0.0)
    return L._device(pr), wqb, att + wqb


def _rel_layers(a: "_pc.Capture", b: "_pc.Capture", keys) -> dict:
    """{layer: rel L2 of b vs a} over each layer's 64 sampled rows (float64)."""
    out = {}
    for k in keys:
        if k not in a.smp or k not in b.smp:
            out[k] = float("inf")
            continue
        x, y = a.smp[k].double(), b.smp[k].double()
        n = float(x.norm())
        out[k] = float((y - x).norm()) / n if n > 0 else float((y - x).norm())
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_wqb_offon(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.common.bringup.testing import batch
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    mesh = mesh_device
    skip = "decode"
    monkeypatch.setenv(settings.env_name("DECODER_SKIP"), skip)  # (call-time)
    full = list(S.layers())
    with _pc.switch("DECODER_SKIP", skip):  # (DS41-314: a held model reads its frozen snapshot)
        layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    assert CAP <= -(-deploy // CHUNK) * CHUNK, f"slots of {CAP} pass the deployment ({deploy})"
    monkeypatch.setitem(S.data["target"], "seq", max(CAP, int(S.data["target"]["seq"])))
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(SLOTS))  # (the runtime's KV cache)
    runtime, adapter, hf, params = SP._runtime(mesh, layers, CHUNK, CAP)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != skip:
        model.use_decoder_skip(skip)
    assert model.max_seq >= CAP, f"the held model's tables reach {model.max_seq} < {CAP}"
    res = {"layers": [layers[0], layers[-1]], "H": H, "cap": CAP, "max_seq": model.max_seq}
    # both wq_b copies on the held model (a test load holds both; the load's own copy kept, the added ones dropped
    # after)
    atts = _atts(model)
    added = {k: [a for a in atts if not a.wqb_has(k)] for k in ("interleaved", "bank")}
    t0 = time.perf_counter()
    for att in atts:
        att.wqb_ensure(("interleaved", "bank"))
    ttnn.synchronize_device(mesh)
    res["ensure_s"] = time.perf_counter() - t0
    res["added"] = {k: len(v) for k, v in added.items()}
    if added["bank"]:
        metrics.record("wqb_load_s_bank", round(res["ensure_s"], 3), layers=len(added["bank"]))
    lay, rows = atts[0].wqb_layout, atts[0].geo.q_rows
    print(f"WQB: layers {layers[0]}-{layers[-1]} ({len(atts)} wq_b), copies added {res['added']} in "
          f"{res['ensure_s']:.1f} s; layout {len(lay.slots)} cores, pn max {lay.pn_max}, rows {rows}, knobs "
          f"{mm.wqb_bank_knobs(rows)}", flush=True)  # fmt: skip
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    text, info = texts.tokens(S, 4 * SHIFT + H + CHUNK + SEG, "agentic_cc")
    print(f"WQB: H {H}, slots of {CAP}, cases {CASES}, text {info}", flush=True)

    def ids(r, s, e):
        return text[r * SHIFT + s : r * SHIFT + e]

    def single(slot, r, s, e):
        inp = runtime.make_chunk_input(ids(r, s, e).tolist(), s)
        runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e)
        return inp

    def run_single(side, base):
        with _pc.switch("WQB_BANK", side):
            inp = single(base, 0, H, H + CHUNK)
            model.sync()
        ttnn.deallocate(inp)

    def run_packed(side, base):
        s0 = H + CHUNK  # (request 0's segment: after its single chunk)
        segs = [Segment(base + 0, s0, s0 + SEG)] + [Segment(base + r, H, H + SEG) for r in (1, 2, 3)]
        lane = [ids(0, s0, s0 + SEG)] + [ids(r, H, H + SEG) for r in (1, 2, 3)]
        desc = ChunkDescriptor.packed(CHUNK, segs)
        with _pc.switch("WQB_BANK", side), _pc.packed("on"):
            inp = runtime.make_packed_input(desc, [t.long() for t in lane])
            runtime.prefill_packed(inp, kv, desc)
            model.sync()
        ttnn.deallocate(inp)

    fns = {"single8192": run_single, "4x2k": run_packed}

    def history(slots_reqs):
        with _pc.switch("WQB_BANK", "off"):
            for slot, r in slots_reqs:
                for s in range(0, H, CHUNK):
                    ttnn.deallocate(single(slot, r, s, s + CHUNK))
            model.sync()

    set_a = [(r, r) for r in range(4)] + [(8, 4)]
    set_b = [(4 + r, r) for r in range(4)] + [(9, 4)]
    pairs_ab = [(0, 4, H + CHUNK + SEG)] + [(r, 4 + r, H + SEG) for r in (1, 2, 3)]
    try:
        t0 = time.time()
        history(set_a + set_b)
        print(f"WQB: histories [0, {H}) in slots {[s for s, _ in set_a + set_b]} in {time.time() - t0:.0f} s",
              flush=True)  # fmt: skip
        o = {}
        for side, base in (("off", 0), ("on", 4)):
            for case, fn in fns.items():
                o[(case, side)] = _mm.Outs()
                o[(case, side)].wrap(model)
                try:
                    fn(side, base)
                finally:
                    _mm.Outs.unwrap(model)
        t0 = time.time()
        history(set_a)
        print(f"WQB: set A re-prefilled to {H} in {time.time() - t0:.0f} s", flush=True)
        for case, fn in fns.items():
            o[(case, "on2")] = _mm.Outs()
            o[(case, "on2")].wrap(model)
            try:
                fn("on", 0)
            finally:
                _mm.Outs.unwrap(model)
        srv_rep, bad_rep = _mm._served_diff(runtime, kv, model, pairs_ab)
        for case in CASES:
            off, on, on2 = (o[(case, s)].cap for s in ("off", "on", "on2"))
            keys = sorted(set(on.dig) | set(off.dig) | set(on2.dig))
            assert len(keys) >= len(layers) - 1, (case, keys)
            out_rep = _pc.Capture.diff(on, on2, keys)
            d_rep = max(out_rep, srv_rep)
            rel = _rel_layers(off, on, keys)
            worst = max(rel.values()) if rel else float("nan")
            res[case] = {
                "repeat_max_abs_diff": d_rep,
                "repeat_out_max_abs_diff": out_rep,
                "repeat_bad_layers": [k for k in keys if on.dig.get(k) != on2.dig.get(k)],
                "offon_rel_l2_layers": {str(k): v for k, v in rel.items()},
                "offon_rel_l2_max": worst,
                "offon_bitequal_layers": [k for k in keys if off.dig.get(k) == on.dig.get(k)],
                "layers_compared": len(keys),
            }
            metrics.record(f"wqb_repeat_max_abs_diff_{case}", d_rep, layers=len(keys), served_bad=len(bad_rep))
            metrics.record(f"wqb_offon_rel_l2_{case}", worst, layers={str(k): f"{v:.3e}" for k, v in rel.items()})
            print(f"WQB {case}: on repeat {d_rep} (outputs {out_rep}, served {srv_rep}: {bad_rep[:4]}); off vs on "
                  f"rel L2 a layer (64 sampled rows): "
                  f"{', '.join(f'L{k} {v:.2e}' for k, v in rel.items())}; {len(keys)} layers", flush=True)  # fmt: skip
        res["served_bad_repeat"] = [list(map(str, b)) for b in bad_rep[:20]]
        # host us a wq_b call (warm, no profiler), each case and side (in-place re-runs: DECODER_SKIP=decode)
        for case, fn in fns.items():
            for side, base in (("off", 0), ("on", 4)):
                fn(side, base)  # (warm)
                got = []
                for _ in range(2):
                    with WqbCalls(atts) as wc:
                        fn(side, base)
                    got.append(wc.host_s / max(1, wc.calls) * 1e6)
                us = round(min(got), 1)
                res[case][f"host_us_{side}"] = us
                res[case]["wqb_calls"] = wc.calls
                metrics.record(f"wqb_host_us_{side}_{case}", us, calls=wc.calls, runs=[round(v, 1) for v in got])
        # device ms (whole chunk; wq_b's own programs signposted), the median of runs a side, alternating
        runs = int(os.environ.get("WQB_OFFON_PROF_RUNS", "3"))  # diagnostic
        for case, fn in fns.items():
            got = {"off": [], "on": []}
            for _ in range(runs):
                for side, base in (("off", 0), ("on", 4)):

                    def go(side=side, base=base, fn=fn):
                        with WqbCalls(atts, sign=True):
                            fn(side, base)

                    got[side].append(_prof(mesh, go))
            for side in ("off", "on"):
                dev = statistics.median(t[0] for t in got[side])
                wqb_med = statistics.median(t[1] for t in got[side])
                att_med = statistics.median(t[2] for t in got[side])
                res[case][f"device_ms_{side}"] = round(dev, 3)
                res[case][f"wqb_ms_{side}"] = round(wqb_med, 3)
                res[case][f"attn_ms_{side}"] = round(att_med, 3)
                res[case][f"device_ms_runs_{side}"] = [round(t[0], 3) for t in got[side]]
                res[case][f"wqb_ms_runs_{side}"] = [round(t[1], 3) for t in got[side]]
                res[case][f"attn_ms_runs_{side}"] = [round(t[2], 3) for t in got[side]]
                metrics.record(f"wqb_device_ms_{case}_{side}", round(dev, 3), skip=skip)
                metrics.record(f"wqb_sec_ms_{case}_{side}", round(wqb_med, 3), skip=skip)
                metrics.record(f"wqb_attn_ms_{case}_{side}", round(att_med, 3), skip=skip)
            r = res[case]
            print(f"WQB {case}: device ms off {r['device_ms_off']} on {r['device_ms_on']} (runs "
                  f"{r['device_ms_runs_off']} / {r['device_ms_runs_on']}); attention ms off {r['attn_ms_off']} on "
                  f"{r['attn_ms_on']}; wq_b ms off {r['wqb_ms_off']} on {r['wqb_ms_on']}; host us a call off "
                  f"{r['host_us_off']} on {r['host_us_on']}", flush=True)  # fmt: skip
    finally:
        for key in [k for k in runtime._states if k[0] == id(kv)]:
            try:
                runtime._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        for t in kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        for kind, lst in added.items():  # (the held model back to its load's copies)
            for att in lst:
                att.wqb_drop(kind)
        if model.skip_mode != loaded:
            model.use_decoder_skip(loaded)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wqb_offon.json").write_text(json.dumps(res, indent=1, default=str))
    bad = {c: res[c]["repeat_max_abs_diff"] for c in CASES}
    assert all(v == 0 for v in bad.values()), f"WQB_BANK=on is not repeatable: {bad}"
