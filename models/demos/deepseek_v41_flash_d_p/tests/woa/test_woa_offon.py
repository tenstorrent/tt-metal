# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WA.3 (OWNER 2026-11-27, wo_a bank-local reads): the model's wo_a with WOA_BANK off vs on in ONE load (WOA_BANK is
call-time; the held model gets both wo_a copies here, tt/attention.woa_ensure, OV-77), on real chunks:

    single8192  request 0's chunk [H, H + 8192) (runtime.prefill_chunk: the single-request path)
    4x2k        request 0's [H + 8192, H + 10240) + requests 1-3's [H, H + 2048), 8 chips each (runtime.prefill_packed)
    single5120  request 4's last, partial chunk [H, H + 5120) (single path, actual_end - start = 5120; the 8192 load
                pads it to 256 rows a chip: the 160-row call of a 5120-chunk model is checked at op level below)

The serving runtime over the held performance model (adapter.build_runtime through the gate's one load, layers 0-20 under
DECODER_SKIP=decode; all 40 with DECODER_SKIP=off when WOA_OFFON_LAYERS=all), the engine's KV cache of SLOTS slots. Five
requests of real text (agentic_cc; request r reads from r x SHIFT) are prefilled from 0 to H (WOA_OFFON_POS, default
139264, the AgentX p50) in two slot sets (A: 0-3 and 8, B: 4-7 and 9), WOA_BANK=off. Then off in set A and on in set B;
set A is re-prefilled to H and on runs AGAIN in it (the repeat; a slot past H keeps its engram tail and compressor state,
so a chunk is never re-run in place).

    woa_offon_max_abs_diff_<case> (gated == 0): off vs on, bitwise (every chip's bytes) over every computed layer's
        output (streams and the pre's used columns, all rows) and every served record of every slot of both sets
        (window rows / snapshot, compressed KV, index keys, every layer; host engram tails); 0 when all match.
    woa_repeat_max_abs_diff_<case> (gated == 0): on vs on (set B vs the repeat in set A), the same comparison.
    woa_offon_max_abs_diff_real160: op level on REAL inputs (DS41-390): every wo_a call's input of the single8192 off
        chunk (chip 0, its first 160 rows, replicated) through today's call and the bank call at the 160-row knobs.
    woa_device_ms_<case>_<off|on> (the whole chunk) and woa_sec_ms_<case>_<off|on> (wo_a's own programs, signposted
        around each call): the real-time profiler, slowest chip, the median of WOA_OFFON_PROF_RUNS runs a side. The
        timing re-runs chunks in place, which DECODER_SKIP=off refuses (window rows not kept): with
        WOA_OFFON_LAYERS=all it runs under DECODER_SKIP=decode (layers 0-20; WA.4), the comparison over all 40.
    woa_host_us_<off|on>_<case>: host us a wo_a call (perf_counter around ttnn.bringup.grouped_matmul, warm, no
        profiler; OV-78).
    woa_attn_ms_<case>_<off|on>: the attention section of the same profiled runs (test_skip_positions' phase
        "attention" + wo_a's "woa" signposts, i.e. the ladders' attention section with wo_a in it), the median.
    woa_load_delta_s: load-time cost of the bank copy over the interleaved one for the executed layers (each copy built
        from the checkpoint read: host prep + conversion + upload, synchronized), woa_load_delta_s_40 (x 40 / layers).

    PYTHONPATH=$PWD BRINGUP_CHUNK=8192 DS41_PACKED=on ... DS41_DECODER_SKIP=decode scripts/run_safe_pytest.sh \\
        --run-all --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_offon.py

FIX.43 (overseer): far histories in the same one load. WOA_OFFON_POS past CAP - 8192 - 2048 (e.g. 1024000) runs the
FAR layout: the slots' capacity and the load's tables are H + 8192 (BATCH_POS_VARS: the batch planner loads the model
for WOA_OFFON_POS + chunk, never past deploy.seq), so no chunk may reach past H + 8192:
    single8192  request 4's chunk [H, H + 8192) in slot 8 (set A) / 9 (set B)
    4x2k        requests 0-3's [H, H + 2048), 8 chips each, slots 0-3 / 4-7
(single5120 is not run). Every history is a real prefill [0, H) of the text (10 slots, set A again for the repeat).
Metric names carry the history: "_1m" at H >= 1,000,000 (woa_offon_max_abs_diff_single8192_1m, ...), "_p<H>" at any
other H but the default.

    PYTHONPATH=$PWD WOA_OFFON_POS=1024000 BRINGUP_CHUNK=8192 ... scripts/run_safe_pytest.sh --run-all \\
        --no-precompile models/demos/deepseek_v41_flash_d_p/tests/woa/test_woa_offon.py
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
    """The helper module as this file sees it now: the resident runner keeps helper modules of an earlier request
    (reload them), or has dropped them from sys.modules after a tests/ edit while the package attribute stayed (then
    reload() raises 'not in sys.modules': import it again; WA.4)."""
    import sys

    if sys.modules.get(mod.__name__) is mod:
        return importlib.reload(mod)
    return importlib.import_module(mod.__name__)


_pc = _fresh(_pc)
_mm = _fresh(_mm)
CHUNK = _pc.CHUNK

S = spec()
BATCH_SHARED = True
BATCH_ITEM_VARS = ("WOA_OFFON_LAYERS", "WOA_OFFON_POS", "WOA_OFFON_PROF_RUNS")
BATCH_POS_VARS = ("WOA_OFFON_POS",)  # F288 (FIX.43): a far history's load reaches WOA_OFFON_POS + chunk
BATCH_LAYERS = "executed"  # F285: layers 0-20 under DECODER_SKIP=decode (all 40: WOA_OFFON_LAYERS=all)
CAP = 155648  # the runtime's max_seq: H + 8192 + 2048 rounded up to a chunk (19 x 8192)
BATCH_MAX_SEQ = CAP
SLOTS = 10
SHIFT = 4096  # request r reads the text from r x SHIFT
SEG = 2048
SHORT = 5120
OUT = pathlib.Path("generated/ds41_woa")
CASES = ("single8192", "4x2k", "single5120")
FAR_CASES = ("single8192", "4x2k")  # FIX.43: the far layout's cases (module docstring)
DEFAULT_POS = 139264


def _pos() -> int:
    h = int(os.environ.get("WOA_OFFON_POS", str(DEFAULT_POS)))  # diagnostic
    assert h % CHUNK == 0 and h >= CHUNK, h
    return h


def _far(h: int) -> bool:
    """FIX.43: the far layout (no chunk past h + CHUNK) where the near one's request-0 2048 tail passes CAP."""
    return h + CHUNK + SEG > CAP


def _cap(h: int) -> int:
    """The slots' capacity and the load's max_seq: CAP (the near layout), else h + CHUNK (the far one)."""
    return h + CHUNK if _far(h) else CAP


def _tag(h: int) -> str:
    """The metric names' history suffix: none at the default, "_1m" at >= 1,000,000, else "_p<h>"."""
    return "" if h == DEFAULT_POS else ("_1m" if h >= 1_000_000 else f"_p{h}")


class WoaCalls:
    """ttnn.bringup.grouped_matmul wrapped for the calls inside: host us a call (perf_counter), optionally a profiler
    section "woa" around each call (the section before restored after it), optionally each call's input (chip 0)."""

    def __init__(self, sign: bool = False, capture: bool = False):
        self.sign, self.capture = sign, capture
        self.host_s, self.calls, self.inputs = 0.0, 0, []

    def __enter__(self):
        import ttnn
        from models.demos.common.bringup.testing import profiler

        self.f = f = ttnn.bringup.grouped_matmul
        me = self

        def gmm(x, w, *a, **k):
            if me.capture:
                me.inputs.append((ttnn.to_torch(ttnn.get_device_tensors(x)[0]), x.dtype, tuple(x.shape), w, dict(k)))
            prev = profiler._P.get("current_full") if me.sign else None
            if me.sign:
                profiler.signpost("woa")
            t0 = time.perf_counter()
            y = f(x, w, *a, **k)
            me.host_s += time.perf_counter() - t0
            me.calls += 1
            if me.sign:
                profiler.signpost(prev or "other")
            return y

        ttnn.bringup.grouped_matmul = gmm
        return self

    def __exit__(self, *exc):
        import ttnn

        ttnn.bringup.grouped_matmul = self.f
        return False


def _atts(model) -> list:
    return [b.attention for _, b in sorted(model.blocks.items()) if hasattr(b.attention, "woa_ensure")]


def _load_delta(mesh, model) -> dict:
    """Each executed layer's wo_a copies built from the checkpoint (host prep, then the interleaved and the bank upload,
    each synchronized); the bank copy kept where the load did not upload it. Returns the seconds and the kinds added."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import attention as A

    host = inter = bank = 0.0
    added = {"bank": [], "interleaved": []}
    atts = _atts(model)
    for att in atts:
        t0 = time.perf_counter()
        t = att._woa_host(att.woa_src())
        t1 = time.perf_counter()
        wi = A._upload(mesh, t, att.wo_a_dtype, ttnn.TILE_LAYOUT, None if att.sp32 else (None, 1))
        ttnn.synchronize_device(mesh)
        t2 = time.perf_counter()
        lay = A.woa_layout(mesh, att.wo_a_shape)
        wb = A.gmm_bank.upload(t, lay, mesh, att.wo_a_dtype)
        ttnn.synchronize_device(mesh)
        t3 = time.perf_counter()
        host, inter, bank = host + (t1 - t0), inter + (t2 - t1), bank + (t3 - t2)
        if att.wo_a_bank is None:
            att.wo_a_bank, att.woa_layout = wb, lay
            added["bank"].append(att)
        else:
            ttnn.deallocate(wb)
        if att.wo_a is None:
            att.wo_a = wi
            added["interleaved"].append(att)
        else:
            ttnn.deallocate(wi)
    return dict(layers=len(atts), host_s=host, interleaved_s=inter, bank_s=bank, added=added)


def _prof(mesh, fn) -> tuple[float, float, float]:
    """(the slowest chip's device ms of the run, wo_a's ms: the "woa" sections, each call's slowest chip, the attention
    section with wo_a in it: phase "attention" + wo_a's ms) of one real-time profiled run."""
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
    woa = sum(v for k, v in (pr.get("kernel_ns") or {}).items() if k == "woa" or str(k).endswith(".woa"))
    att = L._phases(pr).get("attention", 0.0)
    return L._device(pr), woa / 1e6, att + woa / 1e6


def _real160(mesh, inputs) -> tuple[float, int]:
    """Every captured wo_a call (chip 0's input, its first 160 rows, replicated): today's call vs the bank call at the
    160-row knobs, every chip bitwise. Returns (max abs diff, calls)."""
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import mm_configs as mm
    from models.demos.deepseek_v41_flash_d_p.tt.attention import woa_layout
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    worst, n = 0.0, 0
    by_w = {}
    for b in getattr(_real160, "atts", []):
        by_w[id(b.wo_a)] = b
    for xh, xdt, xshape, w, kw in inputs:
        att = by_w.get(id(w))
        if att is None or att.wo_a_bank is None:
            continue
        x = ttnn.from_torch(
            xh[:, :, :160, :].contiguous(),
            dtype=xdt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        bw, depth, rt = mm.gmm_knobs(tuple(x.shape), att.wo_a_shape)
        common = dict(dtype=kw.get("dtype"), compute_kernel_config=kw.get("compute_kernel_config"))
        common["max_grid"] = kw.get("max_grid")
        a = ttnn.bringup.grouped_matmul(x, att.wo_a, block_w=bw, depth=depth, rt=rt, **common)
        d, wi, xa = mm.woa_bank_knobs(x.shape)
        b = ttnn.bringup.grouped_matmul(
            x, att.wo_a_bank, w_layout=woa_layout(mesh, att.wo_a_shape), block_w=bw, depth=d, rt=rt, w_inflight=wi,
            x_ahead=xa, one_group=bool(settings.get("WOA_BANK_ONE_GROUP")), **common,
        )  # fmt: skip
        for p, q in zip(_mm._chips_host(a), _mm._chips_host(b)):
            if not _mm._bits_equal(p, q):
                worst = max(worst, float((p.double() - q.double()).abs().max()), 1e-30)
        n += 1
        for t in (x, a, b):
            ttnn.deallocate(t)
    return worst, n


@mesh_parametrize(timeout_s=3 * 3600)
def test_woa_offon(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192 (the load's chunk)"
    import ttnn
    from models.demos.common.bringup.reference import texts
    from models.demos.common.bringup.testing import batch
    from models.demos.deepseek_v41_flash_d_p.tests import test_skip_positions as SP
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    mesh = mesh_device
    every = os.environ.get("WOA_OFFON_LAYERS", "") == "all"  # diagnostic: all 40 layers (DECODER_SKIP=off)
    skip = "off" if every else "decode"
    monkeypatch.setenv(settings.env_name("DECODER_SKIP"), skip)  # (call-time)
    full = list(S.layers())
    with _pc.switch("DECODER_SKIP", skip):  # (DS41-314: a held model reads its frozen snapshot)
        layers = full if every else batch.run_layers(S, full, last_chunk=False)
    H = _pos()
    far, tag, cap = _far(H), _tag(H), _cap(H)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    assert cap <= -(-deploy // CHUNK) * CHUNK, f"WOA_OFFON_POS {H}: slots of {cap} pass the deployment ({deploy})"
    cases = FAR_CASES if far else CASES
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    monkeypatch.setenv("WALLS_FRESH_SLOTS", str(SLOTS))  # (the runtime's KV cache)
    runtime, adapter, hf, params = SP._runtime(mesh, layers, CHUNK, cap)
    model = runtime.model
    loaded = model.skip_mode
    if loaded != skip:
        model.use_decoder_skip(skip)
    res = {"layers": [layers[0], layers[-1]], "H": H, "far": far, "cap": cap, "max_seq": model.max_seq}
    assert model.max_seq >= cap, f"the held model's tables reach {model.max_seq} < {cap} (WOA_OFFON_POS {H})"
    # both wo_a copies on the held model (OV-77: a test load holds both; the load's own copy kept), timed (load delta)
    ld = _load_delta(mesh, model)
    n = ld["layers"]
    delta = ld["bank_s"] - ld["interleaved_s"]
    res["load"] = {k: v for k, v in ld.items() if k != "added"}
    metrics.record(f"woa_load_delta_s{tag}", round(delta, 3), layers=n, bank_s=round(ld["bank_s"], 3))
    metrics.record(f"woa_load_delta_s_40{tag}", round(delta * 40 / max(1, n), 3), estimate="x 40 / layers")
    metrics.record(f"woa_load_host_s{tag}", round(ld["host_s"], 3), layers=n)
    metrics.record(f"woa_load_interleaved_s{tag}", round(ld["interleaved_s"], 3), layers=n)
    metrics.record(f"woa_load_bank_s{tag}", round(ld["bank_s"], 3), layers=n)
    print(f"WOA: layers {layers[0]}-{layers[-1]} ({n} wo_a), load: {res['load']}", flush=True)
    _real160.atts = _atts(model)
    kv = adapter.allocate_kv_cache(mesh_device=mesh, hf_config=hf, params=params)
    text, info = texts.tokens(S, 4 * SHIFT + H + CHUNK + SEG, "agentic_cc")
    print(f"WOA: H {H} ({'far' if far else 'near'} layout, slots of {cap}, cases {cases}), text {info}", flush=True)

    def ids(r, s, e):
        return text[r * SHIFT + s : r * SHIFT + e]

    def single(slot, r, s, e):
        inp = runtime.make_chunk_input(ids(r, s, e).tolist(), s)
        runtime.prefill_chunk(inp, kv, slot_id=slot, actual_start=s, actual_end=e)
        return inp

    def run_single(side, base):
        with _pc.switch("WOA_BANK", side):
            if far:  # (FIX.43: request 4 in slot 8 / 9, nothing past H + CHUNK)
                inp = single(8 + base // 4, 4, H, H + CHUNK)
            else:
                inp = single(base, 0, H, H + CHUNK)
            model.sync()
        ttnn.deallocate(inp)

    def run_packed(side, base):
        s0 = H if far else H + CHUNK  # (request 0's segment: after its single chunk in the near layout)
        segs = [Segment(base + 0, s0, s0 + SEG)] + [Segment(base + r, H, H + SEG) for r in (1, 2, 3)]
        lane = [ids(0, s0, s0 + SEG)] + [ids(r, H, H + SEG) for r in (1, 2, 3)]
        desc = ChunkDescriptor.packed(CHUNK, segs)
        with _pc.switch("WOA_BANK", side), _pc.packed("on"):
            inp = runtime.make_packed_input(desc, [t.long() for t in lane])
            runtime.prefill_packed(inp, kv, desc)
            model.sync()
        ttnn.deallocate(inp)

    def run_short(side, base):
        with _pc.switch("WOA_BANK", side):
            inp = single(8 + base // 4, 4, H, H + SHORT)  # (set A: slot 8, set B: slot 9)
            model.sync()
        ttnn.deallocate(inp)

    fns = {
        k: v for k, v in {"single8192": run_single, "4x2k": run_packed, "single5120": run_short}.items() if k in cases
    }

    def history(slots_reqs):
        with _pc.switch("WOA_BANK", "off"):
            for slot, r in slots_reqs:
                for s in range(0, H, CHUNK):
                    ttnn.deallocate(single(slot, r, s, s + CHUNK))
            model.sync()

    set_a = [(r, r) for r in range(4)] + [(8, 4)]
    set_b = [(4 + r, r) for r in range(4)] + [(9, 4)]
    if far:
        pairs_ab = [(r, 4 + r, H + SEG) for r in range(4)] + [(8, 9, H + CHUNK)]
    else:
        pairs_ab = [(0, 4, H + CHUNK + SEG)] + [(r, 4 + r, H + SEG) for r in (1, 2, 3)] + [(8, 9, H + SHORT)]
    try:
        t0 = time.time()
        history(set_a + set_b)
        print(
            f"WOA: histories [0, {H}) in slots {[s for s, _ in set_a + set_b]} in {time.time() - t0:.0f} s", flush=True
        )
        o = {}
        cap160 = None
        for side, base in (("off", 0), ("on", 4)):
            for case, fn in fns.items():
                o[(case, side)] = _mm.Outs()
                o[(case, side)].wrap(model)
                try:
                    if case == "single8192" and side == "off":
                        with WoaCalls(capture=True) as cap160:
                            fn(side, base)
                    else:
                        fn(side, base)
                finally:
                    _mm.Outs.unwrap(model)
        srv_offon, bad_offon = _mm._served_diff(runtime, kv, model, pairs_ab)
        t0 = time.time()
        history(set_a)
        print(f"WOA: set A re-prefilled to {H} in {time.time() - t0:.0f} s", flush=True)
        for case, fn in fns.items():
            o[(case, "on2")] = _mm.Outs()
            o[(case, "on2")].wrap(model)
            try:
                fn("on", 0)
            finally:
                _mm.Outs.unwrap(model)
        srv_rep, bad_rep = _mm._served_diff(runtime, kv, model, pairs_ab)
        for case in cases:
            off, on, on2 = (o[(case, s)].cap for s in ("off", "on", "on2"))
            keys = sorted(set(on.dig) | set(off.dig) | set(on2.dig))
            assert len(keys) >= len(layers) - 1, (case, keys)
            out_offon = _pc.Capture.diff(off, on, keys)
            out_rep = _pc.Capture.diff(on, on2, keys)
            d_offon, d_rep = max(out_offon, srv_offon), max(out_rep, srv_rep)
            res[case] = {
                "offon_max_abs_diff": d_offon,
                "offon_out_max_abs_diff": out_offon,
                "offon_bad_layers": [k for k in keys if off.dig.get(k) != on.dig.get(k)],
                "repeat_max_abs_diff": d_rep,
                "repeat_out_max_abs_diff": out_rep,
                "repeat_bad_layers": [k for k in keys if on.dig.get(k) != on2.dig.get(k)],
                "layers_compared": len(keys),
            }
            metrics.record(f"woa_offon_max_abs_diff_{case}{tag}", d_offon, layers=len(keys), served_bad=len(bad_offon))
            metrics.record(f"woa_repeat_max_abs_diff_{case}{tag}", d_rep, layers=len(keys), served_bad=len(bad_rep))
            print(f"WOA {case}{tag}: off vs on {d_offon} (outputs {out_offon}, served {srv_offon}: {bad_offon[:4]}); "
                  f"on repeat {d_rep} (outputs {out_rep}, served {srv_rep}: {bad_rep[:4]}); {len(keys)} layers",
                  flush=True)  # fmt: skip
        res["served_bad_offon"] = [list(map(str, b)) for b in bad_offon[:20]]
        res["served_bad_repeat"] = [list(map(str, b)) for b in bad_rep[:20]]
        # WA.4: the timing below re-runs each chunk in place (a slot already past H). DECODER_SKIP=off refuses that
        # (WINDOW_KV_KEEP=snapshot128: the window rows [H - 128, H) are no longer kept, tt/model.py window cut), so
        # with WOA_OFFON_LAYERS=all the timing runs on the executed layers (DECODER_SKIP=decode, layers 0-20) as WA.3
        # and the ladders time it; the off == on / repeat comparison above stays over all 40 layers.
        timing_skip = "decode" if every else skip
        res["timing_decoder_skip"] = timing_skip
        if timing_skip != model.skip_mode:
            monkeypatch.setenv(settings.env_name("DECODER_SKIP"), timing_skip)
            model.use_decoder_skip(timing_skip)
        # host us a wo_a call (warm, no profiler), each case and side
        for case, fn in fns.items():
            for side, base in (("off", 0), ("on", 4)):
                fn(side, base)  # (warm)
                got = []
                for _ in range(2):
                    with WoaCalls() as wc:
                        fn(side, base)
                    got.append(wc.host_s / max(1, wc.calls) * 1e6)
                us = round(min(got), 1)
                res[case][f"host_us_{side}"] = us
                res[case]["woa_calls"] = wc.calls
                metrics.record(
                    f"woa_host_us_{side}_{case}{tag}",
                    us,
                    calls=wc.calls,
                    runs=[round(v, 1) for v in got],
                    skip=timing_skip,
                )
        # device ms (whole chunk; wo_a's own programs signposted), the median of runs a side, alternating
        runs = int(os.environ.get("WOA_OFFON_PROF_RUNS", "3"))  # diagnostic
        for case, fn in fns.items():
            got = {"off": [], "on": []}
            for _ in range(runs):
                for side, base in (("off", 0), ("on", 4)):

                    def go(side=side, base=base, fn=fn):
                        with WoaCalls(sign=True):
                            fn(side, base)

                    got[side].append(_prof(mesh, go))
            for side in ("off", "on"):
                rs = sorted(got[side])
                dev = rs[len(rs) // 2][0]
                woa_med = statistics.median(t[1] for t in got[side])
                att_med = statistics.median(t[2] for t in got[side])
                res[case][f"device_ms_{side}"] = round(dev, 3)
                res[case][f"woa_ms_{side}"] = round(woa_med, 3)
                res[case][f"device_ms_runs_{side}"] = [round(t[0], 3) for t in got[side]]
                res[case][f"woa_ms_runs_{side}"] = [round(t[1], 3) for t in got[side]]
                res[case][f"attn_ms_{side}"] = round(att_med, 3)
                res[case][f"attn_ms_runs_{side}"] = [round(t[2], 3) for t in got[side]]
                metrics.record(f"woa_device_ms_{case}{tag}_{side}", round(dev, 3), skip=timing_skip)
                metrics.record(f"woa_sec_ms_{case}{tag}_{side}", round(woa_med, 3), skip=timing_skip)
                metrics.record(f"woa_attn_ms_{case}{tag}_{side}", round(att_med, 3), skip=timing_skip)
            print(f"WOA {case}{tag}: device ms off {res[case]['device_ms_off']} on {res[case]['device_ms_on']} (runs "
                  f"{res[case]['device_ms_runs_off']} / {res[case]['device_ms_runs_on']}); attention ms off "
                  f"{res[case]['attn_ms_off']} on {res[case]['attn_ms_on']}; wo_a ms "
                  f"off {res[case]['woa_ms_off']} on {res[case]['woa_ms_on']}; host us a call off "
                  f"{res[case]['host_us_off']} on {res[case]['host_us_on']}", flush=True)  # fmt: skip
        # 160 rows on real inputs (op level)
        r160, n160 = _real160(mesh, cap160.inputs if cap160 is not None else [])
        res["real160"] = {"max_abs_diff": r160, "calls": n160}
        metrics.record(f"woa_offon_max_abs_diff_real160{tag}", r160, calls=n160)
        print(f"WOA real160: {n160} calls, off vs on {r160}", flush=True)
    finally:
        for key in [k for k in runtime._states if k[0] == id(kv)]:
            try:
                runtime._states.pop(key).free()
            except Exception as e:  # noqa: BLE001
                print(f"state free: {e}")
        for t in kv.contract.by_config.values():
            if t.is_allocated():
                ttnn.deallocate(t)
        for kind, atts in ld["added"].items():  # (the held model back to its load's copies)
            for att in atts:
                att.woa_drop(kind)
        if model.skip_mode != loaded:
            model.use_decoder_skip(loaded)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"offon{tag}.json").write_text(json.dumps(res, indent=1, default=str))
    bad = {c: (res[c]["offon_max_abs_diff"], res[c]["repeat_max_abs_diff"]) for c in cases}
    assert all(v == (0, 0) for v in bad.values()) and res["real160"]["max_abs_diff"] == 0, (
        f"WOA_BANK off != on or not repeatable: {bad}, real160 {res['real160']}"
    )  # fmt: skip
