# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.3A diagnostic (not gated): where the host time of a packed chunk goes, PACKED_ATTN_ONECALL as set. The walls'
4 x 2k @ 139264 chunk (test_packed_walls' construction: agentic text, each lane a session stretch, slots rotated a
lane every chunk, the real previous chunk first), warmed up, then: the wall (synced), the host enqueue time of
run_chunk alone (no sync inside), a cProfile of one chunk (the functions with the most own / cumulative time) and the
device time of the ops in the attention sections (real-time profiler, op rows). PB3A_PROBE_POS: the lanes' positions
(default 4 x 139264; e.g. 61440,139264,307200,546816 = the walls' mixed case, each slot sized to its position).
Runs on the walls' load (layers 0-20, DECODER_SKIP=decode)."""

import cProfile
import io
import os
import pstats
import time

import pytest

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = (
    "PB3A_PROBE_POS",
    "PB3A_PROBE_TOP",
    "PB3A_PROBE_OPS",
    "PB3A_PROBE_DUMP",
    "PB3A_PROBE_OPNAMES",
    "PB3A_PROBE_NSEG",
    "PB3A_PROBE_DELAY",
    "PB3A_PROBE_TOP_OPS",
    "PB3A_PROBE_COPY",
    "PB3A_PROBE_CHECK",
    "PB3A_PROBE_LAYERS_T",
    "PB3A_PROBE_LAYER_SITES",
)
CHUNK, LANE, SHIFT = 8192, 2048, 4096


@mesh_parametrize(timeout_s=3600)
def test_onecall_host(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import os

    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import packed
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert settings.get("DECODER_SKIP") == "decode"
    pos = [int(v) for v in os.environ.get("PB3A_PROBE_POS", "139264,139264,139264,139264").split(",")]
    top = max(pos)
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    monkeypatch.setitem(S.data["target"], "seq", max(top + CHUNK, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    text = PositionGolden(S, top + 3 * SHIFT, CHUNK).tokens()
    states = {k: model.new_state(pos[k] + CHUNK) for k in range(4)}
    toks = {k: text[k * SHIFT : k * SHIFT + pos[k] + LANE].long() for k in range(4)}
    it = [0]
    nseg = int(os.environ.get("PB3A_PROBE_NSEG", "4"))

    delay = [None]  # (layer, ms): a host sleep after that layer (PB.3K: is the host on the critical path there?)

    marks = []  # (layer, perf_counter) after each layer (PB.3K: host time a layer)

    def on_layer(i, h):
        marks.append((i, time.perf_counter()))
        if delay[0] is not None and i == delay[0][0]:
            time.sleep(delay[0][1] / 1e3)

    def run(prev: bool, sync: bool = True):
        lanes = [None] * 4
        for k in range(4):
            lanes[(k + it[0]) % 4] = k
        if nseg == 1:  # (PB.3K: one segment in lane 0, three empty lanes: today's single-request path)
            lanes = [0, None, None, None]
        s = {k: pos[k] - (LANE if prev else 0) for k in lanes if k is not None}
        desc = ChunkDescriptor.packed(
            CHUNK, [(Segment(10 + k, s[k], s[k] + LANE) if k is not None else None) for k in lanes]
        )
        inp = make_input(mesh_device, desc, [(toks[k][s[k] : s[k] + LANE] if k is not None else None) for k in lanes])
        model.sync()
        marks.clear()
        marks.append((-1, time.perf_counter()))
        t0 = time.time()
        nat = ids_from_input(mesh_device, inp, desc)
        run_chunk(model, desc, nat, [states[k] for k in lanes if k is not None], layers, last=False, on_layer=on_layer)
        t1 = time.time()
        if sync:
            model.sync()
        t2 = time.time()
        import ttnn

        ttnn.deallocate(inp)
        return (t1 - t0) * 1e3, (t2 - t0) * 1e3

    try:
        with packed("on"):
            for _ in range(4):
                run(True)
                run(False)
                it[0] += 1
            enq, walls = [], []
            for _ in range(3):
                run(True)
                e, w = run(False)
                enq.append(round(e, 1))
                walls.append(round(w, 1))
                it[0] += 1
            print(
                f"PB.3A host probe ONECALL={settings.get('PACKED_ATTN_ONECALL')}: walls {walls} ms, "
                f"run_chunk enqueue (host, no sync) {enq} ms",
                flush=True,
            )
            run(True)
            pr = cProfile.Profile()
            pr.enable()
            run(False)
            pr.disable()
            dump = os.environ.get("PB3A_PROBE_DUMP", "")
            if dump:
                pr.dump_stats(dump)
            for key in ("tottime", "cumulative"):
                buf = io.StringIO()
                pstats.Stats(pr, stream=buf).sort_stats(key).print_stats(int(os.environ.get("PB3A_PROBE_TOP", "45")))
                print(f"PB.3A host probe cProfile by {key}:\n" + buf.getvalue(), flush=True)
            if os.environ.get("PB3A_PROBE_CHECK", "0") == "1":  # PB.3K: PACKED_HOST_FAST on vs off, same chunk
                import hashlib

                import ttnn
                from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import served_digests, switch

                got = {}
                for v in ("on", "off", "on"):
                    with switch("PACKED_HOST_FAST", v):
                        run(True)
                        run(False)
                    rows = {}
                    for i, t in sorted(model._engram_full.items()):
                        b = b"".join(
                            ttnn.to_torch(d).contiguous().view(-1).numpy().tobytes() for d in ttnn.get_device_tensors(t)
                        )
                        rows[i] = hashlib.sha256(b).hexdigest()[:16]
                    dig = {k: served_digests(states[k], layers, pos[k] + LANE) for k in range(4)}
                    got.setdefault(v, []).append((rows, dig))
                    print(f"PB.3K check HOST_FAST={v}: engram rows {rows}", flush=True)
                a, b, c = got["on"][0], got["off"][0], got["on"][1]
                same_rows = a[0] == b[0] == c[0]
                bad = [
                    (k, key) for k in range(4) for key in a[1][k] if not (a[1][k][key] == b[1][k][key] == c[1][k][key])
                ]
                print(
                    f"PB.3K check: engram rows on == off == on: {same_rows}; slot digests differing: {len(bad)} "
                    f"of {sum(len(a[1][k]) for k in range(4))} {bad[:8]}",
                    flush=True,
                )
            if os.environ.get("PB3A_PROBE_COPY", "0") == "1":  # PB.3K: the engram rows' upload alone, idle device
                import ttnn

                rows = model.engram[min(model.engram)]
                for _ in range(2):
                    dev = rows.allocate_tp(CHUNK)
                    host = rows._staging_tp(CHUNK // int(mesh_device.get_num_devices()))[0]
                    model.sync()
                    t0 = time.perf_counter()
                    ttnn.copy_host_to_device_tensor(host, dev)
                    t1 = time.perf_counter()
                    model.sync()
                    t2 = time.perf_counter()
                    print(
                        f"PB.3K engram upload alone: {dev.shape} {dev.dtype}: copy {1e3 * (t1 - t0):.2f} ms, "
                        f"copy + sync {1e3 * (t2 - t0):.2f} ms",
                        flush=True,
                    )
                    ttnn.deallocate(dev)
                half = CHUNK // int(mesh_device.get_num_devices()) // 2
                for _ in range(3):
                    hosth = rows._staging_tp(half)[0]
                    devs = [
                        ttnn.allocate_tensor_on_device(
                            ttnn.Shape([1, 1, half, rows.width]),
                            ttnn.bfloat8_b,
                            ttnn.TILE_LAYOUT,
                            mesh_device,
                            ttnn.DRAM_MEMORY_CONFIG,
                        )
                        for _ in range(2)
                    ]
                    model.sync()
                    t0 = time.perf_counter()
                    for d in devs:
                        ttnn.copy_host_to_device_tensor(hosth, d)
                    t1 = time.perf_counter()
                    model.sync()
                    t2 = time.perf_counter()
                    print(
                        f"PB.3K engram upload as 2 halves: copy {1e3 * (t1 - t0):.2f} ms, copy + sync "
                        f"{1e3 * (t2 - t0):.2f} ms",
                        flush=True,
                    )
                    for d in devs:
                        ttnn.deallocate(d)
                rows._staging_tp(2 * half)
            if os.environ.get("PB3A_PROBE_LAYERS_T", "0") == "1":  # PB.3K: host ms a layer (median of 5 chunks)
                per = []
                for _ in range(5):
                    run(True)
                    run(False)
                    it[0] += 1
                    per.append([(i, 1e3 * (t - marks[j][1])) for j, (i, t) in enumerate(marks[1:])])
                med = [(per[0][j][0], sorted(p[j][1] for p in per)[2]) for j in range(len(per[0]))]
                print(
                    f"PB.3K host ms a layer (from the chunk start for the first): "
                    f"{[(i, round(v, 2)) for i, v in med]} sum {sum(v for _, v in med):.1f}",
                    flush=True,
                )
            dly = os.environ.get("PB3A_PROBE_DELAY", "")
            if dly:
                ms = float(dly)
                for at in [None] + [layers[j] for j in (0, 1, 2, 3, 5, 8, 11, 14, 17, len(layers) - 1)]:
                    delay[0] = None if at is None else (at, ms)
                    ws = []
                    for _ in range(3):
                        run(True)
                        ws.append(run(False)[1])
                        it[0] += 1
                    print(f"PB.3K delay {ms} ms after layer {at}: walls {[round(w, 1) for w in ws]}", flush=True)
                delay[0] = None
            if os.environ.get("PB3A_PROBE_OPNAMES", "0") == "1":
                _opnames(run, model, marks)
            if os.environ.get("PB3A_PROBE_OPS", "1") == "0":
                return
            # the attention sections' ops (device ns, the slowest chip), summed over the layers
            from collections import defaultdict

            from models.demos.common.bringup.testing import profiler

            run(True)
            profiler.enable(mesh_device, ops=True, rt=True)
            try:
                profiler.set_layer(None)
                run(False)
                profiler.signpost("end")
                profiler.finish_rt()
                res = profiler.result()
            finally:
                profiler.disable()
            agg = defaultdict(lambda: [0, 0.0])
            for sec, rows in res.get("op_ns", {}).items():
                name = sec.split(".", 1)[-1]
                if not name.startswith(("attn_", "indexer", "attention", "compressor")):
                    continue
                for row in rows:
                    nd = row.get("ns_dev")
                    ns = max(nd.values()) if isinstance(nd, dict) and nd else row.get("ns", 0)
                    a = agg[(name, row.get("op"), str(row.get("shape", ""))[:60])]
                    a[0] += row.get("calls", 1)
                    a[1] += ns / 1e6
            print(f"PB.3A probe {pos}: attention-section ops (calls, ms, slowest chip), top 40:", flush=True)
            for (name, op, shape), (calls, ms) in sorted(agg.items(), key=lambda x: -x[1][1])[:40]:
                print(f"  {name:12s} {ms:8.3f} ms {calls:4d} x {op} {shape}", flush=True)
    finally:
        for st in states.values():
            st.free()


def _opnames(run, model=None, marks=None) -> None:
    """(PB.3K) Host time of one warm chunk by ttnn op name (every FastOperation call) and by the caller of every
    prepared-program run (ttnn.bringup.run_prepared): count and summed ms, the slowest first."""
    import sys
    from collections import defaultdict

    import ttnn
    from ttnn import decorators

    acc = defaultdict(lambda: [0, 0.0])
    seq = []  # (t0, ms, name) every outermost call
    gcs = []  # (t0, ms, generation)
    import gc

    gc_t = [0.0]

    def gc_cb(phase, info):
        if phase == "start":
            gc_t[0] = time.perf_counter()
        else:
            gcs.append((gc_t[0], (time.perf_counter() - gc_t[0]) * 1e3, info.get("generation")))

    orig_call = decorators.FastOperation.__call__
    orig_rp = ttnn.bringup.run_prepared
    depth = [0]

    def call(self, *a, **k):
        if depth[0]:
            return orig_call(self, *a, **k)
        depth[0] += 1
        t = time.perf_counter()
        try:
            return orig_call(self, *a, **k)
        finally:
            depth[0] -= 1
            fr = sys._getframe(1)
            while fr is not None and "/tt/" not in fr.f_code.co_filename and "deepseek" not in fr.f_code.co_filename:
                fr = fr.f_back
            site = f"{fr.f_code.co_filename.rsplit('/', 1)[-1]}:{fr.f_code.co_name}" if fr is not None else "?"
            e = acc["op " + self.python_fully_qualified_name.replace("ttnn.", "") + " @" + site]
            e[0] += 1
            ms = (time.perf_counter() - t) * 1e3
            e[1] += ms
            seq.append((t, ms, self.python_fully_qualified_name.replace("ttnn.", "") + " @" + site))

    def rp(*a, **k):
        f = sys._getframe(1)
        t = time.perf_counter()
        try:
            return orig_rp(*a, **k)
        finally:
            name = f"run_prepared {f.f_code.co_filename.rsplit('/', 1)[-1]}:{f.f_code.co_name}"
            e = acc[name]
            e[0] += 1
            ms = (time.perf_counter() - t) * 1e3
            e[1] += ms
            if not depth[0]:
                seq.append((t, ms, name))

    jobs = []  # (start, ms, layer?) of the engram worker's jobs
    pool = getattr(model, "_engram_pool", None)
    orig_submit = pool.submit if pool is not None else None

    def submit(fn, *a, **k):
        def wrapped(*aa, **kk):
            t = time.perf_counter()
            try:
                return fn(*aa, **kk)
            finally:
                jobs.append((t, (time.perf_counter() - t) * 1e3, getattr(fn, "__name__", "?")))

        return orig_submit(wrapped, *a, **k)

    run(True)
    if pool is not None:
        pool.submit = submit
    decorators.FastOperation.__call__ = call
    ttnn.bringup.run_prepared = rp
    gc.callbacks.append(gc_cb)
    try:
        e, w = run(False)
    finally:
        decorators.FastOperation.__call__ = orig_call
        ttnn.bringup.run_prepared = orig_rp
        gc.callbacks.remove(gc_cb)
        if pool is not None:
            pool.submit = orig_submit
    seq.sort()
    if seq:
        t0 = seq[0][0]
        big = sorted(range(len(seq)), key=lambda i: -seq[i][1])[:40]
        print(
            f"PB.3K slowest calls (of {len(seq)}; calls > 0.2 ms: "
            f"{sum(1 for x in seq if x[1] > 0.2)}, {sum(x[1] for x in seq if x[1] > 0.2):.1f} ms):",
            flush=True,
        )
        for i in sorted(big):
            prev = seq[i - 1][2] if i else "-"
            print(
                f"  #{i:4d} t={1e3 * (seq[i][0] - t0):7.2f} ms {seq[i][1]:6.2f} ms {seq[i][2]}  (after {prev})",
                flush=True,
            )
        gaps = sum(max(0.0, seq[i + 1][0] - (seq[i][0] + seq[i][1] / 1e3)) for i in range(len(seq) - 1)) * 1e3
        if marks:  # PB.3K: per layer, the host ms by call site (PB3A_PROBE_LAYER_SITES: the layers)
            want = [int(v) for v in os.environ.get("PB3A_PROBE_LAYER_SITES", "").split(",") if v.strip()]
            bounds = [(i, a, b) for (i, b), (_, a) in zip(marks[1:], marks[:-1])]
            for i, a, b in bounds:
                if i not in want:
                    continue
                calls = [x for x in seq if a <= x[0] < b]
                by = defaultdict(lambda: [0, 0.0])
                for _, ms, nm in calls:
                    by[nm][0] += 1
                    by[nm][1] += ms
                tot = sum(x[1] for x in calls)
                print(
                    f"PB.3K layer {i}: {1e3 * (b - a):.2f} ms host, {len(calls)} calls {tot:.2f} ms, python "
                    f"{1e3 * (b - a) - tot:.2f} ms",
                    flush=True,
                )
                for nm, (n, ms) in sorted(by.items(), key=lambda x: -x[1][1])[:40]:
                    print(f"    {ms:6.2f} ms {n:3d} x {nm}", flush=True)
        print(
            f"PB.3K engram jobs (worker): {[(round(1e3 * (a - t0), 1), round(b, 2), n) for a, b, n in jobs]}; "
            f"times {getattr(model, 'engram_times', None)}",
            flush=True,
        )
        print(
            f"PB.3K python between calls {gaps:.1f} ms; gc {[(round(1e3 * (a - t0), 1), round(b, 2), g) for a, b, g in gcs]}",
            flush=True,
        )
    tot_n = sum(v[0] for v in acc.values())
    tot = sum(v[1] for v in acc.values())
    print(f"PB.3K op names: enqueue {e:.1f} ms, wall {w:.1f} ms; {tot_n} calls {tot:.1f} ms", flush=True)
    for name, (n, ms) in sorted(acc.items(), key=lambda x: -x[1][1])[: int(os.environ.get("PB3A_PROBE_TOP_OPS", "70"))]:
        print(f"  {ms:8.2f} ms {n:5d} x {1e3 * ms / max(n, 1):6.1f} us  {name}", flush=True)
