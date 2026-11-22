# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PB.6L diagnostic (not gated): why the history gathers of 4 x 2k near 1M (test_packed_walls' 4x2k_p1m) are not
hidden, and what in the attention section scales with history. On the walls' load (layers 0-20, DECODER_SKIP=decode,
slots at the one capacity): the case's real previous chunk, then for each side of PB6L_PROBE_SIDES (default
"off:off,on:off,on:on" = PACKED_GATHER_HIDE:PACKED_LONG_HIDE) one real-time profiled chunk with the op rows: the
slowest chip's sections, every section's ops summed over layers (the slowest chip), per layer the attention section;
with the hide on the regions' device clock stamps (gather_hide.Stamps): per region the compute span, the stalls, the
last gather's end past the compute (least / worst chip), and the gathers it held. Then the slots' served-cache digests
of every side must be equal. PB6L_PROBE_POS: comma positions (default the walls' 4x2k_p1m)."""

import os
from collections import defaultdict

import pytest
import torch

from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec
from models.demos.common.bringup.testing.profile import PositionGolden

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
BATCH_ITEM_VARS = (
    "PB6L_PROBE_SIDES",
    "PB6L_PROBE_POS",
    "PACKED_WALL_CAP",
    "PB6L_PROBE_OPS",
    "PB6L_PROBE_BENCH",
    "PB6L_PROBE_AB",
)
CHUNK, LANE, SHIFT = 8192, 2048, 4096


@mesh_parametrize(timeout_s=3600)
def test_long_hide_probe(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import packed, served_digests, switch
    from models.demos.deepseek_v41_flash_d_p.tests.packed.test_packed_walls import _cap, _ghz
    from models.demos.deepseek_v41_flash_d_p.tt import gather_hide, history_gather
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    assert int(S.data["target"]["chunk"]) == CHUNK, "run with BRINGUP_CHUNK=8192"
    layers = batch.run_layers(S, list(S.layers()), last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    cap = _cap(deploy)
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    if model.skip_mode != "decode":
        model.use_decoder_skip("decode")
    assert settings.get("DECODER_SKIP") == "decode"
    v = os.environ.get("PB6L_PROBE_POS", "").strip()
    pos = [int(x) for x in v.split(",")] if v else [1040384 - k * SHIFT for k in range(4)]
    sh = 2 * SHIFT if not v else SHIFT
    text = PositionGolden(S, max(k * sh + p + LANE for k, p in enumerate(pos)), CHUNK).tokens()
    states = {k: model.new_state(cap) for k in range(4)}
    toks = {k: text[k * sh : k * sh + pos[k] + LANE].long() for k in range(4)}
    it = [0]
    sides = [tuple(s.split(":")) for s in os.environ.get("PB6L_PROBE_SIDES", "off:off,on:off,on:on").split(",") if s]
    has_long = "PACKED_LONG_HIDE" in settings.table
    ops = os.environ.get("PB6L_PROBE_OPS", "1") == "1"

    last_out = {}

    def run(prev: bool, keep: bool = False):
        lanes = [None] * 4
        for k in range(4):
            lanes[(k + it[0]) % 4] = k
        s = {k: pos[k] - (LANE if prev else 0) for k in lanes}
        desc = ChunkDescriptor.packed(CHUNK, [Segment(10 + k, s[k], s[k] + LANE) for k in lanes])
        inp = make_input(mesh_device, desc, [toks[k][s[k] : s[k] + LANE] for k in lanes])
        model.sync()
        nat = ids_from_input(mesh_device, inp, desc)

        def on_layer(i, h):  # (the last layer output that is hidden states, every chip's rows: its digest)
            if keep and i in layers[-2:] and hasattr(h, "tensors"):
                import hashlib

                d = hashlib.sha256()
                for t in h.tensors():
                    for dt in ttnn.get_device_tensors(t):
                        d.update(ttnn.to_torch(dt).contiguous().view(-1).view(torch.uint8).numpy().tobytes())
                last_out["digest"] = d.hexdigest()

        run_chunk(model, desc, nat, [states[k] for k in lanes], layers, last=False, on_layer=on_layer if keep else None)
        model.sync()
        ttnn.deallocate(inp)
        return lanes

    if os.environ.get("PB6L_PROBE_AB", "").strip():
        _ab(model, mesh_device, layers, text, cap, os.environ["PB6L_PROBE_AB"].strip(), has_long)
        return

    def ctx(h, lh):
        a = switch("PACKED_GATHER_HIDE", h)
        b = switch("PACKED_LONG_HIDE", lh) if has_long else _null()
        return a, b

    # (which regions ran, in order: wrap Hide.attention / tails to tag each region with its layer and items)
    tags = []
    orig_att, orig_tails = gather_hide.Hide.attention, gather_hide.Hide.tails

    def att(self, i, steps, key=None):
        n0, ni = gather_hide.STATS["regions"], len(self.items)
        out = orig_att(self, i, steps, key=key)
        if gather_hide.STATS["regions"] > n0:
            tags.append((("dec" if key is not None else f"attn{i}"), ni - len(self.items)))
        return out

    def tails(self, block, steps):
        n0 = gather_hide.STATS["regions"]
        out = orig_tails(self, block, steps)
        if gather_hide.STATS["regions"] > n0:
            tags.append((f"tails{block.layer}", 0))
        return out

    orig_topk = getattr(gather_hide.Hide, "topk", None)

    def topk(self, i, steps):
        n0 = gather_hide.STATS["regions"]
        out = orig_topk(self, i, steps)
        if gather_hide.STATS["regions"] > n0:
            tags.append((f"topk{i}", 0))
        return out

    if orig_topk is not None:
        monkeypatch.setattr(gather_hide.Hide, "topk", topk)
    monkeypatch.setattr(gather_hide.Hide, "attention", att)
    monkeypatch.setattr(gather_hide.Hide, "tails", tails)
    try:
        with packed("on"):
            for _ in range(2):
                for h, lh in sides:
                    a, b = ctx(h, lh)
                    with a, b:
                        run(True)
                        run(False)
                it[0] += 1
            gather_hide.warm_stamps(mesh_device)
            st = gather_hide.stamps(mesh_device)
            ghz = _ghz(mesh_device, st) if st is not None else 1.0
            digs = {}
            for h, lh in sides:
                a, b = ctx(h, lh)
                with a, b:
                    run(True)
                    tags.clear()
                    gather_hide.LOG.clear()
                    history_gather.TRACE = []
                    profiler.enable(mesh_device, ops=ops, rt=True)
                    try:
                        profiler.set_layer(None)
                        lanes = run(False)
                        profiler.signpost("end")
                        profiler.finish_rt()
                        res = profiler.result()
                    finally:
                        profiler.disable()
                        trace = list(history_gather.TRACE)
                        history_gather.TRACE = None
                    log = list(gather_hide.LOG)
                    clocks = st.read() if (st is not None and log) else []
                print(f"PB.6L probe HIDE={h} LONG={lh}: lanes {lanes}, positions {pos}", flush=True)
                print(f"  gather_hide.STATS {dict(gather_hide.STATS)}", flush=True)
                if lh == "on":
                    print(f"  plan {getattr(gather_hide, 'LAST_PLAN', {})}", flush=True)
                _report(res, ops)
                _gathers(res, trace)
                if log:
                    _regions(log, clocks, ghz, list(tags))
                digs[(h, lh)] = {k: served_digests(states[k], layers, pos[k] + LANE) for k in range(4)}
                run(True)  # (the same chunk once more, the last layer's output kept: its digest a side)
                run(False, keep=True)
                digs[(h, lh)][4] = {("out", layers[-1]): last_out.pop("digest", None)}
            if os.environ.get("PB6L_PROBE_BENCH", "0") == "1":
                run(True)
                run(False)
                _bench(mesh_device, states[0], pos[0])
            base = digs[sides[0]] if sides else {}
            for s in sides[1:]:
                bad = [(k, key) for k in range(5) for key in base[k] if base[k][key] != digs[s][k].get(key)]
                n = sum(len(base[k]) for k in range(5))
                print(f"PB.6L probe digests {s} vs {sides[0]}: {len(bad)} of {n} differ {bad[:8]}", flush=True)
                assert not bad, bad[:8]
    finally:
        for s_ in states.values():
            s_.free()


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _report(res: dict, ops: bool) -> None:
    per_chip = defaultdict(float)
    for d in res["kernel_ns_dev"].values():
        for c, ns in d.items():
            per_chip[c] += ns
    slow = max(per_chip, key=per_chip.get)
    secs = defaultdict(float)
    for sec, d in res["kernel_ns_dev"].items():
        secs[sec.split(".", 1)[-1]] += d.get(slow, 0)
    print(
        f"  device (slowest chip {slow}) {per_chip[slow] / 1e6:.2f} ms; sections on it: "
        + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in sorted(secs.items(), key=lambda x: -x[1])[:16]),
        flush=True,
    )
    per_layer = defaultdict(float)
    for sec, d in res["kernel_ns_dev"].items():
        if sec.split(".", 1)[-1] == "attention":
            per_layer[sec.split(".", 1)[0]] += d.get(slow, 0)
    print("  attention a layer: " + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in per_layer.items()), flush=True)
    if not ops:
        return
    agg = defaultdict(lambda: [0, 0.0])
    for sec, rows in res.get("op_ns", {}).items():
        name = sec.split(".", 1)[-1]
        if not name.startswith(("attention", "attn", "history", "indexer")):
            continue
        for row in rows:
            nd = row.get("ns_dev") or {}
            a = agg[(name.split(".")[0], row.get("op"))]
            a[0] += row.get("calls", 1)
            a[1] += nd.get(slow, nd.get(str(slow), 0))
    print("  ops (section, op, calls, ms on the slowest chip):", flush=True)
    for (sec, op), (calls, ns) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
        print(f"    {sec:16s} {str(op):48s} x{calls:4d} {ns / 1e6:8.3f}", flush=True)


def _gathers(res: dict, trace: list) -> None:
    recs, chip_of = res.get("rt_records") or {}, res.get("dev_to_chip") or {}
    by = defaultdict(lambda: [0, 0.0, 0.0])
    for k, src, lane, a, b in trace:
        for i in range(a, b):
            if not recs.get(i):
                continue
            t = [(e[2] - e[1]) / 1e6 for e in recs[i].values()]
            x = by[(k, src, lane)]
            x[0] += 1
            x[1] += min(t)
            x[2] += max(t)
    print(
        "  traced history ops (kind, source, lane: n, least chip ms, worst chip ms; rt records, serial only exact): "
        + "; ".join(f"{k}/{s}/{l}: {v[0]} {v[1]:.3f} {v[2]:.3f}" for (k, s, l), v in sorted(by.items(), key=str)),
        flush=True,
    )


def _regions(log: list, clocks: list, ghz: float, tags: list) -> None:
    ms = lambda t, a, b: (int(t[b]) - int(t[a])) / ghz / 1e6  # noqa: E731
    print(
        f"  regions {len(log)} (tags {len(tags)}): name items | compute ms | stalls | gather past compute (least/worst chip)"
    )
    tot = [0.0, 0.0, 0.0]
    for j, reg in enumerate(log):
        start = next(v for k, _, v in reg if k == "start")
        endc = next(v for k, _, v in reg if k == "end_compute")
        endg = next(v for k, _, v in reg if k == "end_gather")
        stalls, before = [0.0] * len(clocks), None
        for k, _, v in reg:
            if k == "before":
                before = v
            elif k == "after":
                for c, t in enumerate(clocks):
                    stalls[c] += max(0.0, ms(t, before, v))
        comp = [ms(t, start, endc) for t in clocks]
        past = [max(0.0, ms(t, endc, endg)) for t in clocks]
        exp = [s + p for s, p in zip(stalls, past)]
        c = min(range(len(clocks)), key=lambda c: exp[c])
        w = max(range(len(clocks)), key=lambda c: exp[c])
        ng = sum(1 for k, _, _ in reg if k == "ids")
        tag = tags[j] if j < len(tags) else ("?", 0)
        print(
            f"    {tag[0]:8s} g{ng} | {comp[c]:6.3f} ({min(comp):.3f}-{max(comp):.3f}) | {stalls[c]:6.3f}/{stalls[w]:6.3f}"
            f" | {past[c]:6.3f}/{past[w]:6.3f}",
            flush=True,
        )
        tot[0] += comp[c]
        tot[1] += stalls[c]
        tot[2] += past[c]
    print(
        f"  regions total: compute {tot[0]:.3f} stalls {tot[1]:.3f} past {tot[2]:.3f} ms (least chip each)", flush=True
    )


def _bench(mesh, st, start: int) -> None:
    """Host-timed (synchronized, median of 5): the history gather of one 1M item (the real slot's sizes) as a prefix
    of L = h, h / 2, h / 4 rows an owner, serial (default cores) and on the gather column (alone, and beside a
    ~3 ms local compute), num_links None / 2 / 1; and the index-key decode of the slot's history on several grids."""
    import statistics
    import time

    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt import ccl, history_gather, index_keys
    from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split

    split = Split.of(mesh)
    src = sorted(st.compress)[0]
    ckv = st.compress[src].ckv
    R = int(ckv.slab.rows)
    owners, axis = history_gather.owners_and_axis(mesh, ckv.tp_axis is not None)
    h = (int(start) // CHUNK) * (ckv.stripe // ckv.ratio)
    print(f"[bench] source {src}: R {R} local rows, h {h} an owner, owners {owners}, axis {axis}", flush=True)
    x = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, R, 320]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    x.update_tensor_topology(ckv.slab.tensor.tensor_topology())
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, owners * R, 320]), ttnn.uint8, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    s0, s1 = ccl._sems(mesh, axis)
    xa = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 8192, 4096]), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    from models.demos.deepseek_v41_flash_d_p.tt import cores

    def compute(n=6):
        y = xa
        for _ in range(n):
            z = ttnn.multiply(y, 1.0009765625, memory_config=ttnn.DRAM_MEMORY_CONFIG, sub_core_grids=cores.sub_cores(y))
            if y is not xa:
                ttnn.deallocate(y)
            y = z
        ttnn.deallocate(y)

    def g(L, links, sub=None):
        ttnn.experimental.high_bw_all_gather(
            x,
            2,
            out,
            cluster_axis=axis,
            num_links=links,
            gathered_dim_size=owners * L,
            ready_semaphore=s0,
            data_valid_semaphore=s1,
            **(sub or {}),
        )

    def timed(fn):
        ts = []
        for k in range(2 + 5):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            fn()
            ttnn.synchronize_device(mesh)
            if k >= 2:
                ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    def region(gfn, steps):
        mesh.load_sub_device_manager(split.mgr)
        try:
            with cores.nonmm_on_mm_grid():
                gfn(split.sub)
                for st_ in steps:
                    st_()
        finally:
            mesh.clear_loaded_sub_device_manager()

    t0 = timed(lambda: None)
    tc = timed(lambda: region(lambda sub: None, [compute]))
    print(f"[bench] empty sync {t0:.3f} ms; compute alone in a region {tc:.3f}", flush=True)
    for links in (ccl.AG_LINKS, 2, 1):
        for L in (h, h // 2, h // 4):
            if L <= 0:
                continue
            try:
                ts = timed(lambda: g(L, links))
                tr = timed(lambda: region(lambda sub: g(L, links, sub), []))
                trc = timed(lambda: region(lambda sub: g(L, links, sub), [compute]))
            except Exception as e:  # noqa: BLE001
                print(f"[bench] links {links} L {L}: {e}", flush=True)
                continue
            mib = owners * L * 320 / 2**20
            print(
                f"[bench] links {links} L {L} ({mib:.1f} MiB a chip): serial {ts:.3f} ms, gather column alone {tr:.3f}, "
                f"beside compute {trc:.3f} (compute {tc:.3f})",
                flush=True,
            )
    for t in (x, out, xa):
        ttnn.deallocate(t)
    ik = st.compress[src].index_k
    buf = index_keys._BUFFERS[index_keys._key(ik)]
    lo, _ = index_keys._span(ik, start)
    s = ik.served
    print(f"[bench] index keys: crows {ik.crows}, history rows {lo} a chip", flush=True)
    for grid in ((4, 2), (8, 4), (11, 5), (11, 10)):
        cg = ttnn.CoreCoord(*grid)
        t = timed(
            lambda: ttnn.bringup.unpack_fp8_rows(s.tensor, buf.tensor, row_lo=0, row_hi=lo, batch=s.batch, max_grid=cg)
        )
        print(f"[bench] decode {lo} rows on {grid}: {t:.3f} ms (incl. sync)", flush=True)


AB_CASES = {
    "139k": [139264] * 4,
    "mixed": [61440, 139264, 307200, 546816],
    "1m": [1040384 - k * SHIFT for k in range(4)],
}


def _ab(model, mesh, layers, text0, cap, which: str, has_long: bool) -> None:
    """PB6L_PROBE_AB=139k,mixed,1m: the walls' timing (the real previous chunk first, untimed, synced; slots rotate a
    lane every iteration; wall from the engine's device input through the last layer's sync) with PACKED_LONG_HIDE
    off and on ALTERNATED in one process (PACKED_GATHER_HIDE on), 2 warm-ups a side, then 10 timed a side: the medians
    and the per-iteration pairs (the same process, the same lanes order)."""
    import statistics
    import time

    import ttnn
    from models.demos.common.bringup.testing.profile import PositionGolden
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import packed, switch
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, Segment
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import ids_from_input, make_input, run_chunk

    for case in [c for c in which.split(",") if c]:
        pos = AB_CASES[case]
        sh = 2 * SHIFT if case == "1m" else SHIFT
        text = PositionGolden(S, max(k * sh + p + LANE for k, p in enumerate(pos)), CHUNK).tokens()
        states = {k: model.new_state(cap) for k in range(4)}
        toks = {k: text[k * sh : k * sh + pos[k] + LANE].long() for k in range(4)}
        it = [0]

        def run(prev: bool) -> float:
            lanes = [None] * 4
            for k in range(4):
                lanes[(k + it[0]) % 4] = k
            st = {k: pos[k] - (LANE if prev else 0) for k in lanes}
            desc = ChunkDescriptor.packed(CHUNK, [Segment(10 + k, st[k], st[k] + LANE) for k in lanes])
            inp = make_input(mesh, desc, [toks[k][st[k] : st[k] + LANE] for k in lanes])
            model.sync()
            t0 = time.time()
            nat = ids_from_input(mesh, inp, desc)
            run_chunk(model, desc, nat, [states[k] for k in lanes], layers, last=False)
            model.sync()
            w = (time.time() - t0) * 1e3
            ttnn.deallocate(inp)
            return w

        walls = {"off": [], "on": []}
        try:
            with packed("on"), switch("PACKED_GATHER_HIDE", "on"):
                for k in range(12):
                    for v in ("off", "on"):
                        with switch("PACKED_LONG_HIDE", v) if has_long else _null():
                            run(True)
                            w = run(False)
                        if k >= 2:
                            walls[v].append(round(w, 1))
                    it[0] += 1
        finally:
            for s_ in states.values():
                s_.free()
        med = {v: statistics.median(ws) for v, ws in walls.items()}
        print(
            f"PB.6L A/B {case} {pos}: LONG off {med['off']:.1f} ms, on {med['on']:.1f} ms ({med['on'] - med['off']:+.1f}); "
            f"off {walls['off']}; on {walls['on']}",
            flush=True,
        )
