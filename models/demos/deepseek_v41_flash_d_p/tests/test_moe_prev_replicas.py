# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.56 (owner 2026-11-13, EXPERIMENTAL): MoE replicas learned from the previous chunk (MOE_REPLICAS=prev), off vs prev
on real back-to-back chunks (the deliverable; the orchestrator's random-token A/B profile cannot show it, DS41-228).

Terms (tt/expert_prev.py defines the rest): plan = the device planner's replica table for a layer's next call, made
right after its call from that call's router counts; fill = the plan's new copies carried by the following MoE calls'
combines; ready = the layer's next call came after every copy was carried and fenced (else it ran off, counted);
kept = a receiver that already held the expert (no copy); dropped = a receiver the planner wanted whose copy did not
fit in the fill budget of its window.

Per workload (novel = the ladder's prompt, code = P.45's code prompt), one process, the shipped settings with
MOE_REPLICAS=prev MOE_FILL=piggyback ENGRAM_TABLE=resident (13 slots a chip) and MOE_COUNTS=on (the planner's inputs,
read back for the host mirror; both sides pay its one copy a layer): fresh requests of chunks 0..10 (56,320 tokens),
back to back. Sides: off = prev switched off in the same model (no planner, the stock combine, the empty map: the
off path's routing, P.52) and prev. Passes per side: one warm pass (compiles every chunk position), one profiled
pass (program real-time profiler, no device profiler, no op mode: device ms a chunk = each program's slowest chip,
summed; the experts phase = routed_experts_combine; the planner's programs), one unprofiled pass (the wall a chunk:
first host call to the sync; the chunk's last layer output compared with the off profiled pass, max |diff|). prev's
profiled and unprofiled passes are two runs of the same chunk sequence: their outputs must be equal bit for bit, and
equal to off's (a replica computes the same expert; a replica used before its copy landed would show).
Host mirror (prev's unprofiled pass): every planner run of the pass replayed on the host from the read-back counts
(expert_prev.plan_host); after every chunk each layer's state and slot map, the global state and (at the end) the
pending job pages must equal the device's, every chip the same.

Metrics (suffix _<wl>): prev_device_ms_{off,prev} / prev_wall_ms_{off,prev} / prev_experts_ms_{off,prev} (mean of
chunks 1..10), prev_planner_ms (prev, mean), prev_maxdiff (prev vs off), prev_repeat_maxdiff (prev vs prev),
prev_off_repeat_maxdiff, prev_mirror_mismatch, prev_ready_frac (chunks 1..10), prev_spare_tokens; the side record
results/P.56_prev_replicas.json (per chunk and layer).

P.60 (overseer 2026-11-14): prev runs twice per workload, once per host path (MOE_PREV_HOST, switched in the loaded
model: tt/expert_prev.PrevReplicas.mode): legacy = P.56's (a ring of W job tensors, the host hands each call its
slot and a fresh reader token) and device (one job tensor of W page sets, the ports read G[0] mod W; no reader wait;
the combine's fill args the same every call). The setting's mode runs first (its metrics keep P.56's names), the other
second (metrics prev_*_<mode>_<wl>). Both must give off's outputs bit for bit and the same plans (every layer's
receivers / kept / new / dropped / loads, chunk by chunk). Unprofiled passes now also give host ms a chunk (first host
call to the last layer's enqueue, before any sync) and the clean wall ms a chunk (first host call to the sync after the
last layer; the diff and the final norm after it, untimed), and the combine's host us a call (the op launch,
inclusive, median) beside the planner's (`after`) and the guard's (`before`).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_replicas.py
P.62 (owner 2026-11-14): a third host path, lean (MOE_PREV_HOST=device MOE_PREV_HOST2=lean: the planner as one mesh
program, the combine's fill args set once, the per-call Python trimmed), switched the same way (expert_prev.set_host2);
the modes are legacy (P.56), device (P.60) and lean; the settings' one first (P.56's metric names), the others with
the suffix _<mode>. All three must give off's outputs bit for bit and the same plans.
P56_QUICK=1 (iterating only): 4 layers, chunks 0..3, novel, MOE_FILL_PIECES=2000 (a whole copy a call).
"""

from __future__ import annotations

import os
import statistics
import time

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch, profiler
from models.demos.common.bringup.testing.harness import begin_chunk, mesh_parametrize, spec
from models.demos.common.bringup.testing.positions import _free

S = spec()
QUICK = os.environ.get("P56_QUICK") == "1"
SEQ = 56320
CHUNKS = 4 if QUICK else 11
WORKLOADS = ("novel",) if QUICK else ("novel", "code")
OPS = ("fused_head", "sparse_dispatch_2d", "routed_experts_combine")


class _Spy:
    """The MoE block's ops and the prev planner, wrapped: each call's device-operation id range, under the layer."""

    def __init__(self, monkeypatch):
        from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP

        mb = ttnn.bringup.moe_block_main
        self.calls, self.layer = [], None
        for name in OPS:
            orig = getattr(mb, name)

            def wrap(*a, _o=orig, _n=name, **k):
                lo = ttnn._ttnn.get_device_operation_id()
                t0 = time.perf_counter()
                out = _o(*a, **k)
                if _n == "routed_experts_combine":
                    self.host["combine"].append(time.perf_counter() - t0)
                self.calls.append((self.layer, _n, lo, ttnn._ttnn.get_device_operation_id()))
                return out

            monkeypatch.setattr(mb, name, wrap)
        orig_after, orig_before = XP.PrevReplicas.after, XP.PrevReplicas.before
        self.reset_host()

        def after(pr, layer, counts, _o=orig_after):
            lo = ttnn._ttnn.get_device_operation_id()
            t0 = time.perf_counter()
            _o(pr, layer, counts)
            self.host["after"].append(time.perf_counter() - t0)
            self.calls.append((self.layer, "planner", lo, ttnn._ttnn.get_device_operation_id()))

        def before(pr, layer, _o=orig_before):
            t0 = time.perf_counter()
            _o(pr, layer)
            self.host["before"].append(time.perf_counter() - t0)

        monkeypatch.setattr(XP.PrevReplicas, "after", after)
        monkeypatch.setattr(XP.PrevReplicas, "before", before)

    def reset_host(self):
        # (host s a call: the planner's dispatch, the guard + fill args, the combine's launch)
        self.host = {"after": [], "before": [], "combine": []}


def _times(calls, recs: dict, d2c: dict) -> dict:
    out: dict = {}
    for layer, name, lo, hi in calls:
        per: dict = {}
        for i in range(lo, hi):
            for d, e in recs.get(i, {}).items():
                c = d2c.get(int(d), int(d))
                per[c] = per.get(c, 0.0) + e[0] / 1e6
        if per:
            slot = out.setdefault(layer, {})
            slot[name] = slot.get(name, 0.0) + max(per.values())
    out["_device"] = sum(max(e[0] for e in chips.values()) / 1e6 for chips in recs.values() if chips)
    return out


def _sum(tm: dict, op: str) -> float:
    return sum(v.get(op, 0.0) for k, v in tm.items() if not isinstance(k, str) and k is not None)


def _maxdiff(a: ttnn.Tensor, b: ttnn.Tensor) -> float:
    d = ttnn.max(ttnn.abs(ttnn.sub(a, b)))
    v = max(float(ttnn.to_torch(t).float().abs().max()) for t in ttnn.get_device_tensors(d))
    ttnn.deallocate(d)
    return v


def _ids(wl: str) -> torch.Tensor:
    from models.demos.deepseek_v41_flash_d_p.tests.test_moe_counts import _ids as prompt_ids

    return torch.tensor(prompt_ids(wl))


def _main():
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    return {m.layer: m for m in A._MAIN_LAYERS}


def _prevs(mesh):
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A

    return [pr for _, pr in A._PREV.values() if pr.mesh is mesh]


MODES = ("legacy", "device", "lean")  # (P.56, P.60, P.62: MOE_PREV_HOST / MOE_PREV_HOST2 below)


def _mode_of(host: str, host2: str) -> str:
    """The host path's name: legacy = MOE_PREV_HOST=legacy (P.56); device = MOE_PREV_HOST=device with
    MOE_PREV_HOST2=legacy (P.60); lean = MOE_PREV_HOST=device MOE_PREV_HOST2=lean (P.62)."""
    return "legacy" if host == "legacy" else ("lean" if host2 == "lean" else "device")


def _side(mesh, side: str, mode: str | None = None) -> None:
    """off: prev switched off in the loaded model (no planner, the stock combine, the empty map); prev: on, with host
    path `mode` (MODES; None: the settings')."""
    if side == "prev" and mode is not None:
        from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP

        host = "legacy" if mode == "legacy" else "device"
        XP.set_host2("lean" if mode == "lean" else "legacy")
        prs = _prevs(mesh)
        for pr in prs:
            pr.set_mode(host)
        if not prs:  # (made at the first MoE call: from the setting)
            from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

            assert settings.get("MOE_PREV_HOST") == host, (settings.get("MOE_PREV_HOST"), host)
    for m in _main().values():
        m._prev = side == "prev"
        if side == "off":
            if m._prev_dirty and m.head_w is not None:
                ttnn.copy(m._prev_off_rb, m.head_w.router_bias)
            m._prev_dirty, m._prev_ord = False, None
    if side == "off":
        for pr in _prevs(mesh):
            pr.fill_off()
    ttnn.synchronize_device(mesh)


def _map_tile(m) -> np.ndarray:
    from models.demos.deepseek_v41_flash_d_p.tt.expert_prev import map_words_of_tile

    E = len(m.perm)
    out = []
    for d in ttnn.get_device_tensors(m.head_w.router_bias):
        a = ttnn.to_torch(d).float().numpy().reshape(-1, 32)
        out.append(map_words_of_tile(a[E : E + 32]))
    return np.stack(out)


class _Mirror:
    """Every planner run of a prev pass replayed on the host (expert_prev.plan_host), checked after each chunk."""

    def __init__(self, mesh, layers):
        from models.demos.deepseek_v41_flash_d_p.tt import expert_prev as XP

        self.XP = XP
        self.mesh, self.layers = mesh, list(layers)
        prs = _prevs(mesh)
        self.pr = prs[0] if prs else None
        self.mismatch, self.checks, self.cycles = [], 0, []
        main = _main()
        if self.pr is None:  # (the first prev pass: the planner is made at the first call - its load-time state)
            self.G = None
            self.st = {L: np.zeros(1, dtype=np.uint32) for L in main}
        else:
            G = XP._read(self.pr.G)
            geo = self.pr.geo
            G[1] += 1  # (the reset of chunk 0: a new epoch, stamps NONE, budgets 0)
            G[16 : 16 + geo.W] = XP.NONE
            G[16 + geo.W :] = 0
            self.G = G
            self.st = {L: XP._read(m._prev_state) for L, m in main.items()}
        self.jobs: dict = {}

    def chunk(self, rec: dict, ready: dict) -> None:
        """After a chunk's sync: replay its planner runs (layer order) and compare."""
        XP = self.XP
        main = _main()
        if self.pr is None:
            self.pr = _prevs(self.mesh)[0]
            geo = self.pr.geo
            self.G = np.zeros(geo.gw, dtype=np.uint32)
            self.G[1] = 1
            self.G[16 : 16 + geo.W] = XP.NONE
            self.st = {L: np.zeros(geo.sw, dtype=np.uint32) for L in main}
        geo = self.pr.geo
        spc = int(rec["slots_per_chip"])
        words = {}
        for L in self.layers:
            if L not in main:
                continue
            lc = rec["layers"][str(L)]
            a = np.asarray(lc["chips"], dtype=np.int64)
            rows = np.stack([a[k, k * spc : (k + 1) * spc] for k in range(geo.n)])
            st = self.st[L] if self.st[L].size == geo.sw else np.zeros(geo.sw, dtype=np.uint32)
            w, self.G, self.st[L], jobs = XP.plan_host(geo, rows, self.G, st, self.pr.dirs)
            for (s, h, p, j), v in jobs.items():
                self.jobs[(int(self.G[16 + s]), s, h, p, j)] = v
            words[L] = w
        # compare: global state on every chip; each layer's state and map (chip 0, and every chip for a few layers)
        dG = XP.read_all(self.pr.G)
        if not (dG == dG[0]).all():
            self.mismatch.append("global state differs between chips")
        if not np.array_equal(dG[0], self.G):
            bad = np.nonzero(dG[0] != self.G)[0][:8].tolist()
            self.mismatch.append(
                f"global state != mirror at words {bad}: {dG[0][bad].tolist()} vs {self.G[bad].tolist()}"
            )
        for i, (L, w) in enumerate(words.items()):
            m = main[L]
            dst = XP.read_all(m._prev_state) if i < 3 else XP._read(m._prev_state)[None]
            self.cycles.append([int(x) for x in dst[0][10:14]])
            dst[:, 10:14] = 0  # (the planner's diagnostic cycle stamps)
            if not (dst == dst[0]).all():
                self.mismatch.append(f"layer {L}: state differs between chips")
            if not np.array_equal(dst[0], self.st[L]):
                bad = np.nonzero(dst[0] != self.st[L])[0][:8].tolist()
                self.mismatch.append(
                    f"layer {L}: state != mirror at {bad}: {dst[0][bad].tolist()} vs {self.st[L][bad].tolist()}"
                )
            mt = _map_tile(m) if i < 3 else _map_tile(m)[:1]
            if not (mt == mt[0]).all():
                self.mismatch.append(f"layer {L}: slot map differs between chips")
            if not np.array_equal(mt[0], w):
                bad = np.nonzero(mt[0] != w)[0][:8].tolist()
                self.mismatch.append(
                    f"layer {L}: slot map != mirror at {bad}: {mt[0][bad].tolist()} vs {w[bad].tolist()}"
                )
            self.checks += 1

    def pages(self) -> None:
        """The pending job pages (ordinals not yet carried) equal the mirror's jobs, every chip."""
        XP, geo, pr = self.XP, self.pr.geo, self.pr
        L = int(geo.links)
        for s in range(geo.W):
            o = int(self.G[16 + s])
            if o == XP.NONE or o < int(self.G[0]):
                continue
            pages = pr.read_ring(s)
            for h in range(geo.n):
                for p in range(4):
                    v = int(self.G[16 + geo.W + (s * geo.n + h) * 4 + p])
                    nj = v >> 16
                    pg = pages[h, p * L]
                    if int(pg[0]) != nj:
                        self.mismatch.append(f"ring {s} chip {h} dir {p}: {int(pg[0])} jobs, mirror {nj}")
                        continue
                    for j in range(nj):
                        want = self.jobs.get((o, s, h, p, j))
                        got = [int(x) for x in pg[16 + 8 * j : 16 + 8 * j + 6]]
                        if want is None or got != list(want):
                            self.mismatch.append(f"ring {s} chip {h} dir {p} job {j}: {got} vs mirror {want}")
            self.checks += 1


def _pass(model, mesh, layers, tokens, spy, side, prof=False, ref=None, keep=None, mirror=None) -> list:
    """Chunks 0..CHUNKS-1 of one fresh request, back to back. -> [per chunk {device / ops / wall / diff / prev}]."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A
    from models.demos.deepseek_v41_flash_d_p.tt.model import DevH

    chunk = int(S.get("target.chunk"))
    state = model.new_state(SEQ)
    out = []
    try:
        for c in range(CHUNKS):
            st = c * chunk
            t = tokens[st : st + chunk]
            spy.calls.clear()
            ps0 = dict(A.prev_stats(mesh))
            if prof:
                profiler.enable(mesh, rt=True)
            else:
                ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            row: dict = {"chunk": c}
            try:
                begin_chunk(model, t, st)
                h = model.embed(t)
                for i in layers:
                    spy.layer = i
                    h2 = model.layer(i, h, st, state)
                    model.free(h)
                    h = h2
                spy.layer = None
                if not prof:  # (P.60: host = to the last enqueue; the clean wall = to the sync, before the diff)
                    t1 = time.perf_counter()
                    ttnn.synchronize_device(mesh)
                    row["host_ms"] = (t1 - t0) * 1e3
                    row["wall_ms"] = (time.perf_counter() - t0) * 1e3
                if isinstance(h, DevH):
                    if ref is not None and c in ref:
                        row["diff"] = _maxdiff(h.streams, ref[c])
                    if keep is not None:
                        keep[c] = ttnn.clone(h.streams)
                if layers[-1] == S.num_layers - 1:
                    h2 = model.final_norm(h)
                    model.free(h)
                    h = h2
                model.free(h)
                if prof:
                    profiler.signpost("end")
                    profiler.finish_rt()
                    tm = _times(spy.calls, profiler._P["rt_records"], profiler._P["dev_to_chip"])
                    row["device_ms"] = tm["_device"]
                    for op in OPS + ("planner",):
                        row[f"{op}_ms"] = _sum(tm, op)
                    row["experts_layers"] = {
                        int(k): round(v.get("routed_experts_combine", 0.0), 4)
                        for k, v in tm.items()
                        if not isinstance(k, str) and k is not None
                    }
                else:
                    ttnn.synchronize_device(mesh)
            finally:
                if prof:
                    profiler.disable()
            rec = A.flush_counts(mesh, st, chunk)
            ps1 = A.prev_stats(mesh)
            row["prev"] = {k: ps1.get(k, 0) - ps0.get(k, 0) for k in ps1}
            if rec is not None:
                spc = int(rec["slots_per_chip"])
                epc = spc - 1
                row["spare_tokens"] = int(
                    sum(lc["chips"][k][k * spc + epc] for lc in rec["layers"].values() for k in range(len(lc["chips"])))
                )
            if side != "off":
                plans = {}
                for L, m in _main().items():
                    if m._prev_state is not None and _prevs(mesh):
                        from models.demos.deepseek_v41_flash_d_p.tt.expert_prev import _read

                        w = _read(m._prev_state)
                        plans[L] = [int(x) for x in w[1:10]]
                row[
                    "plans"
                ] = plans  # (each layer's plan for the next chunk: ordinal, receivers, kept, new, dropped, ...)
                if mirror is not None and rec is not None:
                    mirror.chunk(rec, row["prev"])
            out.append(row)
            logger.info(
                f"P.56 {side} chunk {c}: "
                + (
                    f"device {row['device_ms']:.2f} ms, experts {row['routed_experts_combine_ms']:.2f}, planner "
                    f"{row['planner_ms']:.2f}"
                    if prof
                    else f"host {row['host_ms']:.1f} ms, wall {row['wall_ms']:.1f} ms"
                )
                + (f", diff {row['diff']}" if "diff" in row else "")
                + f", prev {row['prev']}, spare tokens {row.get('spare_tokens')}"
            )
        if mirror is not None:
            mirror.pages()
    finally:
        _free(state)
    return out


@mesh_parametrize(timeout_s=3 * 3600)
def test_moe_prev_replicas(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_REPLICAS", "prev")
    monkeypatch.setenv("DS41_MOE_FILL", "piggyback")
    monkeypatch.setenv("DS41_ENGRAM_TABLE", "resident")
    monkeypatch.setenv("DS41_MOE_COUNTS", "on")
    monkeypatch.delenv("DS41_MOE_COUNTS_DIR", raising=False)
    if QUICK:
        monkeypatch.setenv("DS41_MOE_FILL_PIECES", "2000")
    settings.thaw()
    assert settings.get("MOE_REPLICAS") == "prev" and settings.get("MOE_COUNTS") == "on"
    first = _mode_of(settings.get("MOE_PREV_HOST"), settings.get("MOE_PREV_HOST2"))
    modes = [first] + [m for m in MODES if m != first]  # (the settings' host path first)
    layers = S.layers()[:4] if QUICK else S.layers()
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    spy = _Spy(monkeypatch)
    data, fails = {}, []
    for wl in WORKLOADS:
        t0 = time.time()
        tokens = _ids(wl)
        ref: dict = {}
        d: dict = {"modes": {}}
        try:
            _side(mesh_device, "off")
            _pass(model, mesh_device, layers, tokens, spy, "off")  # (warm: every chunk position compiled)
            d["off_prof"] = _pass(model, mesh_device, layers, tokens, spy, "off", prof=True, keep=ref)
            spy.reset_host()
            d["off_wall"] = _pass(model, mesh_device, layers, tokens, spy, "off", ref=ref)
            d["off_host_us"] = {k: 1e6 * statistics.median(v) for k, v in spy.host.items() if v}
            for mode in modes:
                dm: dict = {}
                _side(mesh_device, "prev", mode)
                _pass(model, mesh_device, layers, tokens, spy, mode)  # (warm: the exact combine, the planner)
                dm["prof"] = _pass(model, mesh_device, layers, tokens, spy, mode, prof=True, ref=ref)
                mirror = _Mirror(mesh_device, layers)
                spy.reset_host()
                dm["wall"] = _pass(model, mesh_device, layers, tokens, spy, mode, ref=ref, mirror=mirror)
                dm["host_us"] = {k: 1e6 * statistics.median(v) for k, v in spy.host.items() if v}
                logger.info(f"P.56 {wl} {mode}: host us a call (median, the unprofiled pass): {dm['host_us']}")
                cyc = np.asarray(mirror.cycles, dtype=np.float64)
                dm["mirror"] = {"mismatch": mirror.mismatch, "checks": mirror.checks}
                if cyc.size:  # (planner core, chip 0: median cycles of its phases, at ~1.35 GHz)
                    dm["planner_cycles"] = {
                        k: float(np.median(cyc[:, i]))
                        for i, k in enumerate(("reads", "wait_counts", "plan", "map_writes"))
                    }
                    logger.info(
                        f"P.56 {wl} {mode}: planner core phases (median cycles, chip 0): {dm['planner_cycles']}"
                    )
                d["modes"][mode] = dm
        finally:
            for v in ref.values():
                ttnn.deallocate(v)
        data[wl] = d
        logger.info(f"P.56 {wl}: measured in {time.time() - t0:.0f} s")

    summary = {}
    steady = range(1, CHUNKS)
    mean = lambda rows, k: statistics.mean(rows[c][k] for c in steady)  # noqa: E731
    for wl, d in data.items():
        off_rep = max(r.get("diff", float("nan")) for r in d["off_wall"])
        metrics.record(f"prev_off_repeat_maxdiff_{wl}", off_rep)
        if off_rep != 0.0:
            fails.append(f"{wl}: two off passes differ ({off_rep}): no baseline")
        s_off = {
            "device_ms_off": mean(d["off_prof"], "device_ms"),
            "host_ms_off": mean(d["off_wall"], "host_ms"),
            "wall_ms_off": mean(d["off_wall"], "wall_ms"),
            "experts_ms_off": mean(d["off_prof"], "routed_experts_combine_ms"),
        }
        for k, v in s_off.items():
            metrics.record(f"prev_{k}_{wl}", round(v, 2))
        for k, v in d.get("off_host_us", {}).items():
            metrics.record(f"prev_host_us_{k}_off_{wl}", round(v, 1))
        summary[wl] = {"off": s_off, "off_host_us": d.get("off_host_us"), "modes": {}}
        plans_of = {}
        for mode, dm in d["modes"].items():
            sfx = "" if mode == first else f"_{mode}"  # (the setting's mode keeps P.56's metric names)
            s = {
                "device_ms_prev": mean(dm["prof"], "device_ms"),
                "host_ms_prev": mean(dm["wall"], "host_ms"),
                "wall_ms_prev": mean(dm["wall"], "wall_ms"),
                "experts_ms_prev": mean(dm["prof"], "routed_experts_combine_ms"),
                "planner_ms": mean(dm["prof"], "planner_ms"),
            }
            md = max(r.get("diff", float("nan")) for r in dm["prof"])
            rep = max(r.get("diff", float("nan")) for r in dm["wall"])
            ready = sum(dm["wall"][c]["prev"].get("ready", 0) for c in steady)
            calls = sum(dm["wall"][c]["prev"].get("calls", 0) for c in steady)
            spare = sum(dm["wall"][c].get("spare_tokens", 0) for c in steady)
            mm = len(dm["mirror"]["mismatch"])
            for k, v in s.items():
                metrics.record(f"prev_{k}{sfx}_{wl}", round(v, 2))
            metrics.record(f"prev_maxdiff{sfx}_{wl}", md)
            metrics.record(f"prev_repeat_maxdiff{sfx}_{wl}", rep)
            metrics.record(f"prev_ready_frac{sfx}_{wl}", round(ready / max(calls, 1), 4))
            metrics.record(f"prev_spare_tokens{sfx}_{wl}", spare)
            metrics.record(f"prev_mirror_mismatch{sfx}_{wl}", mm)
            for k, v in dm.get("host_us", {}).items():
                metrics.record(f"prev_host_us_{k}{sfx}_{wl}", round(v, 1))
            if md != 0.0:
                fails.append(f"{wl} {mode}: prev differs from off ({md})")
            if rep != 0.0:
                fails.append(f"{wl} {mode}: two prev passes differ ({rep})")
            if mm:
                fails.append(f"{wl} {mode}: {mm} mirror mismatches, first {dm['mirror']['mismatch'][:3]}")
            if dm["mirror"]["checks"] == 0:
                fails.append(f"{wl} {mode}: the mirror checked nothing")
            if ready == 0 or spare == 0:
                fails.append(f"{wl} {mode}: no replica used (ready {ready} of {calls}, spare tokens {spare})")
            # (the plans without their ordinal word, which counts on across passes)
            plans_of[mode] = [{L: v[1:] for L, v in r.get("plans", {}).items()} for r in dm["wall"]]
            table = []
            for c in range(CHUNKS):
                op, pp, ow, pw = d["off_prof"][c], dm["prof"][c], d["off_wall"][c], dm["wall"][c]
                plans = pw.get("plans", {})
                nxt = dm["wall"][c - 1].get("plans", {}) if c else {}
                used_rcv = sum(v[1] for v in nxt.values()) if c else 0
                dropped = sum(v[4] for v in nxt.values()) if c else 0
                row = {
                    "chunk": c,
                    "device_ms_off": round(op["device_ms"], 2),
                    "device_ms_prev": round(pp["device_ms"], 2),
                    "host_ms_off": round(ow["host_ms"], 1),
                    "host_ms_prev": round(pw["host_ms"], 1),
                    "wall_ms_off": round(ow["wall_ms"], 1),
                    "wall_ms_prev": round(pw["wall_ms"], 1),
                    "experts_ms_off": round(op["routed_experts_combine_ms"], 2),
                    "experts_ms_prev": round(pp["routed_experts_combine_ms"], 2),
                    "dispatch_ms_off": round(op["sparse_dispatch_2d_ms"], 2),
                    "dispatch_ms_prev": round(pp["sparse_dispatch_2d_ms"], 2),
                    "planner_ms": round(pp["planner_ms"], 2),
                    "layers_ready": pw["prev"].get("ready", 0),
                    "layers_off": pw["prev"].get("off", 0),
                    "receivers_used": used_rcv,
                    "receivers_kept": sum(v[2] for v in nxt.values()) if c else 0,
                    "copies_made": sum(v[3] for v in nxt.values()) if c else 0,
                    "copies_dropped": dropped,
                    "complete_frac": round(used_rcv / max(used_rcv + dropped, 1), 3) if c else None,
                    "spare_tokens": pw.get("spare_tokens"),
                    "planned_next": sum(v[1] for v in plans.values()),
                    "experts_layers_off": op["experts_layers"],
                    "experts_layers_prev": pp["experts_layers"],
                }
                table.append(row)
                print(
                    f"P.56 {wl} {mode} chunk {c}: device {row['device_ms_off']:.2f} -> {row['device_ms_prev']:.2f} ms "
                    f"({row['device_ms_prev'] - row['device_ms_off']:+.2f}), host {row['host_ms_off']:.1f} -> "
                    f"{row['host_ms_prev']:.1f}, wall {row['wall_ms_off']:.1f} -> {row['wall_ms_prev']:.1f}, experts "
                    f"{row['experts_ms_off']:.2f} -> {row['experts_ms_prev']:.2f}, planner {row['planner_ms']:.2f}, "
                    f"ready {row['layers_ready']}/{row['layers_ready'] + row['layers_off']}, receivers "
                    f"{row['receivers_used']} (kept {row['receivers_kept']}, new {row['copies_made']}, dropped "
                    f"{row['copies_dropped']}), spare tokens {row['spare_tokens']}"
                )
            summary[wl]["modes"][mode] = {
                "mean_1_10": s,
                "chunks": table,
                "mirror": dm["mirror"],
                "host_us": dm.get("host_us"),
                "planner_cycles": dm.get("planner_cycles"),
            }
            print(
                f"P.56 {wl} {mode} chunks 1..{CHUNKS - 1} mean: device {s_off['device_ms_off']:.2f} -> "
                f"{s['device_ms_prev']:.2f} ms, host {s_off['host_ms_off']:.1f} -> {s['host_ms_prev']:.1f}, wall "
                f"{s_off['wall_ms_off']:.1f} -> {s['wall_ms_prev']:.1f}, experts {s_off['experts_ms_off']:.2f} -> "
                f"{s['experts_ms_prev']:.2f}, planner {s['planner_ms']:.2f}; host us a call {dm.get('host_us')}; "
                f"maxdiff {md}, repeat {rep}, mirror {mm} mismatches of {dm['mirror']['checks']} checks"
            )
        if len(plans_of) > 1:  # (P.60 / P.62: every host path plans the same, chunk by chunk)
            a = plans_of[first]
            same = 1
            for m, b in plans_of.items():
                if b != a:
                    same = 0
                    bad = [c for c in range(min(len(a), len(b))) if a[c] != b[c]]
                    fails.append(f"{wl}: host paths {first} and {m} planned differently at chunks {bad[:5]}")
            metrics.record(f"prev_plans_identical_{wl}", same)
    metrics.record_json(
        "prev_replicas", {"mesh": list(mesh_device.shape), "quick": QUICK, "first": first, "workloads": summary}
    )
    for f in fails:
        logger.error(f"P.56: {f}")
    assert not fails, f"{len(fails)} P.56 checks failed: {fails[:5]}"
