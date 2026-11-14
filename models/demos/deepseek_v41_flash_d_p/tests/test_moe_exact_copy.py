# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.55b (owner 2026-11-13, MoE exact replicas, STEP 2): the streamed expert copy inside one real layer's
routed_experts_combine, as a microbenchmark (MOE_REPLICAS=exact; ttnn.bringup.moe_block_main set_combine_fill_exact,
the port writer's REC_FX path; tt/expert_exact.py defines the terms: piece, path, block / ready flag, tree, relay).

One MoE layer (LAYER, its real routed weights, 13 slots a chip, one chunk of a fixed random input). Every exact call
copies whole experts of THIS layer from their home slot into receivers' spare slots, in the layer's own call: the
sender's port cores re-read the home slot from DRAM (OV-39 (3): the copy outlasts the home's own compute of the expert
in ~94 % of P.55a's planned copies, so the blocks the gate/up cores hold in L1 are gone before the copy needs them;
cost: one extra 19.9 MB DRAM read a sending edge, measured in the added time), half the pieces on each link of the
first hop, a ready flag every MOE_EXACT_BLOCK pieces; several receivers of one expert form a relay tree.

Cases (each: the combine program's device time, real-time profiler, slowest chip, exact calls alternating with the
stock port writer in one window, median; then LAT synced calls whose stamps give the copy latency):
  (0) the exact path with no job vs the stock port writer: outputs bit-identical; the time.
  (1) one chip pair (1 -> 0, one hop): one path (link 0), two paths (both links of the hop), four links (half of the
      pieces through a free neighbour's spare slot, relayed: P.55a's "two paths" of a link pair each).
  (2) all 32 chips, neighbour placement (chip k receives a home slot of chip k + 1): 32 copies at once.
  (3) all 32 chips, a real layer's split: P.55a's chunk-10 counts of layer LAYER (novel prompt; and code), planned by
      tests/moe_exact_model.py (deadline mode, two paths), its hot experts copied as relay trees.
  (4) B2B calls back to back (no sync between) of (3) after (2) overwrote the spare slots: no hang, every status
      clean, the spare slots bit-exact, outputs bit-identical.
Every case's receivers' spare slots == the home slot, bit-exact (decoded tile for tile); every carrier output
bit-identical to the stock port writer's. Latency: the receiver's waiter port stamps when each block's flag is in
(polled every ~1 us) and when the whole copy has landed, from its own start barrier; the receiver's own work = the
last of its ports' main loops (its experts' rows sent, its credits returned) in the same call.
The deadline: P.55a's model (moe_exact_model.py) re-run on its novel and code chunk-10 counts with the measured copy
time a path in place of P.47's (copies on time, surviving ms a chunk).
Metrics: exact_copy_bitexact (1), exact_copy_runs_ok (B2B), exact_combine_identical (1), exact_copy_added_ms and
exact_copy_latency_ms (case (3) novel), per-case extras; the side record results/P.55b_exact_copy.json.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_exact_copy.py
"""

from __future__ import annotations

import json
import statistics
import time

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

S = spec()
LAYER = 20
REPS = 6  # exact calls a timing window (as many stock calls between them)
LAT = 5  # synced exact calls whose stamps are read
B2B = 100
LEAD = 100  # matmuls (4096^3, 11 x 10 = 110 cores: rule 18) ahead of each timed window (DS41-248)
KERNELS = "routed_experts_combine/device/kernels"
MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])


def _settings(monkeypatch, **kv):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    for k, v in kv.items():
        monkeypatch.setenv(f"DS41_{k}", v)
    settings.thaw()
    return settings


def _decode(w, d: int) -> torch.Tensor:
    t = ttnn.to_torch(ttnn.get_device_tensors(w)[d])
    return t.reshape(-1, t.shape[-1])


def _slot(x2, units, tb: int) -> torch.Tensor:
    out = []
    for w_, b, o, n in units:
        out.append(x2[w_][32 * (o // tb) : 32 * ((o + n) // tb), 32 * b : 32 * b + 32].reshape(-1, 32, 32))
    t = torch.cat(out).contiguous()
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _combine_ns(per_program: dict) -> list[float]:
    return [e["duration_ns"] for e in per_program.values() if any(KERNELS in s for s in e["kernel_sources"])]


def _p55a() -> dict:
    """P.55a's per-layer record (the gate's results dir; else this box's ledger beside the tests)."""
    from pathlib import Path

    name = "P.55a_exact_bound.json"
    R, C = S.mesh
    for d in (
        metrics.results_dir(),
        Path(__file__).parents[1] / "bringup" / "ledgers" / f"galaxy_{R}x{C}-full" / "results",
    ):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    raise FileNotFoundError(f"{name}: P.55a's record (run its gate unit first)")


def _real_split(layer: int, wl: str, shape: tuple) -> tuple[list, dict]:
    """P.55a's chunk-10 counts of `layer` (workload wl), planned in deadline mode with two paths: [(home, slot,
    [receivers])] and the plan's numbers."""
    from models.demos.deepseek_v41_flash_d_p.tests import moe_exact_model as M

    d = _p55a()
    A, B = d["fit"]["A_ms"], d["fit"]["B_ms"]
    row = next(r for r in d["layers"][f"{wl}_p51200"] if int(r["layer"]) == layer)
    own = np.array([[int(v) for v in s.split()] for s in row["tokens_slot"]])
    b = M.layer_bound(own, shape, A, B, paths=2)
    hot: dict = {}
    for r in b["replicas_deadline"]:
        hot.setdefault((r["home"], r["slot"]), []).append(r["recv"])
    return [(k, s, rs) for (k, s), rs in sorted(hot.items())], b


def _deadline(t_path_ms: float, shape: tuple) -> dict:
    """P.55a's model (moe_exact_model.py) on its novel and code chunk-10 counts, every layer, with a copy edge taking
    t_path_ms (a measured pair: start barrier to its last flag; P.55a used 0.71 / 2 ms for its two paths): saved ms a
    chunk and copies on time."""
    from models.demos.deepseek_v41_flash_d_p.tests import moe_exact_model as M

    d = _p55a()
    A, B = d["fit"]["A_ms"], d["fit"]["B_ms"]
    old = M.T1_MS
    M.T1_MS = 2.0 * t_path_ms  # (the model's per-link time is T1 / paths)
    out = {}
    try:
        for wl in ("novel", "code"):
            save = 0.0
            on = tot = 0
            for row in d["layers"][f"{wl}_p51200"]:
                own = np.array([[int(v) for v in s.split()] for s in row["tokens_slot"]])
                b = M.layer_bound(own, shape, A, B, paths=2)
                save += b["save_deadline_ms"]
                on += b["on_time"]
                tot += b["n_rep_deadline"]
            out[wl] = {"save_ms": round(save, 2), "on_time": on, "copies": tot}
    finally:
        M.T1_MS = old
    return out


@pytest.mark.timeout(5400)
@BOX
@MESH
def test_moe_exact_copy(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import expert_exact as EX
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_device
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

    fails: list[str] = []
    rec: dict = {"cases": {}}

    def check(ok: bool, what: str) -> None:
        logger.info(f"EXACT {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    n = mesh_device.get_num_devices()
    shape = tuple(mesh_device.shape)
    st = _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="exact", MOE_FILL="serial")
    BLOCK, FAN = int(st.get("MOE_EXACT_BLOCK")), int(st.get("MOE_EXACT_FANOUT"))
    CAP, LIMIT_MS = int(st.get("MOE_EXACT_CAP")), float(st.get("MOE_EXACT_LIMIT_MS"))
    EX.AICLK_MHZ = EX.aiclk_mhz()
    t0 = time.time()
    car = ad.TtMoEBlockMainLayer(mesh_device, ck, cfg, LAYER, chunk, cache_dir=expert_cache_dir(S), own_rows=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"EXACT layer {LAYER} loaded in {time.time() - t0:.1f} s; AICLK {EX.AICLK_MHZ:.0f} MHz")
    mb, res = ad._mb(), car.block().resources
    links = mb.combine_fill_links(res)
    sl = car.slot_tables().lay_spc
    tb, epc = sl.tb, car.epc
    piece = mb.expert_slot_copy_chunk_bytes(res)
    xt = EX.ExactTables(sl, epc, piece, mesh_device)
    P = xt.pieces
    logger.info(
        f"EXACT a slot {sl.slot_bytes() / 2**20:.2f} MiB = {P} pieces of {piece} B; block {BLOCK} pieces "
        f"({BLOCK * piece / 2**10:.0f} KiB), fan-out {FAN}, cap {CAP}, links {links}"
    )
    rec.update({"layer": LAYER, "pieces": P, "piece_bytes": piece, "block": BLOCK, "fanout": FAN, "cap": CAP})
    g = torch.Generator().manual_seed(55)
    x_host = torch.randn(chunk, int(cfg.dim), generator=g).bfloat16()
    x = own_rows_to_device(mesh_device, x_host, ttnn.bfloat16)
    lead = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096, generator=g) * 0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    limit = EX.ms_cycles(LIMIT_MS)

    # ---- outputs kept on the device: every call's against the first stock call's (max of ne, folded)
    ref = {}
    marker = [None, 0]

    def fold(y):
        if "y" not in ref:
            ref["y"] = ttnn.clone(y)
        else:
            m = ttnn.max(ttnn.ne(ref["y"], y, dtype=ttnn.bfloat16))
            marker[0] = m if marker[0] is None else ttnn.maximum(marker[0], m)
            marker[1] += 1
        ttnn.deallocate(y)

    def identical() -> bool:
        if marker[0] is None:
            return True
        return max(float(ttnn.to_torch(t).float().max()) for t in ttnn.get_device_tensors(marker[0])) == 0.0

    def dir_of(a, b):
        return mb.combine_fill_port(res, a, b)

    def jobs_for(hot, paths=2):
        host, info = EX.build_jobs(hot, n, shape, links, dir_of, P, BLOCK, CAP, xt.relay_table, paths=paths, fanout=FAN)
        return EX.jobs_tensor(host, mesh_device), info

    empty_jobs, _ = jobs_for([])

    def set_exact(jobs):
        if jobs is None:
            mb.set_combine_fill_exact(res, False)
        else:
            mb.set_combine_fill_exact(res, True, car.w_gu, car.w_d, xt.runs, jobs, limit)

    def window(jobs, count: int) -> tuple[list, list]:
        """2 count calls in one profiler window, stock and `jobs` alternating: the combine's device us of each side."""

        def run():
            for _ in range(LEAD):
                ttnn.deallocate(ttnn.matmul(lead, lead, core_grid=ttnn.CoreGrid(y=10, x=11)))
            for i in range(2 * count):
                set_exact(None if i % 2 == 0 else jobs)
                fold(car(x))

        _, per = profile_realtime_program_merged(mesh_device, run)
        ns = _combine_ns(per)
        assert len(ns) == 2 * count, f"{len(ns)} combine programs profiled, {2 * count} calls"
        return [v / 1e3 for v in ns[0::2]], [v / 1e3 for v in ns[1::2]]

    # warm: both programs built
    set_exact(None)
    fold(car(x))
    set_exact(empty_jobs)
    fold(car(x))
    ttnn.synchronize_device(mesh_device)

    def stamped(jobs, receivers: list, targets: dict) -> list[dict]:
        """LAT synced calls: per call every chip's stamps; checks every status and wait."""
        out = []
        for _ in range(LAT):
            set_exact(jobs)
            fold(car(x))
            ttnn.synchronize_device(mesh_device)
            w = EX.read_words(res)
            s = [EX.stamps(w, d, links) for d in range(n)]
            out.append(s)
        last = out[-1]
        bad = [d for d in range(n) if last[d]["status"] != 0]
        check(not bad, f"no relay gate gave up (chips {bad})")
        for r in receivers:
            ok = last[r].get("wait_ok") and last[r]["wait_final"][: len(targets[r])] == [t for _, t in targets[r]]
            if not ok:
                logger.info(f"EXACT receiver {r}: {last[r]}")
            check(bool(ok), f"receiver {r}: every block flag in ({targets[r]})")
        return out

    home_bits: dict = {}

    def verify(hot, label: str) -> bool:
        dec: dict = {}

        def chip(d):
            if d not in dec:
                dec[d] = [_decode(car.w_gu, d), _decode(car.w_d, d)]
            return dec[d]

        bad = []
        for e in hot:
            k, s, rcv = e[:3]
            if (k, s) not in home_bits:
                home_bits[(k, s)] = _slot(chip(k), sl.units(s), tb)
            for r in rcv:
                got = _slot(chip(r), sl.units(epc), tb)
                if not torch.equal(got, home_bits[(k, s)]):
                    diff = (got != home_bits[(k, s)]).reshape(got.shape[0], -1).any(-1)
                    logger.info(f"EXACT {label}: {k}/{s} -> {r}: {int(diff.sum())} of {diff.numel()} tiles differ")
                    bad.append((k, s, r))
                dec.pop(r, None)
            dec.pop(k, None)
        check(not bad, f"{label}: every receiver's spare slot == its home slot, bit-exact (differ: {bad})")
        return not bad

    def case(label: str, hot, paths: int = 2, timed: bool = True, check_bytes: bool = True) -> dict:
        jobs, info = jobs_for(hot, paths)
        receivers = sorted(info["waits"])
        r: dict = {
            "hot": [(e[0], e[1], list(e[2])) + tuple(e[3:]) for e in hot],
            "copies": len(receivers),
            "paths": paths,
            "depth_max": info["depth_max"],
            "jobs_max_port": info["jobs_max_port"],
        }
        if timed:
            off, on = window(jobs, REPS)
            mo, mn = statistics.median(off), statistics.median(on)
            r.update({"off_us": round(mo, 1), "on_us": round(mn, 1), "added_ms": round((mn - mo) / 1e3, 4)})
        ss = stamped(jobs, receivers, info["waits"]) if receivers else []
        if ss:
            land = [max(s[r_]["land_us"] for r_ in receivers) for s in ss]
            per_r = {r_: statistics.median(s[r_]["land_us"] for s in ss) for r_ in receivers}
            own = {r_: statistics.median(s[r_]["loop_end_us"] for s in ss) for r_ in receivers}
            first = {
                r_: statistics.median(min(a[0] for a in s[r_]["arrivals_us"] if a) for s in ss) for r_ in receivers
            }
            r.update(
                {
                    "latency_ms": round(statistics.median(land) / 1e3, 4),
                    "latency_ms_min": round(min(land) / 1e3, 4),
                    "land_ms": {r_: round(v / 1e3, 4) for r_, v in per_r.items()},
                    "first_block_ms": {r_: round(v / 1e3, 4) for r_, v in first.items()},
                    "own_work_ms": {r_: round(v / 1e3, 4) for r_, v in own.items()},
                    "arrivals_us_last": {r_: ss[-1][r_]["arrivals_us"] for r_ in receivers[:2]},
                }
            )
        if check_bytes and receivers:
            r["bitexact"] = verify(hot, label)
        logger.info(
            f"EXACT {label}: {r['copies']} copies, {paths} path(s), tree depth {r['depth_max']}, "
            f"jobs/port <= {r['jobs_max_port']}: combine {r.get('on_us', 0):.1f} us vs {r.get('off_us', 0):.1f} stock "
            f"(+{r.get('added_ms', 0):.3f} ms); latency {r.get('latency_ms', 0):.3f} ms (min "
            f"{r.get('latency_ms_min', 0):.3f}); first block {min(r.get('first_block_ms', {0: 0}).values()):.3f} ms; "
            f"receivers' own work {statistics.median(r.get('own_work_ms', {0: 0}).values()):.3f} ms (median)"
        )
        rec["cases"][label] = r
        return r

    # ---- (0) the exact path with no job vs the stock port writer
    off, emp = window(empty_jobs, REPS)
    check(identical(), "(0) the exact path with no job: outputs bit-identical to the stock port writer")
    rec["cases"]["(0) no job"] = {
        "off_us": round(statistics.median(off), 1),
        "on_us": round(statistics.median(emp), 1),
        "added_ms": round((statistics.median(emp) - statistics.median(off)) / 1e3, 4),
    }
    logger.info(f"EXACT (0) no job: {rec['cases']['(0) no job']}")
    set_exact(empty_jobs)
    fold(car(x))
    ttnn.synchronize_device(mesh_device)
    w0 = EX.read_words(res)
    own0 = [EX.stamps(w0, d, links)["loop_end_us"] / 1e3 for d in range(n)]
    rec["own_work_ms_nojob"] = [round(v, 4) for v in own0]
    logger.info(
        f"EXACT (0) every chip's own work, no job: median {statistics.median(own0):.3f} ms, max {max(own0):.3f}"
    )

    # ---- (1) one pair, one path then two
    pair1 = case("(1) pair 1 path", [(1, 3, [0])], paths=1)
    pair2 = case("(1) pair 2 paths", [(1, 4, [0])], paths=2)
    # four links: half straight on both links, half through a free neighbour's spare slot whose route to 0 avoids the
    # straight link (5 -> 4 -> 0 when the fabric routes X first from 5, else 1 -> 5 -> 4 then 4 -> 0)
    R, C = shape
    via = C + 1 if dir_of(C + 1, 0) == dir_of(1, 0) else C
    pair4 = case("(1) pair 4 links (detour)", [(1, 6, [0], via)], paths=2)
    # ---- (2) 32 chips, neighbour placement
    neigh = [((k // C) * C + (k % C + 1) % C, 5, [k]) for k in range(n)]
    case("(2) neighbour 32", neigh)
    # ---- (3) a real layer's split
    hot_novel, plan_novel = _real_split(LAYER, "novel", shape)
    hot_code, plan_code = _real_split(LAYER, "code", shape)
    rec["plan_novel"] = {k: plan_novel[k] for k in ("before_ms", "deadline_ms", "save_deadline_ms", "n_rep_deadline")}
    rec["plan_code"] = {k: plan_code[k] for k in ("before_ms", "deadline_ms", "save_deadline_ms", "n_rep_deadline")}
    novel = case("(3) real split novel", hot_novel)
    case("(3) real split code", hot_code)
    check(identical(), "(1)-(3) every output with copies bit-identical to the stock port writer")

    # ---- (4) back to back: the spare slots overwritten by (2)'s neighbour copies first
    runs_ok = 0
    if not fails:
        jn, info = jobs_for(neigh)
        set_exact(jn)
        fold(car(x))
        jobs, info = jobs_for(hot_novel)
        t0 = time.perf_counter()
        for _ in range(B2B):
            set_exact(jobs)
            fold(car(x))
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        logger.info(f"EXACT (4) {B2B} calls back to back in {dt:.2f} s: no hang")
        w = EX.read_words(res)
        last = [EX.stamps(w, d, links) for d in range(n)]
        ok = all(s["status"] == 0 for s in last) and all(
            last[r]["wait_ok"] and last[r]["wait_final"][:2] == [t for _, t in info["waits"][r]] for r in info["waits"]
        )
        check(ok, f"(4) the last of {B2B} back-to-back calls: every gate and wait clean")
        check(identical(), f"(4) {B2B} back-to-back calls: outputs bit-identical")
        ok &= verify(hot_novel, "(4) back to back")
        runs_ok = B2B if ok else 0

    # ---- the deadline with the measured copy time a path (two paths, one hop: the pair)
    rec["deadline"] = {}
    for tag, c in (("2links", pair2), ("4links", pair4)):
        t_path = c.get("latency_ms", float("nan"))
        if t_path != t_path:
            continue
        dl = _deadline(t_path, shape)
        rec["deadline"][tag] = {"t_path_ms": t_path, **dl}
        for wl, v in dl.items():
            logger.info(
                f"EXACT deadline {tag} ({wl}, chunk 10 at 51,200, a copy {t_path:.3f} ms): saves {v['save_ms']:.2f} "
                f"ms a chunk, {v['on_time']} of {v['copies']} copies on time"
            )
            metrics.record(f"exact_deadline_save_ms_{wl}_{tag}", v["save_ms"])
            metrics.record(f"exact_deadline_on_time_{wl}_{tag}", v["on_time"])
            metrics.record(f"exact_deadline_copies_{wl}_{tag}", v["copies"])

    set_exact(None)
    bitexact = all(c.get("bitexact", True) for c in rec["cases"].values()) and not any("bit-exact" in f for f in fails)
    metrics.record("exact_copy_bitexact", 1 if bitexact else 0)
    metrics.record("exact_copy_runs_ok", runs_ok)
    metrics.record("exact_combine_identical", 1 if identical() else 0)
    metrics.record("exact_copy_added_ms", novel.get("added_ms", float("nan")))
    metrics.record("exact_copy_latency_ms", novel.get("latency_ms", float("nan")))
    for key, c in (
        ("pair1", pair1),
        ("pair2", pair2),
        ("pair4", pair4),
        ("neigh32", rec["cases"]["(2) neighbour 32"]),
    ):
        metrics.record(f"exact_added_ms_{key}", c.get("added_ms", float("nan")))
        metrics.record(f"exact_latency_ms_{key}", c.get("latency_ms", float("nan")))
    metrics.record("exact_added_ms_nojob", rec["cases"]["(0) no job"]["added_ms"])
    metrics.record("exact_checks_failed", len(fails))
    rec["fails"] = fails
    metrics.record_json("exact_copy", rec)
    for t in [ref.get("y"), marker[0], x, lead]:
        if t is not None:
            ttnn.deallocate(t)
    for t in (car.gidx, car.w_gu, car.w_d):
        ttnn.deallocate(t)
    for _, r_, _ in list(ad._BLOCKS.values()):
        r_.release()
    ad._BLOCKS.clear()
    ad._SLOT_TABLES.clear()
    ad._FILLQ.clear()
    ad._MAIN_LAYERS.clear()
    assert not fails, f"{len(fails)} MOE_REPLICAS=exact checks failed: {fails}"
