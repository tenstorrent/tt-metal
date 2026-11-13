# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.53 step 1 (owner 2026-11-12): MOE_FILL=piggyback, the microbenchmark - expert replica copies carried by the
combine's fabric port senders (routed_experts_combine's port writer, REC_FILL; ttnn.bringup.moe_block_main
set_combine_fill) on a real layer's combine.

Two MoE layers on the spec's mesh, MOE_REPLICAS=runtime (13 slots a chip, the spare slot empty): the carrier
(CARRIER, its real routed weights, one chunk of a fixed random input) runs its experts + combine; the copies fill the
spare slots of another layer (FILLED: never the carrier's own). Every copy: chip a's home slot s of FILLED -> chip b's
spare slot of FILLED, pieces of expert_slot_copy's run table s.

(0) the carrier with the fill path off (the stock port writer) vs the fill path built with no job: outputs
    bit-identical; the combine's device time (real-time profiler, each program's slowest chip, median) of both.
(1) cases, each at BUDGETS (a call carries 1 / k of a slot) x CAPS: one chip (a pair of row neighbours), all 32 chips
    with neighbour placement (chip (r, c) from (r, c + 1)), all 32 chips with random placement (sources repeat, at
    most 4 jobs a port). Per case: the combine's device time of the carrying calls vs the fill path off, the calls
    alternating in one profiler window (the combine's time drifts between windows), the added %; the pieces sent
    inside the combine's main loop vs after it (hidden %); every done word counted the copy; every carrier output
    bit-identical to the fill path off. Then at VERIFY one case a placement with every destination's spare slot ==
    its source's home slot (decoded tile for tile, bit-exact) and the source slots unchanged.
(2) B2B calls back to back (no sync between), random placement, at VERIFY: no hang, outputs identical, the final
    copies bit-exact, the done words.
(3) step 2, the model's path (MOE_FILL=piggyback, tt/expert_fill.py): FILLED.set_replicas(a neighbour table) queues
    the copies, none mapped; the carrier's calls carry them (the filled layer's own call while pending carries none
    and its output is unchanged); the table is mapped after the copies; the spare slots bit-exact; the filled layer's
    output with the table bit-identical to none mapped and rows reach the spare slots; a table with half the chips
    changed unmaps them at once and maps them after their copies; the table emptied.
Metrics: fill_checks_failed (== 0), fill_combine_us_* (median device us), fill_added_pct_* (vs the fill path off).
The owner's pass mark (< 5 % added device time) is logged per case (FILL PASS / FILL OVER), not asserted.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_fill_piggyback.py
"""

from __future__ import annotations

import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

S = spec()
CARRIER, FILLED = 20, 19
REPS = 8  # carrier calls a timing window
B2B = 100
BUDGETS = (4, 8, 16, 32)  # a call carries 1 / k of a slot (0: the whole slot)
CAPS = (1, 4)  # a piece in the main loop while the channel holds fewer packets than this
VERIFY = (16, 4)  # (budget, cap) of the byte-checked cases and the back-to-back run
MIN_FILL_CALLS = 16  # carrying calls a timed case (at least)
LEAD = 100  # matmuls (4096^3, 110 cores) ahead of each timed window
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
    """A slot's tiles (both weights, units order) from decoded [w_gu, w_d] of one chip, as integer bits."""
    out = []
    for w_, b, o, n in units:
        out.append(x2[w_][32 * (o // tb) : 32 * ((o + n) // tb), 32 * b : 32 * b + 32].reshape(-1, 32, 32))
    t = torch.cat(out).contiguous()
    return t.view(torch.int16 if t.element_size() == 2 else torch.int32)


def _pieces(table, chunk: int) -> int:
    return sum(-(-r[5] // chunk) for r in table)


def _combine_ns(per_program: dict) -> list[float]:
    return [e["duration_ns"] for e in per_program.values() if any(KERNELS in s for s in e["kernel_sources"])]


@pytest.mark.timeout(5400)
@BOX
@MESH
def test_moe_fill_piggyback(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_device
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program_merged

    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        logger.info(f"FILL {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    n = mesh_device.get_num_devices()
    R, C = tuple(mesh_device.shape)
    # (the microbenchmark drives set_combine_fill itself: MOE_FILL=serial keeps the model's fill queue out of it)
    _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="runtime", MOE_FILL="serial")
    t0 = time.time()
    car = ad.TtMoEBlockMainLayer(mesh_device, ck, cfg, CARRIER, chunk, cache_dir=expert_cache_dir(S), own_rows=True)
    fil = ad.TtMoEBlockMainLayer(mesh_device, ck, cfg, FILLED, chunk, cache_dir=expert_cache_dir(S), own_rows=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"FILL layers {CARRIER} (carrier) and {FILLED} (filled) loaded in {time.time() - t0:.1f} s")
    mb, res = ad._mb(), car.block().resources
    assert fil.block().resources is res, "the two layers share one MoE block"
    tabs = fil.slot_tables()
    sl, tb, epc = tabs.lay_spc, tabs.lay_spc.tb, fil.epc
    piece = mb.expert_slot_copy_chunk_bytes(res)
    total = {s: _pieces(tabs.host[s], piece) for s in range(epc)}
    logger.info(
        f"FILL a slot {sl.slot_bytes() / 2**20:.2f} MiB = {total[0]} pieces of {piece} B "
        f"(tables {sorted(set(total.values()))} pieces)"
    )
    g = torch.Generator().manual_seed(53)
    x_host = torch.randn(chunk, int(cfg.dim), generator=g).bfloat16()
    x = own_rows_to_device(mesh_device, x_host, ttnn.bfloat16)

    # ---- outputs kept on the device: every carrier call's against the first fill-off call's (max of ne, folded)
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

    def calls(jobs_of_call, count: int) -> tuple[list[float], list[float]]:
        """count carrier calls in one profiler window, call i carrying jobs_of_call(i) (None: the fill path off);
        the combine's device us of the calls with the fill path off, and of the others (paired in one window: the
        combine's time drifts between windows, 785-1006 us medians on the same code)."""

        def run():
            for _ in range(LEAD):  # (device work ahead: no measured program waits on the host's dispatch)
                ttnn.deallocate(ttnn.matmul(lead, lead, core_grid=ttnn.CoreGrid(y=10, x=11)))
            for i in range(count):
                j = jobs_of_call(i)
                if j is None:
                    mb.set_combine_fill(res, False)
                else:
                    mb.set_combine_fill(res, True, fil.w_gu, fil.w_d, tabs.runs, j, cap=CAP[0])
                fold(car(x))

        _, per = profile_realtime_program_merged(mesh_device, run)
        ns = _combine_ns(per)
        assert len(ns) == count, f"{len(ns)} combine programs profiled, {count} calls"
        off = [v / 1e3 for i, v in enumerate(ns) if jobs_of_call(i) is None]
        on = [v / 1e3 for i, v in enumerate(ns) if jobs_of_call(i) is not None]
        return off, on

    CAP = [1]
    lead = ttnn.from_torch(
        torch.randn(1, 1, 4096, 4096, generator=g) * 0.01,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    # warm: both programs built
    mb.set_combine_fill(res, False)
    fold(car(x))
    mb.set_combine_fill(res, True)
    fold(car(x))
    ttnn.synchronize_device(mesh_device)

    # ---- (0) the fill path with no job vs off, alternating calls
    offs, emps = calls(lambda i: None if i % 2 == 0 else [], 4 * REPS)
    base, empty = statistics.median(offs), statistics.median(emps)
    check(identical(), "(0) the fill path with no job: outputs bit-identical to the stock port writer")
    logger.info(
        f"FILL (0) combine device time: fill off {base:.1f} us (min {min(offs):.1f}), fill path no job {empty:.1f} us "
        f"(min {min(emps):.1f}): {100 * (empty - base) / base:+.2f} %"
    )
    metrics.record("fill_combine_us_off", round(base, 1))
    metrics.record("fill_combine_us_nojob", round(empty, 1))
    metrics.record("fill_added_pct_nojob", round(100 * (empty - base) / base, 2))

    # ---- copies
    home = {}  # (chip, slot) -> its home slot's bits (FILLED)
    done_seen = torch.zeros(n, n, dtype=torch.long)  # [dst, src] completed copies so far

    def done_words() -> torch.Tensor:
        t = mb.combine_fill_done(res)
        return torch.stack(
            [ttnn.to_torch(d).reshape(-1)[: n + 16].to(torch.int64) & 0xFFFFFFFF for d in ttnn.get_device_tensors(t)]
        )

    def verify(copies, label: str) -> None:
        """copies [(src, dst, slot)]: each dst's spare slot == src's home slot s, bit-exact."""
        dec = {}

        def chip(d):
            if d not in dec:
                dec[d] = [_decode(fil.w_gu, d), _decode(fil.w_d, d)]
            return dec[d]

        bad = []
        for a, b, s in copies:
            if (a, s) not in home:
                home[(a, s)] = _slot(chip(a), sl.units(s), tb)
            elif a in dec or len(dec) < 4:  # (a few sources re-read: their home slots unchanged)
                check(
                    torch.equal(_slot(chip(a), sl.units(s), tb), home[(a, s)]), f"{label}: chip {a} slot {s} unchanged"
                )
            got = _slot(chip(b), sl.units(epc), tb)
            if not torch.equal(got, home[(a, s)]):
                diff = (got != home[(a, s)]).reshape(got.shape[0], -1).any(-1)
                logger.info(f"FILL {label}: {a} -> {b} slot {s}: {int(diff.sum())} of {diff.numel()} tiles differ")
                bad.append((a, b, s))
            dec.pop(b, None)
        check(not bad, f"{label}: {len(copies)} spare slots == their sources' home slots, bit-exact (differ: {bad})")

    def case(label: str, copies, budget_div: int, cap: int = 1, check_bytes: bool = True) -> None:
        """copies [(src, dst, slot)] carried by the carrier's calls, 1 / budget_div of a slot a call (0: whole)."""
        CAP[0] = cap
        before = done_words()
        per_call = {c: (total[c[2]] if budget_div == 0 else -(-total[c[2]] // budget_div)) for c in copies}
        n_calls = max(-(-total[s] // per_call[(a, b, s)]) for a, b, s in copies)
        reps = max(1, -(-MIN_FILL_CALLS // n_calls))

        def jobs(i):  # (even calls: the fill path off; odd: the copies' pieces of round (i // 2) % n_calls)
            if i % 2 == 0:
                return None
            k = (i // 2) % n_calls
            out = []
            for a, b, s in copies:
                p0 = min(k * per_call[(a, b, s)], total[s])
                p1 = min(p0 + per_call[(a, b, s)], total[s])
                if p0 < p1:
                    out.append((a, b, fil.slot_tables().fill_table(s), p0, p1, p1 == total[s]))
            return out

        off, us = calls(jobs, 2 * n_calls * reps)
        ttnn.synchronize_device(mesh_device)
        for a, b, s in copies:
            done_seen[b, a] += reps
        med, mo = statistics.median(us), statistics.median(off)
        add = 100.0 * (med - mo) / mo
        key = label.replace(" ", "_").replace(",", "")
        verdict = "PASS" if add < 5.0 else "OVER"
        mx = max(sum(1 for a_, _, _ in copies if a_ == a) for a, _, _ in copies)
        logger.info(
            f"FILL {verdict} {label}: {len(copies)} copies (busiest source {mx}), {per_call[copies[0]]} pieces a call, "
            f"{n_calls} call(s) a copy x {reps}: combine {med:.1f} us (min {min(us):.1f}, max {max(us):.1f}) vs "
            f"{mo:.1f} off in the same window (min {min(off):.1f}): {add:+.2f} %"
        )
        metrics.record(f"fill_combine_us_{key}", round(med, 1))
        metrics.record(f"fill_added_pct_{key}", round(add, 2))
        dw = done_words()
        split = dw[:, n:] - before[:, n:]
        p_in, p_tl = int(split[:, :8].sum()), int(split[:, 8:].sum())
        logger.info(
            f"FILL {label}: pieces sent inside the combine's main loop {p_in}, after it {p_tl} "
            f"({100 * p_in / max(1, p_in + p_tl):.1f} % hidden)"
        )
        metrics.record(f"fill_hidden_pct_{key}", round(100 * p_in / max(1, p_in + p_tl), 1))
        check(torch.equal(dw[:, :n], done_seen), f"{label}: the done words count every completed copy")
        if not torch.equal(dw[:, :n], done_seen):
            logger.info(f"FILL {label}: done words differ at {(dw[:, :n] != done_seen).nonzero().tolist()[:16]}")
        if check_bytes:
            verify(copies, label)

    def placement(kind: str, seed: int):
        gg = torch.Generator().manual_seed(seed)
        out = []
        for b in range(n):
            r, c = divmod(b, C)
            if kind == "neighbour":
                a = r * C + (c + 1) % C
            else:
                while True:
                    a = int(torch.randint(0, n, (1,), generator=gg))
                    if a != b:
                        break
            out.append((a, b, int(torch.randint(0, epc, (1,), generator=gg))))
        return out

    # timing sweep (bytes checked once a placement below)
    for bd in BUDGETS:
        for cap in CAPS:
            if fails:
                break
            tag = ("whole" if bd == 0 else f"1of{bd}") + f" cap{cap}"
            case(f"one pair {tag}", [(1, 0, 3 + bd % epc)], bd, cap, check_bytes=False)
            case(f"neighbour 32 {tag}", placement("neighbour", 530 + bd), bd, cap, check_bytes=False)
            case(f"random 32 {tag}", placement("random", 540 + bd), bd, cap, check_bytes=False)
    for kind in ("one pair", "neighbour 32", "random 32"):
        if fails:
            break
        cp = [(1, 0, 6)] if kind == "one pair" else placement(kind.split()[0], 600 + len(kind))
        case(f"{kind} verified", cp, VERIFY[0], VERIFY[1], check_bytes=True)
    check(identical(), "(1) every carrier output with copies bit-identical to the fill path off")

    # ---- (2) back to back
    if not fails:
        copies = placement("random", 77)
        CAP[0] = VERIFY[1]
        per = {s: -(-total[s] // VERIFY[0]) for s in range(epc)}
        t0 = time.perf_counter()
        for i in range(B2B):
            k = i % VERIFY[0]
            jobs = []
            for a, b, s in copies:
                p0, p1 = min(k * per[s], total[s]), min((k + 1) * per[s], total[s])
                if p0 < p1:
                    jobs.append((a, b, s, p0, p1, p1 == total[s]))
            mb.set_combine_fill(res, True, fil.w_gu, fil.w_d, tabs.runs, jobs, cap=CAP[0])
            fold(car(x))
        ttnn.synchronize_device(mesh_device)
        for a, b, s in copies:
            done_seen[b, a] += B2B // VERIFY[0]
        logger.info(f"FILL (2) {B2B} calls back to back in {time.perf_counter() - t0:.2f} s: no hang")
        check(identical(), f"(2) {B2B} back-to-back calls: outputs bit-identical")
        verify(copies, "(2) back to back")
        check(torch.equal(done_words()[:, :n], done_seen), "(2) the done words count every completed copy")
    # ---- (3) step 2: set_replicas under MOE_FILL=piggyback (tt/expert_fill.py): the queue drives the carrier's calls
    if not fails:
        from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_host

        _settings(monkeypatch, MOE_FILL="piggyback")
        q = ad.fill_queue(res)
        spc = fil.spc

        def fil_call():
            y = fil(x)
            counts = torch.stack(
                [ttnn.to_torch(d).reshape(-1).long() for d in ttnn.get_device_tensors(fil._last_head.counts)]
            )
            yh = own_rows_to_host(mesh_device, y).float()
            ttnn.deallocate(y)
            fil._last_head = None
            spare = int(sum(counts[d, d * spc + epc] for d in range(n)))
            return yh, spare

        y0, sp0 = fil_call()
        check(sp0 == 0, f"(3) layer {FILLED} with nothing mapped: no row in a spare slot ({sp0})")

        def run_until_flip(table, label: str, own_call_at: int = 3) -> None:
            for i in range(64):
                if i == own_call_at:  # (the filled layer runs while its copy is pending: it carries none of it)
                    y_, sp_ = fil_call()
                    check(torch.equal(y_, y0), f"{label}: layer {FILLED} while pending: output unchanged")
                fold(car(x))
                if q.pending(fil) == 0:
                    logger.info(f"FILL {label}: mapped after {i + 1} carrier calls ({q.stats})")
                    break
            check(q.pending(fil) == 0 and fil.copies == sorted(table), f"{label}: the table mapped after the copies")

        where = torch.empty(len(fil.perm), dtype=torch.long)
        where[fil.perm] = torch.arange(len(fil.perm))
        s0 = 2
        table = [(int(fil.perm[((k + 1) % n) * epc + s0]), k) for k in range(n)]
        made = fil.set_replicas(table)
        check(
            made == n and fil.copies == [] and q.pending(fil) == 1,
            f"(3) set_replicas queued {made} copies, none mapped",
        )
        run_until_flip(table, "(3) neighbour table")
        ttnn.synchronize_device(mesh_device)
        verify([(int(where[x_]) // epc, k, int(where[x_]) % epc) for x_, k in table], "(3) piggyback set_replicas")
        y1, sp1 = fil_call()
        check(torch.equal(y1, y0), f"(3) layer {FILLED} with the table mapped: output bit-identical to none mapped")
        check(sp1 > 0, f"(3) rows reached the filled spare slots ({sp1})")
        # half the chips change (unmapped at once), then the table emptied
        table2 = [(x_, k) if k % 2 else (int(fil.perm[((k + 1) % n) * epc + s0 + 1]), k) for x_, k in table]
        made = fil.set_replicas(table2)
        check(
            made == n // 2 and fil.copies == sorted(t for t in table if t[1] % 2),
            f"(3) a changed table: {made} copies queued, the changed chips unmapped at once",
        )
        run_until_flip(table2, "(3) half changed")
        ttnn.synchronize_device(mesh_device)
        verify([(int(where[x_]) // epc, k, int(where[x_]) % epc) for x_, k in table2 if k % 2 == 0], "(3) half changed")
        y2, sp2 = fil_call()
        check(torch.equal(y2, y0) and sp2 > 0, f"(3) half changed: output bit-identical ({sp2} rows in spare slots)")
        check(fil.set_replicas([]) == 0 and fil.copies == [], "(3) the table emptied: unmapped, nothing queued")
        y3, sp3 = fil_call()
        check(torch.equal(y3, y0) and sp3 == 0, "(3) emptied: output bit-identical, no spare row")
        check(identical(), "(3) every carrier output while the queue ran bit-identical to the fill path off")
    mb.set_combine_fill(res, False)
    metrics.record("fill_checks_failed", len(fails))
    for t in [ref.get("y"), marker[0], x, lead]:
        if t is not None:
            ttnn.deallocate(t)
    for m in (car, fil):
        for t in (m.gidx, m.w_gu, m.w_d):
            ttnn.deallocate(t)
    for _, r_, _ in list(ad._BLOCKS.values()):
        r_.release()
    ad._BLOCKS.clear()
    ad._SLOT_TABLES.clear()
    ad._FILLQ.clear()
    ad._MAIN_LAYERS.clear()
    assert not fails, f"{len(fails)} MOE_FILL checks failed: {fails}"
