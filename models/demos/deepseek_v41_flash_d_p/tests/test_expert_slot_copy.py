# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.47 gate 1 (owner 2026-11-12): ttnn.bringup.moe_block_main.expert_slot_copy, the runtime fill of the spare
(13th) expert slot. One MoE layer (LAYER) on the spec's mesh with MOE_REPLICAS=runtime: its real routed weights (the
off path's weight cache, re-laid on the device into 13 slots a chip, the spare slot empty).

(0) host: the run tables' slot geometry (tt/expert_slots.py SlotLayout, from the plan's facts) == the flat expert's own
    layout (flat_weight_regions + bank_host_tensor on 16 x 16 block ids, 12- and 13-slot plans): every slot's units
    hold exactly that slot's tiles, in the same order for every slot; the load-time regroup emulated tile for tile on
    the ids == the 13-slot layout; every fill table writes only inside the spare slot (also on the device's read-back
    tables).
(1) one pair, then pairs on a row, a column and a diagonal, 1 / 2 / 4 hops, odd and even source slots: chip A's home
    slot s -> chip B's spare slot. Bit-exact: B's spare slot read back (to_torch per device) == A's slot s tile for
    tile; every other tile of A and B unchanged (exact per-tile integer checksums of the decoded values against the
    state before). Timed: median of 20 calls after 2 warm-up calls (op + synchronize), ms and GB/s.
(2) all chips at once (each chip receives one copy from a random other chip's random home slot; sources repeat): the
    same checks on every chip, timed the same way.
Metrics: slot_copy_checks_failed (== 0), slot_copy_ms_* / slot_copy_gbps_*.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_expert_slot_copy.py
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

S = spec()
LAYER = 20
REPS, WARM = 20, 2
MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])
# (name, dr, dc, source slot): pairs relative to chip (r0, c0) = (i % R, 0) on the mesh (wrapping)
PAIRS = [
    ("row 1 hop, even slot", 0, 1, 0),
    ("row 2 hops, odd slot", 0, 2, 5),
    ("column 1 hop, odd slot", 1, 0, 7),
    ("column 2 hops, even slot", 2, 0, 2),
    ("column 4 hops, odd slot", 4, 0, 11),
    ("diagonal 2 hops, even slot", 1, 1, 4),
    ("diagonal 4 hops, odd slot", 2, 2, 9),
]


def _settings(monkeypatch, **kv):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    for k, v in kv.items():
        monkeypatch.setenv(f"DS41_{k}", v)
    settings.thaw()
    return settings


def _ints(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32)


def _decode(w, d: int) -> torch.Tensor:
    """Chip d's tensor decoded (to_torch of that device), [rows, banks * 32]."""
    t = ttnn.to_torch(ttnn.get_device_tensors(w)[d])
    return t.reshape(-1, t.shape[-1])


def _checksums(x: torch.Tensor, banks: int) -> torch.Tensor:
    """Per tile (index t x banks + bank) two exact integer sums of the decoded values' bits: [tiles, 2] int64."""
    out = []
    wv = torch.arange(1, 1025, dtype=torch.int64)
    for r0 in range(0, x.shape[0], 32 * 2048):
        xi = _ints(x[r0 : r0 + 32 * 2048]).to(torch.int64)
        t = xi.reshape(-1, 32, banks, 32).permute(0, 2, 1, 3).reshape(-1, 1024)
        out.append(torch.stack([t.sum(-1), (t * wv).sum(-1)], dim=-1))
    return torch.cat(out)


def _slot_tiles(x: torch.Tensor, units, w: int, tb: int) -> torch.Tensor:
    """The tiles of weight w's units (slot order), [n, 32, 32]."""
    parts = []
    for w_, b, o, n in units:
        if w_ == w:
            parts.append(x[32 * (o // tb) : 32 * ((o + n) // tb), 32 * b : 32 * b + 32].reshape(-1, 32, 32))
    return torch.cat(parts)


def _slot_index(units, w: int, tb: int, banks: int) -> torch.Tensor:
    """The tile indices (t x banks + bank) of weight w's units."""
    return torch.cat([torch.arange(o // tb, (o + n) // tb) * banks + b for w_, b, o, n in units if w_ == w])


def _layout_ok(lay_spc, lay_epc, check) -> None:
    """(0): the slot geometry against the flat expert's own layout, on 16 x 16 block ids."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad

    fw = ad._flat_weights_module()
    H, I, banks = lay_spc.H, lay_spc.I, lay_spc.banks
    nt = lambda r, c: (r // 16) * (c // 16)  # noqa: E731
    per = 2 * nt(I, H) + nt(H, I)

    def ids(base, r, c):
        g = torch.arange(base, base + nt(r, c), dtype=torch.int32).reshape(r // 16, 1, c // 16, 1)
        return g.expand(r // 16, 16, c // 16, 16).reshape(r, c)

    def laid(E, lay):
        W_l = [
            (ids(1 + s * per, I, H), ids(1 + s * per + nt(I, H), I, H), ids(1 + s * per + 2 * nt(I, H), H, I))
            for s in range(E)
        ]
        regions, dregs = fw.flat_weight_regions(W_l, lay, nk=True)
        return [fw.bank_host_tensor(regions, lay["banks"]), fw.bank_host_tensor(dregs, lay["banks"])]

    t0 = time.time()
    for sl in (lay_spc, lay_epc):
        X = laid(sl.E, sl.lay)
        tb = 1  # (byte offsets in tile units below: units' offsets / sl.tb)
        ok = True
        ref = None
        for s in range(sl.E):
            units = [(w, b, o // sl.tb * tb, n // sl.tb * tb) for w, b, o, n in sl.units(s)]
            got = []
            for w in (0, 1):
                g = _slot_tiles(X[w], units, w, tb)
                q = g.reshape(-1, 2, 16, 2, 16)[:, :, 0, :, 0].reshape(-1, 4).long()  # (each 16 x 16 block's id)
                ok &= bool(((q - 1) // per == s).all())
                got.append((q - 1) % per)
                n_all = int(((X[w].long() - 1) // per == s).logical_and(X[w] > 0).sum())
                ok &= n_all == g.numel()
            key = torch.cat(got)
            if ref is None:
                ref = key
            ok &= torch.equal(key, ref)
        check(ok, f"(0) {sl.E}-slot layout: every slot's units are exactly its tiles, in one order for all slots")
        if sl is lay_spc:
            X13 = X
        else:
            X12 = X
    from models.demos.deepseek_v41_flash_d_p.tt.expert_slots import copy_runs, regroup_runs

    tb = lay_spc.tb
    emu = [torch.zeros_like(X13[0]), torch.zeros_like(X13[1])]
    for w, sb, so, db, do, n in regroup_runs(lay_epc, lay_spc, lay_epc.E):
        src = X12[w][32 * (so // tb) : 32 * ((so + n) // tb), 32 * sb : 32 * sb + 32]
        emu[w][32 * (do // tb) : 32 * ((do + n) // tb), 32 * db : 32 * db + 32] = src
    ok = True
    for w in (0, 1):
        want = torch.where(((X13[w].long() - 1) // per) == lay_epc.E, torch.zeros_like(X13[w]), X13[w])
        ok &= torch.equal(emu[w], want)
    check(ok, "(0) the load-time regroup (12 -> 13 slots), emulated on the ids == the 13-slot layout, spare slot empty")
    # a fill emulated: slot 5 onto the spare slot
    emu = [X13[0].clone(), X13[1].clone()]
    for w, sb, so, db, do, n in copy_runs(lay_spc, 5, lay_spc, lay_epc.E):
        emu[w][32 * (do // tb) : 32 * ((do + n) // tb), 32 * db : 32 * db + 32] = X13[w][
            32 * (so // tb) : 32 * ((so + n) // tb), 32 * sb : 32 * sb + 32
        ]
    ok = True
    for w in (0, 1):
        q = X13[w].long()
        spare_m = ((q - 1) // per) == lay_epc.E
        want = torch.where(spare_m, q - (lay_epc.E - 5) * per, q)
        ok &= torch.equal(emu[w].long(), want)
    check(
        ok, "(0) a fill (slot 5 -> the spare slot), emulated on the ids: the spare slot == slot 5, nothing else moved"
    )
    del emu, X13, X12
    logger.info(f"SLOT_COPY (0) layout checks {time.time() - t0:.1f} s")


@pytest.mark.timeout(5400)
@BOX
@MESH
def test_expert_slot_copy(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir

    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        logger.info(f"SLOT_COPY {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="runtime")
    t0 = time.time()
    moe = ad.TtMoEBlockMainLayer(mesh_device, ck, cfg, LAYER, chunk, cache_dir=expert_cache_dir(S), own_rows=True)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"SLOT_COPY layer {LAYER} loaded (13 slots, the spare empty) in {time.time() - t0:.1f} s")
    tabs = moe.slot_tables()
    sl = tabs.lay_spc
    tb, banks, epc = sl.tb, sl.banks, moe.epc
    R, C = tuple(mesh_device.shape)
    n = R * C
    mb, res, W = ad._mb(), moe.block().resources, (moe.w_gu, moe.w_d)
    slot_bytes = sl.slot_bytes()
    logger.info(
        f"SLOT_COPY a slot: {slot_bytes / 2**20:.2f} MiB in {len(tabs.host[0])} runs "
        f"({sum(1 for r in tabs.host[0] if r[0] == 0)} gate/up, {sum(1 for r in tabs.host[0] if r[0] == 1)} down); "
        f"chunk {mb.expert_slot_copy_chunk_bytes(res)} B"
    )

    # ---- (0) host layout
    _layout_ok(sl, tabs.lay_epc, check)
    tabs.verify()  # (the device's tables read back == the host's, fills only inside the spare slot; asserts)

    spare = sl.units(epc)
    sums = {}  # chip -> [checksums gu, checksums d]

    def state(d):
        return [_checksums(_decode(w, d), banks) for w in W]

    def run(jobs):
        mb.expert_slot_copy(W[0], W[1], W[0], W[1], tabs.runs, res, jobs)

    def timed(jobs) -> float:
        for _ in range(WARM):
            run(jobs)
        ttnn.synchronize_device(mesh_device)
        ts = []
        for _ in range(REPS):
            t1 = time.perf_counter()
            run(jobs)
            ttnn.synchronize_device(mesh_device)
            ts.append(1e3 * (time.perf_counter() - t1))
        return sorted(ts)[len(ts) // 2]

    def verify(jobs, label) -> None:
        """Every destination's spare slot == its source's home slot; every other tile of every chip involved as it was."""
        dst_of = {b: (a, s) for a, b, s in jobs}
        chips = sorted({a for a, _, _ in jobs} | set(dst_of))
        homes, spares = {}, {}
        for d in chips:
            for w in (0, 1):
                x = _decode(W[w], d)
                cs = _checksums(x, banks)
                keep = torch.ones(cs.shape[0], dtype=torch.bool)
                if d in dst_of:
                    keep[_slot_index(spare, w, tb, banks)] = False
                    spares[(d, w)] = _slot_tiles(x, spare, w, tb)
                for a, b, s in jobs:
                    if a == d:
                        homes[(d, s, w)] = _slot_tiles(x, sl.units(s), w, tb)
                ok = torch.equal(cs[keep], sums[d][w][keep])
                check(ok, f"{label}: chip {d} {('w_gu', 'w_d')[w]}: every tile outside its spare slot unchanged")
                sums[d][w] = cs
                del x
        for b, (a, s) in dst_of.items():
            for w in (0, 1):
                ok = torch.equal(_ints(spares[(b, w)]), _ints(homes[(a, s, w)]))
                if not ok:
                    bad = (_ints(spares[(b, w)]) != _ints(homes[(a, s, w)])).reshape(-1, 1024).any(-1)
                    logger.info(
                        f"SLOT_COPY {label}: {int(bad.sum())} of {bad.numel()} tiles differ, first "
                        f"{bad.nonzero()[:8].flatten().tolist()}"
                    )
                check(ok, f"{label}: chip {b}'s spare slot == chip {a}'s slot {s} ({('w_gu', 'w_d')[w]}, bit-exact)")

    # ---- (1) pairs
    for i, (name, dr, dc, s) in enumerate(PAIRS):
        r0, c0 = i % R, 0
        a, b = r0 * C + c0, ((r0 + dr) % R) * C + (c0 + dc) % C
        if a == b or s >= epc:
            logger.info(f"SLOT_COPY skip {name} (mesh {R}x{C})")
            continue
        for d in (a, b):
            if d not in sums:
                sums[d] = state(d)
        jobs = [(a, b, tabs.fill_table(s))]
        run(jobs)
        ttnn.synchronize_device(mesh_device)
        verify(jobs, f"(1) {name} {a}->{b}")
        if fails:
            break  # (the owner's order: stop at the first failure)
        ms = timed(jobs)
        gbps = slot_bytes / ms / 1e6
        logger.info(f"SLOT_COPY (1) {name} chip {a} -> {b}, slot {s}: {ms:.3f} ms, {gbps:.2f} GB/s (median of {REPS})")
        key = name.split(",")[0].replace(" ", "_")
        metrics.record(f"slot_copy_ms_{key}", round(ms, 4))
        metrics.record(f"slot_copy_gbps_{key}", round(gbps, 3))

    # ---- (2) every chip at once
    if not fails:
        g = torch.Generator().manual_seed(47)
        jobs = []
        for b in range(n):
            a = int(torch.randint(0, n - 1, (1,), generator=g))
            a = a + (a >= b)
            jobs.append((a, b, tabs.fill_table(int(torch.randint(0, epc, (1,), generator=g)))))
        logger.info(f"SLOT_COPY (2) jobs (src, dst, slot): {jobs}")
        for d in range(n):
            if d not in sums:
                sums[d] = state(d)
        run(jobs)
        ttnn.synchronize_device(mesh_device)
        verify(jobs, f"(2) all {n} chips")
        if not fails:
            ms = timed(jobs)
            logger.info(
                f"SLOT_COPY (2) {n} copies at once: {ms:.3f} ms, {n * slot_bytes / ms / 1e6:.2f} GB/s total, "
                f"{slot_bytes / ms / 1e6:.2f} GB/s a copy (median of {REPS}); busiest source sends "
                f"{max(sum(1 for a, _, _ in jobs if a == c) for c in range(n))} copies"
            )
            metrics.record("slot_copy_ms_all", round(ms, 4))
            metrics.record("slot_copy_gbps_all_total", round(n * slot_bytes / ms / 1e6, 3))
    metrics.record("slot_copy_slot_mib", round(slot_bytes / 2**20, 3))
    metrics.record("slot_copy_checks_failed", len(fails))
    for t in (moe.gidx, moe.w_gu, moe.w_d):
        ttnn.deallocate(t)
    for _, r_, _ in list(ad._BLOCKS.values()):
        r_.release()
    ad._BLOCKS.clear()
    ad._SLOT_TABLES.clear()
    assert not fails, f"{len(fails)} expert_slot_copy checks failed: {fails}"
