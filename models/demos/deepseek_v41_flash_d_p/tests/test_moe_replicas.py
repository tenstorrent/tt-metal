# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.46 / P.47 (owner 2026-11-12): MOE_REPLICAS=runtime - one more expert slot a chip, empty at load, filled at
runtime by a chip-to-chip copy (expert_slot_copy) from the expert's home slot, the router splitting the expert's tokens
over its holders (tt/moe_block_main_adapter.py set_replicas; the fused head's slot map: router_rows.hpp slot_of). One
MoE layer (LAYER) on the spec's mesh, one chunk of a fixed random input (OWN_ROWS rows), MOE_BLOCK_SRC=main:

(a) runtime, nothing filled (the spare slots empty, the map empty) vs off: the layer output bit-identical; every
    expert's count at its own slot; every home slot's tiles on the device (re-laid at load from the off path's cache)
    == the off path's tiles of that slot, on the first and the last chip.
(b) set_replicas(this input's hottest experts, the hottest with 3 copies, on the least-loaded chips), then the same
    layer object run again (fill, then the map written into its live head weights) vs off: the same routing (every
    token's experts and gate-weight bits, read back from the head's rows, slot -> expert); the output bit-identical
    (max |diff| 0). The routed part of both vs the CPU (exact fp4 experts) on 24 tokens.
(c) every copy's tiles on the device equal its original's (decoded values, slot for slot).
(d) each replicated expert's tokens split over its holders exactly as the host's hash of each token says (per-slot
    counts read back from the head; each token's slot from its routing row), each holder's share printed.
(e) the table changed at runtime (other experts on part of the chips, a chip unmapped) and then emptied: each run
    bit-identical to off; an unchanged table enqueues nothing.

Metric replicas_checks_failed (gate: == 0): the failed checks.

    PYTHONPATH=$PWD DS41_MOE_REPLICAS=runtime scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_replicas.py
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
_TIME_CALLS, _TIMES = [0], []
SAMPLE = list(range(0, 16)) + list(range(2560, 2568))  # (tokens checked against the CPU)
MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])


def _settings(monkeypatch, **kv):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    for k, v in kv.items():
        monkeypatch.setenv(f"DS41_{k}", v)
    settings.thaw()
    return settings


# The owner's oracle table (P.46 current.json, learned on chunk 10 of the novel, 32 copies a layer) for layers 0 and 20,
# as data: (checkpoint expert, chip r C + c). The model reads no table file (P.47); tests drive set_replicas.
ORACLE = {
    0: [(355, 0), (209, 1), (370, 2), (286, 3), (357, 4), (264, 5), (211, 7), (62, 8), (369, 9), (62, 10), (67, 11),
        (351, 12), (64, 14), (61, 15), (137, 16), (204, 17), (210, 18), (366, 19), (135, 20), (204, 21), (205, 22),
        (208, 23), (210, 24), (204, 25), (215, 26), (210, 27), (63, 28), (209, 29), (204, 30), (353, 31)],
    20: [(204, 0), (204, 1), (246, 2), (204, 3), (204, 4), (226, 5), (96, 6), (217, 7), (241, 8), (246, 9), (125, 10),
         (297, 11), (120, 12), (383, 13), (204, 14), (221, 15), (192, 16), (124, 17), (124, 18), (352, 19), (30, 20),
         (131, 21), (142, 22), (236, 23), (216, 24), (225, 25), (223, 26), (60, 27), (238, 28), (204, 29), (134, 30),
         (139, 31)],
}  # fmt: skip


def _build(mesh, ck, cfg, chunk, cache_dir):
    """LAYER's MoE layer under the current settings (load timed)."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad

    t0 = time.time()
    moe = ad.TtMoEBlockMainLayer(mesh, ck, cfg, LAYER, chunk, cache_dir=cache_dir, own_rows=True)
    ttnn.synchronize_device(mesh)
    logger.info(
        f"REPLICAS layer {LAYER} built (MOE_REPLICAS={'runtime' if moe.replicas else 'off'}) in {time.time() - t0:.1f} s"
    )
    return moe


def _call(moe, mesh, cfg, x_host):
    """Run the chunk on moe: (y host, shared expert host, counts [chips, NG], routing rows [chips, pages, 32, 2K]
    (uint16 values as int64))."""
    from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_device, own_rows_to_host

    x = own_rows_to_device(mesh, x_host, ttnn.bfloat16)
    y = moe(x)
    ttnn.synchronize_device(mesh)
    if _TIME_CALLS[0]:  # (the layer's warm call, synced wall: median of _TIME_CALLS calls)
        ts = []
        for _ in range(_TIME_CALLS[0]):
            ttnn.deallocate(y)
            t0 = time.perf_counter()
            y = moe(x)
            ttnn.synchronize_device(mesh)
            ts.append(1e3 * (time.perf_counter() - t0))
        _TIMES.append(sorted(ts)[len(ts) // 2])
    head = moe._last_head
    assert head is not None, "the OWN_ROWS path's fused head outputs"
    y_host = own_rows_to_host(mesh, y).float()
    se_host = own_rows_to_host(mesh, head.se).float()
    n = mesh.get_num_devices()
    counts = torch.stack([ttnn.to_torch(d).reshape(-1).long() for d in ttnn.get_device_tensors(head.counts)])
    K = int(cfg.topk_experts)
    rows = torch.stack(
        [ttnn.to_torch(d).reshape(-1).to(torch.int64) & 0xFFFF for d in ttnn.get_device_tensors(head.rows)]
    ).reshape(n, -1, 32, 2 * K)
    ttnn.deallocate(x)
    ttnn.deallocate(y)
    moe._last_head = None
    return y_host, se_host, counts, rows


def _release(moe, keep_weights: bool = False):
    """The block released (one MoE block on the mesh at a time - DS41-202); the weights too unless keep_weights."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad

    moe.head_w = None
    if not keep_weights:
        for t in (moe.gidx, moe.w_gu, moe.w_d):
            ttnn.deallocate(t)
    for _, res, _ in list(ad._BLOCKS.values()):
        res.release()
    ad._BLOCKS.clear()
    ad._INPUTS.clear()
    ad._FEED.clear()
    ad._SLOT_TABLES.clear()


def _decode(w, d):
    t = ttnn.to_torch(ttnn.get_device_tensors(w)[d])
    return t.reshape(-1, t.shape[-1])


def _slot_data(x, units, tb):
    """A slot's tiles (both weights, units order) from decoded [w_gu, w_d] of one chip."""
    out = []
    for w_, b, o, n in units:
        out.append(x[w_][32 * (o // tb) : 32 * ((o + n) // tb), 32 * b : 32 * b + 32].reshape(-1, 32, 32))
    return torch.cat(out)


def _slot_totals(counts, spc):
    """Per slot (global), the head's planner count: chip d holds its own slots' [d spc, (d + 1) spc)."""
    return torch.cat([counts[d, d * spc : (d + 1) * spc] for d in range(counts.shape[0])])


def _cpu_routed(ck, cfg, perm, rows_tok, x_host, K):
    """The routed sum on the CPU (exact fp4 experts) for SAMPLE, from routing rows given as router candidates."""
    from models.demos.deepseek_v41_flash_d_p.reference.weights import fp4_dequant

    lim = float(cfg.swiglu_limit)
    wts = (rows_tok[:, K:].to(torch.int32) << 16).view(torch.float32)
    ref = torch.zeros(len(SAMPLE), int(cfg.dim))
    cache_e: dict = {}
    for i, t in enumerate(SAMPLE):
        xt = x_host[t].float()
        for q in range(K):
            e_ck = int(perm[int(rows_tok[t, q])])
            if e_ck not in cache_e:
                st = ck.expert(LAYER, e_ck)
                cache_e[e_ck] = {k: fp4_dequant(*st[k]) for k in ("w1", "w2", "w3")}
            W = cache_e[e_ck]
            gate = (W["w1"] @ xt).clamp(max=lim)
            up = (W["w3"] @ xt).clamp(min=-lim, max=lim)
            ref[i] += float(wts[t, q]) * (W["w2"] @ (torch.nn.functional.silu(gate) * up))
    return ref


P46_BUSIEST = {
    0: 1.12,
    20: (5.19, 1.19),
}  # (P.46's measured busiest chip / mean on golden chunk 10: L0 table; L20 off, table)


@pytest.mark.timeout(5400)
@BOX
@MESH
def test_moe_replicas(mesh_device, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.expert_slots import SlotLayout, tile_bytes
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir

    fails: list[str] = []

    def check(ok: bool, what: str) -> None:
        logger.info(f"REPLICAS {'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            fails.append(what)

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    n = mesh_device.get_num_devices()
    epc = int(cfg.n_experts) // n
    spc = epc + 1
    K = int(cfg.topk_experts)
    H, I = int(cfg.dim), int(cfg.inter)
    g = torch.Generator().manual_seed(46)
    x_host = torch.randn(chunk, H, generator=g).bfloat16()
    cache_dir = expert_cache_dir(S)
    probe = (0, n - 1)  # (chips whose home slots are compared tile for tile in (a))

    # ---- off
    _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="off")
    off = _build(mesh_device, ck, cfg, chunk, cache_dir)
    y_off, se_off, c_off, r_off = _call(off, mesh_device, cfg, x_host)
    loops = max(1, int(off.block().loops))
    perm = off.perm
    tb = tile_bytes(off.dtypes.routed_weights)
    lay_off = SlotLayout(off.lay, epc, H, I, tb)
    off_slots = {d: [_slot_data([_decode(off.w_gu, d), _decode(off.w_d, d)], lay_off.units(s), tb) for s in range(epc)]
                 for d in probe}  # fmt: skip
    off_layout = ad._weight_layout(off.lay)
    _release(off)
    load = _slot_totals(c_off, epc)  # (block slot b = router candidate b: checkpoint expert perm[b])
    assert int(load.sum()) == chunk * K, f"off: counts sum {int(load.sum())}, expected {chunk * K}"
    chip_load = load.reshape(n, epc).sum(dim=1)
    own = (torch.arange(n * epc) // epc) * spc + torch.arange(n * epc) % epc

    # ---- (a) runtime, nothing filled
    _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="runtime")
    rt = _build(mesh_device, ck, cfg, chunk, cache_dir)
    tabs = rt.slot_tables()
    check(ad._weight_layout(rt.lay_index) == off_layout, "(a) the runtime layer's 12-slot plan == the off path's")
    y_e, _, c_e, _ = _call(rt, mesh_device, cfg, x_host)
    d_a = (y_e - y_off).abs().max().item()
    check(torch.equal(y_e, y_off), f"(a) nothing filled vs off bit-identical (max |diff| {d_a})")
    check(torch.equal(_slot_totals(c_e, spc)[own], load), "(a) nothing filled: every expert's count at its own slot")
    check(int(_slot_totals(c_e, spc).sum()) == int(load.sum()), "(a) nothing filled: no row in a spare slot")
    dec = {}
    for d in probe:
        dec[d] = [_decode(rt.w_gu, d), _decode(rt.w_d, d)]
        bad = [s for s in range(epc) if not torch.equal(_slot_data(dec[d], tabs.lay_spc.units(s), tb), off_slots[d][s])]
        check(not bad, f"(a) chip {d}: the 13-slot home slots == the off path's slots (differ: {bad})")
    del off_slots, dec

    # ---- (b) fill at runtime, then run the same layer object
    hot = torch.argsort(load, descending=True)[:8].tolist()  # (candidates)
    free = torch.argsort(chip_load).tolist()  # (least loaded first)
    copies = []
    for i, b in enumerate(hot):
        for _ in range(3 if i == 0 else 1):
            k = next(k for k in free if k != b // epc)
            free.remove(k)
            copies.append((int(perm[b]), k))
    logger.info(f"REPLICAS copies (checkpoint expert, chip): {copies}")
    t0 = time.perf_counter()
    made = rt.set_replicas(copies)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"REPLICAS (b) set_replicas: {made} copies + the slot map in {1e3 * (time.perf_counter() - t0):.2f} ms")
    check(made == len(copies), f"(b) set_replicas made {made} copies of {len(copies)}")
    y_rep, se_rep, c_rep, r_rep = _call(rt, mesh_device, cfg, x_host)

    where = torch.empty(len(perm), dtype=torch.long)
    where[perm] = torch.arange(len(perm))
    slot_cand = torch.full((n * spc,), -1, dtype=torch.long)
    slot_cand[own] = torch.arange(n * epc)
    for x, k in copies:
        slot_cand[k * spc + epc] = where[x]
    ids_off, ids_rep = r_off[..., :K], r_rep[..., :K]
    same_w = torch.equal(r_off[..., K:], r_rep[..., K:])
    same_ids = torch.equal(slot_cand[ids_rep], ids_off)
    check(same_ids and same_w, f"(b) routing: same experts {same_ids}, same gate weight bits {same_w}")
    moved_any = bool(torch.isin(ids_rep, torch.tensor([k * spc + epc for _, k in copies])).any())
    check(moved_any, "(b) rows reached the filled spare slots")
    d_b = (y_rep - y_off).abs()
    logger.info(f"REPLICAS (b) copies vs off: max |diff| {d_b.max().item()}, {int((d_b > 0).sum())} values differ")
    metrics.record("replicas_max_abs_diff", float(d_b.max().item()))
    check(d_b.max().item() == 0.0, f"(b) filled vs off bit-identical (max |diff| {d_b.max().item()})")
    rows_off = torch.cat([ids_off, r_off[..., K:]], dim=-1).reshape(-1, 2 * K)
    ref = _cpu_routed(ck, cfg, perm, rows_off, x_host, K)
    for nm, y_, se_ in (("off", y_off, se_off), ("copies", y_rep, se_rep)):
        got = (y_ - se_)[SAMPLE]
        rl = ((got - ref).norm() / ref.norm()).item()
        logger.info(f"REPLICAS (b) routed vs CPU (exact fp4 experts), {len(SAMPLE)} tokens: {nm} rel L2 {rl:.4f}")
        metrics.record(f"replicas_routed_rel_cpu_{nm}", round(rl, 5))

    # ---- (c) the copies' tiles on the device equal their originals'
    cache: dict = {}

    def chip_w(d):
        if d not in cache:
            cache[d] = [_decode(rt.w_gu, d), _decode(rt.w_d, d)]
        return cache[d]

    bad = []
    for x, k in copies:
        od, ole = divmod(int(where[x]), epc)
        if not torch.equal(
            _slot_data(chip_w(k), tabs.lay_spc.units(epc), tb), _slot_data(chip_w(od), tabs.lay_spc.units(ole), tb)
        ):
            bad.append((x, k))
    check(not bad, f"(c) {len(copies)} copies' tiles equal their originals' (differ: {bad})")
    del cache

    # ---- (d) the split: device counts and slots == the host's hash of each token, per holder
    tot = _slot_totals(c_rep, spc)
    check(
        torch.equal(tot, torch.bincount(ids_rep.reshape(-1), minlength=n * spc)),
        "(d) the head's per-slot counts == its routing rows' slots",
    )
    pages = ids_off.shape[1]
    rb = pages // loops
    keys = (torch.arange(n).reshape(n, 1, 1) << 16) + (torch.arange(pages) % rb).reshape(1, pages, 1) * 32
    keys = (keys + torch.arange(32).reshape(1, 1, 32)).unsqueeze(-1).expand(n, pages, 32, K)
    for b in sorted({int(where[x]) for x, _ in copies}):
        hs = [int(own[b])] + sorted(k * spc + epc for x, k in copies if int(where[x]) == b)
        sel = ids_off == b
        hold = torch.tensor([ad.holder_of(int(kk), len(hs) - 1) for kk in keys[sel].tolist()], dtype=torch.long)
        expect = torch.bincount(hold, minlength=len(hs))
        got = torch.stack([tot[h] for h in hs])
        logger.info(
            f"REPLICAS (d) expert {int(perm[b])} (candidate {b}): {int(sel.sum())} tokens over {len(hs)} holders "
            f"{hs}: {got.tolist()} (expected {expect.tolist()}, even {int(sel.sum()) / len(hs):.1f})"
        )
        check(torch.equal(got, expect), f"(d) expert {int(perm[b])}: holder counts as the hash says")
        check(torch.equal(ids_rep[sel], torch.tensor(hs)[hold]), f"(d) expert {int(perm[b])}: each token at its holder")
    rep_chip = torch.stack([tot[d * spc : (d + 1) * spc].sum() for d in range(n)])
    b_off = chip_load.max().item() / chip_load.float().mean().item()
    b_rep = rep_chip.max().item() / rep_chip.float().mean().item()
    logger.info(f"REPLICAS busiest chip / mean (this input, layer {LAYER}): off {b_off:.3f}, copies {b_rep:.3f}")
    metrics.record("replicas_busiest_over_mean_off", round(b_off, 4))
    metrics.record("replicas_busiest_over_mean_runtime", round(b_rep, 4))

    # ---- (e) the table changed at runtime, then emptied
    nxt = torch.argsort(load, descending=True)[8:16].tolist()
    copies2 = [(int(perm[b]), k) for b, (_, k) in zip(nxt, copies[:5]) if k != b // epc] + copies[6:]
    made2 = rt.set_replicas(copies2)
    y2, _, c2, _ = _call(rt, mesh_device, cfg, x_host)
    changed = sum(1 for x, k in copies2 if (x, k) not in copies)
    check(made2 == changed, f"(e) changed table: {made2} copies made, {changed} destinations changed")
    check(
        torch.equal(y2, y_off), f"(e) changed table vs off bit-identical (max |diff| {(y2 - y_off).abs().max().item()})"
    )
    unmapped = {k for _, k in copies} - {k for _, k in copies2}
    sp = _slot_totals(c2, spc).reshape(n, spc)[:, epc]
    check(all(int(sp[k]) == 0 for k in unmapped), f"(e) chips dropped from the table ({sorted(unmapped)}) get no rows")
    check(rt.set_replicas(copies2) == 0, "(e) the same table again enqueues nothing")
    check(rt.set_replicas([]) == 0, "(e) an emptied table copies nothing (the map only)")
    y3, _, c3, _ = _call(rt, mesh_device, cfg, x_host)
    check(torch.equal(y3, y_off), "(e) emptied table vs off bit-identical")
    check(torch.equal(_slot_totals(c3, spc)[own], load), "(e) emptied table: every expert's count at its own slot")
    _release(rt)

    metrics.record("replicas_checks_failed", len(fails))
    assert not fails, f"{len(fails)} MOE_REPLICAS checks failed: {fails}"


@pytest.mark.timeout(3600)
@BOX
@MESH
def test_replicas_golden_layer(mesh_device, monkeypatch):
    """The owner's oracle table (ORACLE, P.46's current.json) on the golden's own MoE input: layer LAYER's ffn_norm at
    chunk 10 of the s56320 golden (the chunk the table was learned on). Off vs runtime (built empty, run, filled with
    set_replicas, run again): the per-chip loads (busiest chip / mean, from the head's counts) as P.46 measured, the
    output bit-identical, the routed part vs the golden's experts_out; the fill's time."""
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    n = mesh_device.get_num_devices()
    g = Golden.for_rung(S, "s56320")
    d = g.layer(10, LAYER)
    x_host = d["ffn_norm"].bfloat16()
    out = {}
    for mode in ("off", "runtime"):
        _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS=mode)
        m_ = _build(mesh_device, ck, cfg, chunk, expert_cache_dir(S))
        if mode == "runtime":
            y0, _, _, _ = _call(m_, mesh_device, cfg, x_host)
            assert torch.equal(y0, out["off"][0]), "golden: runtime with nothing filled differs from off"
            ts = []
            for i in range(3):  # (a fill, then the same table again (nothing), then emptied and filled again)
                if i == 2:
                    m_.set_replicas([])
                t0 = time.perf_counter()
                made = m_.set_replicas(ORACLE[LAYER])
                ttnn.synchronize_device(mesh_device)
                ts.append((made, 1e3 * (time.perf_counter() - t0)))
            logger.info(f"REPLICAS golden L{LAYER}: set_replicas (copies made, synced ms): {ts}")
            metrics.record("replicas_fill_ms", round(ts[2][1], 3))
            metrics.record("replicas_fill_copies", ts[2][0])
        _TIME_CALLS[0] = 9
        y, se, counts, _ = _call(m_, mesh_device, cfg, x_host)
        _TIME_CALLS[0] = 0
        s_ = m_.spc
        tot = _slot_totals(counts, s_)
        chip = tot.reshape(n, s_).sum(dim=1).float()
        hot = tot.max().item()
        routed = (y - se).double()
        ref = d["experts_out"].double()
        rel = ((routed - ref).norm() / ref.norm()).item()
        busiest = chip.max().item() / chip.mean().item()
        logger.info(
            f"REPLICAS golden L{LAYER} c10 {mode}: {len(m_.copies)} copies, busiest chip / mean {busiest:.3f}, hottest "
            f"slot {hot} rows, routed vs golden experts_out rel {rel:.4f}; the layer's MoE call {_TIMES[-1]:.3f} ms "
            f"(synced wall, median of 9)"
        )
        metrics.record(f"replicas_golden_busiest_over_mean_{mode}", round(busiest, 4))
        metrics.record(f"replicas_golden_moe_call_ms_{mode}", round(_TIMES[-1], 3))
        out[mode] = (y, busiest)
        _release(m_)
    d_ = (out["runtime"][0] - out["off"][0]).abs().max().item()
    logger.info(f"REPLICAS golden L{LAYER} c10: runtime vs off max |diff| {d_}")
    metrics.record("replicas_golden_max_abs_diff", d_)
    assert d_ == 0.0, f"golden L{LAYER}: runtime vs off max |diff| {d_}"
    p_off, p_tab = P46_BUSIEST[LAYER]
    assert round(out["off"][1], 2) == p_off and round(out["runtime"][1], 2) == p_tab, (
        f"golden L{LAYER}: busiest / mean off {out['off'][1]:.3f}, runtime {out['runtime'][1]:.3f}; P.46 measured "
        f"{p_off} / {p_tab} with this table"
    )


@pytest.mark.timeout(3600)
@BOX
@MESH
def test_replicas_shared_program(mesh_device, monkeypatch):
    """Two layers on one block (one shared head program, as the model's 40), each filled at runtime with its oracle
    table: layer 0 first, then LAYER on its golden input - each layer's own slot map must apply (busiest chip / mean as
    P.46's single-layer runs: L0 1.12, L20 1.19)."""
    from models.demos.common.bringup.reference.golden import Golden
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir

    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    n = mesh_device.get_num_devices()
    g = Golden.for_rung(S, "s56320")
    _settings(monkeypatch, MOE_BLOCK_SRC="main", MOE_REPLICAS="runtime")
    layers = [
        ad.TtMoEBlockMainLayer(mesh_device, ck, cfg, L, chunk, cache_dir=expert_cache_dir(S), own_rows=True)
        for L in (0, LAYER)
    ]
    for moe, L in zip(layers, (0, LAYER)):
        moe.set_replicas(ORACLE[L])
    for moe, L in zip(layers, (0, LAYER)):
        _, _, counts, _ = _call(moe, mesh_device, cfg, g.layer(10, L)["ffn_norm"].bfloat16())
        tot = _slot_totals(counts, moe.spc)
        chip = tot.reshape(n, moe.spc).sum(dim=1).float()
        busiest = chip.max().item() / chip.mean().item()
        logger.info(
            f"REPLICAS shared program L{L}: {len(moe.copies)} copies, busiest chip / mean {busiest:.3f}, hottest "
            f"slot {tot.max().item()} rows, spare slots' rows {int(tot.reshape(n, moe.spc)[:, -1].sum())}"
        )
        assert int(tot.reshape(n, moe.spc)[:, -1].sum()) > 0, f"L{L}: no rows reached its copies (its slot map unused?)"
        want = P46_BUSIEST[0] if L == 0 else P46_BUSIEST[LAYER][1]
        assert round(busiest, 2) == want, f"L{L}: busiest / mean {busiest:.3f}, P.46 measured {want}"
    for moe in layers:
        _release(moe)
