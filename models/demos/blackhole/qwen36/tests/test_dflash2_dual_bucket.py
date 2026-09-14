# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DUAL-BUCKET serving decoder (tt/dflash2_serving.DFlash2DualBucketDecoder): a bucket switch must be a bit-exact
STATE MOVE, the HOLD must stay exact in both buckets, and every request must stay lossless across switches in both
directions -- on ONE set of captures for the server's lifetime (profiles/dual_bucket_spec.json, test_plan Device
A-E + the warm-up guards).

One model load (max_batch_size=8, buckets 8x4 = K 3 and 4x8 = K 7) drives, in order:

  warm-up  alloc, warm (every seed / switch variant eagerly), capture (both verify traces), one traced step per
           bucket (the drafter's draft/extend traces); GUARD: the program-cache count must not change over a repeat
           of the warm sweep + one traced step per bucket, and the TRACE region must stay <= 85%.
  B) cross-bucket verify equality: slot 0's prefilled state seeded into 4x8 row 0 and 8x4 row 0, ONE replay each on
     the same tokens / positions: the ids of the 4 shared rows must agree (near-tie gate on the 4x8 logits);
     bitwise equality of the written ring blocks / rebuilt conv window / KV rows is RECORDED, not asserted (a
     difference -- expected only from the depthwise conv1d's per-shape rounding -- makes a switch a 'valid
     continuation' of the decode_cfg class, still lossless).
  A) exact move + exact HOLD: A, B, C join 4x8 and step at different cadences (assorted mi); F is prefilled and
     ingested but NOT begun. Snapshot every live user's ring block / conv-window taps / attention K blocks, every
     empty 8x4 row's slot-0 block and F's drafter context pages; set_bucket(8x4): every live user's bytes must
     reappear at its 8x4 row (ring slot 0, window rows 0..Kc) with mi == 0; idle rows and F untouched. One step
     with only A stepping: B and C's verify state and B, C, F's drafter context pages bit-identical (exact HOLD in
     the new bucket). C ends; set_bucket(4x8) back: stable rows for A and B, F (begun in 8x4) takes the freed row,
     same checks in the down direction; a step with only F stepping holds A and B.
  C) lossless across plan()-driven switches: D and E join (plan() forces 8x4 with A, B, F in flight); A and B run
     out and end; plan() switches down after DOWN_STEPS; G joins in 4x8; everything runs to MAX_NEW.
  D) cadence stress: H, I, J join and set_bucket flips on EVERY step (mi == 0 after each) until they are done.
  Every request (A..J) is checked against the teacher-forced max_batch_size=1 reference (test_spec_lossless.
  _reference_greedy + the near-tie gate).

Run: MESH_DEVICE=P150x4 QWEN36_GDN_SPEC_FUSED=1 QWEN36_DFLASH_FOLD_SEED=1 \
       pytest models/demos/blackhole/qwen36/tests/test_dflash2_dual_bucket.py -v -s
Needs the DFlash2 drafter matched to the served weights (DFLASH_WEIGHTS) and the full 64-layer model. The
dual-bucket decoder is implemented for the FUSED GDN verify (QWEN36_GDN_SPEC_FUSED=1, the served path).
"""

import gc
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _MESH_SHAPE, _MULTI, BLOCK_SIZE, DEVICE_PARAMS
from models.demos.blackhole.qwen36.tests.test_dflash2_serving import _prefill
from models.demos.blackhole.qwen36.tests.test_spec_batched import (
    _assert_lossless,
    _batch_prompts,
    _blocks_per_user,
    _reference_model,
    _release,
)
from models.demos.blackhole.qwen36.tests.test_spec_lossless import MAX_NEW, NEAR_TIE_GAP, NUM_BLOCKS, _reference_greedy
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

S = 8  # physical slots
BUCKETS = ((8, 4), (4, 8))
K_MAX = 7
DOWN_STEPS = 4  # hysteresis constants for this scenario (env-tunable in the decoder; short here)
COOLDOWN = 4
TRACE_REGION_MAX_FRAC = 0.85
STRESS_STEPS = 64
STRESS_MAX_NEW = 2 * MAX_NEW


# ------------------------------------------------------------------------------------------------ helpers
def _bid(dec, B, T):
    """The decoder's id for bucket (B, T) (its registry; 'BxT' by convention)."""
    for bid, b in dec.buckets.items():
        if (int(b.B), int(b.T)) == (B, T):
            return bid
    raise AssertionError(f"bucket {B}x{T} not registered: {list(dec.buckets)}")


def _snap_layers(dec):
    """GDN layers snapshotted per check: first, middle, last (each ring read is 24 MiB per layer)."""
    n = len(dec._gdn)
    return sorted({0, n // 2, n - 1})


def _dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _assert_same(before, after, what):
    """torch.equal on every snapshotted byte: per layer the ring block and the conv window taps, then the K blocks."""
    for li, (rb, ra) in enumerate(zip(before["ring"], after["ring"])):
        assert torch.equal(
            rb, ra
        ), f"{what}: GDN ring block differs (snap layer {li}, max |delta| {float((rb - ra).abs().max()):.3e})"
    for li, (wb, wa) in enumerate(zip(before["win"], after["win"])):
        assert torch.equal(wb, wa), (
            f"{what}: conv window taps differ (snap layer {li}, max |delta| "
            f"{float((wb.float() - wa.float()).abs().max()):.3e})"
        )
    if "kv" in before and "kv" in after:
        assert torch.equal(before["kv"], after["kv"]), f"{what}: attention K blocks differ"


def _drafter_ctx(dec, phys, n_layers=2):
    """The drafter ring KV rows that hold slot ``phys``'s COMMITTED context (positions max(0, ctx_len-ring) ..
    ctx_len-1): what a non-stepping slot must keep bit-identical across a step or a switch (its p+1.. rows are
    scratch the slot's next real draft rewrites before reading)."""
    d = dec.drafter
    bs, ring = int(d.block_size), int(d.ring)
    n = int(dec.ctx_len[phys])
    if n == 0:
        return []
    # The compared range is [max(0, n-ring), n); once ctx_len + a block's reach exceeds the ring the next draft's
    # scratch rows (p+1..p+block) wrap onto the OLDEST committed positions in this range, so a byte diff there would
    # be a false HOLD violation. Every prompt here stays well under the ring; assert it so a longer-prompt reuse of
    # the helper fails loudly instead of mis-reporting.
    assert not ring or n + int(d.block_max) <= ring, (
        f"_drafter_ctx: ctx_len {n} + block {int(d.block_max)} > ring {ring}: the compared range would wrap onto "
        f"older committed rows -- restrict the range before reusing this helper at longer contexts"
    )
    pos = torch.arange(max(0, n - ring), n)
    slots = pos % ring if ring else pos
    blocks = d.phys_tables[phys][slots // bs].long()
    rows = (slots % bs).long()
    out = []
    for li in range(min(n_layers, len(d.kv))):
        kc, vc = d.kv[li]
        k, v = _dev0(kc), _dev0(vc)  # [n_blocks, NKV, bs, HD]
        out.append((k[blocks, :, rows].clone(), v[blocks, :, rows].clone()))
    return out


def _assert_drafter_ctx_same(before, after, what):
    assert len(before) == len(after)
    for li, ((kb, vb), (ka, va)) in enumerate(zip(before, after)):
        assert torch.equal(kb, ka) and torch.equal(vb, va), f"{what}: drafter context pages differ (layer {li})"


def _kv_rows(att0, pt_row, positions):
    """K rows [len(positions), NKV, HD] of one request at absolute positions, through its page-table row."""
    k = _dev0(att0.paged_k)  # [n_blocks, NKV, bs, HD]
    blocks = pt_row[[p // BLOCK_SIZE for p in positions]].long()
    rows = torch.tensor([p % BLOCK_SIZE for p in positions])
    return k[blocks, :, rows].clone()


def _trace_region(model, device):
    used = model._trace_region_bytes(device)
    try:
        mv = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
        total = int(mv.total_bytes_per_bank) * int(mv.num_banks)
    except Exception:
        total = None
    return used, total


class _Session:
    """Request bookkeeping over the decoder: name -> slot, per-request outputs, plan()-before-begin joins."""

    def __init__(self, model, dec, page_tables, prompts):
        self.model, self.dec, self.pt, self.prompts = model, dec, page_tables, prompts
        self.out = {}  # name -> committed ids (incl. first)
        self.name_of = {}  # phys -> name
        self.limit = {}  # name -> stop after this many tokens

    def live(self):
        return {u for u in range(S) if self.dec.active[u]}

    def prefill(self, name, phys, prompt_i):
        first = _prefill(self.model, self.dec, phys, self.prompts[prompt_i], self.pt)
        return first

    def begin(self, name, phys, prompt_i, first, limit=MAX_NEW):
        """plan() with the live set after this join, then begin -- decode_forward's order."""
        bucket = self.dec.plan(self.live() | {phys})
        self.dec.begin(phys, first, len(self.prompts[prompt_i]), self.pt[phys])
        assert self.dec.cur_id == bucket and self.dec.row_of(phys) is not None
        self.out[name] = [first]
        self.name_of[phys] = name
        self.limit[name] = limit
        logger.info(f"[dual] {name} joined slot {phys} in {bucket} (row {self.dec.row_of(phys)})")

    def join(self, name, phys, prompt_i, limit=MAX_NEW):
        first = self.prefill(name, phys, prompt_i)
        self.begin(name, phys, prompt_i, first, limit)

    def absorb(self, com):
        done = []
        for u, ids in com.items():
            name = self.name_of[u]
            self.out[name].extend(int(t) for t in ids)
            if len(self.out[name]) >= self.limit[name]:
                done.append(u)
        return done

    def end(self, phys):
        name = self.name_of.pop(phys)
        self.dec.end(phys)
        logger.info(f"[dual] {name} left slot {phys} with {len(self.out[name])} tokens (bucket {self.dec.cur_id})")

    def step(self, only=None, plan=True):
        """One engine step: plan(live) then step(); requests that reached their limit end. Returns the ended slots."""
        if plan:
            self.dec.plan(self.live())
        com = self.dec.step(only=only)
        done = self.absorb(com)
        for u in done:
            self.end(u)
        return done


# ------------------------------------------------------------------------------------------------ the test
@run_for_blackhole()
@pytest.mark.timeout(7200)
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
def test_dual_bucket_exact_switch_hold_and_lossless(mesh_device, monkeypatch):
    if not _MULTI:
        pytest.skip("spec decode is the TP path; run with MESH_DEVICE=P150x4")
    if os.environ.get("QWEN36_GDN_SPEC_FUSED", "0") != "1":
        pytest.skip("the dual-bucket decoder runs the fused GDN verify: set QWEN36_GDN_SPEC_FUSED=1")
    monkeypatch.setenv("QWEN36_DFLASH_BUCKET_DOWN_STEPS", str(DOWN_STEPS))
    monkeypatch.setenv("QWEN36_DFLASH_BUCKET_COOLDOWN", str(COOLDOWN))
    from transformers import AutoTokenizer

    from models.demos.blackhole.qwen36.tt.dflash2_serving import DFlash2DualBucketDecoder

    device = mesh_device
    device.enable_program_cache()
    model = Qwen36Model.from_pretrained(device, max_batch_size=S, max_seq_len=NUM_BLOCKS * BLOCK_SIZE)
    assert len(model.layers) >= 62, "the DFlash2 taps live at layers 5..61: run the full model"
    model.set_gdn_fused_decode(True)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompts = _batch_prompts(S, tokenizer)  # 8 distinct, ragged prompts (130 .. 255 tokens)
    bpu = _blocks_per_user(max(len(p) for p in prompts), K_MAX, STRESS_MAX_NEW + 8)
    page_tables = torch.stack([torch.arange(u * bpu, (u + 1) * bpu, dtype=torch.int32) for u in range(S)])
    kv_shape = [S * bpu, model.args.n_local_kv_heads, BLOCK_SIZE, model.args.head_dim]
    model.free_kv_caches()
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=S)
    att0 = next(layer.attention for layer in model.layers if layer.is_full_attention)

    dec = DFlash2DualBucketDecoder(model, num_blocks=bpu, buckets=BUCKETS)
    assert dec.K_max == K_MAX and dec.multi
    BIG, SMALL = _bid(dec, 8, 4), _bid(dec, 4, 8)
    assert all(getattr(dn, "_spec_fused", False) for dn in dec._gdn), "fused GDN spec path expected"
    L = _snap_layers(dec)
    ses = _Session(model, dec, page_tables, prompts)
    equality_report = {}
    try:
        # ---- warm-up: allocate, compile everything eagerly, capture both verify traces, trace each layout ---- #
        dec.alloc()
        dec.warm(warm_position=len(prompts[0]) + 1)
        first_w = _prefill(model, dec, 0, prompts[0], page_tables)  # compiles the prefill bucket + ingest
        used0, total = _trace_region(model, device)
        dec.capture(warm_position=len(prompts[0]) + 1)
        used1, _ = _trace_region(model, device)
        if used0 is not None:
            logger.info(
                f"[dual] verify traces: TRACE region +{(used1 - used0) / 2**20:.1f} MiB -> {used1 / 2**20:.1f} MiB"
            )
        assert dec.cur_id == BIG, f"the decoder starts in the identity bucket, got {dec.cur_id}"
        dec.plan({0})
        dec.begin(0, first_w, len(prompts[0]), page_tables[0])
        for _ in range(2):
            dec.step()  # captures BIG's draft/extend traces
        dec.set_bucket(SMALL)
        for _ in range(2):
            dec.step()  # captures SMALL's draft/extend traces
        dec.set_bucket(BIG)
        dec.step()
        dec.end(0)
        used2, _ = _trace_region(model, device)
        if used2 is not None and total:
            logger.info(
                f"[dual] all spec traces parked: TRACE region {used2 / 2**20:.1f} / {total / 2**20:.0f} MiB "
                f"({used2 / total:.0%}; drafter traces +{(used2 - used1) / 2**20:.1f} MiB)"
            )
            assert (
                used2 <= TRACE_REGION_MAX_FRAC * total
            ), f"TRACE region {used2 / total:.0%} > {TRACE_REGION_MAX_FRAC:.0%}"
        # GUARD: nothing spec-related may compile once the traces are parked.
        ttnn.synchronize_device(device)
        n0 = device.num_program_cache_entries()
        dec.warm_sweep()
        first_g = _prefill(model, dec, 0, prompts[0], page_tables)
        dec.plan({0})
        dec.begin(0, first_g, len(prompts[0]), page_tables[0])
        dec.step()
        dec.set_bucket(SMALL if dec.cur_id == BIG else BIG)
        dec.step()
        dec.end(0)
        ttnn.synchronize_device(device)
        n1 = device.num_program_cache_entries()
        assert n1 == n0, (
            f"{n1 - n0} program(s) compiled AFTER the spec traces were captured (cache {n0} -> {n1}) over the warm "
            f"sweep + one traced step per bucket: a seed / switch / draft variant is missing from warm()"
        )
        logger.info(f"[dual] post-capture guard: program cache unchanged at {n0} entries")

        # ---- B) cross-bucket verify equality: the identical committed state replayed once in each cfg -------- #
        first0 = _prefill(model, dec, 0, prompts[0], page_tables)
        T0 = len(prompts[0])
        toks = [first0] + [int(t) for t in prompts[0][:7]]  # row 0 = the pending token, rows 1.. fixed candidates
        for B_cfg, T_cfg in ((4, 8), (8, 4)):
            cid = _bid(dec, B_cfg, T_cfg)
            row = 0
            for dn in dec._gdn:
                dn.seed_spec_row(cid, row, 0)
            pt = torch.zeros(B_cfg, dec.nb_v, dtype=torch.int32)
            pt[row] = dec.fit_row(page_tables[0])
            model.refresh_verify_page_tables(pt, cfg_id=cid)
            tokens = [[0] * T_cfg for _ in range(B_cfg)]
            tokens[row] = toks[:T_cfg]
            positions, mi_prev = [0] * B_cfg, [0] * B_cfg
            positions[row] = T0
            hold = [r for r in range(B_cfg) if r != row]
            ids, _feed, logits = model.verify_traced(
                tokens, positions, mi_prev, read_logits=True, hold=hold, cfg_id=cid
            )
            rings, wins = [], []
            for li in L:
                dn = dec._gdn[li]
                Nv, Kc = dn.Nv, dn.K
                ring = _dev0(dn._spec_ring)
                rings.append(torch.stack([ring[(t * B_cfg + row) * Nv : (t * B_cfg + row + 1) * Nv] for t in range(4)]))
                cfg = dn.spec_cfg(cid)
                wins.append(_dev0(cfg.win_pair[cfg.parity])[row, : Kc - 1 + 4].clone())
            kv = _kv_rows(att0, page_tables[0], [T0 + j for j in range(4)])
            equality_report[cid] = {
                "ids": [int(x) for x in ids[row * T_cfg : row * T_cfg + 4]],
                "logits": logits[row * T_cfg : row * T_cfg + 4].float().clone(),
                "rings": rings,
                "wins": wins,
                "kv": kv,
            }
            model.refresh_verify_page_tables(torch.zeros(B_cfg, dec.nb_v, dtype=torch.int32), cfg_id=cid)
        a, b = equality_report[SMALL], equality_report[BIG]
        # Collect any confident id disagreement but DEFER the assert to the end (next to the lossless checks): a
        # conv1d per-shape rounding tie here must not abort scenarios A/C/D and hide their exact-move / HOLD results
        # behind a ~2 h run. Logged now, asserted last.
        cross_bucket_bad = []
        for j in range(4):
            if a["ids"][j] != b["ids"][j]:
                top2 = torch.topk(a["logits"][j].reshape(-1), 2).values
                gap = float(top2[0] - top2[1])
                cross_bucket_bad.append((j, int(a["ids"][j]), int(b["ids"][j]), gap))
        equality_report["_bad"] = cross_bucket_bad
        if cross_bucket_bad:
            logger.info(
                "[dual] cross-bucket verify id disagreements (deferred): "
                + "; ".join(
                    f"row {j}: {i0} (4x8) vs {i1} (8x4), top-2 gap {g:.3f}" for j, i0, i1, g in cross_bucket_bad
                )
            )
        same_ring = all(torch.equal(x, y) for x, y in zip(a["rings"], b["rings"]))
        same_win = all(torch.equal(x, y) for x, y in zip(a["wins"], b["wins"]))
        same_kv = torch.equal(a["kv"], b["kv"])
        same_logits = torch.equal(a["logits"], b["logits"])
        logger.info(
            f"[dual] cross-bucket verify equality (4x8 vs 8x4, same state, 4 shared rows): ids {a['ids']} vs {b['ids']}; "
            f"bitwise: logits {same_logits}, ring blocks {same_ring}, conv window {same_win}, KV rows {same_kv}"
            + (
                ""
                if same_ring and same_win
                else " -> a switch is a 'valid continuation' (decode_cfg class), see docstring"
            )
        )

        # ---- A) exact move + exact HOLD ------------------------------------------------------------------- #
        dec.set_bucket(SMALL)
        assert dec.cur_id == SMALL and not any(dec.active)
        ses.join("A", 0, 0)
        ses.join("B", 1, 1)
        ses.join("C", 2, 2)
        assert dec.cur_id == SMALL and [dec.row_of(u) for u in (0, 1, 2)] == [0, 1, 2]
        # assorted cadences -> assorted mi (held slots keep theirs)
        ses.step(only=None)
        ses.step(only=[0, 1])
        ses.step(only=[2])
        ses.step(only=[0])
        logger.info(
            f"[dual] before the up-switch: mi={[dec.mi[u] for u in (0, 1, 2)]} p={[dec.p[u] for u in (0, 1, 2)]}"
        )
        # F: prefilled + ingested, NOT begun (a pending request at the switch)
        firstF = ses.prefill("F", 5, 5)
        ctxF = _drafter_ctx(dec, 5)
        live = [0, 1, 2]
        before = {u: dec.snapshot(u, gdn_layers=L) for u in live}
        idle_rows = [r for r in range(8) if r not in live]  # 8x4 rows == slots: 3..7 are empty after the switch
        idle_before = [dec.snapshot_row(BIG, r, gdn_layers=L, mi=0) for r in idle_rows]
        host_before = ([dec.p[u] for u in live], [dec.pending[u] for u in live], [dec.ctx_len[u] for u in live])
        t = time.perf_counter()
        dec.set_bucket(BIG)
        ttnn.synchronize_device(device)
        logger.info(
            f"[dual] up-switch {SMALL} -> {BIG} with {len(live)} live: {(time.perf_counter() - t) * 1e3:.1f} ms"
        )
        assert dec.cur_id == BIG
        for u in live:
            assert (
                dec.row_of(u) == u and dec.mi[u] == 0
            ), f"slot {u}: row {dec.row_of(u)} mi {dec.mi[u]} after the switch"
        assert host_before == ([dec.p[u] for u in live], [dec.pending[u] for u in live], [dec.ctx_len[u] for u in live])
        for u in live:
            _assert_same(before[u], dec.snapshot(u, gdn_layers=L), f"up-switch, slot {u}")
        for r, snap in zip(idle_rows, idle_before):
            after = dec.snapshot_row(BIG, r, gdn_layers=L, mi=0)
            for li, (x, y) in enumerate(zip(snap["ring"], after["ring"])):
                assert torch.equal(x, y), f"empty 8x4 row {r}: slot-0 ring block changed (identity move expected)"
            for li, w in enumerate(after["win"]):
                assert torch.isfinite(w.float()).all(), f"empty 8x4 row {r}: window row not finite"
        _assert_drafter_ctx_same(ctxF, _drafter_ctx(dec, 5), "up-switch, pending slot 5")
        logger.info(
            "[dual] up-switch: every live slot's ring block / window taps / KV moved bit-for-bit; idle rows untouched"
        )
        # one step with only A stepping: exact HOLD for B, C in the new bucket; drafter context of B, C, F untouched
        # AND the EMPTY rows (no active slot) copy through unchanged -- the ctrl-page HOLD sentinel must fire on a
        # row whose phys is None too (the window is read at the flipped parity: copy-through semantics).
        idle_step_rows = [r for r in range(8) if r not in (0, 1, 2, 5)]  # 5 is pending F; 0 steps; 1,2 held
        idle_hold_before = {r: dec.snapshot_row(BIG, r, gdn_layers=L, mi=0) for r in idle_step_rows}
        hold_before = {u: dec.snapshot(u, gdn_layers=L) for u in (1, 2)}
        dctx_before = {u: _drafter_ctx(dec, u) for u in (1, 2, 5)}
        com = dec.step(only=[0])
        assert set(com) == {0}
        ses.absorb(com)
        for u in (1, 2):
            _assert_same(hold_before[u], dec.snapshot(u, gdn_layers=L), f"HOLD in {BIG} while slot 0 steps, slot {u}")
        for r in idle_step_rows:
            _assert_same(
                idle_hold_before[r],
                dec.snapshot_row(BIG, r, gdn_layers=L, mi=0),
                f"empty {BIG} row {r} HOLD across a step",
            )
        for u in (1, 2, 5):
            _assert_drafter_ctx_same(dctx_before[u], _drafter_ctx(dec, u), f"HOLD in {BIG}, drafter slot {u}")
        logger.info(f"[dual] HOLD in {BIG}: B, C verify state, empty rows, and B, C, F drafter context bit-identical")
        # F begins in the new bucket (a pending request joining right after a switch)
        ses.begin("F", 5, 5, firstF)
        ses.step()
        # C leaves; down-switch with A, B, F live: A, B keep rows 0, 1 (stable), F takes the lowest free row (2)
        ses.end(2)
        live = [0, 1, 5]
        before = {u: dec.snapshot(u, gdn_layers=L) for u in live}
        idle_before = dec.snapshot_row(SMALL, 3, gdn_layers=L, mi=0)
        t = time.perf_counter()
        dec.set_bucket(SMALL)
        ttnn.synchronize_device(device)
        logger.info(
            f"[dual] down-switch {BIG} -> {SMALL} with {len(live)} live: {(time.perf_counter() - t) * 1e3:.1f} ms"
        )
        assert dec.cur_id == SMALL
        assert [dec.row_of(u) for u in live] == [0, 1, 2], f"rows after the down-switch: {dec.row_map()}"
        assert all(dec.mi[u] == 0 for u in live)
        for u in live:
            _assert_same(before[u], dec.snapshot(u, gdn_layers=L), f"down-switch, slot {u}")
        after = dec.snapshot_row(SMALL, 3, gdn_layers=L, mi=0)
        for x, y in zip(idle_before["ring"], after["ring"]):
            assert torch.equal(x, y), "empty 4x8 row 3: slot-0 ring block changed (identity move expected)"
        idle_hold_before = dec.snapshot_row(SMALL, 3, gdn_layers=L, mi=0)  # 4x8 row 3 is empty (live are rows 0,1,2)
        hold_before = {u: dec.snapshot(u, gdn_layers=L) for u in (0, 1)}
        dctx_before = {u: _drafter_ctx(dec, u) for u in (0, 1)}
        com = dec.step(only=[5])
        assert set(com) == {5}
        ses.absorb(com)
        for u in (0, 1):
            _assert_same(hold_before[u], dec.snapshot(u, gdn_layers=L), f"HOLD in {SMALL} while slot 5 steps, slot {u}")
            _assert_drafter_ctx_same(dctx_before[u], _drafter_ctx(dec, u), f"HOLD in {SMALL}, drafter slot {u}")
        _assert_same(
            idle_hold_before, dec.snapshot_row(SMALL, 3, gdn_layers=L, mi=0), f"empty {SMALL} row 3 HOLD across a step"
        )
        logger.info("[dual] down-switch exact; HOLD in 4x8 exact (live rows, drafter context, and the empty row)")

        # ---- C) lossless across plan()-driven switches ------------------------------------------------------ #
        # The direct set_bucket() churn above backed the planner's cooldown off (a server never calls set_bucket()
        # itself; it resets the policy once after its warm-up, before the first request): the same clean start here.
        dec.reset_bucket_policy()
        assert dec._planner.cooldown == COOLDOWN and dec.switch_count == 0
        n_sw = dec.switch_count
        # D and E are long-lived (STRESS_MAX_NEW): the planner's step clock only advances while a slot is live, and
        # they must still be live when the plan()-driven down-switch is due (that switch is what the lossless check
        # below exercises; a 48-token E would end before a 4-step cooldown + 4 fitting steps elapse).
        ses.join("D", 3, 3, limit=STRESS_MAX_NEW)  # 4 live: still fits 4x8
        assert dec.cur_id == SMALL
        ses.join("E", 4, 4, limit=STRESS_MAX_NEW)  # 5 live: plan() must force 8x4 with A, B, F, D in flight
        assert dec.cur_id == BIG and dec.switch_count == n_sw + 1, "plan() must force the up-switch for a 5th user"
        # run until A and B are done (they started first)
        for _ in range(200):
            ses.step()
            if not dec.active[0] and not dec.active[1]:
                break
        assert not dec.active[0] and not dec.active[1], "A and B did not finish"
        # <= 3 live (F, D, E): plan() switches down after DOWN_STEPS consecutive fitting calls AND the cooldown since
        # the forced up-switch (the planner's CURRENT cooldown: the forced up did not back it off after the reset).
        down_budget = int(dec._planner.cooldown) + DOWN_STEPS + 8
        for i in range(down_budget):
            assert ses.live(), "every request ended before the plan()-driven down-switch was due: lengthen D / E"
            ses.step()
            if dec.cur_id == SMALL:
                break
        assert dec.cur_id == SMALL, (
            f"plan() never switched down in {down_budget} steps with {len(ses.live())} live: "
            f"planner {dec._planner.stats()}, switches {dec.bucket_stats()['_switches']}"
        )
        assert dec.active[3] and dec.active[4], "D and E must be live across the plan()-driven down-switch"
        logger.info(f"[dual] plan()-driven down-switch after the leaves; planner {dec.bucket_stats()['_switches']}")
        ses.join("G", 6, 6)  # 4 live in 4x8
        assert dec.cur_id == SMALL
        for _ in range(400):
            if not ses.live():
                break
            ses.step()
        assert not ses.live(), f"requests still live: {ses.name_of}"

        # ---- D) cadence stress: a switch on EVERY step ------------------------------------------------------- #
        ses.join("H", 0, 0, limit=STRESS_MAX_NEW)
        ses.join("I", 1, 1, limit=STRESS_MAX_NEW)
        ses.join("J", 2, 2, limit=STRESS_MAX_NEW)
        n_sw = dec.switch_count
        for i in range(STRESS_STEPS):
            if not ses.live():
                break
            dec.set_bucket(SMALL if dec.cur_id == BIG else BIG)
            assert all(dec.mi[u] == 0 for u in ses.live())
            ses.step(plan=False)
        stress_switches = dec.switch_count - n_sw
        logger.info(f"[dual] cadence stress: {stress_switches} switches, live now {sorted(ses.live())}")
        assert stress_switches >= 16, f"only {stress_switches} switches in the stress phase"
        for _ in range(400):
            if not ses.live():
                break
            ses.step()
        assert not ses.live()
        dec.log_bucket_stats()
        outs = dict(ses.out)
        for name, ids in outs.items():
            logger.info(f"[dual] {name}: {len(ids)} tokens: {tokenizer.decode(ids)!r}")
    finally:
        dec.release()
        _release(model)
    del dec, model
    gc.collect()

    # ---- every request against the teacher-forced single-user reference ------------------------------------ #
    prompt_of = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 0, "I": 1, "J": 2}
    ref_model, pt1, kv1 = _reference_model(device)
    try:
        for name, ids in outs.items():
            ref, gaps = _reference_greedy(ref_model, prompts[prompt_of[name]], pt1, kv1, ids)
            _assert_lossless(ids, ref, gaps, tokenizer, f"dual-bucket request {name}")
    finally:
        _release(ref_model)
    # The deferred scenario B check: a cross-bucket id disagreement is a failure only when CONFIDENT (top-2 gap
    # past the near-tie gate on the 4x8 logits); a near-tie is an accepted 'valid continuation' of the decode_cfg.
    bad = [x for x in equality_report.get("_bad", []) if x[3] >= NEAR_TIE_GAP]
    assert not bad, "cross-bucket verify: CONFIDENT id disagreement(s) beyond bf16 noise: " + "; ".join(
        f"row {j}: {i0} (4x8) vs {i1} (8x4), top-2 gap {g:.3f} >= {NEAR_TIE_GAP}" for j, i0, i1, g in bad
    )
    logger.info("[dual] every request lossless across bucket switches in both directions, plan()-driven and forced")
