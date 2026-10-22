# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""APC_CHUNKED_PREFILL_PLAN.md milestone M2 prototype (tt/prefix_cache.py): snapshot a user's state at a chunk boundary s, restore it into a user on a DIFFERENT mesh row (and, with
DSV41_M2_UP=2, a different prefill slot) and continue the prompt from s. Checks, bit-exactly:

  (a) the carried tensors read back from the target device shards right after the restore equal the snapshot,
  (b) the first-token logits and token of the continued prompt equal those of the uninterrupted reference (same prompt, one user, pieces [0, s) and [s, e)),
  (c) the final decode state of the restored user (pool latents of every kv-source layer, window ring rows of every ring layer, ratio-2 ``prev_cs``, decode index keys) equals the
      reference user's, tensor by tensor,
  (d) the reference user (another mesh row) and a bystander are untouched by the restore,
  (e) no "recomputing from" line (the restored user really resumed), and the times of snapshot / restore / continuation vs the full prefill.

    DSV41_LAYERS=0-3 pytest models/demos/blackhole/deepseek_v41_flash/tests/test_apc_m2_snapshot_restore.py -s
Env: DSV41_M2_U (users per row, 4), DSV41_M2_LEN (prompt length, 2600), DSV41_M2_S (snapshot positions, "1024,1536"), DSV41_M2_UP (prefill slots per row, 1)
"""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as P
from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC
from models.demos.blackhole.deepseek_v41_flash.tt.common import create_tt_model, default_page_params
from models.tt_transformers.tt.common import PagedAttentionConfig

U = int(os.environ.get("DSV41_M2_U", "4"))
UP = int(os.environ.get("DSV41_M2_UP", "1"))
E = int(os.environ.get("DSV41_M2_LEN", "2600"))
SNAPS = [int(x) for x in os.environ.get("DSV41_M2_S", "1024,1536").split(",")]
CHUNK = 512
LAYERS = os.environ.get("DSV41_LAYERS", "0-3")


def bits(x):
    x = x.contiguous()
    return x.view(torch.int16) if x.element_size() == 2 else x.view(torch.int32)


def same(x, y):
    return x.shape == y.shape and bool(torch.equal(bits(x), bits(y)))


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": int(os.environ.get("DSV41_IL_TRACE_REGION", "1600000000")),
            },
            id="ring",
        )
    ],
    indirect=True,
)
@pytest.mark.timeout(7200)
@torch.no_grad()
def test_snapshot_restore_into_another_row(mesh_device):
    os.environ.update({"DSV41_PF_UMASK": "1", "DSV41_PREFILL_UP": str(UP), "DSV41_MO_OVERLAP": "0"})
    md = mesh_device
    log = lambda m: print(m, flush=True)
    a, _, b = LAYERS.partition("-")
    layer_ids = list(range(int(a), int(b or a) + 1))
    B = 4 * U
    S_pad = -(-E // CHUNK) * CHUNK
    max_ctx = S_pad + (
        0 if os.environ.get("DSV41_PF_SINGLE") == "1" else 256
    )  # (single trace: S_pad = max_ctx rounded to the chunk must stay <= max_ctx + 128)
    max_ctx = max(
        max_ctx, int(os.environ.get("DSV41_M2_MAXCTX", "0"))
    )  # (single trace: every column slab needs >= 512 local slots for the top-k: max_ctx >= 16384)
    pp = default_page_params(max_ctx, U)
    args, m, pool, _ = create_tt_model(
        md,
        B,
        max_ctx,
        PagedAttentionConfig(block_size=pp["page_block_size"], max_num_blocks=pp["page_max_num_blocks_per_dp"]),
        layer_ids=layer_ids,
        log=log,
    )
    assert m.Up == UP
    rows, cols = m.rows, m.cols
    recompute = []
    orig_log = m.log

    def log_spy(msg):
        if "recomputing from" in str(msg):
            recompute.append(str(msg))
        return orig_log(msg)

    m.log = log_spy
    g = torch.Generator().manual_seed(99)
    prompt = torch.randint(1000, 100000, (E,), generator=g, dtype=torch.int64)
    other = torch.randint(1000, 100000, (E - 300,), generator=g, dtype=torch.int64)

    def run(items):
        return m.prefill_interleaved(items, CHUNK, s_pad_max=S_pad, want_logits=True)

    def user_state(b_, e):
        """decode state of user b_ after a prompt of e tokens: {name: torch}"""
        r = b_ // U
        out = {}
        full = PC.read_shard(m.pool.pool, r * cols)[0, 0]
        for src in PC._sources(m):
            ent = torch.arange(e // P.SRC_RATIO[src])
            out[f"pool.lat{src}"] = full[m.pool.phys_rows(b_, src, ent).long()].clone()
        for l in range(m.pool.n_ring_layers):
            base = m.pool.ring_base(l) + (b_ % U) * m.pool.ring_rows
            out[f"ring{l}"] = full[base : base + m.pool.ring_rows].clone()
        for L, at in m.attns.items():
            if getattr(at, "prev_cs", None) is not None:
                out[f"prev_cs{L}"] = PC.read_shard(at.prev_cs, r * cols).reshape(U, -1)[b_ % U].clone()
        for L, d in m.index_owner.items():
            rr = m.attns[L].ratio
            out[f"idxkeys{L}"] = (
                PC.read_shard(d.k_cache, r * cols).reshape(U, d.k_cache.shape[2], -1)[b_ % U, : e // rr].clone()
            )
        return out

    def valid_slot(r_, i_, s_):
        """the valid part (what a snapshot at s_ would hold) of every carried tensor of slot i_ of mesh row r_ (the window scratch rows of the kv table are rewritten for every slot by every chunk, valid or not)"""
        out = {}
        for n_, sp in PC.carried_specs(m).items():
            vals = []
            for c in range(cols):
                x = PC.read_shard(sp.tensor, r_ * cols + c)[i_ : i_ + 1]
                if sp.kind == "tail":
                    lo = sp.row0(x.shape[2], sp.count(s_))
                    x = x[:, :, lo : lo + sp.count(s_)]
                vals.append(x.clone())
            out[n_] = vals
        return out

    def cmp_state(tag, s0_, s1_):
        bad = [k for k in s0_ if not same(s0_[k], s1_[k])]
        log(f"STATE {tag}: {len(s0_)} tensors, " + ("all bit-identical" if not bad else f"DIFFER: {bad}"))
        return bad

    # ---- reference: one user (row 0), pieces [0, s) ... [s, E) -------------------------------------------------------------------------------
    u_ref, snaps = 1, {}
    if UP > 1:
        # the numerics of a prefill depend on the prefill SLOT of the row (Up > 1: slot 0 and slot 1 differ by bf16 noise, max abs logit diff ~0.3, same token; a pre-existing property of the traced chunk), so the donor works in
        # slot 1 like the restored users (a resumable blocker holds slot 0 of the donor's row)
        run([(0, other, 0, 1024)])
        assert m.pf_resume.get(0) == 1024
    ref_state = {}
    t0 = time.perf_counter()
    res = run([(u_ref, prompt, 0, E)])
    t_full = time.perf_counter() - t0
    L_ref, tok_ref = res[u_ref][1].clone(), int(res[u_ref][0])
    ref_state = user_state(u_ref, E)
    log(f"carried tensors: {list(PC.carried_specs(m))}")
    log(f"reference (one call, 0 -> {E}): {t_full:.2f} s, first token {tok_ref}, {len(ref_state)} state tensors")
    for s in SNAPS:
        m.release_user(u_ref)
        run([(u_ref, prompt, 0, s)])
        snaps[s] = PC.snapshot_user(m, u_ref, s, prompt)
        log(
            f"snapshot at {s}: {snaps[s].seconds:.2f} s, {snaps[s].nbytes / 2**20:.1f} MiB ({len(snaps[s].carried)} carried tensors x {cols} column shards + {len(snaps[s].pool)} pool sources)"
        )
        res = run([(u_ref, prompt, s, E)])  # continuation of the same user (pieces vs one call)
        assert same(res[u_ref][1], L_ref), f"pieces [0,{s}) [{s},{E}) differ from the one-call prefill"
        assert same(ref_state["pool.lat%d" % PC._sources(m)[0]], user_state(u_ref, E)["pool.lat%d" % PC._sources(m)[0]])
    ref_state = user_state(u_ref, E)

    # a bystander on row 1 (must stay untouched by the restores)
    tgt_users = {
        u
        for _, u in (
            [(0, 3 * U + (U - 1)), (0, 2 * U + 2)]
            if not os.environ.get("DSV41_M2_TARGETS")
            else [tuple(int(y) for y in x.split(":")) for x in os.environ["DSV41_M2_TARGETS"].split(",")]
        )
    }
    u_by = next(u for u in (6, 7, 4, 8) if u not in tgt_users and u != 0)
    run([(u_by, other, 0, len(other))])
    by_state = user_state(u_by, len(other))
    ref_before = user_state(u_ref, E)

    # ---- restore into a user of another mesh row ----------------------------------------------------------------------------------------------
    # the reference of a restored user is a FULL prefill of the SAME user (one call): numerics may differ from one mesh row / user to another, a restore must not add to that
    tg = os.environ.get(
        "DSV41_M2_TARGETS"
    )  # "s:user,s:user": explicit (snapshot, target user) pairs (localisation runs)
    targets = [(SNAPS[0], 3 * U + (U - 1)), (SNAPS[-1], 2 * U + 2)]  # row 3 / in-row U-1 ; row 2 / in-row 2
    if tg:
        targets = [tuple(int(y) for y in x.split(":")) for x in tg.split(",")]
    results = []
    for s, u2 in targets:
        snap = snaps[s]
        m.release_user(u2)
        run(
            [(u2, other, 0, len(other))]
        )  # a different prompt leaves its state in the slot / user: the restore has to overwrite it
        blk, blk_before = None, None
        if (
            UP > 1
        ):  # a resumable prompt of the same mesh row holds slot 0: u2 (control run and restore) works in slot 1 and must leave the blocker's carried state alone
            blk = u2 - 1 if u2 % U else u2 + 1
            m.release_user(blk)
            run([(blk, other, 0, 1024)])
            assert m.pf_resume.get(blk) == 1024
            rb, ib = m.pf_slot_of[blk] // UP, m.pf_slot_of[blk] % UP
            blk_before = valid_slot(rb, ib, 1024)
        m.release_user(u2)
        # the reference of a restored user is a FULL prefill of the SAME user in the SAME slot (one call): numerics may differ between users / slots, a restore must not add to that
        t0 = time.perf_counter()
        res = run([(u2, prompt, 0, E)])
        t_full2 = time.perf_counter() - t0
        L_full2, tok_full2 = res[u2][1].clone(), int(res[u2][0])
        st_full2 = user_state(u2, E)
        cross = [k for k in ref_state if not same(ref_state[k], st_full2[k])]
        log(
            f"control: full prefill of user {u2} (row {u2 // U}, slot {(m.pf_slot_of[u2] % UP) if UP < U else '-'}) takes {t_full2:.2f} s; vs user {u_ref} (row 0): first-token logits bit-identical {same(L_full2, L_ref)} "
            f"(max abs diff {(L_full2.float() - L_ref.float()).abs().max():.3g}), state tensors that differ across users/rows/slots: {cross}"
        )
        if UP > 1:
            assert m.pf_resume.get(blk) == 1024, "the control prefill evicted the resumable blocker"
        m.release_user(u2)
        recompute.clear()
        t_rest = PC.restore_user(m, snap, u2)
        if UP > 1:
            assert (
                m.pf_slot_of[u2] != m.pf_slot_of[blk] and m.pf_resume.get(blk) == 1024
            ), "the restore evicted the resumable blocker"
            blk_after = valid_slot(rb, ib, 1024)
            bad_b = [n for n in blk_before if not all(same(x, y) for x, y in zip(blk_before[n], blk_after[n]))]
            log(
                f"    slot variant: user {u2} took slot {m.pf_slot_of[u2] % UP} of its row (blocker user {blk} in slot {ib}); blocker's carried tensors untouched: {not bad_b} {bad_b}"
            )
            assert not bad_b
        # (a) carried tensors straight from the target shards
        r2 = (u2 // U) if UP == U else m.pf_slot_of[u2] // UP
        i2 = (u2 % U) if UP == U else m.pf_slot_of[u2] % UP
        bad_c = []
        for name, sp in PC.carried_specs(m).items():
            n = sp.count(s)
            for c in range(cols):
                x = PC.read_shard(sp.tensor, r2 * cols + c)[i2 : i2 + 1]
                if sp.kind == "tail":
                    lo = sp.row0(x.shape[2], n)
                    x = x[:, :, lo : lo + n]
                if not same(x, snap.carried[name][c]):
                    bad_c.append((name, c))
        pool_now = {
            src: PC.read_shard(m.pool.pool, (u2 // U) * cols)[0, 0][
                m.pool.phys_rows(u2, src, torch.arange(snap.pool[src].shape[0])).long()
            ]
            for src in snap.pool
        }
        bad_p = [src for src in snap.pool if not same(pool_now[src], snap.pool[src])]
        log(
            f"(a) restore of the snapshot at {s} into user {u2} (mesh row {u2 // U}, slot {i2}): {t_rest:.2f} s {m.__dict__.get('_restore_timing')}; carried tensors bit-identical to the snapshot: {not bad_c} {bad_c[:3]}; pool latents: {not bad_p}"
        )
        t0 = time.perf_counter()
        res = run([(u2, prompt, s, E)])
        t_cont = time.perf_counter() - t0
        L2, tok2 = res[u2][1], int(res[u2][0])
        ok_logits, ok_tok = same(L2, L_full2), tok2 == tok_full2
        st2 = user_state(u2, E)
        bad_s = cmp_state(f"restored user {u2} vs a full prefill of user {u2} at the end of the prompt", st_full2, st2)
        for k in bad_s:
            a_, b_ = st_full2[k].float().reshape(st_full2[k].shape[0], -1), st2[k].float().reshape(st2[k].shape[0], -1)
            nz = (a_ != b_).any(dim=1).nonzero().reshape(-1)
            log(
                f"    {k}: {len(nz)} of {a_.shape[0]} rows differ (first {nz[:8].tolist()}, last {nz[-3:].tolist()}), max abs diff {(a_ - b_).abs().max():.3g}"
            )
        log(
            f"(b) first-token logits bit-identical to the full prefill of the same user {ok_logits} (max abs diff {(L2.float() - L_full2.float()).abs().max():.3g}), token {tok2} vs {tok_full2}; (e) recompute lines: {len(recompute)}; "
            f"times (warm): full prefill {t_full2:.2f} s, restore {t_rest:.2f} s, continuation [{s},{E}) {t_cont:.2f} s"
        )
        results.append((s, u2, bad_c, bad_p, ok_logits, ok_tok, bad_s, list(recompute)))
    # (d) untouched
    bad_ref = cmp_state("donor user after the restores", ref_before, user_state(u_ref, E))
    bad_by = cmp_state("bystander after the restores", by_state, user_state(u_by, len(other)))
    for s, u2, bad_c, bad_p, ok_logits, ok_tok, bad_s, rc in results:
        assert (
            not bad_c and not bad_p
        ), f"restore at {s} into user {u2}: carried tensors differ from the snapshot {bad_c[:3]} {bad_p}"
        assert not rc, f"restore at {s}: the continuation recomputed from 0: {rc}"
        assert (
            ok_logits and ok_tok
        ), f"restore at {s} into user {u2}: first-token logits / token differ from the full prefill of the same user"
        assert (
            not bad_s
        ), f"restore at {s} into user {u2}: final decode state differs from the full prefill of the same user: {bad_s}"
    assert not bad_ref and not bad_by
