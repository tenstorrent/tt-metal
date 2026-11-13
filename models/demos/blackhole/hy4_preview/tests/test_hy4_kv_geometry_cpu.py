# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) of the Hy4 KV pool geometry / byte budget (tt/kv_geometry.py), the column sequence-sharding reference (tt/seqshard.py) and the pool admission policy (tt/apc_blocks.py::PoolAdmission).
Run (worktree root):  python3 -m pytest --noconftest -c /dev/null --rootdir=/tmp models/demos/blackhole/hy4_preview/tests/test_hy4_kv_geometry_cpu.py -q"""
import importlib.util
import os
import random

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


G = _load("hy4_kv_geometry_under_test", "../tt/kv_geometry.py")
S = _load("hy4_seqshard_under_test", "../tt/seqshard.py")
AB = _load("hy4_apc_blocks_under_test2", "../tt/apc_blocks.py")
C = 512


# ---- byte budget ---------------------------------------------------------------------------------------------------------------------------------------
def test_checkpoint_shapes_give_the_known_parameter_counts():
    p = G.params()
    assert abs(p["attn"] / 1e9 - 20.99) < 0.01  # 78 layers + MTP, q_a q_b kv_a kv_b linear_gate o_proj
    assert abs(p["routed"] / 1e9 - 753.77) < 0.01  # 256 x 3 x 6144 x 2048 x 78 (77 MoE layers + MTP)
    assert abs(p["shared"] / 1e9 - 2.944) < 0.001
    assert G.INDEX_LAYERS == 22 and G.KV_LAYERS == 79
    cfg_full = [0, 1] + list(range(5, 78, 4))
    assert len(cfg_full) == G.INDEX_FULL_LAYERS


def test_row_formats_and_the_704_byte_alignment_finding():
    f = G.FORMATS
    assert f["scaled_fp8"].logical_row == 656 == 512 + 4 * 4 + 128  # 512 fp8 + 4 fp32 scales + 64 bf16 rope
    assert f["scaled_fp8"].stored_row == 704  # round_up(656, 64): sparse_sdpa strides by buffer()->aligned_page_size()
    assert f["bf16"].stored_row == 1152 and f["fp8"].stored_row == 576  # already 64 B aligned
    assert G.INDEX_KEY_ROW == 136


def test_page_bytes_per_chip():
    assert G.page_bytes_per_chip("scaled_fp8") == 79 * 64 * 704 + 22 * 64 * 136 == 3750912
    assert G.page_bytes_per_chip("scaled_fp8", packed=True) == 79 * 64 * 656 + 22 * 64 * 136
    # per-token cost over the 8 chips of a row, KiB: matches port-plan's 53.5 KiB (packed 656) and the real 57.2 (aligned 704)
    assert abs(G.kv_bytes_per_token_mesh("scaled_fp8", True) / 1024 - 53.5) < 0.05
    assert abs(G.kv_bytes_per_token_mesh("scaled_fp8") / 1024 - 57.2) < 0.05


def test_default_budget_numbers():
    b = G.Budget()
    assert abs(b.weights_gib - 16.0) < 0.05 and abs(b.free_gib - 7.5) < 0.06
    assert b.pages_per_row == 2147 and b.tokens_per_row == 2147 * 512
    # D3-1 primary: B=4 x 1M (1 user per row) fits with 99 spare pages (49k tokens) at the aligned stride; no room for a 10% cache
    assert b.guaranteed_ctx(1) == 1 << 20 and b.pages_per_row - 2048 == 99
    assert b.guaranteed_ctx(1, 0.10) < 1 << 20
    assert b.guaranteed_ctx(2) < 1 << 20  # B=8 x 1M does not fit with SCALED_FP8
    # the 656 packed layout would give 2296 pages
    assert G.Budget(packed=True).pages_per_row == 2296
    # nothing but bfp4 holds B=8 x 1M
    assert G.Budget(fmt="bfp4").guaranteed_ctx(2) == 1 << 20 and G.Budget(fmt="fp8").guaranteed_ctx(2) < 1 << 20
    # D3-1 secondary: B=8 x 1M = 4096 pages/row: bfp4 KV 4402 pages (306 spare); with attention bfp4 too 5238 (1142 spare)
    assert G.Budget(fmt="bfp4").pages_per_row == 4402
    assert G.Budget(fmt="bfp4", attn_bytes=G.BFP4).pages_per_row - 4096 > 1000


def test_reserve_items_sum_below_the_5_gib_reserve():
    assert 3.5 < sum(G.RESERVE_ITEMS.values()) < G.Budget().reserve_gib


def test_sensitivity_one_gib_of_reserve_is_about_286_pages():
    b0, b1 = G.Budget(), G.Budget(reserve_gib=6.0)
    d = b0.pages_per_row - b1.pages_per_row
    assert 280 <= d <= 290  # 8 chips x 1 GiB / 3.75 MB per page-chip... per chip: 1 GiB / 3.75 MB


def test_every_assumption_has_a_measurement():
    ids = [a[0] for a in G.ASSUMPTIONS]
    assert ids == sorted(set(ids)) and all(len(a) == 4 and a[3] for a in G.ASSUMPTIONS)


# ---- page geometry / token placement -------------------------------------------------------------------------------------------------------------------
def test_locate_is_a_bijection_and_roundtrips():
    t = np.arange(0, 40 * C)
    page, col, loc = S.locate(t)
    assert (S.global_pos(page, col, loc) == t).all()
    keys = set(zip(col.tolist(), loc.tolist()))
    assert len(keys) == t.size  # no two tokens share a (column, local row)
    assert loc.max() == 40 * 64 - 1 and col.max() == 7


def test_every_page_is_balanced_over_columns_and_slices_are_tile_aligned():
    t = np.arange(0, 512)
    _, col, loc = S.locate(t)
    assert np.bincount(col).tolist() == [64] * 8
    assert S.SLICE % 32 == 0  # a column slice is whole 32-row tiles (bfp8 index keys, bf16/bfp4 formats)
    # any 512 consecutive tokens (decode window, indexer's recent tokens) hit every column 64 times
    for start in (0, 1, 100, 4097, 99999):
        _, c, _ = S.locate(np.arange(start, start + 512))
        assert np.bincount(c, minlength=8).tolist() == [64] * 8


def test_page_alternative_page128_mod8_is_unbalanced_per_block():
    # the alternative of the task text: 128-token pages, column = page % 8: one 512-token chunk-block lands on 4 columns only
    t = np.arange(512)
    col = (t // 128) % 8
    assert np.bincount(col, minlength=8).tolist() == [128] * 4 + [0] * 4


# ---- sequence-sharded attention: the merge contract --------------------------------------------------------------------------------------------------
def _setup(seed, n_pages=6, H=4, D=576, V=512, K=300):
    rng = np.random.default_rng(seed)
    kv = rng.standard_normal((n_pages * C, D)).astype(np.float64)
    q = rng.standard_normal((H, D))
    ids = rng.choice(n_pages * C, size=K, replace=False).astype(np.int64)
    ids = np.concatenate([ids, np.full(16, S.SENTINEL, dtype=np.int64)])
    return rng, kv, q, ids, V, D**-0.5


@pytest.mark.parametrize("seed", range(5))
def test_per_column_partials_merge_to_the_full_softmax_with_sink(seed):
    rng, kv, q, ids, V, scale = _setup(seed)
    sink = rng.standard_normal(4)
    loc, cnt = S.split_ids(ids)
    assert int(cnt.sum()) == 300
    shards = S.shard_kv(kv, 6)
    parts = [S.partial_attention(q, shards[c], loc[c], V, scale) for c in range(8)]
    got = S.merge(parts, sink)
    want = S.reference_attention(q, kv, ids, V, scale, sink)
    assert np.abs(got - want).max() < 1e-10


def test_sentinels_are_a_contiguous_tail_on_every_column():
    _, _, _, ids, _, _ = _setup(0)
    loc, cnt = S.split_ids(ids)
    for c in range(8):
        valid = loc[c] != S.SENTINEL
        n = int(valid.sum())
        assert valid[:n].all() and not valid[n:].any() and n == cnt[c]


def test_column_with_no_selected_key_is_harmless_and_early_tokens_work():
    # a query at position 100 selects tokens < 100 only: columns 2..7 own none (tokens 0..63 -> column 0, 64..99 -> column 1)
    rng = np.random.default_rng(1)
    kv = rng.standard_normal((C, 576))
    q = rng.standard_normal((4, 576))
    ids = np.concatenate([np.arange(100), np.full(28, S.SENTINEL)]).astype(np.int64)
    loc, cnt = S.split_ids(ids)
    assert cnt[:,].tolist() == [64, 36, 0, 0, 0, 0, 0, 0]
    shards = S.shard_kv(kv, 1)
    parts = [S.partial_attention(q, shards[c], loc[c], 512, 576**-0.5) for c in range(8)]
    assert all(
        np.isinf(p[1][0]) and p[2][0] == 0 for p in parts[2:]
    )  # the op contract: max=-inf, sum=0 on an empty column
    sink = np.zeros(4)
    assert np.abs(S.merge(parts, sink) - S.reference_attention(q, kv, ids, 512, 576**-0.5, sink)).max() < 1e-10


def test_two_stage_topk_equals_global_topk():
    rng = np.random.default_rng(3)
    for n in (5000, 70000):
        sc = rng.standard_normal(n)
        want = np.argsort(-sc, kind="stable")[:2048]
        got = S.two_stage_topk(sc, 2048)
        assert set(got.tolist()) == set(want.tolist()) and (sc[got][:-1] >= sc[got][1:]).all()


def test_selected_rows_balance_over_columns_for_a_recent_window_and_random_ids():
    rng = np.random.default_rng(4)
    recent = np.arange(300000 - 2048, 300000)
    _, c, _ = S.locate(recent)
    assert np.bincount(c, minlength=8).tolist() == [256] * 8
    rand = rng.choice(1 << 20, size=2048, replace=False)
    _, c, _ = S.locate(rand)
    assert np.bincount(c, minlength=8).max() < 330  # mean 256, 5 sigma


# ---- admission --------------------------------------------------------------------------------------------------------------------------------------
def _adm(pages=2147, users=4, **kw):
    idx = AB.Hy4BlockIndex(rows=2, users_per_row=users, pages_per_row=pages)
    return idx, AB.PoolAdmission(idx, **kw)


def _start(idx, adm, rid, row, user, n, max_new=8192, hit=0):
    assert adm.admit(rid, row, n, max_new, hit)
    idx.bind(rid, 0, user=user, prompt_len=n)
    idx.page_table(rid, n)


def test_b8_bfp4_two_million_token_users_per_row_fit():
    idx, adm = _adm(pages=4402, users=2)
    for u in range(2):
        assert adm.admit(f"r{u}", 0, (1 << 20) - 8192, 8192)
        idx.bind(f"r{u}", 0, user=u)
        idx.page_table(f"r{u}", (1 << 20) - 8192)
    assert adm.avail(0) == 4402 - 4096
    idx.check()


def test_b4_scaled_fp8_one_million_user_per_row_leaves_99_pages():
    idx, adm = _adm(pages=2147, users=1)
    assert adm.admit("g", 0, (1 << 20) - 8192, 8192)
    idx.bind("g", 0, user=0)
    idx.page_table("g", (1 << 20) - 8192)
    assert adm.avail(0) == 99
    idx.check()


def test_a_million_token_user_is_admitted_on_an_empty_row_only():
    idx, adm = _adm()
    assert adm.admit("g", 0, (1 << 20) - 8192, 8192)  # 2048 pages of 2147
    idx.bind("g", 0, user=0)
    idx.page_table("g", (1 << 20) - 8192)
    assert not adm.admit("g2", 0, 1 << 19, 8192)  # a 524288 user does not fit beside it
    assert adm.admit("small", 0, 40000, 4096)  # 88 pages: still fits (2147 - 2048 = 99 left)
    adm.release("g")
    idx.release("g")
    assert adm.admit("g2", 0, 1 << 19, 8192)


def test_pick_row_prefers_hit_row_then_most_room_and_waits_when_nothing_fits():
    idx, adm = _adm()
    _start(idx, adm, "a", 0, 0, 900000)
    assert adm.pick_row(100000, 4096, rows_with_free_user=[0, 1]) == 1  # row 1 has more room
    assert (
        adm.pick_row(100000, 4096, hit_row=0, hit_tokens=50000, rows_with_free_user=[0, 1]) == 0
    )  # the hit row, if it fits
    assert adm.pick_row(100000, 4096, hit_row=0, hit_tokens=0, rows_with_free_user=[0]) == -1 or adm.avail(0) >= 200
    _start(idx, adm, "b", 1, 4, 900000)
    assert adm.pick_row(500000, 4096, rows_with_free_user=[0, 1]) == -1  # wait


def test_draining_row_takes_only_giants():
    idx, adm = _adm()
    adm.start_drain(0)
    assert adm.pick_row(10000, 1024, rows_with_free_user=[0]) == -1
    assert adm.pick_row((1 << 20) - 8192, 8192, rows_with_free_user=[0]) == 0
    adm.end_drain(0)
    assert adm.pick_row(10000, 1024, rows_with_free_user=[0]) == 0


def test_promise_is_never_broken_decode_growth_inside_it_cannot_fail():
    idx, adm = _adm(pages=300, users=4)
    rng = random.Random(1)
    live = []
    for k in range(200):
        n = rng.choice([5000, 20000, 60000])
        if adm.admit(k, 0, n, 4096) and any(True for u in range(4) if u not in [x[1] for x in live]):
            user = next(u for u in range(4) if u not in [x[1] for x in live])
            idx.bind(k, 0, user=user, prompt_len=n)
            idx.page_table(k, n)
            live.append((k, user, n))
        if live and rng.random() < 0.4:
            rid, user, n = live.pop(rng.randrange(len(live)))
            for t in range(n, n + 4096, 700):
                adm.grow(rid, t)  # inside the promise: never raises
            adm.release(rid)
            idx.release(rid)
        idx.check()
        assert adm.avail(0) >= 0


def test_grow_beyond_the_promise_extends_or_names_a_victim():
    idx, adm = _adm(pages=100, users=4)
    _start(idx, adm, "giant", 0, 0, 30 * C, max_new=C)  # 31 pages
    _start(idx, adm, "a", 0, 1, 30 * C, max_new=C)
    _start(idx, adm, "b", 0, 2, 30 * C, max_new=C)
    adm.entitlement_pages = 10
    adm.promises["giant"].giant = True
    adm.grow("a", 31 * C)  # one page beyond its promise (the last 7 pages of the row are free): extends
    with pytest.raises(AB.PoolExhausted) as ei:  # allow-pytest.raises: CPU-only module
        adm.grow("b", 80 * C)
    assert ei.value.victim in ("giant", "a", "b") and ei.value.row == 0
    assert ei.value.victim != "b" or not [r for r, p in adm.promises.items() if p.pages > 10 and r != "b"]


def test_cached_blocks_count_as_room_but_pinned_hit_blocks_do_not():
    idx, adm = _adm(pages=40, users=4)
    t = np.random.default_rng(5).integers(1, 1000, size=30 * C + 5)
    idx.bind("d", 0, user=0, prompt_len=len(t))
    idx.page_table("d", len(t))
    for st in range(0, 30 * C, C):
        idx.finish_chunk(idx.begin_chunk("d", st, st + C, t, len(t)))
    idx.release("d")
    assert idx.free_pages(0) == 10 and idx.reclaimable_pages(0) == 30 and adm.avail(0) == 40
    follow = np.concatenate([t[: 30 * C], np.arange(1, 2000)])
    plan = idx.plan_hit("f", follow, len(follow))
    assert plan.hit_tokens == 30 * C
    assert idx.reclaimable_pages(0) == 0  # the pinned chain is not reclaimable while the hit lives
    assert adm.avail(0) == 10
    assert adm.admit("f", 0, len(follow), 512, hit_tokens=plan.hit_tokens)  # needs 4 new pages beyond the 30 shared
    assert adm.avail(0) == 10 - (len(adm.promises) and adm.promises["f"].pages - 30)


def test_total_pages_caps_at_one_million_tokens():
    _, adm = _adm()
    assert adm.total_pages((1 << 20) - 10, 1 << 20) == 2048
    assert adm.total_pages(100, 1) == 1
