# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the host-side bookkeeping of the MTP spec-decode engine.

Covers the pure helpers in tt/spec_engine.py and the page-table helpers on ``Qwen36Model``
(``_fit_spec_page_table``). No device.

Convention: after a verify at start_pos s with m accepted drafts, committed_tokens[u, :m+1] =
accepted drafts then the bonus token, at positions s+1 .. s+m+1.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_engine_host.py -q
"""
import json
import os
import types

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen36.tt.model import Qwen36Model, spec_kv_write_conflicts
from models.demos.blackhole.qwen36.tt.spec_engine import (
    PLACEHOLDER_TOKEN_ID,
    MTPSpecEngine,
    _elem_bytes,
    accepted_plan,
    anchor_selector,
    draft_rows_fit,
    draft_step_plan,
    last_row_selector,
    mask_unused_columns,
    note_withheld_drafts,
    parked_rows_host,
    propose_skip_plan,
    reseed_row_plan,
    reseed_rows_host,
    sanitize_verify_block,
    spec_memory_costs,
    spec_round_up,
    spec_table_width,
    table_capacity,
    verify_write_table,
)

SHAPES = [(1, 12), (2, 8), (4, 4), (8, 4), (3, 5)]


# --------------------------------------------------------------------------- #
# Selectors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B,T", SHAPES)
def test_anchor_selector_one_hot_layout(B, T):
    """One 1.0 per user row, at column u*T + m[u]; everything else is zero."""
    m = [(u * 3 + 1) % T for u in range(B)]
    sel = anchor_selector(m, B, T)
    assert sel.shape == (1, 1, B, B * T) and sel.dtype == torch.bfloat16
    assert torch.all(sel.float().sum(-1) == 1.0)
    for u in range(B):
        assert sel[0, 0, u, u * T + m[u]] == 1.0


@pytest.mark.parametrize("B,T", SHAPES)
def test_anchor_selector_gathers_rows(B, T):
    """sel @ feed returns exactly the named verify rows (fp32 matmul of a one-hot is exact)."""
    torch.manual_seed(0)
    D = 16
    m = [T - 1 - (u % T) for u in range(B)]
    feed = torch.randn(1, 1, B * T, D)
    out = anchor_selector(m, B, T).float() @ feed
    assert out.shape == (1, 1, B, D)
    for u in range(B):
        assert torch.equal(out[0, 0, u], feed[0, 0, u * T + m[u]])


@pytest.mark.parametrize("bucket,i", [(32, 0), (128, 99), (512, 511)])
def test_last_row_selector_picks_row(bucket, i):
    """One-hot at column i; sel @ feed is row i of the bucket."""
    torch.manual_seed(1)
    sel = last_row_selector(i, bucket)
    assert sel.shape == (1, 1, 1, bucket) and sel.dtype == torch.bfloat16
    assert int(sel.float().sum()) == 1 and sel[0, 0, 0, i] == 1.0
    feed = torch.randn(bucket, 8)
    assert torch.equal(sel[0, 0].float() @ feed, feed[i : i + 1])


# --------------------------------------------------------------------------- #
# accepted_plan / reseed_row_plan
# --------------------------------------------------------------------------- #
def test_accepted_plan_only_bonus():
    """accepted_counts == 1 (m = 0): the bonus alone; anchor is the user's first verify row."""
    tokens = [[50, 0, 0, 0]]
    pos = [[11, 0, 0, 0]]
    assert accepted_plan(tokens, pos, [1], 1, 4) == ([0], [0], [50], [10])


def test_accepted_plan_partial():
    tokens = [[7, 8, 9, 0]]
    pos = [[11, 12, 13, 0]]
    assert accepted_plan(tokens, pos, [3], 1, 4) == ([2], [2], [9], [12])


def test_accepted_plan_full_acceptance():
    """accepted_counts == T (m == K): every draft taken, bonus is the last committed token."""
    tokens = [[7, 8, 9, 10]]
    pos = [[11, 12, 13, 14]]
    assert accepted_plan(tokens, pos, [4], 1, 4) == ([3], [3], [10], [13])


def test_accepted_plan_two_users_different_counts():
    tokens = torch.tensor([[7, 8, 9], [20, 21, 22]])
    pos = torch.tensor([[11, 12, 13], [101, 102, 103]])
    m, rows, pending, apos = accepted_plan(tokens, pos, torch.tensor([2, 3]), 2, 3)
    assert m == [1, 2]
    assert rows == [1, 3 + 2]
    assert pending == [8, 22]
    assert apos == [11, 102]


@pytest.mark.parametrize("bad", [0, 5])
def test_accepted_plan_rejects_out_of_range(bad, expect_error):
    tokens, pos = [[1, 2, 3, 4]], [[1, 2, 3, 4]]
    with expect_error(AssertionError, ".*"):
        accepted_plan(tokens, pos, [bad], 1, 4)


def test_reseed_row_plan_real_rows_and_padding():
    B, T = 2, 4
    tokens = [[7, 8, 9, 10], [20, 21, 22, 23]]
    pos = [[11, 12, 13, 14], [101, 102, 103, 104]]
    plan = reseed_row_plan(tokens, pos, [2, 0], B, T)
    assert len(plan) == B * T
    assert plan[:4] == [(7, 10, True), (8, 11, True), (0, 0, False), (0, 0, False)]
    assert plan[4:] == [(0, 0, False)] * 4


@pytest.mark.parametrize("B,T", SHAPES)
def test_reseed_row_plan_counts(B, T):
    """Exactly m[u] real rows per user, all at the front of the user's T rows."""
    tokens = [[100 * u + j + 1 for j in range(T)] for u in range(B)]
    pos = [[50 * u + j + 5 for j in range(T)] for u in range(B)]
    m = [u % T for u in range(B)]
    plan = reseed_row_plan(tokens, pos, m, B, T)
    assert len(plan) == B * T
    for u in range(B):
        real = [r[2] for r in plan[u * T : (u + 1) * T]]
        assert real == [i < m[u] for i in range(T)]
    assert sum(r[2] for r in plan) == sum(m)


def test_reseed_row_plan_m_zero_has_no_real_rows():
    plan = reseed_row_plan([[1, 2, 3]], [[4, 5, 6]], [0], 1, 3)
    assert not any(r[2] for r in plan)


@pytest.mark.parametrize("B,T", [(1, 4), (2, 5), (4, 4)])
@pytest.mark.parametrize("seed", range(4))
def test_round_trip_verify_to_plan(B, T, seed):
    """Simulate a greedy verify: draft d[i] is accepted iff a[i] == d[i] (a[j] = argmax after input j);
    m is the accepted prefix length and the bonus is a[m]. The plans must agree."""
    g = torch.Generator().manual_seed(seed)
    K = T - 1
    starts = [int(x) for x in torch.randint(5, 500, (B,), generator=g)]
    committed_tokens = [[0] * T for _ in range(B)]
    committed_positions = [[0] * T for _ in range(B)]
    drafts, argmax, m_true = [], [], []
    for u in range(B):
        d = [int(x) for x in torch.randint(1, 20, (K,), generator=g)]
        a = [int(x) for x in torch.randint(1, 20, (T,), generator=g)]
        # force a spread of acceptance lengths across seeds/users
        for i in range(min((seed + u) % T, K)):
            a[i] = d[i]
        m = 0
        while m < K and a[m] == d[m]:
            m += 1
        for i in range(m + 1):
            committed_tokens[u][i] = d[i] if i < m else a[m]
            committed_positions[u][i] = starts[u] + 1 + i
        drafts.append(d)
        argmax.append(a)
        m_true.append(m)

    m, rows, pending, anchor_pos = accepted_plan(committed_tokens, committed_positions, [x + 1 for x in m_true], B, T)
    assert m == m_true
    for u in range(B):
        assert rows[u] == u * T + m_true[u]
        assert pending[u] == argmax[u][m_true[u]]
        assert anchor_pos[u] == starts[u] + m_true[u]

    plan = reseed_row_plan(committed_tokens, committed_positions, m, B, T)
    for u in range(B):
        real = [r for r in plan[u * T : (u + 1) * T] if r[2]]
        assert [r[1] for r in real] == [starts[u] + i for i in range(m_true[u])]  # slots s .. s+m-1
        assert [r[0] for r in real] == drafts[u][: m_true[u]]  # slot s+i pairs with d[i]


# --------------------------------------------------------------------------- #
# Qwen36Model page-table helpers
# --------------------------------------------------------------------------- #
def test_fit_page_table_pads_with_zeros():
    out = Qwen36Model._fit_spec_page_table(torch.tensor([[3, 4], [5, 6]]), 5)
    assert out.tolist() == [[3, 4, 0, 0, 0], [5, 6, 0, 0, 0]]


def test_fit_page_table_clips():
    out = Qwen36Model._fit_spec_page_table(torch.arange(12).reshape(2, 6), 4)
    assert out.tolist() == [[0, 1, 2, 3], [6, 7, 8, 9]]


def test_fit_page_table_exact_width_unchanged():
    pt = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    assert torch.equal(Qwen36Model._fit_spec_page_table(pt, 3), pt)


def test_fit_page_table_1d_is_one_row():
    out = Qwen36Model._fit_spec_page_table(torch.tensor([1, 2, 3]), 4)
    assert out.shape == (1, 4) and out.tolist() == [[1, 2, 3, 0]]


@pytest.mark.parametrize("src", [torch.arange(6, dtype=torch.int64).reshape(2, 3), [[1, 2, 3], [4, 5, 6]]])
@pytest.mark.parametrize("nb", [2, 3, 7])
def test_fit_page_table_int32_contiguous(src, nb):
    out = Qwen36Model._fit_spec_page_table(src, nb)
    assert out.dtype == torch.int32 and out.is_contiguous() and out.shape == (2, nb)


def test_fit_page_table_clip_of_noncontiguous_is_contiguous():
    out = Qwen36Model._fit_spec_page_table(torch.arange(12, dtype=torch.int32).reshape(2, 6), 3)
    assert out.is_contiguous()


# --------------------------------------------------------------------------- #
# Runner contract helpers
# --------------------------------------------------------------------------- #
K4 = 4


def _block(rows):
    """rows: list of (start, tokens) -> padded [B, 1+K4] tokens/positions; start None = inactive row."""
    toks, poss = [], []
    for start, real in rows:
        if start is None:
            toks.append([0] * (K4 + 1))
            poss.append([-1] * (K4 + 1))
            continue
        pad = K4 + 1 - len(real)
        toks.append(list(real) + [-1] * pad)
        poss.append([start + j for j in range(len(real))] + [-1] * pad)
    return torch.tensor(toks, dtype=torch.int32), torch.tensor(poss, dtype=torch.int32)


def test_sanitize_full_partial_zero_rows():
    tokens, pos = _block([(10, [5, 6, 7, 8, 9]), (20, [3, 4, 5]), (30, [2])])
    clean, start0, active = sanitize_verify_block(tokens, pos, [4, 2, 0], K4)
    assert clean.dtype == torch.int64 and start0.dtype == torch.int64
    assert clean.tolist() == [[5, 6, 7, 8, 9], [3, 4, 5, 3, 3], [2, 2, 2, 2, 2]]
    assert start0.tolist() == [10, 20, 30] and active.tolist() == [True] * 3


def test_sanitize_inactive_row():
    tokens, pos = _block([(10, [5, 6]), (None, [])])
    clean, start0, active = sanitize_verify_block(tokens, pos, [1, 0], K4)
    assert active.tolist() == [True, False]
    assert clean[1].tolist() == [0] * 5 and start0.tolist() == [10, 0]


def test_sanitize_position_zero_is_active():
    tokens, pos = _block([(0, [5, 6])])
    assert sanitize_verify_block(tokens, pos, [1], K4)[2].tolist() == [True]


@pytest.mark.parametrize("n", [-1, K4 + 1])
def test_sanitize_n_out_of_range(n, expect_error):
    tokens, pos = _block([(10, [5, 6])])
    with expect_error(ValueError, "row 0.*num_valid_drafts"):
        sanitize_verify_block(tokens, pos, [n], K4)


def test_sanitize_non_consecutive_positions(expect_error):
    tokens, pos = _block([(10, [5, 6, 7])])
    pos[0, 2] += 1
    with expect_error(ValueError, "row 0.*consecutive"):
        sanitize_verify_block(tokens, pos, [2], K4)


def test_sanitize_positions_past_n_ignored():
    tokens, pos = _block([(10, [5, 6, 7])])
    pos[0, 3:] = 999
    tokens[0, 3:] = 77
    assert sanitize_verify_block(tokens, pos, [2], K4)[0][0].tolist() == [5, 6, 7, 5, 5]


def test_sanitize_negative_real_token(expect_error):
    tokens, pos = _block([(10, [5, 6, 7])])
    tokens[0, 1] = -1
    with expect_error(ValueError, "row 0.*negative"):
        sanitize_verify_block(tokens, pos, [2], K4)


def test_sanitize_inactive_row_with_drafts(expect_error):
    tokens, pos = _block([(None, [])])
    with expect_error(ValueError, "row 0.*inactive"):
        sanitize_verify_block(tokens, pos, [1], K4)


def test_sanitize_wrong_shapes(expect_error):
    tokens, pos = _block([(10, [5, 6]), (20, [1])])
    with expect_error(ValueError, "tokens"):
        sanitize_verify_block(tokens[:, :3], pos[:, :3], [1, 0], K4)
    with expect_error(ValueError, "start_pos"):
        sanitize_verify_block(tokens, pos[:, :3], [1, 0], K4)
    with expect_error(ValueError, "num_valid_drafts"):
        sanitize_verify_block(tokens, pos, [1], K4)
    with expect_error(ValueError, "tokens"):
        sanitize_verify_block(tokens[0], pos[0], [1], K4)


def test_mask_unused_columns():
    ids = torch.arange(15, dtype=torch.int32).reshape(3, 5)
    out = mask_unused_columns(ids, torch.tensor([4, 1, 0], dtype=torch.int32))
    P = PLACEHOLDER_TOKEN_ID
    assert out.dtype == torch.int32
    assert out.tolist() == [[0, 1, 2, 3, 4], [5, 6, P, P, P], [10, P, P, P, P]]
    assert ids[2, 1].item() == 11  # input untouched


@pytest.mark.parametrize("K,cap", [(1, 16), (4, 128), (7, 4096)])
def test_draft_rows_fit_boundary(K, cap):
    last_ok = cap - 2 - K
    assert draft_rows_fit([last_ok, last_ok + 1, 0, cap], K, cap) == [True, False, True, False]


# --------------------------------------------------------------------------- #
# spec_memory_costs (pure formula; device equality is checked in test_spec_engine)
# --------------------------------------------------------------------------- #
def _cached_ckpt(repo="Qwen/Qwen3.8-27B"):
    """Local snapshot dir of ``repo`` from the HF cache, or None when it is not cached."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(os.environ.get("HF_MODEL", repo), local_files_only=True)
    except Exception:
        return None


_MIB = 1 << 20


@pytest.fixture(scope="module")
def qwen38_text_config():
    ckpt = _cached_ckpt()
    if ckpt is None or not os.path.exists(os.path.join(ckpt, "config.json")):
        pytest.skip("Qwen3.8-27B config not in the local HF cache")
    with open(os.path.join(ckpt, "config.json")) as f:
        return json.load(f)["text_config"]


def _mib(d):
    return {k: round(v / _MIB, 3) for k, v in d.items()}


def test_spec_memory_costs_per_token_and_log(qwen38_text_config):
    tc = qwen38_text_config
    assert tc["num_key_value_heads"] // 4 == 1 and tc["head_dim"] == 256
    c = spec_memory_costs(tc, 4, 1, 3, nb=72)
    assert c["per_token"]["mtp_kv"] == 2 * 1 * 256 * 2 == 1024
    assert spec_memory_costs(tc, 4, 1, 3, kv_dtype_bytes=1)["per_token"]["mtp_kv"] == 512
    assert c["fixed"]["mtp_scratch_block"] == 64 * 1024
    for k in ("per_seq", "fixed", "per_token"):
        logger.info(f"[spec_memory_costs] B=1 K=3 {k} (MiB/device): {_mib(c[k])}")


def test_spec_memory_costs_scaling_and_totals(qwen38_text_config):
    tc, nb = qwen38_text_config, 72
    for K in (1, 3):
        c1 = spec_memory_costs(tc, 4, 1, K, nb=nb)
        for B in (2, 4):
            if B * (K + 1) > 32:
                continue
            cb = spec_memory_costs(tc, 4, B, K, nb=nb)
            for k, v in c1["per_seq"].items():
                assert cb["per_seq"][k] == B * v, (K, B, k)  # linear in users
            assert cb["per_token"] == c1["per_token"]
            assert cb["fixed"]["mtp_scratch_block"] == c1["fixed"]["mtp_scratch_block"]
            assert cb["fixed"]["verify_io"] < 2 * B * c1["fixed"]["verify_io"]  # feed rows / selectors only
    # per_seq: GDN state grows linearly in T (ring exactly; windows by the padded row count)
    r = [spec_memory_costs(tc, 4, 1, K, nb=nb)["per_seq"]["gdn_ring"] for K in (1, 3, 7)]
    assert r[1] * 2 == r[0] * 4 and r[2] * 2 == r[1] * 4
    for c in (spec_memory_costs(tc, 4, 2, 3, nb=nb), spec_memory_costs(tc, 4, 1, 1, nb=nb)):
        for k in ("per_seq", "fixed", "per_token"):
            d = dict(c[k])
            assert d.pop("total") == sum(d.values())


def test_spec_memory_costs_page_tables_only_depend_on_nb(qwen38_text_config):
    a = spec_memory_costs(qwen38_text_config, 4, 1, 3, nb=0)
    b = spec_memory_costs(qwen38_text_config, 4, 1, 3, nb=100)
    c = spec_memory_costs(qwen38_text_config, 4, 1, 3, max_seq_len=6400, block_size=64)
    assert b == c and a["per_seq"] == b["per_seq"] and a["per_token"] == b["per_token"]
    assert (
        b["fixed"]["verify_io"] > a["fixed"]["verify_io"]
        and b["fixed"]["engine_buffers"] > a["fixed"]["engine_buffers"]
    )


def test_elem_bytes_lookup(expect_error):
    import ttnn

    assert [_elem_bytes(d) for d in (ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32)] == [2, 4, 4, 4]
    with expect_error(ValueError, "BFLOAT8_B|bfloat8_b"):
        _elem_bytes(ttnn.bfloat8_b)


# --------------------------------------------------------------------------- #
# Page-table width / capacity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "max_seq,K,blk,want",
    [
        (1, 3, 64, 1),
        (2048, 3, 64, 33),
        (2048 - 4, 3, 64, 32),
        (2048 - 3, 3, 64, 33),
        (4096, 7, 64, 65),
        (131072, 3, 64, 2049),
        (100, 3, 32, 4),
    ],
)
def test_spec_table_width(max_seq, K, blk, want):
    w = spec_table_width(max_seq, K, blk)
    assert w == want
    assert w * blk >= max_seq + K + 1


def test_spec_round_up_and_capacity():
    assert [spec_round_up(n) for n in (1, 32, 33, 64)] == [32, 32, 64, 64]
    assert table_capacity(5, 64) == 320
    # capacity uses the caller's width, not the padded one: a 5-block table padded to 32 still ends at 320
    assert not draft_rows_fit([319 - 1 - 3 + 1], 3, table_capacity(5, 64))[0]
    assert draft_rows_fit([319 - 1 - 3], 3, table_capacity(5, 64))[0]


def test_reseed_rows_host_and_draft_step_plan():
    B, T, K, nb, scratch = 2, 4, 3, 5, 99
    ctok = torch.tensor([[11, 12, 13, 14], [21, 22, 23, 24]])
    cpos = torch.tensor([[10, 11, 12, 13], [30, 31, 32, 33]])
    pt = torch.arange(B * nb, dtype=torch.int32).reshape(B, nb)
    plan = reseed_row_plan(ctok, cpos, [2, 0], B, T)
    tok, pos, rows = reseed_rows_host(plan, pt, T, nb, scratch)
    assert tok.flatten().tolist() == [11, 12, 0, 0, 0, 0, 0, 0]
    assert pos.tolist() == [9, 10, 0, 0, 0, 0, 0, 0]
    assert torch.equal(rows[0], pt[0]) and torch.equal(rows[1], pt[0])
    assert bool((rows[2:] == scratch).all())
    # padding-only plan never reads pt
    _, _, rows0 = reseed_rows_host(reseed_row_plan(ctok, cpos, [0, 0], B, T), None, T, nb, scratch)
    assert bool((rows0 == scratch).all())
    t0, poss = draft_step_plan([7, 8], [5, 40], K)
    assert t0.tolist() == [[7], [8]] and t0.dtype == torch.int32
    assert [v.tolist() for v in poss] == [[5, 40], [6, 41], [7, 42]]


# --------------------------------------------------------------------------- drafter skip
def _note(offered, marked, spec, streak, n, limit=16):
    return note_withheld_drafts(offered, marked, spec, streak, torch.tensor(n), limit)


def test_sampled_like_slot_marked_at_streak_limit():
    """Offered, withheld every step, never verified with drafts -> marked exactly at the 16th."""
    marked, spec, streak = set(), set(), {}
    for i in range(1, 17):
        offered = {0}
        marked = _note(offered, marked, spec, streak, [0])
        assert offered == set()
        assert marked == (set() if i < 16 else {0})


def test_greedy_like_slot_never_marked():
    marked, spec, streak = set(), set(), {}
    marked = _note({0}, marked, spec, streak, [2])
    for _ in range(50):
        marked = _note({0}, marked, spec, streak, [0])
    assert marked == set() and 0 in spec


def test_zero_stretch_then_drafts_never_marked():
    marked, spec, streak = set(), set(), {}
    for _ in range(8):
        marked = _note({0}, marked, spec, streak, [0])
    marked = _note({0}, marked, spec, streak, [3])
    assert marked == set() and streak[0] == 0
    for _ in range(40):
        marked = _note({0}, marked, spec, streak, [0])
    assert marked == set()


def test_streak_resets_on_drafts_and_not_offered_ignored():
    marked, spec, streak = set(), set(), {}
    for _ in range(15):
        marked = _note({0}, marked, spec, streak, [0])
    assert streak[0] == 15 and marked == set()
    assert _note(set(), marked, spec, streak, [0]) == set() and streak[0] == 15  # not offered: unchanged
    assert _note(set(), {1}, spec, streak, [0, 0]) == {1}  # mark persists


def test_propose_skip_plan():
    assert propose_skip_plan(set(), 2) == ([False, False], False)
    assert propose_skip_plan({1}, 2) == ([False, True], False)
    assert propose_skip_plan({0, 1}, 2) == ([True, True], True)


def test_mark_cleared_on_release_and_prefill(expect_error):
    def mk():
        return types.SimpleNamespace(
            _active={0},
            _fresh={},
            _parked={},
            _prompts={},
            _fresh_dirty=set(),
            _offered={0},
            _no_draft={0},
            _speculated={0},
            _withheld_streak={0: 5},
            B=1,
            model=types.SimpleNamespace(),
            _last_hidden=[],
        )

    def clean(e):
        return e._no_draft == set() and e._offered == set() and e._speculated == set() and e._withheld_streak == {}

    eng = mk()
    MTPSpecEngine.release(eng, 0)
    assert clean(eng)
    eng = mk()
    with expect_error(AssertionError, ".*"):  # stops at "prefill before prepare", after the state is cleared
        MTPSpecEngine.prefill(eng, 0, [1, 2, 3], None)
    assert clean(eng)


def _bare_engine(B, K=2, **kw):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size = B, K, K + 1, 64
    e._active, e._fresh, e._offered = set(range(B)), {}, set()
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._no_draft, e._speculated, e._withheld_streak = set(range(B)), set(range(B)), {u: 3 for u in range(B)}
    e._row_active, e._last_pt, e._last_cap_blocks, e.skipped_propose_calls = None, None, 4, 0
    e.__dict__.update(kw)
    return e


def test_release_reoffers_drafts_to_all_rows():
    e = _bare_engine(3, _fresh={0: 1, 1: 1})
    MTPSpecEngine.release(e, 0)
    assert e._no_draft == set() and e._withheld_streak == {}
    assert e._speculated == {1, 2} and e._active == {1, 2} and e._fresh == {1: 1}


@pytest.mark.parametrize("B", [1, 2])
def test_propose_all_marked_skips_only_at_b1(B):
    e, K = _bare_engine(B), 2
    calls = []

    def fake_propose(*a, **kw):
        if B == 1:
            raise AssertionError("propose called")
        calls.append(kw)
        return torch.zeros(B, K, dtype=torch.int32)

    e.propose = fake_propose
    ctok = torch.ones(B, K + 1, dtype=torch.int32)
    cpos = (torch.arange(K + 1) + 1).repeat(B, 1).to(torch.int32)
    acc = torch.ones(B, dtype=torch.int32)
    res = e.propose_draft_tokens(K, ctok, cpos, acc, page_table=torch.zeros(B, 4, dtype=torch.int32))
    assert res.draft_token_ids.shape == (B, K) and not bool(res.num_valid.any())
    if B == 1:
        assert e.skipped_propose_calls == 1 and not calls
    else:
        assert (
            e.skipped_propose_calls == 0 and calls[0]["decline"] == [True, True] and calls[0]["keep_kv"] == [True, True]
        )


def test_decode_forward_remap_after_validation(expect_error):
    B, K = 2, 2
    order = []
    e = _bare_engine(B, _gdn=[])
    e.no_draft_streak = 16
    e.remap_slots = lambda r: order.append("remap")
    e.verify = lambda *a, **k: (order.append("verify"), (torch.zeros(B, K + 1, dtype=torch.int32), None))[1]
    tok = torch.ones(B, K + 1, dtype=torch.int32)
    pos = (torch.arange(K + 1) + 5).repeat(B, 1).to(torch.int32)
    acc = torch.ones(B, dtype=torch.int32)
    kw = dict(accepted_counts=acc, page_table=None, slot_remap=[1, 0])
    with expect_error(ValueError, ".*"):
        e.decode_forward(tok, pos, num_valid_drafts=torch.full((B,), K + 5, dtype=torch.int32), **kw)
    assert order == []
    e.decode_forward(tok, pos, num_valid_drafts=torch.zeros(B, dtype=torch.int32), **kw)
    assert order == ["remap", "verify"]


def test_parked_rows_host():
    D, T = 2, 4
    ring = torch.tensor([10.0 * d + t for d in range(D) for t in range(T)]).reshape(D * T, 1, 1, 1)
    e_prev = torch.arange(D * (3 + T), dtype=torch.float32).reshape(D, 3 + T, 1)
    slot, e = parked_rows_host(ring, e_prev, 0, T)
    assert slot.flatten().tolist() == [0.0, 10.0] and torch.equal(e, e_prev)
    slot, e = parked_rows_host(ring, e_prev, 3, T)
    assert slot.flatten().tolist() == [3.0, 13.0] and slot.shape == (D, 1, 1, 1)
    assert torch.equal(e[:, :4], e_prev[:, 3:]) and not bool(e[:, 4:].any())


class _FakeGdn:
    def __init__(self, T):
        self.T, self.saved, self.loaded = T, [], []

    def save_spec_user(self, u):
        self.saved.append(u)
        return torch.full((2 * self.T, 1, 1, 1), float(u)), torch.zeros(2, 3, 1)

    def load_spec_users(self, states):
        self.loaded.append(states)


def test_verify_parks_and_restores_left_out_row():
    B, K = 2, 2
    T = K + 1
    gdn, calls = _FakeGdn(T), []

    def verify_traced(*a, **kw):
        calls.append(a)
        return list(range(B * T)), "feed", None

    model = types.SimpleNamespace(
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32), verify_traced=verify_traced
    )
    e = _bare_engine(B, K, captured=True, nb=4, _timing=False, _gdn=[gdn], model=model)
    tok = torch.ones(B, T, dtype=torch.int32)
    pos = torch.full((B,), 5, dtype=torch.int32)
    pt = torch.zeros(B, 4, dtype=torch.int32)
    for _ in range(2):
        e.verify(tok, pos, torch.ones(B, dtype=torch.int32), pt, active=[True, False])
    assert gdn.saved == [1] and set(e._parked) == {1} and not gdn.loaded
    e.verify(tok, pos, torch.tensor([1, 3], dtype=torch.int32), pt)
    assert list(gdn.loaded[0]) == [1] and gdn.loaded[0][1][0].flatten().tolist() == [1.0, 1.0]
    assert calls[-1][2] == [0, 0] and e._parked == {}


def test_release_and_remap_cover_parked_state():
    e = _bare_engine(2, _parked={0: ["a"]}, _prompts={0: ("p", None)}, _fresh_dirty={0})
    MTPSpecEngine.release(e, 0)
    assert e._parked == {} and e._prompts == {} and e._fresh_dirty == set()
    e = _bare_engine(2, _parked={0: ["a"]}, _prompts={0: ("p", None)}, _fresh_dirty={0}, _last_hidden=[0, 1])
    e._gdn = [types.SimpleNamespace(remap_spec_slots=lambda r: None)]
    MTPSpecEngine.remap_slots(e, [1, 0])
    assert set(e._parked) == {1} and set(e._prompts) == {1} and e._fresh_dirty == {1}


def test_verify_defers_fresh_row_left_out():
    B, K = 2, 2
    T = K + 1
    row = torch.zeros(4, dtype=torch.int32)
    model = types.SimpleNamespace(_fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32))
    e = _bare_engine(
        B,
        K,
        captured=True,
        nb=4,
        _timing=False,
        _gdn=[],
        model=model,
        _fresh={0: 5, 1: 7},
        _prompts={0: ([1] * 5, row), 1: ([2] * 7, row)},
        _mtp_pt_rows=[None] * B,
        _last_hidden=[None] * B,
    )
    e._stage_mtp_tables = e._warm_mtp_last = lambda *a, **k: None
    seeds, joins, prefills = [], [], []
    e._seed = lambda *a: (seeds.append(1), (torch.zeros(B, T, dtype=torch.int32), None))[1]
    e._join = lambda *a: (joins.append(a[5]), (torch.zeros(B, T, dtype=torch.int32), None))[1]

    def prefill(slot, prompt, r):
        prefills.append((slot, prompt))
        e._fresh[slot] = len(prompt)
        e._fresh_dirty.discard(slot)

    e.prefill = prefill
    tok = torch.ones(B, T, dtype=torch.int32)
    pt = torch.zeros(B, 4, dtype=torch.int32)
    one = torch.ones(B, dtype=torch.int32)
    e.verify(tok, torch.tensor([5, 0]), one, pt, active=[True, False])
    assert seeds == [1] and e._fresh == {1: 7} and e._fresh_dirty == {1} and 0 not in e._prompts
    e.verify(tok, torch.tensor([6, 7]), one, pt)
    assert prefills == [(1, [2] * 7)] and joins == [[1]] and e._fresh == {} and e._fresh_dirty == set()


def test_verify_write_table():
    pt = torch.tensor([[5, 9, 7], [3, 4, 8], [0, 0, 0]], dtype=torch.int32)
    # row 0: n=0 at the last slot of block 0 (no lookahead): stale block 9 is not written
    # row 1: n=2 from 62 ends at 64 (block 1); row 2: padding row
    out = verify_write_table(pt, [63, 62, 0], [0, 2, 0], 64)
    assert out.tolist() == [[5, 0, 0], [3, 4, 0], [0, 0, 0]]
    assert pt[0, 1].item() == 9  # input untouched
    full = torch.cat([torch.arange(1, 65), torch.zeros(1)]).to(torch.int32).reshape(1, 65)
    assert torch.equal(verify_write_table(full, [4094], [0], 64), full)


@pytest.mark.parametrize(
    "pos,nb,stale",
    [
        ([4094, 4094], 65, False),  # two narrow rows at max_model_len 4096: 4096..4097 hit column 64 = null block
        ([63, 127], 3, True),  # no lookahead (ngram) at a 64-token boundary, stale ids past the reservation
    ],
)
def test_decode_forward_padded_columns_stay_in_reservation(pos, nb, stale):
    B, K = 2, 3
    T = K + 1
    seen = {}

    def verify_traced(tokens, positions, mi_prev, **kw):
        seen["pt"] = kw["page_tables"].clone()
        assert spec_kv_write_conflicts(kw["page_tables"], positions, T, 64) == []
        return [0] * (B * T), "feed", None

    model = types.SimpleNamespace(
        _fit_spec_page_table=lambda pt, n: torch.as_tensor(pt, dtype=torch.int32), verify_traced=verify_traced
    )
    e = _bare_engine(B, K, captured=True, nb=nb, _timing=False, _gdn=[], model=model, no_draft_streak=16)
    pt = torch.zeros(B, nb, dtype=torch.int32)
    for u in range(B):
        live = (pos[u] // 64) + 1
        pt[u, :live] = torch.arange(1 + 100 * u, 1 + 100 * u + live)
        if stale:
            pt[u, live:] = 99  # a block another request owns
    tok = torch.full((B, T), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    tok[:, 0] = 7
    start = torch.full((B, T), -1, dtype=torch.int32)
    start[:, 0] = torch.tensor(pos, dtype=torch.int32)
    e.decode_forward(
        tok,
        start,
        num_valid_drafts=torch.zeros(B, dtype=torch.int32),
        accepted_counts=torch.ones(B, dtype=torch.int32),
        page_table=pt,
    )
    assert not bool((seen["pt"] == 99).any()), "verify got a block past the reservation"
    assert torch.equal(e._last_pt, pt), "the drafter must keep the full table"


def test_prepare_decode_after_record_prefill_refused(expect_error):
    e = _bare_engine(
        1, prefill_prepared=True, prepared=False, prefill_recorded=True, captured=False, model=types.SimpleNamespace()
    )
    with expect_error(AssertionError, "prepare_decode after record_prefill"):
        MTPSpecEngine.prepare_decode(e)


@pytest.mark.parametrize("recorded", [False, True])
def test_propose_writes_anchor_kv_of_kept_rows(recorded, monkeypatch):
    """A declined row in keep_kv still gets MTP KV at its anchor slot p (one more reseed row, i = m)."""
    import ttnn

    B, K = 3, 2
    staged, reseeds = {}, []
    model = types.SimpleNamespace(
        _fit_spec_page_table=lambda pt, nb: torch.as_tensor(pt, dtype=torch.int32),
        _rope_tp_cos_sin_decode_torch=lambda pos: (None, None),
    )
    e = _bare_engine(
        B,
        K,
        captured=True,
        nb=4,
        model=model,
        _vfeed="feed",
        _scratch_block=99,
        eager_reseed_len=10**9,
        recorded_drafter=recorded,
        _dr_trace="trace",
        mesh=None,
        mtp_extra_steps=0,
        _hp_buf=None,
    )
    for name in ("_anchor_sel", "_dr_rs_tok", "_dr_rs_pos", "_dr_rs_pt", "_dr_rs_cos", "_dr_rs_sin", "_dr_tok0"):
        setattr(e, name, name)
    e._mtp_pt = "_mtp_pt"
    e._dr_pos = e._dr_cos = e._dr_sin = e._dr_outs = [None] * K
    e._stage = lambda host, buf, *a: staged.__setitem__(buf, host) if isinstance(buf, str) else None
    e._stage_mtp_tables = e._set_anchor = lambda *a, **k: None
    e._reseed_mtp_batched = lambda ctok, cpos, m, pt: reseeds.append(list(m))
    e._draft = lambda *a: torch.zeros(B, K, dtype=torch.int32)
    e._id_to_host = lambda t: [0] * B
    monkeypatch.setattr(ttnn, "execute_trace", lambda *a, **k: None)
    ctok = torch.tensor([[11, -1, -1], [21, -1, -1], [31, -1, -1]])
    cpos = torch.tensor([[10, -1, -1], [20, -1, -1], [30, -1, -1]])
    pt = torch.arange(1, 13, dtype=torch.int32).reshape(B, 4)
    acc = torch.ones(B, dtype=torch.int32)
    MTPSpecEngine.propose(
        e, ctok, cpos, acc, pt, decline=[True, True, False], cap_blocks=4, keep_kv=[True, False, False]
    )
    if recorded:
        tok = staged["_dr_rs_tok"].reshape(-1).tolist()
        rows = list(zip(tok, staged["_dr_rs_pos"].tolist(), staged["_dr_rs_pt"][:, 0].tolist()))
        assert rows[0] == (11, 9, 1), "kept row: pending token at anchor slot p = 9 over its own table"
        assert all(r == (0, 0, 99) for r in rows[1:]), "every other reseed row pads to scratch"
    else:
        assert reseeds == [[1, 0, 0]]
