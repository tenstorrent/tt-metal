# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the host-side bookkeeping of the MTP spec-decode engine.

Covers the pure helpers in tt/spec_engine.py and the page-table helpers on ``Qwen36Model``
(``_fit_spec_page_table``, ``_verify_page_table_forms``, ``_check_verify_write_blocks``). No device.

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

from models.demos.blackhole.qwen36.tt.model import Qwen36Model
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
    propose_skip_plan,
    reseed_row_plan,
    reseed_rows_host,
    sanitize_verify_block,
    spec_memory_costs,
    spec_round_up,
    spec_table_width,
    table_capacity,
)

SHAPES = [(1, 12), (2, 8), (4, 4), (8, 4), (3, 5)]
BLOCK = 64


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


def _bare_model(groups=0, grouped=False, block=BLOCK):
    """Qwen36Model with only the attributes the page-table helpers read."""
    model = object.__new__(Qwen36Model)
    model._vfy_spec_groups = groups
    model._vfy_kv_grouped = grouped
    # get_block_size reads kv_cache[0][0].shape[2]
    model._paged_kv_caches = [[types.SimpleNamespace(shape=(8, 1, block, 32))]]
    return model


def test_page_table_forms_row_and_user():
    pt = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    kvpt, users, _ = _bare_model(groups=0)._verify_page_table_forms(pt, 4)
    assert torch.equal(kvpt, pt.repeat_interleave(4, dim=0)) and kvpt.shape == (8, 3)
    assert torch.equal(users, pt)
    assert kvpt.is_contiguous()


@pytest.mark.parametrize("B,groups", [(2, 2), (2, 6), (4, 8), (1, 3)])
def test_page_table_forms_group_aliasing(B, groups):
    """groups >= B and groups % B == 0: each user's row repeated groups // B times."""
    pt = torch.arange(B * 3, dtype=torch.int32).reshape(B, 3) + 1
    _, _, kvpt1 = _bare_model(groups=groups)._verify_page_table_forms(pt, 4)
    assert torch.equal(kvpt1, pt.repeat_interleave(groups // B, dim=0))
    assert kvpt1.shape == (groups, 3) and kvpt1.is_contiguous()


@pytest.mark.parametrize("B,groups", [(2, 0), (2, 3), (4, 2)])
def test_page_table_forms_group_fallback(B, groups):
    """Fused path off or groups not a multiple of B: row 0 repeated ``groups`` times (unread)."""
    pt = torch.arange(B * 3, dtype=torch.int32).reshape(B, 3) + 1
    _, _, kvpt1 = _bare_model(groups=groups)._verify_page_table_forms(pt, 4)
    assert torch.equal(kvpt1, pt[:1].repeat(groups, 1))


def test_write_blocks_disjoint_passes():
    pt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    _bare_model(grouped=True)._check_verify_write_blocks(pt, [0, BLOCK - 2], 4)


def test_write_blocks_shared_block_raises(expect_error):
    pt = torch.tensor([[1, 2, 3], [1, 5, 6]], dtype=torch.int32)
    with expect_error(AssertionError, "block 1"):
        _bare_model(grouped=True)._check_verify_write_blocks(pt, [0, 0], 4)


def test_write_blocks_shared_zero_padding_not_written_passes():
    """vLLM zero-pads tables; both users own one real block and only write inside it."""
    pt = torch.tensor([[1, 0, 0], [2, 0, 0]], dtype=torch.int32)
    _bare_model(grouped=True)._check_verify_write_blocks(pt, [3, 10], 4)


def test_write_blocks_shared_zero_padding_written_raises(expect_error):
    """If a user's write crosses into its zero padding while another user also writes block 0."""
    pt = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.int32)
    with expect_error(AssertionError, ".*"):
        _bare_model(grouped=True)._check_verify_write_blocks(pt, [BLOCK - 2, 0], 4)


def test_write_blocks_same_user_repeat_is_fine():
    pt = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    _bare_model(grouped=True)._check_verify_write_blocks(pt, [5], 8)


@pytest.mark.parametrize("grouped", [True, False])
def test_write_blocks_position_past_table_raises(grouped, expect_error):
    pt = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    model = _bare_model(grouped=grouped)
    model._check_verify_write_blocks(pt, [0, 2 * BLOCK - 4], 4)  # last write at 2*BLOCK-1: fits
    with expect_error(AssertionError, "past the page table"):
        model._check_verify_write_blocks(pt, [0, 2 * BLOCK - 3], 4)


def test_write_blocks_overlap_allowed_when_not_grouped():
    pt = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.int32)
    _bare_model(grouped=False)._check_verify_write_blocks(pt, [0, 0], 4)


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
_QWEN38_CFG = (
    "/local/ttuser/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
    "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json"
)
_MIB = 1 << 20


@pytest.fixture(scope="module")
def qwen38_text_config():
    if not os.path.exists(_QWEN38_CFG):
        pytest.skip("Qwen3.8-27B config not in the local HF cache")
    with open(_QWEN38_CFG) as f:
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
