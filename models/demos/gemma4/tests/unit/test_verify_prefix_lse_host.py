# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of GEMMA4_SPEC_VERIFY_PREFIX_LSE (tt/verify_prefix_lse.py): the serving verify's global layers as
a causal decode over each row's prefix plus the candidates' own keys, merged through their log-sum-exp. The constant
fresh mask and its padded keys, the per-replay prefix positions and row biases, the Q / output layouts and the head
splits, the planner against the op's own CB list, and "prefix [0, c_b - 1] + fresh [c_b, c_b + p] + LSE merge" ==
the masked packed verify over the rung -- through a torch emulation of both attentions over a paged cache, with rows
at position 0, rows crossing a block boundary and mirrored padding rows -- and the pair split's halves + three-way
merge on a (1, 8) row == the same rows. Pure torch: neither this file nor the modules
it imports touch ttnn (the record's mask helpers are compiled from tt/spec_decode.py's source, the sdpa_decode_lse
planner is loaded from its file: both packages import ttnn).

    pytest models/demos/gemma4/tests/unit/test_verify_prefix_lse_host.py --noconftest -p no:cacheprovider
"""
from __future__ import annotations

import ast
import dataclasses
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt import verify_prefix_lse as vp

_GEMMA4 = Path(__file__).resolve().parents[2]
_SPEC_DECODE = _GEMMA4 / "tt" / "spec_decode.py"
_PLANNER = _GEMMA4 / "custom_ops" / "sdpa_decode_lse" / "planner.py"
_PURE = (
    "SPEC_MASK_NEG",
    "serving_width_for",
    "serving_pad_rows",
    "serving_select_rows",
    "serving_mask_plan",
    "serving_mask_rows",
    "serving_masks_host",
    "serving_masks_from_plan",
)


def _spec_decode_pure():
    """The record's pure position / mask / table helpers, compiled from the source of tt/spec_decode.py (the module
    imports ttnn): the functions checked here are the served ones, not copies."""
    tree = ast.parse(_SPEC_DECODE.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _PURE:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in _PURE for t in node.targets):
            keep.append(node)
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_SPEC_DECODE), "exec"), namespace)
    missing = set(_PURE) - set(namespace)
    assert not missing, f"tt/spec_decode.py no longer defines {sorted(missing)}"
    return SimpleNamespace(**{name: namespace[name] for name in _PURE})


def _served_effective_block_size():
    """``attention/operations.py: effective_block_size``, compiled from its source (the module imports ttnn): the
    helper the served calls pass as the block-size override."""
    path = _GEMMA4 / "tt" / "attention" / "operations.py"
    tree = ast.parse(path.read_text())
    fn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "effective_block_size"]
    assert fn, "attention/operations.py no longer defines effective_block_size"
    namespace = {}
    exec(compile(ast.Module(body=fn, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["effective_block_size"]


def _load_planner():
    """custom_ops/sdpa_decode_lse/planner.py on its own (pure python; the package's __init__ imports the op, ttnn)."""
    spec = importlib.util.spec_from_file_location("_sdpa_decode_lse_planner_host", _PLANNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the dataclasses resolve their module through sys.modules
    spec.loader.exec_module(module)
    return module


SD = _spec_decode_pure()
LP = _load_planner()

# Blackhole: 15/16 of the 1,572,864 B L1 above the 112,000 B unreserved base (custom_ops/grid.py
# sdpa_decode_cb_budget_bytes, which imports ttnn)
BH_BUDGET = (1536 * 1024 - 112_000) * 15 // 16


def _pack(q, P):
    """packed_decode_forward step 6: [1, H, B*P, D] -> head-major [1, B, H*P, D] (row h*P + p)."""
    _one, H, rows, D = q.shape
    B = rows // P
    return q.reshape(1, H, B, P, D).permute(0, 2, 1, 3, 4).reshape(1, B, H * P, D)


def _unpack(o, H, P):
    """packed_decode_forward step 7: [1, B, H*P, D] -> [1, B*P, H, D], the rows the o_proj tail takes."""
    _one, B, _hp, D = o.shape
    return o.reshape(1, B, H, P, D).permute(0, 1, 3, 2, 4).reshape(1, B * P, H, D)


# ── the constant fresh mask and the fresh K / V rows ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("B, H, P", [(5, 4, 4), (1, 4, 6), (8, 4, 4), (16, 8, 2), (5, 8, 3)])
def test_fresh_mask_is_causal_per_candidate_and_hides_padded_keys(B, H, P):
    mask = vp.fresh_causal_mask(B, H, P)
    assert tuple(mask.shape) == (1, B, H * P, vp.TILE) and mask.dtype == torch.float32
    for b in range(B):
        assert torch.equal(mask[0, b], mask[0, 0]), "the mask is independent of the record row"
    for h in range(H):
        for p in range(P):
            row = mask[0, 0, h * P + p]
            visible = (row == 0).nonzero().reshape(-1).tolist()
            assert visible == list(range(p + 1)), (h, p, visible)
            assert bool((row[p + 1 :] == vp.FRESH_MASK_NEG).all())
    assert bool((mask[..., P:] == vp.FRESH_MASK_NEG).all()), "a padded key (q >= P) is visible somewhere"


def test_fresh_kv_rows_take_each_rows_candidates_and_zero_the_padding():
    B, P, D = 5, 4, 16
    kv = torch.randn(1, 1, 32, D)
    kv[:, :, B * P :] = float("nan")  # the split's tile-padded tail: whatever the tile held
    rows = vp.fresh_kv_rows(kv, B, P)
    assert tuple(rows.shape) == (1, B, vp.TILE, D)
    for b in range(B):
        assert torch.equal(rows[0, b, :P], kv[0, 0, b * P : (b + 1) * P])
    assert torch.equal(rows[:, :, P:], torch.zeros(1, B, vp.TILE - P, D)), "padded keys must be zero, not the tail"


def test_fresh_attention_ignores_padded_keys():
    """The padded keys (zero K / V rows past P) never reach a softmax: their masked scores underflow to an exact 0,
    so the fresh part equals a softmax over the visible candidates alone."""
    B, H, P, D = 3, 4, 4, 32
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(1, B, H * P, D, generator=gen)
    k = vp.fresh_kv_rows(torch.randn(1, 1, B * P, D, generator=gen), B, P)
    v = vp.fresh_kv_rows(torch.randn(1, 1, B * P, D, generator=gen), B, P)
    o_un, m, l = vp.fresh_attention_torch(q, k, v, vp.fresh_causal_mask(B, H, P))
    for b in range(B):
        for h in range(H):
            for p in range(P):
                r = h * P + p
                s = q[0, b, r] @ k[0, b, : p + 1].T
                want = torch.softmax(s, dim=-1) @ v[0, b, : p + 1]
                assert torch.allclose(o_un[0, b, r] / l[0, b, r], want, atol=1e-6)
                assert torch.allclose(m[0, b, r] + torch.log(l[0, b, r]), torch.logsumexp(s, dim=-1), atol=1e-5)


# ── per-replay inputs, layouts, splits ─────────────────────────────────────────────────────────────────────────────


def test_prefix_positions_and_biases():
    pos = SD.serving_pad_rows([300, 0, 64], 5)  # rows 3, 4 mirror row 0
    cur = vp.prefix_positions(pos)
    assert cur.dtype == torch.int32 and cur.tolist() == [299, 0, 63, 299, 299]
    bias = vp.prefix_bias(pos, 16)
    assert tuple(bias.shape) == (1, 5, 16, 1) and bias.dtype == torch.float32
    assert bool((bias[0, 1] == vp.EMPTY_PREFIX_LSE).all()), "the row at 0 has no prefix"
    assert not bias[0, [0, 2, 3, 4]].any()
    # a row at position 1 has a one-key prefix (key 0) and no bias
    assert vp.prefix_positions([1]).tolist() == [0] and not vp.prefix_bias([1], 4).any()
    with pytest.raises(ValueError, match="negative"):  # allow-pytest.raises (contract check, no conftest here)
        vp.prefix_positions([3, -1])


@pytest.mark.parametrize(
    "H, P, splits", [(4, 2, 1), (4, 4, 1), (4, 6, 1), (8, 2, 1), (8, 4, 1), (8, 6, 2), (32, 2, 2), (5, 7, 5)]
)
def test_head_splits_keep_whole_heads_in_one_q_tile(H, P, splits):
    assert vp.prefix_head_splits(H, P) == splits
    assert H % splits == 0 and (H // splits) * P <= vp.MAX_Q_ROWS
    assert splits == 1 or (H // (splits - 1)) * P > vp.MAX_Q_ROWS or H % (splits - 1)
    assert vp.prefix_head_splits(1, 33) == 0


@pytest.mark.parametrize("H, B, P", [(4, 5, 4), (8, 1, 6), (8, 5, 6), (4, 16, 2)])
def test_q_layout_and_unpack_are_the_masked_verifys(H, B, P):
    D = 16
    q = torch.randn(1, H, B * P, D)
    splits = vp.prefix_head_splits(H, P)
    parts = vp.prefix_q_rows(q, P, splits)
    assert len(parts) == splits and all(tuple(t.shape) == (1, B, (H // splits) * P, D) for t in parts)
    assert torch.equal(torch.cat(parts, dim=2), _pack(q, P))
    # an attention that returned its q would leave the o_proj tail exactly where the masked path leaves it
    assert torch.equal(vp.prefix_out_rows(parts, H, P), _unpack(_pack(q, P), H, P))
    assert torch.equal(vp.prefix_out_rows(parts, H, P), q.permute(0, 2, 1, 3))


# ── the split is the masked row ────────────────────────────────────────────────────────────────────────────────────

# (positions of the live rows, record rows, P, block, S_k, local heads): rows at position 0 and 1, rows whose
# candidates cross a block boundary (62..65 over 64), a row whose prefix ends on a block edge (64), a row near the
# rung's top, mirrored padding rows, P = 2 / 4 / 6, and 8 heads at P = 6 (two prefix calls)
_CASES = [
    ([62, 0, 125], 3, 4, 64, 256, 4),
    ([62, 130], 4, 4, 64, 256, 4),
    ([1, 64, 250], 3, 4, 64, 256, 4),
    ([63, 64, 31, 0], 5, 6, 32, 128, 4),
    ([5, 40], 3, 2, 32, 64, 4),
    ([0, 97, 33], 3, 6, 32, 128, 8),
    ([200], 1, 6, 64, 256, 8),
]


def test_prefix_and_fresh_keys_are_the_record_mask_rows():
    """Per packed row (b, h, p): prefix keys [0, c_b - 1] (none at c_b = 0) + fresh keys c_b + q, q <= p, are exactly
    the keys the record's full-attention mask leaves (serving_masks_from_plan, the in-trace build)."""
    H = 4
    for live_pos, B, P, _bs, S_k, _h in _CASES:
        pos = SD.serving_pad_rows(live_pos, B)
        mask_full, _ = SD.serving_masks_from_plan(SD.serving_mask_plan(pos, P), S_k, 1024, ring=None, h=H)
        fresh = vp.fresh_causal_mask(B, H, P)
        for b, c in enumerate(pos):
            prefix_keys = set(range(c))
            for h in range(H):
                for p in range(P):
                    r = h * P + p
                    fresh_keys = {c + q for q in range(vp.TILE) if fresh[0, b, r, q] == 0}
                    record = set((mask_full[b, 0, r] == 0).nonzero().reshape(-1).tolist())
                    assert prefix_keys | fresh_keys == record, (live_pos, b, h, p)
                    assert not prefix_keys & fresh_keys


def _paged_cache(n_blocks, bs, D, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n_blocks, 1, bs, D, generator=gen), torch.randn(n_blocks, 1, bs, D, generator=gen)


def _write_candidates(k_cache, v_cache, table, pos, P, k_new, v_new, bs):
    """Step 5 of packed_decode_forward: candidate p of row b written at position c_b + p through row b of the table
    (no modulo on a global layer); padding rows write row 0's bytes to row 0's slots."""
    for b, c in enumerate(pos):
        for p in range(P):
            x = c + p
            blk = int(table[b, x // bs])
            k_cache[blk, 0, x % bs] = k_new[b * P + p]
            v_cache[blk, 0, x % bs] = v_new[b * P + p]


def _masked_packed_attention(q_packed, k_cache, v_cache, table, mask, bs):
    """Today's call: per record row b, every packed row attends ALL table_cols * bs keys through row b, plus the
    additive mask row. Returns [1, B, H*P, D]."""
    _one, B, rows, D = q_packed.shape
    S_k = int(mask.shape[-1])
    keys = torch.arange(S_k)
    out = torch.zeros(1, B, rows, D)
    for b in range(B):
        blocks = table[b, keys // bs].to(torch.int64)
        k = k_cache[blocks, 0, keys % bs]
        v = v_cache[blocks, 0, keys % bs]
        scores = q_packed[0, b] @ k.T + mask[b, 0].float()
        out[0, b] = torch.softmax(scores, dim=-1) @ v
    return out


def _case(live_pos, B, P, bs, S_k, H, D=32, seed=0):
    """One record's global layer over a paged cache: the adapter's shuffled tables fitted to the rung (padding rows
    mirror row 0), the candidates' K / V written first, Q, and the masked verify's rows (the reference)."""
    gen = torch.Generator().manual_seed(seed + 1000 * len(live_pos) + S_k + P + H)
    pos = SD.serving_pad_rows(live_pos, B)
    cols = S_k // bs
    adapter_cols = cols + 3
    n_blocks = 1 + len(live_pos) * adapter_cols  # block 0 is the null block
    perm = torch.randperm(n_blocks - 1, generator=gen) + 1
    adapter = torch.zeros(B, adapter_cols, dtype=torch.int64)
    for r in range(len(live_pos)):
        adapter[r] = perm[r * adapter_cols : (r + 1) * adapter_cols]
    adapter[len(live_pos) :] = adapter[0]
    table = SD.serving_select_rows(adapter, list(range(B)), B, cols, allow_clip=True)
    k_cache, v_cache = _paged_cache(n_blocks, bs, D, seed=S_k + P + seed)
    k_new = torch.randn(B * P, D, generator=gen)
    v_new = torch.randn(B * P, D, generator=gen)
    q = torch.randn(1, H, B * P, D, generator=gen)
    for r in range(len(live_pos), B):  # a mirrored row computes the same Q / K / V as row 0
        k_new[r * P : (r + 1) * P] = k_new[:P]
        v_new[r * P : (r + 1) * P] = v_new[:P]
        q[:, :, r * P : (r + 1) * P] = q[:, :, :P]
    _write_candidates(k_cache, v_cache, table, pos, P, k_new, v_new, bs)
    mask_full, _ = SD.serving_masks_from_plan(SD.serving_mask_plan(pos, P), S_k, 1024, ring=None, h=H)
    today = _unpack(_masked_packed_attention(_pack(q, P), k_cache, v_cache, table, mask_full, bs), H, P)
    # the prefill-style split's K / V: [1, 1, B*P (tile-padded), D], whatever the tile held past the rows
    pad = -(-(B * P) // 32) * 32 - B * P
    k_split = torch.cat([k_new, torch.full((pad, D), float("nan"))]).reshape(1, 1, -1, D)
    v_split = torch.cat([v_new, torch.full((pad, D), float("nan"))]).reshape(1, 1, -1, D)
    return SimpleNamespace(
        pos=pos, table=table, k_cache=k_cache, v_cache=v_cache, q=q, k_split=k_split, v_split=v_split, today=today
    )


def _prefix_lse_rows(c, B, P, bs, H, op_dtype=None):
    """The new subgraph in torch, step by step: Q split per prefix call, the prefix call at max(c_b - 1, 0), the fresh
    part from the split's K / V, the merge with the row biases, the un-pack. ``op_dtype`` (bf16) rounds inside the
    prefix call as the op's CBs do (scores, l, O)."""
    splits = vp.prefix_head_splits(H, P)
    rows = (H // splits) * P
    cur = vp.prefix_positions(c.pos)
    bias = vp.prefix_bias(c.pos, rows)
    mask = vp.fresh_causal_mask(B, H // splits, P)
    k_fresh = vp.fresh_kv_rows(c.k_split, B, P)
    v_fresh = vp.fresh_kv_rows(c.v_split, B, P)
    parts = []
    for q_s in vp.prefix_q_rows(c.q, P, splits):
        o1, m1, l1 = vp.prefix_attention_torch(q_s, c.k_cache, c.v_cache, c.table, cur, bs, dtype=op_dtype)
        o2, m2, l2 = vp.fresh_attention_torch(q_s, k_fresh, v_fresh, mask)
        parts.append(vp.merge_prefix_fresh_torch(o1, m1, l1, bias, o2, m2, l2))
    return vp.prefix_out_rows(parts, H, P)


@pytest.mark.parametrize("live_pos, B, P, bs, S_k, H", _CASES)
def test_prefix_fresh_merge_equals_the_masked_packed_verify(live_pos, B, P, bs, S_k, H):
    """The whole global-layer attention both ways over one paged cache: today's masked packed SDPA over the rung vs
    the prefix call + the fresh part + the LSE merge. Same rows out, padding rows equal to row 0, all finite."""
    c = _case(live_pos, B, P, bs, S_k, H)
    got = _prefix_lse_rows(c, B, P, bs, H)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, c.today, atol=2e-5, rtol=1e-5), f"max diff {(got - c.today).abs().max()}"
    for r in range(len(live_pos), B):  # equal up to the host BLAS's batch blocking (not bitwise on every torch)
        assert torch.allclose(got[0, r * P : (r + 1) * P], got[0, :P], atol=1e-6), f"padding row {r} differs from row 0"


def _rounded_prefix_reference(c, B, P, bs, H, dtype=torch.bfloat16):
    """Attention over [0, c_b + p] per candidate with the PREFIX keys' scores rounded as the op's CBs round them (the
    fresh scores are fp32 matmul outputs on device): the reference the merge of the op's bf16 outputs is measured
    against, apart from the rounding of the scores themselves (the stock op's, shared with the masked verify).
    Returns [1, B*P, H, D]."""
    D = int(c.q.shape[-1])
    out = torch.zeros(1, B * P, H, D)
    for b, cb in enumerate(c.pos):
        for p in range(P):
            keys = torch.arange(cb + p + 1)
            blocks = c.table[b, keys // bs].to(torch.int64)
            k = c.k_cache[blocks, 0, keys % bs]
            v = c.v_cache[blocks, 0, keys % bs]
            for h in range(H):
                s = c.q[0, h, b * P + p] @ k.T
                s[:cb] = s[:cb].to(dtype).float()
                out[0, b * P + p, h] = torch.softmax(s, dim=-1) @ v
    return out


@pytest.mark.parametrize("live_pos, B, P, bs, S_k, H", _CASES)
def test_merge_with_the_ops_bf16_statistics_stays_at_bf16_error(live_pos, B, P, bs, S_k, H):
    """The op's scores, l and O are bf16 (m is one of the bf16 scores; the merge forms the LSE in fp32 from the two
    columns): the merged rows stay within bf16 error of the attention over the same rounded prefix scores."""
    c = _case(live_pos, B, P, bs, S_k, H, seed=7)
    got = _prefix_lse_rows(c, B, P, bs, H, op_dtype=torch.bfloat16)
    want = _rounded_prefix_reference(c, B, P, bs, H)
    err = (got - want).abs().max() / want.abs().max()
    assert err < 1e-2, err
    pcc = torch.corrcoef(torch.stack([got.reshape(-1), want.reshape(-1)]))[0, 1]
    assert pcc > 0.99999, pcc


def test_a_row_at_position_zero_takes_the_fresh_part_only():
    """c_b = 0: the prefix call runs at cur_pos 0 (key 0, the row's own first candidate: written, finite) and its lse
    gets EMPTY_PREFIX_LSE, so the merged row is the fresh part's softmax over keys [0, p] alone, whatever finite O / m /
    l the prefix call wrote."""
    B, H, P, D = 2, 4, 4, 16
    gen = torch.Generator().manual_seed(3)
    q = torch.randn(1, B, H * P, D, generator=gen)
    k_fresh = vp.fresh_kv_rows(torch.randn(1, 1, B * P, D, generator=gen), B, P)
    v_fresh = vp.fresh_kv_rows(torch.randn(1, 1, B * P, D, generator=gen), B, P)
    o2, m2, l2 = vp.fresh_attention_torch(q, k_fresh, v_fresh, vp.fresh_causal_mask(B, H, P))
    want = o2 / l2
    bias = vp.prefix_bias([0, 0], H * P)
    for o1, m1, l1 in (
        (torch.randn(1, B, H * P, D) * 100, torch.full((1, B, H * P, 1), 50.0), torch.full((1, B, H * P, 1), 3.0)),
        (torch.zeros(1, B, H * P, D), torch.full((1, B, H * P, 1), 1e4), torch.ones(1, B, H * P, 1)),
    ):
        got = vp.merge_prefix_fresh_torch(o1, m1, l1, bias, o2, m2, l2)
        assert torch.isfinite(got).all()
        assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()


# ── per-layer gate ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_layer_reason():
    ok = dict(is_sliding=False, cache_position_modulo=None, kv_modulo=None, nkv_local=1, head_dim=512)
    assert vp.prefix_lse_layer_reason(**ok, cache_shape=(4113, 1, 64, 512)) is None
    assert "sliding" in vp.prefix_lse_layer_reason(**{**ok, "is_sliding": True}, cache_shape=(9, 1, 64, 512))
    assert "ring" in vp.prefix_lse_layer_reason(**{**ok, "cache_position_modulo": 1024}, cache_shape=(9, 1, 64, 512))
    assert "ring" in vp.prefix_lse_layer_reason(**{**ok, "kv_modulo": 1280}, cache_shape=(9, 1, 64, 512))
    assert "KV-shared" in vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 1, 64, 512), is_kv_shared=True)
    assert "KV heads" in vp.prefix_lse_layer_reason(**{**ok, "nkv_local": 2}, cache_shape=(9, 2, 64, 512))
    assert "head_dim" in vp.prefix_lse_layer_reason(**{**ok, "head_dim": 128}, cache_shape=(9, 1, 64, 128))
    # a hybrid-group buffer in the sliding view (2 KV heads x 256) is read as one 512-wide head of block 64
    # (test_shared_hybrid_group_view_is_read_through_the_effective_block); views without whole tile blocks are not
    assert vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 2, 64, 256)) is None
    assert "multiple" in vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 1, 16, 512))
    assert "multiple" in vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 2, 24, 256))  # effective block 24
    assert "whole blocks" in vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 3, 64, 100))  # 19200 / 512
    assert "blocks, heads" in vp.prefix_lse_layer_reason(**ok, cache_shape=(9, 64, 512))
    assert vp.kv_dtype_name("DataType.BFLOAT8_B") == "bfloat8_b" and vp.kv_dtype_name("DataType.BFLOAT16") == "bfloat16"
    assert vp.kv_dtype_name("DataType.FLOAT32") is None


# ── planner ────────────────────────────────────────────────────────────────────────────────────────────────────────


def _geometry(B, P, H, kv_dtype="bfloat8_b", block=64, blocks=4113, pair=False):
    return vp.PrefixLseGeometry(
        batch=B,
        packed_p=P,
        h_local=H,
        head_dim=512,
        splits=vp.prefix_head_splits(H, P),
        block_size=block,
        num_cache_blocks=blocks,
        kv_dtype=kv_dtype,
        pair=pair,
    )


@pytest.mark.parametrize("grid", [(11, 10), (12, 10), (13, 10)], ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("kv_dtype", ["bfloat16", "bfloat8_b"])
@pytest.mark.parametrize("H, pair", [(4, False), (8, False), (8, True)], ids=["tp8", "tp4", "tp8_pair"])
@pytest.mark.parametrize("shape", [(1, 5), (5, 5), (5, 3), (8, 3), (16, 1), (1, 3)], ids=lambda s: f"B{s[0]}K{s[1]}")
def test_plan_fits_the_31b_records(shape, H, pair, kv_dtype, grid):
    """The 31B's global layers (head_dim 512, one KV head per chip; 4 local q heads on a Galaxy row at TP 8, 8 at
    TP 4 or under the pair view at TP 8) at the top rung's table width (263168 / 64 columns): the prefix call fits for
    every served shape, its plan is the op planner's own, and the cores per row are the plain decode's."""
    B, K = shape
    P = K + 1
    geom = _geometry(B, P, H, kv_dtype, pair=pair)
    plan = vp.plan_verify_prefix_lse(geom, 263168 // 64, grid, BH_BUDGET, LP, env={})
    assert plan.fits, plan.describe()
    assert plan.rows <= vp.MAX_Q_ROWS and plan.k_chunk == 64 and plan.max_cores_per_head == 16
    if B == 1 and kv_dtype == "bfloat16":
        assert plan.grid == (8, 4)
    else:
        assert plan.grid == grid
    n = plan.grid[0] * plan.grid[1]
    assert plan.cores_per_head == max(1, min(n, 16 * B) // B)
    assert plan.cb_bytes <= BH_BUDGET


def test_plan_knobs_and_refusals():
    geom = _geometry(5, 4, 4)
    env = {vp.VERIFY_PREFIX_LSE_MAX_CORES_ENV: "8", vp.DECODE_K_CHUNK_ENV: "128"}
    plan = vp.plan_verify_prefix_lse(geom, 4112, (11, 10), BH_BUDGET, LP, env=env)
    assert plan.fits and (plan.k_chunk, plan.max_cores_per_head, plan.cores_per_head) == (128, 8, 8), plan.describe()
    plan = vp.plan_verify_prefix_lse(geom, 4112, (11, 10), BH_BUDGET, LP, env={vp.DECODE_MAX_CORES_ENV: "32"})
    assert (plan.max_cores_per_head, plan.cores_per_head) == (32, 22), plan.describe()
    # the dynamic chunk (0) is not ported to the op: refused with the planner's reason, as is a tiny budget
    assert not vp.plan_verify_prefix_lse(geom, 4112, (11, 10), BH_BUDGET, LP, env={vp.DECODE_K_CHUNK_ENV: "0"}).fits
    assert "budget" in vp.plan_verify_prefix_lse(geom, 4112, (11, 10), 100_000, LP, env={}).note
    # more record rows than cores
    assert not vp.plan_verify_prefix_lse(_geometry(16, 2, 4), 4112, (3, 3), BH_BUDGET, LP, env={}).fits


# ── pair split ─────────────────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("k_chunk", [32, 64])
def test_pair_half_keys_mirror_the_planner_and_cover_the_row(k_chunk):
    for cur in range(0, 300):
        lo0, hi0 = vp.pair_half_keys(cur, k_chunk, 0)
        lo1, hi1 = vp.pair_half_keys(cur, k_chunk, 1)
        assert (lo0, hi0) == LP.pair_key_range(cur, k_chunk, 0) and (lo1, hi1) == LP.pair_key_range(cur, k_chunk, 1)
        assert lo0 == 0 and hi0 == lo1 and hi1 == cur + 1, (cur, (lo0, hi0), (lo1, hi1))
        assert (lo1 == hi1) == (cur < k_chunk), "only a one-chunk row has an empty half"


# (positions of the live rows, record rows, P, block, S_k, K chunk): a row at 0 (no prefix), prefixes inside one chunk
# (the second half empty: the op's sentinel), prefixes over several chunks and blocks, mirrored padding rows, P = 4 / 6
_PAIR_CASES = [
    ([62, 0, 125], 3, 4, 32, 256, 32),
    ([1, 20, 250], 4, 4, 64, 256, 64),
    ([63, 64, 31, 0], 5, 6, 32, 128, 32),
    ([200], 1, 6, 64, 256, 64),
]


@pytest.mark.parametrize("live_pos, B, P, bs, S_k, k_chunk", _PAIR_CASES)
def test_pair_split_three_way_merge_is_the_masked_row(live_pos, B, P, bs, S_k, k_chunk):
    """The 31B's (1, 8) row under the pair view: 32 q / 4 KV heads, chip c holds the 8 q heads of KV group c // 2 and
    attends prefix half c % 2 (the empty half writes the op's sentinel), the fresh part is the same on both chips of
    a pair, and each chip merges three ways (pair_prefix_merge_torch). The pair's two outputs add up (the pair o_proj
    is the same rows on both chips, the row all-reduce sums them) to the masked verify's rows of the group."""
    TP, NKV, G = 8, 4, 8
    for g in range(NKV):
        c = _case(live_pos, B, P, bs, S_k, G, seed=11 * g)  # group g's own KV head, candidates and q heads
        D = int(c.q.shape[-1])
        splits = vp.prefix_head_splits(G, P)
        rows = (G // splits) * P
        cur = vp.prefix_positions(c.pos)
        bias = vp.prefix_bias(c.pos, rows)
        mask = vp.fresh_causal_mask(B, G // splits, P)
        k_fresh = vp.fresh_kv_rows(c.k_split, B, P)
        v_fresh = vp.fresh_kv_rows(c.v_split, B, P)
        pair_sums = []
        for q_s in vp.prefix_q_rows(c.q, P, splits):
            gen = torch.Generator().manual_seed(g)
            # every chip of the row: the other pairs hold other groups' halves (finite values the selector never reads)
            o = torch.randn(TP, B, rows, D, generator=gen)
            m = torch.randn(TP, B, rows, 1, generator=gen) * 5
            l = torch.rand(TP, B, rows, 1, generator=gen) + 0.5
            for h in (0, 1):
                oh, mh, lh = vp.prefix_attention_torch(
                    q_s, c.k_cache, c.v_cache, c.table, cur, bs, half=h, k_chunk=k_chunk
                )
                o[2 * g + h], m[2 * g + h], l[2 * g + h] = oh[0], mh[0], lh[0]
            o2, m2, l2 = vp.fresh_attention_torch(q_s, k_fresh, v_fresh, mask)
            out = vp.pair_prefix_merge_torch(o, m, l, bias, o2, m2, l2)
            assert torch.isfinite(out).all()
            pair_sums.append((out[2 * g] + out[2 * g + 1])[None])
        got = vp.prefix_out_rows(pair_sums, G, P)
        assert torch.allclose(got, c.today, atol=2e-5, rtol=1e-5), f"group {g}: max diff {(got - c.today).abs().max()}"


def test_pair_merge_weights_split_the_fresh_term_evenly_and_drop_empty_halves():
    """One pair, one packed row: an empty half (the sentinel) gets weight 0 on its own chip, the partner's half and the
    fresh part share the rest as their LSEs say, and each chip carries half of the fresh term."""
    o = torch.tensor([[3.0, 3.0], [0.0, 0.0]]).reshape(2, 1, 1, 2)  # [tp, B, R, D]: chip 1's half is empty
    m = torch.tensor([2.0, vp.EMPTY_HALF_M]).reshape(2, 1, 1, 1)
    l = torch.tensor([4.0, 1.0]).reshape(2, 1, 1, 1)
    o2_un, m2, l2 = torch.full((1, 1, 1, 2), 10.0), torch.tensor(1.0).reshape(1, 1, 1, 1), torch.full((1, 1, 1, 1), 5.0)
    out = vp.pair_prefix_merge_torch(o, m, l, torch.zeros(1, 1, 1, 1), o2_un, m2, l2)
    lse_half, lse_fresh = 2.0 + math.log(4.0), 1.0 + math.log(5.0)
    w = 1.0 / (1.0 + math.exp(lse_fresh - lse_half))
    want = w * 3.0 + (1.0 - w) * 10.0 / 5.0
    assert torch.allclose(out.sum(dim=0), torch.full((1, 1, 2), want), atol=1e-6)
    half_fresh = torch.full((1, 1, 2), 0.5 * (1.0 - w) * 2.0)
    assert torch.allclose(out[1], half_fresh, atol=1e-6), "chip 1 carries half the fresh term"


# ── vLLM's shared hybrid-group cache view ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cache_shape, head_dim, block",
    [
        ((27936, 2, 64, 256), 512, 64),  # the served 31B global layer on the sliding layers' [blocks, 2, 64, 256]
        ((4113, 1, 64, 512), 512, 64),  # its own view
        ((100, 4, 32, 256), 512, 64),  # 26B-style: more sliding heads per block
        ((100, 1, 64, 256), 256, 64),
    ],
)
def test_shared_hybrid_group_view_is_read_through_the_effective_block(cache_shape, head_dim, block):
    """vLLM's hybrid kv-cache groups allocate the 31B's global layers (one 512-wide KV head per chip) in the sliding
    layers' view [blocks, 2, 64, 256]; the layer reads it through the stock geometry override at the block that keeps
    the elements per block (operations.effective_block_size, what pair_split_sdpa and the masked verify pass). The gate
    takes it, the record's geometry and plan use that block, and the planner's table arithmetic is the own view's."""
    served = _served_effective_block_size()
    assert vp.prefix_effective_block(cache_shape, head_dim) == block
    assert served(SimpleNamespace(padded_shape=cache_shape), head_dim, 1) == block
    ok = dict(is_sliding=False, cache_position_modulo=None, kv_modulo=None, nkv_local=1, head_dim=head_dim)
    assert vp.prefix_lse_layer_reason(**ok, cache_shape=cache_shape) is None
    geom = vp.PrefixLseGeometry(
        batch=5,
        packed_p=4,
        h_local=4,
        head_dim=head_dim,
        splits=1,
        block_size=vp.prefix_effective_block(cache_shape, head_dim),
        num_cache_blocks=cache_shape[0],
        kv_dtype="bfloat8_b",
    )
    cols = 131072 // block
    plan = vp.plan_verify_prefix_lse(geom, cols, (12, 10), BH_BUDGET, LP, env={})
    assert plan.fits, plan.describe()
    # the plan of the same layer over a cache in its own view [blocks, 1, block, head_dim]: identical
    own_view = (cache_shape[0], 1, block, head_dim)
    assert vp.prefix_effective_block(own_view, head_dim) == block
    own = vp.plan_verify_prefix_lse(dataclasses.replace(geom, block_size=block), cols, (12, 10), BH_BUDGET, LP, env={})
    assert (plan.cb_bytes, plan.cores_per_head, plan.k_chunk) == (own.cb_bytes, own.cores_per_head, own.k_chunk)
    # the masked verify's table spans its mask at the same block: S_k / block columns
    assert cols * block == 131072


def test_shared_view_needs_whole_tiles_of_the_effective_block():
    served = _served_effective_block_size()
    for shape in ((9, 2, 24, 256), (9, 2, 16, 512)):  # effective blocks 24 and 32 of a 512-wide head
        b = vp.prefix_effective_block(shape, 512)
        assert b == served(SimpleNamespace(padded_shape=shape), 512, 1)
        reason = vp.prefix_lse_layer_reason(
            is_sliding=False, cache_position_modulo=None, kv_modulo=None, nkv_local=1, head_dim=512, cache_shape=shape
        )
        assert (reason is None) == (b % vp.TILE == 0), (shape, b, reason)
    assert vp.prefix_effective_block((9, 3, 64, 100), 512) is None
    with pytest.raises(ValueError, match="not exactly divisible"):  # allow-pytest.raises (the served helper's contract)
        served(SimpleNamespace(padded_shape=(9, 3, 64, 100)), 512, 1)


def test_two_call_prefixes_keep_the_masked_verify_below_the_rung_threshold():
    """One prefix call takes the prefix + fresh verify at every rung; two calls only from S_k 131072 (the env moves it)."""
    for s_k in (8192, 32768, 131072, 263168):
        assert vp.prefix_lse_rung_reason(1, s_k, env={}) is None
    assert vp.prefix_lse_rung_reason(2, 8192, env={}) is not None
    assert vp.prefix_lse_rung_reason(2, 32768, env={}) is not None
    assert vp.prefix_lse_rung_reason(2, 131072, env={}) is None
    assert vp.prefix_lse_rung_reason(2, 263168, env={}) is None
    env = {vp.VERIFY_PREFIX_LSE_SPLIT_MIN_SK_ENV: "8192"}
    assert vp.prefix_lse_rung_reason(2, 8192, env=env) is None
