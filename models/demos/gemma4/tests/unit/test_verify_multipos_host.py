# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of GEMMA4_SPEC_VERIFY_MULTIPOS (tt/verify_multipos.py): the serving verify's global layers on the
causal multi-position paged SDPA. The Q re-layout and its un-pad, the per-candidate positions, the group page tables,
the planner against the op's CB arithmetic, and the equivalence "candidate b*P + p attends [0, c_b + p]" == the rows of
the additive mask the records build today -- per key and through a torch emulation of both attentions over a paged
cache. Pure torch: neither this file nor the modules it imports touch ttnn (the record's mask and table helpers are
compiled from tt/spec_decode.py's source, whose module imports ttnn).

    pytest models/demos/gemma4/tests/unit/test_verify_multipos_host.py --noconftest -p no:cacheprovider
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt import verify_multipos as vm

_SPEC_DECODE = Path(__file__).resolve().parents[2] / "tt" / "spec_decode.py"
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


SD = _spec_decode_pure()

# Blackhole: 15/16 of the 1,572,864 B L1 above the 112,000 B unreserved base (custom_ops/grid.py
# sdpa_decode_cb_budget_bytes, which imports ttnn)
BH_BUDGET = (1536 * 1024 - 112_000) * 15 // 16
BF16, BFP8 = 2048, 1088


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _today_pack(q, P):
    """packed_decode_forward step 6: [1, H, B*P, D] -> head-major [1, B, H*P, D] (row h*P + p)."""
    _one, H, rows, D = q.shape
    B = rows // P
    return q.reshape(1, H, B, P, D).permute(0, 2, 1, 3, 4).reshape(1, B, H * P, D)


def _today_unpack(o, H, P):
    """packed_decode_forward step 7: [1, B, H*P, D] -> [1, B*P, H, D], the rows the o_proj tail takes."""
    _one, B, _hp, D = o.shape
    return o.reshape(1, B, H, P, D).permute(0, 1, 3, 2, 4).reshape(1, B * P, H, D)


# ── Q re-layout and un-pad ──────────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("H, B, P, tg", [(4, 5, 4, 2), (4, 5, 4, 4), (4, 1, 6, 3), (8, 8, 4, 2), (4, 16, 2, 1)])
def test_q_layout_places_each_candidate_in_its_row_tile(H, B, P, tg):
    D = 64
    G = P // tg
    q = torch.randn(1, H, B * P, D)
    q_mp = vm.multipos_q_layout(q, P, tg)
    assert tuple(q_mp.shape) == (1, B * G, tg * vm.TILE, D)
    for b in range(B):
        for p in range(P):
            tile = q_mp[0, b * G + p // tg, (p % tg) * vm.TILE : (p % tg + 1) * vm.TILE]
            assert torch.equal(tile[:H], q[0, :, b * P + p]), (b, p)
            assert not tile[H:].any(), f"padding rows of candidate ({b}, {p}) are not zero"


@pytest.mark.parametrize("H, B, P, tg", [(4, 5, 4, 2), (4, 1, 6, 3), (8, 8, 4, 4), (32, 2, 2, 1)])
def test_q_layout_round_trip_is_the_tail_layout(H, B, P, tg):
    """Un-padding the re-laid Q gives the [1, B*P, H, D] rows today's unpack hands the o_proj tail, so an op that
    returned its input would leave the tail exactly where the masked path leaves it."""
    D = 32
    q = torch.randn(1, H, B * P, D)
    rows = vm.multipos_out_rows(vm.multipos_q_layout(q, P, tg), H)
    assert torch.equal(rows, q.permute(0, 2, 1, 3))
    assert torch.equal(rows, _today_unpack(_today_pack(q, P), H, P))


# ── positions and page tables ───────────────────────────────────────────────────────────────────────────────────────


def test_positions_vector_is_the_verify_positions_in_int32():
    pos, P = [10, 62, 0, 10], 4
    cur = vm.multipos_positions(pos, P)
    assert cur.dtype == torch.int32 and tuple(cur.shape) == (len(pos) * P,)
    assert cur.tolist() == [10, 11, 12, 13, 62, 63, 64, 65, 0, 1, 2, 3, 10, 11, 12, 13]
    # the same values the record uploads as v_pos (row b*P + p) and compares its mask iota against
    assert cur.tolist() == [c + p for c in pos for p in range(P)]
    assert cur.tolist() == SD.serving_mask_plan(pos, P)["q"].reshape(-1).tolist()
    # every group of Tg (any divisor of P) is ascending, and group b*G + g holds candidates g*Tg .. of row b
    for tg in _divisors(P):
        G = P // tg
        assert vm.multipos_groups_ascending(cur, tg)
        for b, c in enumerate(pos):
            for g in range(G):
                assert cur[(b * G + g) * tg : (b * G + g + 1) * tg].tolist() == [c + g * tg + j for j in range(tg)]


def test_positions_vector_mirrors_padding_rows_and_rejects_skips():
    pos = SD.serving_pad_rows([300, 7], 4)  # two live rows of a 4-row record: rows 2, 3 mirror row 0
    cur = vm.multipos_positions(pos, 3).reshape(4, 3)
    assert torch.equal(cur[2], cur[0]) and torch.equal(cur[3], cur[0])
    assert not vm.multipos_groups_ascending(torch.tensor([5, 4, 6, 7]), 2)
    with pytest.raises(ValueError, match="never -1"):  # allow-pytest.raises (contract check, no conftest here)
        vm.multipos_positions([3, -1], 2)


def test_group_page_table_repeats_each_row_per_group():
    table = torch.arange(3 * 7, dtype=torch.int32).reshape(3, 7)
    assert vm.multipos_page_table(table, 1) is table
    grouped = vm.multipos_page_table(table, 2)
    assert tuple(grouped.shape) == (6, 7)
    for b in range(3):
        for g in range(2):
            assert torch.equal(grouped[b * 2 + g], table[b])


# ── mask equivalence ────────────────────────────────────────────────────────────────────────────────────────────────

# (positions of the live rows, record rows, P, block, S_k): rows whose candidates cross a block boundary (62..65 over
# 64, 125..128 over 128), a row at 0, a row at the rung's top (S_k - P - 2) and padding rows mirroring row 0
_CASES = [
    ([62, 0, 125], 3, 4, 64, 256),
    ([62, 130], 4, 4, 64, 256),
    ([250], 1, 4, 64, 256),
    ([63, 64, 31], 3, 6, 32, 128),
    ([5, 40], 3, 2, 32, 64),
]


@pytest.mark.parametrize("live_pos, B, P, bs, S_k", _CASES)
def test_causal_bound_is_the_record_mask_rows(live_pos, B, P, bs, S_k):
    """Every packed row (b, h, p) of the full-attention mask the record builds in-trace (serving_masks_from_plan, the
    torch emulation of _srv_masks_device at the packed height) leaves exactly the keys [0, cur_pos[b*P + p]]."""
    H = 4
    pos = SD.serving_pad_rows(live_pos, B)
    assert SD.serving_width_for([S_k], max(pos), P - 1) == S_k, "the case sits on its rung"
    mask_full, _slide = SD.serving_masks_from_plan(SD.serving_mask_plan(pos, P), S_k, 1024, ring=None, h=H)
    host_full, _ = SD.serving_masks_host(pos, P, 1024, S_k)
    assert torch.equal(mask_full, host_full.repeat(1, 1, H, 1)), "the device build equals the host reference"
    assert tuple(mask_full.shape) == (B, 1, H * P, S_k)
    cur = vm.multipos_positions(pos, P)
    keys = torch.arange(S_k)
    for b in range(B):
        for h in range(H):
            for p in range(P):
                visible = mask_full[b, 0, h * P + p] == 0
                assert torch.equal(visible, keys <= int(cur[b * P + p])), (b, h, p)
                assert bool((mask_full[b, 0, h * P + p][~visible] == SD.SPEC_MASK_NEG).all())
        assert int(cur[b * P + P - 1]) < S_k, "the row's last candidate is inside the table span"


def _paged_cache(n_blocks, bs, D, seed):
    gen = torch.Generator().manual_seed(seed)
    k = torch.randn(n_blocks, 1, bs, D, generator=gen)
    v = torch.randn(n_blocks, 1, bs, D, generator=gen)
    return k, v


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


@pytest.mark.parametrize("live_pos, B, P, bs, S_k", _CASES)
def test_multipos_attention_equals_the_masked_packed_verify(live_pos, B, P, bs, S_k):
    """The whole global-layer attention both ways over one paged cache: the candidates' K/V written first (the
    per-candidate KV write), then today's masked packed SDPA over the rung vs the multi-position SDPA (Q re-layout,
    group tables, per-candidate bounds, un-pad) for every group size the planner may pick. Same rows out."""
    H, D = 4, 32
    gen = torch.Generator().manual_seed(len(live_pos) * 1000 + S_k + P)
    pos = SD.serving_pad_rows(live_pos, B)
    cols = S_k // bs
    adapter_cols = cols + 3  # the adapter's table is wider than the rung: the record fits it (allow_clip)
    n_blocks = 1 + len(live_pos) * adapter_cols  # block 0 is the null block
    # the adapter's per-row tables: shuffled distinct blocks per live row; padding rows mirror row 0
    # (_srv_upload_tables), then fitted to the rung's width
    perm = torch.randperm(n_blocks - 1, generator=gen) + 1
    adapter = torch.zeros(B, adapter_cols, dtype=torch.int64)
    for r in range(len(live_pos)):
        adapter[r] = perm[r * adapter_cols : (r + 1) * adapter_cols]
    adapter[len(live_pos) :] = adapter[0]
    table = SD.serving_select_rows(adapter, list(range(B)), B, cols, allow_clip=True)
    k_cache, v_cache = _paged_cache(n_blocks, bs, D, seed=S_k + P)
    k_new = torch.randn(B * P, D, generator=gen)
    v_new = torch.randn(B * P, D, generator=gen)
    for r in range(len(live_pos), B):  # a mirrored row computes the same K/V as row 0
        k_new[r * P : (r + 1) * P] = k_new[:P]
        v_new[r * P : (r + 1) * P] = v_new[:P]
    _write_candidates(k_cache, v_cache, table, pos, P, k_new, v_new, bs)

    q = torch.randn(1, H, B * P, D, generator=gen)
    for r in range(len(live_pos), B):
        q[:, :, r * P : (r + 1) * P] = q[:, :, :P]
    mask_full, _ = SD.serving_masks_from_plan(SD.serving_mask_plan(pos, P), S_k, 1024, ring=None, h=H)
    today = _today_unpack(_masked_packed_attention(_today_pack(q, P), k_cache, v_cache, table, mask_full, bs), H, P)

    cur = vm.multipos_positions(pos, P)
    for tg in _divisors(P):
        q_mp = vm.multipos_q_layout(q, P, tg)
        out = vm.multipos_reference_attention(
            q_mp, k_cache, v_cache, vm.multipos_page_table(table, P // tg), cur, tg, bs
        )
        got = vm.multipos_out_rows(out, H)
        assert torch.allclose(got, today, atol=1e-5, rtol=1e-5), f"Tg={tg}: max diff {(got - today).abs().max()}"
        for r in range(len(live_pos), B):
            assert torch.equal(got[0, r * P : (r + 1) * P], got[0, :P]), f"padding row {r} differs from row 0"


# ── per-layer gate ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_layer_reason():
    ok = dict(is_sliding=False, cache_position_modulo=None, kv_modulo=None, nkv_local=1, h_local=4, head_dim=512)
    assert vm.multipos_layer_reason(**ok, cache_shape=(4113, 1, 64, 512)) is None
    assert "sliding" in vm.multipos_layer_reason(**{**ok, "is_sliding": True}, cache_shape=(9, 1, 64, 512))
    assert "ring" in vm.multipos_layer_reason(**{**ok, "cache_position_modulo": 1024}, cache_shape=(9, 1, 64, 512))
    assert "ring" in vm.multipos_layer_reason(**{**ok, "kv_modulo": 1280}, cache_shape=(9, 1, 64, 512))
    assert "KV heads" in vm.multipos_layer_reason(**{**ok, "nkv_local": 2}, cache_shape=(9, 2, 64, 512))
    assert "tile" in vm.multipos_layer_reason(**{**ok, "h_local": 40}, cache_shape=(9, 1, 64, 512))
    # a hybrid-group buffer allocated in the sliding view (2 KV heads x 256) read by the global layer
    assert "view" in vm.multipos_layer_reason(**ok, cache_shape=(9, 2, 64, 256))
    assert "multiple" in vm.multipos_layer_reason(**ok, cache_shape=(9, 1, 16, 512))


# ── planner ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_cb_model_reproduces_the_op_numbers():
    """The spec-mode CB totals the op's own sweep documents (tests/ttnn/unit_tests/operations/sdpa/
    spec_multi_pos_test_utils.py, head_dim 256, bf16): T=4 chunk 128 at 64 cores, T=7 chunk 64 at 4, T=11 chunk 32
    at 1."""
    assert vm.spec_sdpa_cb_bytes(4, 8, 4, 64) == 1_310_720
    assert vm.spec_sdpa_cb_bytes(7, 8, 2, 4) == 1_329_152
    assert vm.spec_sdpa_cb_bytes(11, 8, 1, 1) == 1_447_936
    assert [vm.spec_sdpa_tree_rounds(c) for c in (1, 2, 3, 4, 5, 8, 9, 64)] == [0, 1, 2, 2, 3, 3, 4, 6]


def _check_plan(plan, B, P, grid, budget):
    n = grid[0] * grid[1]
    assert plan.fits, plan.describe()
    assert P % plan.tg == 0 and plan.groups == P // plan.tg and plan.sdpa_batch == B * plan.groups
    sk = plan.k_chunk // vm.TILE
    assert sk >= 1 and sk & (sk - 1) == 0, "spec mode needs a power-of-two chunk in tiles"
    assert min(plan.tg * sk, vm.DST_TILES) & (min(plan.tg * sk, vm.DST_TILES) - 1) == 0, "MUL_BCAST_GRANULARITY"
    assert plan.cores_per_head == vm.spec_sdpa_cores_per_head(n, plan.sdpa_batch, plan.max_cores_per_head_batch)
    assert plan.sdpa_batch * plan.cores_per_head <= n
    assert vm.spec_sdpa_tree_rounds(plan.cores_per_head) <= vm.MAX_TREE_ROUNDS
    want = vm.spec_sdpa_cb_bytes(plan.tg, 512 // 32, sk, plan.cores_per_head, plan.kv_tile_bytes, plan.extra_bytes)
    assert plan.cb_bytes == want <= budget


@pytest.mark.parametrize("grid", [(11, 10), (12, 10), (13, 10)], ids=lambda g: f"{g[0]}x{g[1]}")
@pytest.mark.parametrize("kv_tile", [BF16, BFP8], ids=["bf16", "bfp8"])
@pytest.mark.parametrize("shape", [(1, 5), (5, 5), (5, 3), (8, 3), (16, 1), (1, 3)], ids=lambda s: f"B{s[0]}K{s[1]}")
def test_plan_fits_the_31b_galaxy_records(shape, kv_tile, grid):
    """The 31B on a Galaxy row (TP 8: 4 local q heads, head_dim 512, one KV head) at the top rung's table width
    (263168 / 64 columns): a fitting config for every served shape."""
    B, K = shape
    P = K + 1
    extra = vm.multipos_extra_bytes(263168 // 64, B * P)
    plan = vm.plan_verify_multipos(B, P, 512, 4, grid, BH_BUDGET, kv_tile_bytes=kv_tile, extra_bytes=extra, env={})
    _check_plan(plan, B, P, grid, BH_BUDGET)


def test_plan_weighs_cores_against_kv_rereads():
    extra = vm.multipos_extra_bytes(4112, 20)
    # (5, 3) bf16 on 11x10: one group of 4 per row fits 4 cores (32-token chunk); two groups of 2 would keep 11 but
    # read every row's KV twice, which the five rows' DRAM stream does not afford
    plan = vm.plan_verify_multipos(5, 4, 512, 4, (11, 10), BH_BUDGET, extra_bytes=extra, env={})
    assert (plan.tg, plan.groups, plan.k_chunk, plan.cores_per_head) == (4, 1, 32, 4), plan.describe()
    two = vm.plan_verify_multipos(
        5, 4, 512, 4, (11, 10), BH_BUDGET, extra_bytes=extra, env={vm.VERIFY_MULTIPOS_TG_ENV: "2"}
    )
    assert (two.tg, two.cores_per_head) == (2, 11) and two.cost_ns > plan.cost_ns
    # one row (1, 3): its KV stream is a fifth of that, so 2 groups of 2 at 32 cores beat one group at 4
    lone = vm.plan_verify_multipos(1, 4, 512, 4, (11, 10), BH_BUDGET, extra_bytes=extra, env={})
    assert (lone.tg, lone.groups, lone.cores_per_head) == (2, 2, 32), lone.describe()
    # pins: a group size that does not divide P does not fit; a pinned one is taken as is
    assert not vm.plan_verify_multipos(5, 4, 512, 4, (11, 10), BH_BUDGET, env={vm.VERIFY_MULTIPOS_TG_ENV: "3"}).fits
    pinned = vm.plan_verify_multipos(5, 4, 512, 4, (11, 10), BH_BUDGET, env={vm.VERIFY_MULTIPOS_TG_ENV: "4"})
    assert pinned.fits and pinned.tg == 4
    # odd groups need a 4-tile chunk (min(3 x Sk, 8) a power of two)
    odd = vm.plan_verify_multipos(1, 6, 512, 4, (11, 10), BH_BUDGET, env={vm.VERIFY_MULTIPOS_TG_ENV: "3"})
    assert odd.fits and odd.k_chunk == 128, odd.describe()
    # nothing fits a tiny budget
    assert not vm.plan_verify_multipos(5, 4, 512, 4, (11, 10), 100_000, env={}).fits
