# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the multi-row verify KV write (QWEN36_KV_WRITE_MULTIROW, plan item KV-WRITE1).

Host-checkable parts only:
* the flag (default off) and its gating on the ttnn capability probe;
* the attention write issues ONE paged_fused_update_cache(rows_per_user=T) with the full per-row position vector and
  the per-user page table, K on cores [n, 2n) and V on [0, n), no slices, instead of T calls + 4T data moves;
* the verify skips its T per-candidate position slices when the flag is on;
* spec_kv_write_conflicts(across_calls=True): the stricter rule the one-call write needs (old == new, and it is a
  superset of the per-call rule);
* a host model of the multi-row kernels' row -> cache-tile -> pass grouping (reader/writer_paged_fused_update_cache_
  multi_row.cpp), run against T sequential single-row read-modify-writes through an idempotent per-row block-float
  repack: identical caches for tile straddles, page-block straddles, -1 rows, duplicate positions and aliased blocks.
The device byte-identity check is tests/ttnn/unit_tests/operations/transformers/test_paged_fused_update_cache_multi_row.py.
No device. Run: pytest models/demos/blackhole/qwen36/tests/test_kv_write_multirow_host.py -q
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import tp as tp_mod
from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention
from models.demos.blackhole.qwen36.tt.model import _spec_kv_write_conflicts_old, spec_kv_write_conflicts

TILE = 32


# ---- flag ---------------------------------------------------------------------------------------------------------
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_KV_WRITE_MULTIROW", raising=False)
    assert tpc.kv_write_multirow_enabled() is False
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "1")
    assert tpc.kv_write_multirow_enabled() is True
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "0")
    assert tpc.kv_write_multirow_enabled() is False


def test_flag_needs_capability(monkeypatch):
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "1")
    monkeypatch.setattr(tp_mod, "KV_MULTIROW_WRITE_SUPPORTED", False)
    assert tp_mod.kv_multirow_write_ok() is False
    monkeypatch.setattr(tp_mod, "KV_MULTIROW_WRITE_SUPPORTED", True)
    assert tp_mod.kv_multirow_write_ok() is True
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "0")
    assert tp_mod.kv_multirow_write_ok() is False


# ---- attention write (recording fake of the ttnn calls) -------------------------------------------------------------
class FT:
    def __init__(self, shape, name="t"):
        self.shape = tuple(shape)
        self.name = name

    def buffer_address(self):
        return id(self)


@pytest.fixture
def rec(monkeypatch):
    calls = []

    def op(name, ret=None):
        def f(*a, **k):
            calls.append((name, a, k))
            return ret(*a, **k) if ret is not None else FT((1,), name)

        return f

    monkeypatch.setattr(ttnn, "reshape", op("reshape", lambda t, s, p=None, **k: FT(tuple(s), "view")))
    monkeypatch.setattr(ttnn, "slice", op("slice"))
    monkeypatch.setattr(ttnn, "to_memory_config", op("to_memory_config"))
    monkeypatch.setattr(ttnn, "deallocate", op("deallocate", lambda *a, **k: None))
    monkeypatch.setattr(ttnn.experimental, "paged_fused_update_cache", op("fused", lambda *a, **k: None), raising=False)
    monkeypatch.setattr(ttnn, "Shape", lambda s: tuple(s), raising=False)
    return calls


def _attn():
    a = TPAttention.__new__(TPAttention)
    a.HD, a.NH, a.NKV = 256, 3, 1
    a.mesh = types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=11, y=10))
    a.paged_k, a.paged_v = FT((1,), "pk"), FT((1,), "pv")
    a.B = 8
    a._kv_key_shard_cfg = lambda n: ("kcfg", n)
    a._kv_shard_cfg = lambda n: ("vcfg", n)
    a._kv_rows_shard_cfg = lambda first, n, rows: ("rows", first, n, rows)
    return a


@pytest.mark.parametrize("n,T", [(8, 4), (1, 4), (2, 4), (4, 8)])
def test_multirow_write_is_one_fused_call(rec, monkeypatch, n, T):
    monkeypatch.setenv("QWEN36_KV_WRITE_MULTIROW", "1")
    monkeypatch.setattr(tp_mod, "KV_MULTIROW_WRITE_SUPPORTED", True)
    a, B = _attn(), n * T
    pos = [FT((n,), f"pos{j}") for j in range(T)]
    pt, cur = FT((n, 4), "pt"), FT((B,), "cur")
    a._write_kv_fused_groups(FT((1, B, 1, 256), "k"), FT((1, B, 1, 256), "v"), cur, n, pt, pos, B)
    names = [c[0] for c in rec]
    assert names.count("fused") == 1 and "slice" not in names
    _, args, kw = next(c for c in rec if c[0] == "fused")
    assert args[0] is a.paged_k and args[2] is a.paged_v
    assert kw["rows_per_user"] == T and kw["update_idxs_tensor"] is cur and kw["page_table"] is pt
    # [1, B, 1, HD] viewed as one user per shard, T tiles each
    views = [c[1][1] for c in rec if c[0] == "reshape"]
    assert views == [(1, n, T * TILE, 256)] * 2
    cfgs = [c[1][1] for c in rec if c[0] == "to_memory_config"]
    assert cfgs == [("rows", n, n, T * TILE), ("rows", 0, n, T * TILE)]  # K on [n, 2n), V on [0, n)
    assert names.index("fused") > max(i for i, c in enumerate(names) if c == "to_memory_config")


def test_flag_off_keeps_grouped_calls(rec, monkeypatch):
    monkeypatch.delenv("QWEN36_KV_WRITE_MULTIROW", raising=False)
    monkeypatch.setattr(tp_mod, "KV_MULTIROW_WRITE_SUPPORTED", True)
    a, n, T = _attn(), 8, 4
    B = n * T
    pos = [FT((n,), f"pos{j}") for j in range(T)]
    a._write_kv_fused_groups(FT((1, B, 1, 256), "k"), FT((1, B, 1, 256), "v"), FT((B,), "cur"), n, FT((n, 4)), pos, B)
    fused = [c for c in rec if c[0] == "fused"]
    assert len(fused) == T and all("rows_per_user" not in c[2] for c in fused)


def test_rows_shard_cfg_cores_disjoint(monkeypatch):
    """Real _kv_rows_shard_cfg: V cores [0, n) and K cores [n, 2n) never overlap and keep the user order."""
    a = _attn()
    del a._kv_rows_shard_cfg
    a._kv_rows_cfg_cache = {}
    monkeypatch.setattr(ttnn, "CoreCoord", lambda x, y: (x, y), raising=False)
    monkeypatch.setattr(ttnn, "CoreRange", lambda s, e: (s, e), raising=False)
    monkeypatch.setattr(ttnn, "CoreRangeSet", lambda s: frozenset(s), raising=False)
    monkeypatch.setattr(ttnn, "ShardSpec", lambda cores, shape, o: (cores, tuple(shape)), raising=False)
    monkeypatch.setattr(ttnn, "MemoryConfig", lambda lay, buf, spec: spec, raising=False)
    for n in (1, 2, 4, 8, 16):
        v_cores, v_shape = a._kv_rows_shard_cfg(0, n, 4 * TILE)
        k_cores, k_shape = a._kv_rows_shard_cfg(n, n, 4 * TILE)
        assert v_shape == k_shape == (4 * TILE, 256)
        assert len(v_cores) == len(k_cores) == n and not (v_cores & k_cores)
    assert a._kv_rows_shard_cfg(0, 8, 4 * TILE) is a._kv_rows_shard_cfg(0, 8, 4 * TILE)  # cached


# ---- conflict rule --------------------------------------------------------------------------------------------------
def test_across_calls_default_unchanged():
    pt = torch.full((2, 4), 9, dtype=torch.int32)
    # user 0 writes tile 0 rows 28..31, user 1 writes tile 1 rows 32..35 -> never the same tile
    assert spec_kv_write_conflicts(pt, [28, 32], 4, 64) == []
    assert spec_kv_write_conflicts(pt, [28, 32], 4, 64, across_calls=True) == []


def test_across_calls_catches_cross_candidate_tile_sharing():
    pt = torch.full((2, 4), 9, dtype=torch.int32)  # both users on block 9 (only legal if they never share a tile)
    # user 0 rows 26..29, user 1 rows 30..33: call j writes 26+j vs 30+j -> same tile only at j = 0, 1 ...
    per_call = spec_kv_write_conflicts(pt, [26, 30], 4, 64)
    across = spec_kv_write_conflicts(pt, [26, 30], 4, 64, across_calls=True)
    assert set(per_call) <= set(across)
    # block 128: user 0 rows 62..65 is on tile 1 at j = 0, 1; user 1 rows 30..33 reaches tile 1 at j = 2, 3. They share
    # tile 1 but never in the same call: safe for T grouped calls, a lost row for one multi-row call.
    assert spec_kv_write_conflicts(pt, [62, 30], 4, 128) == []
    got = spec_kv_write_conflicts(pt, [62, 30], 4, 128, across_calls=True)
    assert got == [(2, 0, 1, 9, 1), (3, 0, 1, 9, 1)]
    assert got == _spec_kv_write_conflicts_old(pt, [62, 30], 4, 128, across_calls=True)


@pytest.mark.parametrize("seed", range(30))
def test_across_calls_old_matches_new_and_is_superset(seed):
    g = torch.Generator().manual_seed(seed)
    B, T, bs, nb = 8, 4, 64, int(torch.randint(2, 20, (1,), generator=g))
    pt = torch.randint(0, 4, (B, nb), generator=g, dtype=torch.int32)
    pos = torch.randint(-3, nb * bs - T, (B,), generator=g).tolist()
    new = spec_kv_write_conflicts(pt, pos, T, bs, across_calls=True)
    assert new == _spec_kv_write_conflicts_old(pt, pos, T, bs, across_calls=True)
    per_call = spec_kv_write_conflicts(pt, pos, T, bs)
    # every same-call collision is still reported (as a (block, tile) pair) by the stricter rule
    assert {(c[3], c[4]) for c in per_call} <= {(c[3], c[4]) for c in new}


# ---- host model of the multi-row kernels vs T sequential single-row updates -----------------------------------------
def _bfp_repack(rows):
    """Per-row block-float repack (shared exponent per 16 datums of a row, 7-bit magnitude, saturating): the
    property the cache relies on is that it is per-row and idempotent, which is what makes one final repack of a
    tile equal to T successive ones. Rows never mix exponents (BFP8_B shares within a face row)."""
    x = rows.reshape(*rows.shape[:-1], -1, 16).double()
    amax = x.abs().amax(-1, keepdim=True)
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp_min(1e-30))), torch.zeros_like(amax))
    step = torch.pow(2.0, e - 6)
    q = torch.clamp(torch.round(x / step), -127, 127) * step
    return q.reshape(rows.shape).float()


def _cache_id(pos, user, pt, block_size):
    blk = int(pt[user][pos // block_size])
    return (blk, (pos % block_size) // TILE)


def _seq_reference(cache, x, idx, pt, block_size):
    """T sequential single-row RMWs (the shipped grouped path): call j updates row j of every user."""
    cache = cache.clone()
    n, T = x.shape[:2]
    for j in range(T):
        for u in range(n):
            p = idx[u * T + j]
            if p < 0:
                continue
            blk, t = _cache_id(p, u, pt, block_size)
            tile = cache[blk, t * TILE : (t + 1) * TILE].clone()
            tile[p % TILE] = x[u, j]
            cache[blk, t * TILE : (t + 1) * TILE] = _bfp_repack(tile)
    return cache


def _multirow_model(cache, x, idx, pt, block_size):
    """Mirror of the multi-row reader/writer: per user, group rows by physical cache tile in first-seen order, then one
    RMW per group with rows copied in increasing j (later row wins). num_passes = T; extra passes do nothing."""
    cache = cache.clone()
    n, T = x.shape[:2]
    reads = 0
    for u in range(n):
        rows = [idx[u * T + j] for j in range(T)]
        pass_tile, row_pass = [], [None] * T
        for j, p in enumerate(rows):
            if p < 0:
                continue
            key = _cache_id(p, u, pt, block_size)
            if key not in pass_tile:
                pass_tile.append(key)
            row_pass[j] = pass_tile.index(key)
        assert len(pass_tile) <= T
        for pi, (blk, t) in enumerate(pass_tile):
            tile = cache[blk, t * TILE : (t + 1) * TILE].clone()
            reads += 1
            for j in range(T):
                if row_pass[j] == pi:
                    tile[rows[j] % TILE] = x[u, j]
            cache[blk, t * TILE : (t + 1) * TILE] = _bfp_repack(tile)
    return cache, reads


def _setup(n, T, block_size, nb, seed):
    g = torch.Generator().manual_seed(seed)
    n_blocks = 1 + n * nb  # block 0 = vLLM null block
    perm = torch.randperm(n_blocks - 1, generator=g) + 1
    pt = perm.reshape(n, nb).tolist()
    cache = _bfp_repack(torch.randn(n_blocks, block_size, 64, generator=g))
    x = torch.randn(n, T, 64, generator=g).bfloat16().float()
    return g, pt, cache, x


@pytest.mark.parametrize("block_size", [32, 64, 128])
@pytest.mark.parametrize("T", [2, 4, 8])
@pytest.mark.parametrize("seed", range(6))
def test_multirow_model_matches_sequential(block_size, T, seed):
    n, nb = 4, 6
    g, pt, cache, x = _setup(n, T, block_size, nb, seed)
    # consecutive candidates p..p+T-1 (the verify layout), chosen to straddle tiles and page blocks
    starts = [TILE - 1, block_size - 2, 5, -1]
    idx = []
    for u, p in enumerate(starts):
        idx += [p + j if p >= 0 else -1 for j in range(T)] if u != 3 else [-1] + [j for j in range(T - 1)]
    ref = _seq_reference(cache, x, idx, pt, block_size)
    got, reads = _multirow_model(cache, x, idx, pt, block_size)
    assert torch.equal(ref, got)
    # each user touches at most ceil((31 + T) / 32) + 1 tiles, never T when the rows fit in two
    assert reads <= n * 2


def test_multirow_model_random_positions_and_duplicates():
    for seed in range(40):
        n, T, block_size, nb = 3, 4, 64, 5
        g, pt, cache, x = _setup(n, T, block_size, nb, 100 + seed)
        idx = torch.randint(-1, nb * block_size, (n * T,), generator=g).tolist()
        if seed % 3 == 0:
            idx[1] = idx[2] = max(idx[1], 0)  # duplicate position in one user: the later row must win
        ref = _seq_reference(cache, x, idx, pt, block_size)
        got, _ = _multirow_model(cache, x, idx, pt, block_size)
        assert torch.equal(ref, got)


def test_multirow_model_aliased_virtual_blocks():
    """Two virtual blocks of one user mapping to ONE physical block (a padded table): grouping is by physical tile, so
    rows that alias one tile share a pass and none is lost (a stale second read would drop the first pass's row)."""
    n, T, block_size, nb = 1, 4, 32, 4
    _, _, cache, x = _setup(n, T, block_size, nb, 7)
    pt = [[3, 3, 3, 3]]
    idx = [30, 31, 32, 33]  # virtual blocks 0 and 1 -> physical block 3, tile 0 both
    ref = _seq_reference(cache, x, idx, pt, block_size)
    got, reads = _multirow_model(cache, x, idx, pt, block_size)
    assert torch.equal(ref, got) and reads == 1
