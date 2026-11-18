# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Hy4 context-parallel KV stripe (``tt/hy4/kv_cache.py``, #65886).

* the token -> (chip, row) map, its inverse and the host stripe / de-stripe;
* the mesh coordinate of each chip on the physical 8x4 Galaxy mesh for the TP32 layout, which is also the dev
  layout's mesh (sequence over SP 8 on axis 0, hidden over TP 4 on axis 1: chip = 4 * row + col, row-major), and for
  SP on axis 1 (SP 4 on axis 1 x TP 8 on axis 0, the first TP32 split and GLM-5.3's PREFILL_SP_AXIS=1: chip =
  8 * col + row);
* a Python model of ``update_padded_kv_cache``'s writer and reader (the per-chip offset and source-row math of
  ``writer_update_padded_kv_cache.cpp`` / ``reader_update_padded_kv_cache.cpp``, with the producer's chunk rotation
  ``chunk_layout.rotate_chunk_tokens``): chunked prefill from any tile-aligned start, prefix-cache resume and a
  partial last chunk all leave the cache the stripe describes, identical to a one-shot prefill;
* the migration table (``populate_kv_chunk_address_table_block_cyclic``) publishes every position on the chip the
  stripe names, in row order;
* the chunk-size contract (128-token blocks need a 4096-token chunk on 32 chips), the 21 index-key slots
  (layer 77 included) and the per-token bytes.
"""

from __future__ import annotations

import json
import types
from pathlib import Path

import pytest
import torch

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens
from models.demos.deepseek_v3_d_p.reference.hy4_config import Hy4Config
from models.demos.deepseek_v3_d_p.tt.hy4.kv_cache import (
    HY4_KV_BLOCK,
    KVPE_SCALED_FP8_ROW_BYTES,
    KvBudget,
    KvStripe,
    bank_aligned_max_seq_len,
    full_indexer_layers,
    hy4_stripe,
    index_slot,
    merge_local_topk,
    stripe_key_axis,
)

TP32 = KvStripe(sp=8, tp=4, sp_axis=0)  # tp32 (#65884: sequence over the 8-chip axis 0, hidden over the 4-chip axis 1)
DEV = TP32  # glm_prefill_sp8_tp4: the same 8x4 mesh and axes, so the same stripe
SP_AXIS1 = KvStripe(sp=4, tp=8, sp_axis=1)  # SP over the 4 columns: the module still supports it
STRIPES = [pytest.param(TP32, id="tp32-sp8tp4"), pytest.param(SP_AXIS1, id="sp4-axis1")]
MANIFEST = Path(__file__).resolve().parents[1] / "tt" / "runners" / "manifests" / "hy4_preview.json"


# ----------------------------------------------------------------------------------------------------------------
# The map
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stripe", STRIPES)
def test_geometry(stripe):
    assert (stripe.block, stripe.num_chips, stripe.period) == (128, 32, 4096)
    assert stripe.mesh_shape == (8, 4)


@pytest.mark.parametrize("stripe", STRIPES)
def test_token_to_chip_row_round_trip(stripe):
    pos = torch.arange(5 * stripe.period + 77)
    chip, row = stripe.chip(pos), stripe.row(pos)
    assert torch.equal(stripe.position(chip, row), pos)
    # 128-token blocks, cyclic over all 32 chips
    assert torch.equal(chip, (pos // 128) % 32)
    # every block sits whole on one chip, in consecutive rows
    blocks = pos[: 4 * stripe.period].reshape(-1, 128)
    for b in blocks[::7]:
        assert len(set(stripe.chip(b).tolist())) == 1
        r = stripe.row(b)
        assert torch.equal(r - r[0], torch.arange(128))
    # (chip, row) pairs are unique: no two tokens share a cache cell
    cells = set(zip(chip.tolist(), row.tolist()))
    assert len(cells) == pos.numel()


@pytest.mark.parametrize("stripe", STRIPES)
def test_each_chip_holds_one_32nd(stripe, expect_error):
    seq = 3 * stripe.period
    counts = torch.bincount(stripe.chip(torch.arange(seq)), minlength=32)
    assert torch.all(counts == seq // 32)
    assert stripe.rows_per_chip(seq) == seq // 32
    with expect_error(ValueError, "whole number"):
        stripe.rows_per_chip(seq + 128)


def test_mesh_coords(expect_error):
    # TP32 (and dev): SP position = mesh row, TP position = mesh column, chip = row * 4 + col, the row-major mesh order
    # the shared layers' sparse KV push assumes (ttMLA.push_matches_stripe_order holds with SP on axis 0)
    assert [TP32.coord(c) for c in (0, 1, 3, 4, 31)] == [(0, 0), (0, 1), (0, 3), (1, 0), (7, 3)]
    assert [TP32.chip_of_coord((r, c)) for r in range(8) for c in range(4)] == list(range(32))
    # SP on axis 1: SP position = mesh column, TP position = mesh row, chip = col * 8 + row (not row-major)
    assert [SP_AXIS1.coord(c) for c in (0, 1, 7, 8, 31)] == [(0, 0), (1, 0), (7, 0), (0, 1), (7, 3)]
    for s in (TP32, SP_AXIS1):
        coords = [s.coord(c) for c in range(32)]
        assert sorted(coords) == [(r, c) for r in range(8) for c in range(4)]
        assert [s.chip_of_coord(x) for x in coords] == list(range(32))
    with expect_error(ValueError, "outside"):
        DEV.coord(32)


@pytest.mark.parametrize("stripe", STRIPES)
@pytest.mark.parametrize("length", [1, 127, 128, 4095, 4096, 4097, 9000, 3 * 4096])
def test_stripe_destripe_round_trip(stripe, length):
    rows = torch.randn(length, 3)
    slabs = stripe.stripe(rows)
    assert slabs.shape[0] == 32 and slabs.shape[1] % 128 == 0
    assert torch.equal(stripe.destripe(slabs, length), rows)
    # the slab rows are where chip / row say they are
    pos = torch.arange(length)
    assert torch.equal(slabs[stripe.chip(pos), stripe.row(pos)], rows)


@pytest.mark.parametrize("stripe", STRIPES)
def test_populated_rows_and_local_positions(stripe):
    for kv_len in (0, 1, 31, 128, 129, 4000, 4096, 4097, 8192 + 1000, 5 * 4096 + 4095):
        pos = torch.arange(kv_len)
        held = torch.bincount(stripe.chip(pos), minlength=32) if kv_len else torch.zeros(32, dtype=torch.long)
        for chip in range(32):
            n = stripe.populated_rows(chip, kv_len)
            assert n == held[chip]
            lp = stripe.local_positions(chip, n)
            # the populated rows are a prefix holding exactly this chip's positions < kv_len, ascending
            assert torch.equal(lp, pos[stripe.chip(pos) == chip])
            assert torch.equal(stripe.row(lp), torch.arange(n))


@pytest.mark.parametrize("stripe", STRIPES)
def test_decode_append_location(stripe):
    # decode appends token p at chip(p), row(p); consecutive tokens fill a block, then move to the next chip
    for p in (0, 127, 128, 4095, 4096, 4096 * 7 + 300):
        coord, row = stripe.locate(p)
        assert stripe.chip_of_coord(coord) == stripe.chip(p) and row == stripe.row(p)
        assert row == stripe.populated_rows(stripe.chip(p), p)  # the next free row of that chip


# ----------------------------------------------------------------------------------------------------------------
# update_padded_kv_cache model: chunked prefill, resume, partial chunks
# ----------------------------------------------------------------------------------------------------------------
def _local_cache_offset(prefix, coordinate, factor, chunk):
    """writer_update_padded_kv_cache.cpp: local_cache_offset."""
    slab = prefix // (factor * chunk)
    boundary = (prefix // chunk) % factor
    offset = prefix % chunk
    return slab * chunk + (chunk if coordinate < boundary else (offset if coordinate == boundary else 0))


def _write_chunk(cache, chunk_tokens, start, valid_end, stripe):
    """One update_padded_kv_cache call with tp_axis set (KV dedup), tokens as values (row units, tile_height 1).

    The producer hands SP row s the s-th slice of rotate_chunk_tokens(chunk, start, sp), TP-replicated. Each chip's
    reader picks its rows out of that slice and its writer stores them from local_cache_offset on, clamped to
    valid_end. Pad tokens are -1."""
    sp, tp, n = stripe.sp, stripe.tp, stripe.num_chips
    chunk = len(chunk_tokens)
    input_ht = chunk // sp  # the SP slice every chip of SP row s holds
    b = input_ht // tp  # chunk_local_t: rows a chip writes
    rotated = rotate_chunk_tokens(list(chunk_tokens), start, sp)
    for lin in range(n):
        s_coord = lin // tp
        window = rotated[s_coord * input_ht : (s_coord + 1) * input_ht]
        # reader_update_padded_kv_cache.cpp source-row mapping
        start_in_stripe = start % b if lin == (start // b) % n else 0
        start_in_window = start % input_ht if s_coord == (start // input_ht) % sp else 0
        base = ((lin % tp) * b + start_in_stripe + input_ht - start_in_window) % input_ht
        jump = b * (tp - 1)
        update = _local_cache_offset(start, lin, n, b)
        end = _local_cache_offset(-(-valid_end // 32) * 32, lin, n, b)
        rows = min(max(end - update, 0), b)
        for j in range(rows):
            src = base + j + (jump if j + start_in_stripe >= b else 0)
            cache[lin][update + j] = window[src]


def _prefill(stripe, seq_cache, length, starts, prefix_cache=None):
    """Prefill tokens [starts[0], length) in chunks beginning at ``starts``; value of a token = its position."""
    cache = [[-2] * (seq_cache // stripe.num_chips) for _ in range(stripe.num_chips)]
    if prefix_cache is not None:
        cache = [list(c) for c in prefix_cache]
    c = stripe.period
    bounds = list(starts) + [length]
    for a, z in zip(bounds[:-1], bounds[1:]):
        while a < z:
            valid = min(z, a + c)
            toks = [p if p < valid else -1 for p in range(a, a + c)]
            _write_chunk(cache, toks, a, valid, stripe)
            a = valid
    return cache


def _expected(stripe, seq_cache, length, unwritten=-2):
    cache = [[unwritten] * (seq_cache // stripe.num_chips) for _ in range(stripe.num_chips)]
    for p in range(length):
        cache[stripe.chip(p)][stripe.row(p)] = p
    return cache


def _strip_tail(cache, stripe, length):
    """Positions >= length on each chip may hold the pad of a clamped 32-row block; ignore them."""
    return [
        [v if (stripe.position(chip, r) < length) else None for r, v in enumerate(rows)]
        for chip, rows in enumerate(cache)
    ]


@pytest.mark.parametrize("stripe", STRIPES)
@pytest.mark.parametrize(
    "length, starts",
    [
        (4096, [0]),  # one full chunk
        (3 * 4096, [0]),  # whole chunks
        (2 * 4096 + 1000, [0]),  # partial last chunk
        (3 * 4096, [0, 1312]),  # resume mid-block (tile aligned, not block aligned)
        (3 * 4096, [0, 256, 4096 + 2048]),  # resume at block boundaries mid-period
        (2 * 4096 + 64, [0, 4096 - 32]),  # resume one tile before a period boundary
    ],
)
def test_chunked_prefill_writes_the_stripe(stripe, length, starts):
    seq_cache = 4 * stripe.period
    got = _prefill(stripe, seq_cache, length, starts)
    want = _expected(stripe, seq_cache, length)
    assert _strip_tail(got, stripe, length) == _strip_tail(want, stripe, length)
    # history independence: same cache as a one-shot (chunk-aligned) prefill of the same tokens
    assert _strip_tail(got, stripe, length) == _strip_tail(_prefill(stripe, seq_cache, length, [0]), stripe, length)


@pytest.mark.parametrize("stripe", STRIPES)
@pytest.mark.parametrize("hit", [128, 1312, 4096, 4096 + 640])
def test_prefix_cache_hit_resume(stripe, hit):
    """A prefix-cache hit: the first ``hit`` tokens are already cached (by another request); prefill starts there."""
    seq_cache = 4 * stripe.period
    length = 2 * stripe.period + 500
    prefix = _prefill(stripe, seq_cache, hit, [0])
    got = _prefill(stripe, seq_cache, length, [hit], prefix_cache=prefix)
    assert _strip_tail(got, stripe, length) == _strip_tail(_expected(stripe, seq_cache, length), stripe, length)
    # the chunk starting at the hit fills exactly the rows chunk_positions names on each chip
    for chip, pos in enumerate(stripe.chunk_positions(hit)):
        assert torch.all(pos >= hit) and torch.equal(stripe.chip(pos), torch.full_like(pos, chip))
        assert pos.numel() == stripe.block


def test_resume_must_be_tile_aligned(expect_error):
    KvStripe.check_resume(32 * 41)
    with expect_error(ValueError, "multiple of 32"):
        KvStripe.check_resume(1000)


# ----------------------------------------------------------------------------------------------------------------
# Migration table
# ----------------------------------------------------------------------------------------------------------------
class _FakeTable:
    def __init__(self):
        self.groups, self.entries = [], {}

    def add_device_group(self, fids):
        self.groups.append(list(fids))
        return len(self.groups) - 1

    def set_fabric_node_host(self, *a, **kw):
        pass

    def set(self, row, position, slot, location, config_id):
        self.entries[(config_id, row, slot, position)] = location


@pytest.mark.parametrize("stripe", STRIPES)
def test_migration_table_matches_stripe(stripe, monkeypatch):
    import ttnn
    from models.demos.deepseek_v3_d_p.utils import kv_cache_utils as kcu

    class _Loc:
        pass

    monkeypatch.setattr(ttnn.experimental.disaggregation, "KvCacheLocation", _Loc, raising=False)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)
    monkeypatch.setattr(ttnn, "distributed_context_get_size", lambda: 1)
    monkeypatch.setattr(ttnn, "MeshCoordinate", lambda *c: tuple(c))
    banks = 8
    monkeypatch.setattr(kcu, "get_num_dram_banks", lambda mesh: banks)
    mesh = types.SimpleNamespace(
        get_fabric_node_id=lambda coord: types.SimpleNamespace(mesh_id=0, chip_id=coord[0] * 4 + coord[1], coord=coord)
    )
    num_layers, users, seq = 2, 2, 2 * stripe.period
    cache = types.SimpleNamespace(shape=[users * num_layers, 1, seq // 32, 656], buffer_address=lambda: 0)
    table = kcu.populate_kv_chunk_address_table_block_cyclic(
        lookup_table=_FakeTable(),
        config=types.SimpleNamespace(num_layers=num_layers),
        mesh_device=mesh,
        mesh_shape=stripe.mesh_shape,
        seq_len=seq,
        sp_axis=stripe.sp_axis,
        tt_kvpe_cache=cache,
        chunk_size_bytes=1000,
        num_users=users,
        tp_axis=stripe.tp_axis,
        chunk_size_global=stripe.period,
    )
    positions = range(0, seq, 32)
    for slot in range(users):
        for layer in range(num_layers):
            for p in positions:
                loc = table.entries[(0, layer, slot, p)]
                # the device group is the single chip the stripe names
                assert [f.coord for f in table.groups[loc.device_group_index]] == [stripe.coord(stripe.chip(p))]
                # its address is the 32-row shard holding row(p) of slot (slot, layer): bank round-robin
                shard = (slot * num_layers + layer) * (seq // 32 // 32) + stripe.row(p) // 32
                assert loc.noc_addr >> 32 == shard % banks
                assert loc.noc_addr & 0xFFFFFFFF == (shard // banks) * 1000


# ----------------------------------------------------------------------------------------------------------------
# Contracts: chunk size, index-key slots, bytes per token
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_shape, sp_axis", [((8, 4), 0), ((8, 4), 1)])
def test_chunk_size_contract(mesh_shape, sp_axis, expect_error):
    s = hy4_stripe(mesh_shape, sp_axis, 4096)
    assert s.block == HY4_KV_BLOCK and s.num_chips == 32
    for bad in (2048, 5120, 8192):
        with expect_error(ValueError, "chunk must be 4096"):
            hy4_stripe(mesh_shape, sp_axis, bad)
    # a test mesh below 32 chips is not the SC1 stripe and is not pinned
    assert hy4_stripe((2, 4), 0, 4096).block == 512


def test_from_mesh_matches_the_op_block():
    # update_padded_kv_cache writes chunk / (sp * tp) rows per chip
    assert KvStripe.from_mesh((8, 4), 0, 5120).block == 160
    assert KvStripe.from_mesh((8, 4), 0, 4096) == TP32
    assert KvStripe.from_mesh((8, 4), 1, 4096) == SP_AXIS1


def test_tp32_layout_writes_the_tp32_stripe():
    """The tp32 layout's mesh knobs (sequence on SP, axis 0) give the stripe decode reads; dev's are the same."""
    from models.demos.deepseek_v3_d_p.tt.hy4 import layout as hy4_layout

    for name in (hy4_layout.TP32_LAYOUT, hy4_layout.DEV_LAYOUT):
        lay = hy4_layout.HY4_PREFILL_LAYOUTS[name]
        mesh = [0, 0]
        mesh[lay.sp_axis], mesh[1 - lay.sp_axis] = lay.sp, lay.tp
        assert hy4_stripe(tuple(mesh), lay.sp_axis, 4096) == TP32, name


def test_manifest_chunk_writes_the_stripe():
    env = json.loads(MANIFEST.read_text())["env"]
    chunk, max_seq = int(env["PREFILL_CHUNK_SIZE"]), int(env["PREFILL_MAX_SEQ_LEN"])
    s = hy4_stripe((int(env["PREFILL_SP"]), int(env["PREFILL_TP"])), int(env["PREFILL_SP_AXIS"]), chunk)
    assert s.block == 128 and max_seq % s.period == 0
    assert bank_aligned_max_seq_len(s, max_seq) == max_seq


def test_index_key_slots(expect_error):
    types_ = list(Hy4Config.INDEXER_TYPES) if hasattr(Hy4Config, "INDEXER_TYPES") else None
    if types_ is None:
        from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config

        types_ = list(hy4_hf_config(max_seq=8192).indexer_types)
    full = full_indexer_layers(types_)
    assert full == [0, 1] + list(range(5, 78, 4))
    assert len(full) == 21 and full[-1] == 77  # layer 77 is the kv-only last layer and still owns a slot
    assert [index_slot(types_, layer) for layer in full] == list(range(21))
    with expect_error(ValueError, "shared indexer layer"):
        index_slot(types_, 2)
    from models.demos.deepseek_v3_d_p.reference.hy4_config import hy4_hf_config
    from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank

    cfg = hy4_hf_config(max_seq=8192)
    assert [full_indexer_rank(cfg, layer) for layer in full] == list(range(21))


def test_bytes_per_token():
    b = KvBudget()
    assert b.values_per_token == 47_616  # 78 x 576 + 21 x 128
    assert KvBudget(num_layers=79, num_index_layers=22).values_per_token - b.values_per_token == 704  # MTP layer
    # the 656-byte packed KVPE row occupies 704 bytes of DRAM (64-byte page alignment)
    assert KVPE_SCALED_FP8_ROW_BYTES == 656 and b.kvpe_row_bytes == 704 and b.index_row_bytes == 136
    assert b.bytes_per_token == 78 * 704 + 21 * 136 == 57_768
    assert KvBudget(num_layers=79, num_index_layers=22).bytes_per_token == 57_768 + 704 + 136  # with the MTP layer
    # per chip, every token costs 1/32 of that
    free = 17_600_000_000
    assert b.tokens(free) == (free // 57_768) * 32


# ----------------------------------------------------------------------------------------------------------------
# Distributed top-k: local top-k per chip + gather + merge == global top-k
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("stripe", STRIPES)
@pytest.mark.parametrize("k", [16, 128, 2048])
def test_local_topk_merge_equals_global(stripe, k):
    torch.manual_seed(k)
    q, s_len = 48, 3 * stripe.period
    scores = torch.randn(q, s_len)
    # causal-style validity: query i sees keys [0, kv_len_i); some see fewer than k keys
    kv_len = torch.linspace(100, s_len, q).long()
    scores = scores.masked_fill(torch.arange(s_len).unsqueeze(0) >= kv_len.unsqueeze(1), float("-inf"))
    per_chip = stripe_key_axis(scores, stripe)  # [mesh rows, mesh cols, Q, rows]
    rows = per_chip.shape[-1]
    k_local = min(k, rows)
    lv, lr = [], []
    for chip in range(32):
        r, c = stripe.coord(chip)
        v, i = per_chip[r, c].topk(k_local, dim=-1)
        lv.append(v)
        lr.append(i)
    v, p = merge_local_topk(torch.stack(lv), torch.stack(lr), stripe, k)
    gv, gi = scores.topk(k, dim=-1)
    gi = gi.masked_fill(torch.isneginf(gv), -1)
    assert torch.equal(v, gv)
    # same selection; equal scores may come back in either order
    assert torch.equal(p.sort(dim=-1).values, gi.sort(dim=-1).values)


@pytest.mark.parametrize("stripe", STRIPES)
def test_stripe_key_axis_places_each_chips_keys(stripe):
    scores = torch.arange(2 * stripe.period, dtype=torch.float32).unsqueeze(0).repeat(3, 1)  # value = key position
    per_chip = stripe_key_axis(scores, stripe)
    for chip in (0, 9, 31):
        r, c = stripe.coord(chip)
        assert torch.equal(per_chip[r, c, 0].long(), stripe.local_positions(chip, per_chip.shape[-1]))
