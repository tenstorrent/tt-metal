# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests (no device) of tt/apc_paged_store.py: key block index math against the single-trace ``SingleCtx.values`` / key_append row-block offsets, store sizes, slot accounting."""

import pytest

from models.demos.blackhole.deepseek_v41_flash.tt import apc_paged_store as S


@pytest.mark.parametrize("ratio,nrb", [(2, 1), (1, 2)])
def test_geom_chunk_512(ratio, nrb):
    g = S.KeyGeom(512, ratio, 8)
    assert g.Cc8 == 32 * nrb and g.nrb == nrb and g.np == 4 * nrb
    assert g.block_bytes == 8192 * nrb  # 8 KiB (ratio 2) / 16 KiB (ratio 1) per chip


def test_block_bytes_total_40k():
    st = S.store_bytes_per_chip(128, 48, 2048)
    assert st["key_block_bytes"] == 3 * 8192 + 16384 == 40 * 1024
    assert st["halo_a"] == 128 * 21 * 128 * 1024 and st["halo_b"] == 48 * 19 * 128 * 1024
    # design note: 66 MiB/bank = 128 A + 48 B + 2048 key blocks (per bank = per chip / 8 DRAM banks)
    per_bank = (st["halo_a"] + st["halo_b"] + 2048 * st["key_block_bytes"]) / 8 / 2**20
    assert 60 < per_bank < 70


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("i", [0, 1, 7, 100])
def test_hist_row_block_matches_kofs(ratio, i):
    """kofs of a chunk starting at s = i * 512 as ``SingleCtx.values``: (Cc8 + (s / ratio) // cols) // 32"""
    cols, C = 8, 512
    g = S.KeyGeom(C, ratio, cols)
    s = i * S.GRID
    nloc = g.Cc8 + (s // ratio) // cols
    assert nloc % 32 == 0
    assert g.hist_row_block(i) == nloc // 32


def test_slab_capacity():
    g = S.KeyGeom(512, 2, 8)
    rows = g.slab_rows(1 << 20)  # 1M tokens: (256 + 524288) / 8 + 32
    assert rows == (256 + 524288) // 8 + 32
    # history holds 1M / 512 = 2048 chunk blocks (the pad rows are extra)
    assert g.max_blocks(rows) >= 2048


def test_slot_pool():
    p = S.SlotPool(3)
    a = [p.alloc() for _ in range(3)]
    assert sorted(a) == [0, 1, 2] and p.alloc() is None
    p.free(1)
    assert p.alloc() == 1
    bad = []
    for s in (7, 0, 0):  # a foreign slot, then a double free of 0 (the first free of 0 is fine)
        try:
            p.free(s)
        except ValueError:
            bad.append(s)
    assert bad == [7, 0]
    assert p.free_count == 1 and p.used_count == 2


def test_row_slots():
    r = S.RowSlots(4, halo_a=2, halo_b=1, key_blocks=5)
    assert r.alloc("A", 0) == 0 and r.alloc("A", 0) == 1 and r.alloc("A", 0) is None
    assert r.alloc("A", 1) == 0  # rows are independent
    assert r.alloc("B", 3) == 0 and r.alloc("B", 3) is None


def test_layer_parts():
    assert len(S.PART_A) == 21 and len(S.PART_B) == 19 and set(S.PART_A) | set(S.PART_B) == set(range(40))
    assert S.halo_slot_bytes("A") + S.halo_slot_bytes("B") == 5 * 1024 * 1024


def test_halo_geometry_pages():
    assert S.HALO_PAGES == 64 and S.HALO_PAGES * S.TILE_BYTES == 128 * 1024
