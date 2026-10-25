# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only check of the fixed KV layout slab (GLM_KV_SLAB_TOKENS) position maps.

Python replicas of the device-side index math (KV writer/reader, indexed RoPE, indexer causal geometry, invP remap,
sparse push owner map, gather extent) are checked against the slab layout's ground truth
(rotated_chip_positions / blockcyclic_positions with slab S) on the 8x4 GLM geometry, for chunk C in {1024, 2048, 5120}
and slab S in {1024, C}, at chunk-aligned and rotated (mid-slab) starts. The S == C cases must reproduce today's
kernels exactly. No device: `pytest -q models/demos/deepseek_v3_d_p/tests/test_kv_slab_layout_maps.py`.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    block_cyclic_reorder,
    blockcyclic_positions,
    rotated_chip_positions,
    slab_local_cache_offset as lco,
)

SP, TP, TW = 8, 4, 32
F = SP * TP  # linear (TP-deduped) block-cyclic chip count


def cache_pos(L, lr, factor, block):
    """Position held by linear chip L's local cache row lr (the update_padded_kv_cache writer layout)."""
    return (lr // block) * (factor * block) + L * block + lr % block


def reader_old(start, L, cl, R):
    """reader_update_padded_kv_cache.cpp source-row map as of 52aece4265d (rows, ROW_MAJOR units)."""
    sis = start % cl if L == (start // cl) % F else 0
    siw = start % R if L // TP == (start // R) % SP else 0
    base = ((L % TP) * cl + sis + R - siw) % R
    jump = cl * (TP - 1)
    return [base + j + (jump if j + sis >= cl else 0) for j in range(cl)]


def reader_new(start, L, cl, b):
    """Slab-aware source-row map: write row U1+j of linear chip L <- SP-window row of the same position."""
    c, t = divmod(L, TP)
    s = b * TP
    u1, uc = lco(start, L, F, b), lco(start, c, SP, s)
    return [((u1 + j) // b) * s + t * b + (u1 + j) % b - uc for j in range(cl)]


def causal_geometry_tiles(start, sp, chunk_local, device_index, tp_index, Sq):
    """indexer_score_causal_geometry.hpp, rotation_exact branch (tokens in, tiles out)."""
    chunk_global = sp * chunk_local
    boundary_slab = start // chunk_global
    boundary_chip = (start // chunk_local) % sp
    offset = start % chunk_local
    if device_index < boundary_chip:
        u = (boundary_slab + 1) * chunk_local
    elif device_index == boundary_chip:
        u = boundary_slab * chunk_local + offset
    else:
        u = boundary_slab * chunk_local
    lr0 = u + tp_index * Sq
    loff = lr0 % chunk_local
    logical_start = (lr0 // chunk_local) * chunk_global + device_index * chunk_local + loff
    sq_tile, jump = 0, 0
    # Slab layout: Sq may exceed the SP-level block, so a straddle can start at loff == 0 (the header's condition
    # was `loff != 0 && ...`, equivalent while Sq <= chunk_local was enforced).
    if loff + Sq > chunk_local:
        assert loff + Sq <= 2 * chunk_local, "more than one straddle"
        sq_tile, jump = (chunk_local - loff) // TW, (chunk_global - chunk_local) // TW
    return logical_start // TW, sq_tile, jump


def invp(n, chunk_local, sp, T):
    """block_cyclic_remap.hpp logical_to_chunked_physical (shard-major gathered buffer of T rows)."""
    shard_gap, slab_gap = T // sp - chunk_local, chunk_local * (sp - 1)
    blk = n // chunk_local
    slab, shard = divmod(blk, sp)
    return n + shard * shard_gap - slab * slab_gap


CASES = [(1024, 1024), (2048, 2048), (2048, 1024), (5120, 5120), (5120, 1024)]


def starts_for(C, S):
    T = 4 * 5120
    aligned = [k * C for k in range(3)]
    rotated = [k * S + o for k in range(3) for o in (0, 32, 96, S // 2, S - 32)]
    return T, sorted({x for x in aligned + rotated if x + C <= T})


@pytest.mark.parametrize("C,S", CASES)
def test_positions_tile_chunk(C, S):
    R, s = C // SP, S // SP
    T, starts = starts_for(C, S)
    for start in starts:
        pos = rotated_chip_positions(start, SP, R, s)
        flat = sorted(p for row in pos for p in row)
        assert flat == list(range(start, start + C)), (C, S, start)
        for c in range(SP):  # positions grow with the row (real rows are a per-chip prefix)
            assert all(a < b for a, b in zip(pos[c], pos[c][1:]))
            # SP-level ownership: chip c's rows are its own cache rows from lco(start) on
            uc = lco(start, c, SP, s)
            assert pos[c] == [cache_pos(c, uc + r, SP, s) for r in range(R)]
    if S == C:  # default knob == today's function
        for start in starts:
            assert rotated_chip_positions(start, SP, R, s) == rotated_chip_positions(start, SP, R)


@pytest.mark.parametrize("C,S", CASES)
def test_kv_write_tp_dedup(C, S):
    """update_padded_kv_cache with tp_axis: linear chip L writes R/tp rows from its SP chip's TP-replicated window."""
    R, s, b, cl = C // SP, S // SP, S // F, C // F
    T, starts = starts_for(C, S)
    for start in starts:
        pos = rotated_chip_positions(start, SP, R, s)
        for L in range(F):
            c = L // TP
            src = reader_new(start, L, cl, b)
            u1 = lco(start, L, F, b)
            assert all(0 <= r < R for r in src)
            assert [pos[c][r] for r in src] == [cache_pos(L, u1 + j, F, b) for j in range(cl)], (C, S, start, L)
            if S == C:  # bit-identical to today's reader
                assert src == reader_old(start, L, cl, R), (C, start, L)
        # every (L, row) of the chunk is written exactly once and the cache image is the slab layout
        img = {}
        for L in range(F):
            u1 = lco(start, L, F, b)
            for j in range(cl):
                img[(L, u1 + j)] = cache_pos(L, u1 + j, F, b)
        assert sorted(img.values()) == list(range(start, start + C))
    # whole-cache map (test harness preload / KV PCC): blockcyclic_positions(stripes=32, S, T)
    bc = blockcyclic_positions(F, S, T)
    Tl = T // F
    assert bc.tolist() == [cache_pos(L, lr, F, b) for L in range(F) for lr in range(Tl)]


@pytest.mark.parametrize("C,S", CASES)
def test_rope_table(C, S):
    """Indexed RoPE reads cos/sin row update_idxt + r of the block_cyclic_reorder(s) table for input row r."""
    R, s = C // SP, S // SP
    T, starts = starts_for(C, S)
    table = block_cyclic_reorder(torch.arange(T).view(1, 1, T, 1), s, SP, seq_dim=2).view(SP, T // SP)
    for start in starts:
        pos = rotated_chip_positions(start, SP, R, s)
        for c in range(SP):
            u_new = lco(start, c, SP, s)
            assert table[c, u_new : u_new + R].tolist() == pos[c]
            if start % C == 0:  # today's kernel (block = input rows) lands on the same rows at chunk-aligned starts
                assert lco(start, c, SP, R) == u_new


@pytest.mark.parametrize("C,S", CASES)
def test_indexer_causal_geometry(C, S):
    """ring_indexer_score_dsa full-mesh: chip (c, t) scores Sq = R/tp query rows (SP rows t*Sq..), geometry from the
    SP-level block (bc stripe b x mesh_cols split = s); one straddle must describe every q row."""
    R, s, b = C // SP, S // SP, S // F
    Sq = R // TP
    T, starts = starts_for(C, S)
    for start in starts:
        pos = rotated_chip_positions(start, SP, R, s)
        for c in range(SP):
            for t in range(TP):
                cst, sq_tile, jump = causal_geometry_tiles(start, F // TP, b * TP, c, t, Sq)
                for j in range(0, Sq, TW):
                    qt = j // TW
                    diag = cst + qt + (jump if (jump and qt >= sq_tile) else 0)
                    assert diag * TW == pos[c][t * Sq + j], (C, S, start, c, t, j)
                    # rows of a q tile are consecutive positions
                    assert pos[c][t * Sq + j : t * Sq + j + TW] == list(range(diag * TW, diag * TW + TW))
        # derived kv_len (trace path) must be the chunk end: start + Sq * F, NOT start + b * F (the slab)
        assert start + Sq * F == start + C


@pytest.mark.parametrize("C,S", CASES)
def test_invp_push_gather(C, S):
    """sparse_sdpa / indexer invP with stripe b over F shards; sparse push owner map; gather extent."""
    b = S // F
    T = 4 * 5120
    Tl = T // F
    for n in range(0, T, 7):
        phys = invp(n, b, F, T)
        L, lr = divmod(phys, Tl)
        assert cache_pos(L, lr, F, b) == n
        blk = n // b  # sparse_push_plan: natural row n -> block n / stripe, owner block % F
        owner, local = blk % F, (blk // F) * b + n % b
        assert (owner, local) == (L, lr)
    _, starts = starts_for(C, S)
    for start in starts:  # KVPE gather extent with gathered_slab_global = C (python keeps the chunk)
        gathered = min(T, -(-(start + C) // C) * C)
        for L in range(F):
            assert lco(start + C, L, F, b) <= gathered // F <= Tl
