# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free conformance for the Gemma 4 prefill KV chunk table.

Locks the prefill↔decode contract (config order, protobuf names, dual-family
geometry, SWA extent-not-fold) and the NdShard ROUND_ROBIN_1D closed form
against an independent dim-0→1→2 walk. No device / ttnn import.
"""

from __future__ import annotations

import pytest

from models.demos.gemma4.tt.runners.kv_chunk_table import (
    DEFAULT_GLOBAL_N_KV,
    DEFAULT_LOCAL_N_KV,
    TILE,
    Gemma4PrefillGeom,
    PrefillConfigSpec,
    block_cyclic_local_pos,
    chunk_noc_addr,
    chunk_size_bytes,
    config_names,
    config_specs,
    default_layer_types,
    global_layer_index,
    global_row_dim,
    layer_owns_config,
    locate,
    nkv_per_device,
    num_global_layers,
    shard_id,
    stable_config_name,
)

LOCAL_N_KV, GLOBAL_N_KV, TP = DEFAULT_LOCAL_N_KV, DEFAULT_GLOBAL_N_KV, 4
LOCAL_HEAD_DIM, GLOBAL_HEAD_DIM = 256, 512
N_SLOTS, SW, SEQ = 2, 1024, 2048
NUM_LAYERS = 12  # two full 5+1 periods
NUM_BANKS = 8
DTYPE = "bfloat8_b"
GLOBAL_ROW = global_row_dim(GLOBAL_HEAD_DIM, 0.25)
LAYER_TYPES = default_layer_types(NUM_LAYERS)
WINDOWED = {i for i, t in enumerate(LAYER_TYPES) if t == "sliding_attention"}
FULL = {i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"}


def _geom(**kw) -> Gemma4PrefillGeom:
    base = dict(
        num_layers=NUM_LAYERS,
        num_users=N_SLOTS,
        seq_len=SEQ,
        layer_types=LAYER_TYPES,
        mesh_shape=(1, TP),
        sp_axis=0,
        chunk_size=SEQ,
        sliding_window=SW,
        num_dram_banks=NUM_BANKS,
    )
    base.update(kw)
    return Gemma4PrefillGeom(**base)


def _spec_by_name(geom: Gemma4PrefillGeom, name: str) -> PrefillConfigSpec:
    return next(s for s in config_specs(geom) if s.label == name)


def _walk_ndshard(n_batch: int, n_heads: int, seq_local: int, tile: int = TILE):
    """Independent restatement of iterate_over_shards dim 0→1→2."""
    n_blocks = seq_local // tile
    sid = 0
    for b in range(n_batch):
        for h in range(n_heads):
            for blk in range(n_blocks):
                yield b, h, blk * tile, sid
                sid += 1


# ── contract ─────────────────────────────────────────────────────────────────


def test_config_order_matches_decode_contract():
    names = config_names(LOCAL_N_KV, GLOBAL_N_KV)
    assert names[:16] == tuple(f"k_h{h}" for h in range(16))
    assert names[16:32] == tuple(f"v_h{h}" for h in range(16))
    assert names[32:] == ("kv_h0", "kv_h1", "kv_h2", "kv_h3")
    geom = _geom()
    assert tuple(s.label for s in config_specs(geom)) == names
    assert config_specs(geom)[0].row_dim == LOCAL_HEAD_DIM
    assert config_specs(geom)[32].row_dim == GLOBAL_ROW
    assert config_specs(geom)[0].chunk_size_bytes == 8 * 1088
    assert config_specs(geom)[32].chunk_size_bytes == 20 * 1088
    assert config_specs(geom)[0].seq_extent == SW
    assert config_specs(geom)[32].seq_extent == SEQ


def test_protobuf_names_are_zero_padded():
    n = len(config_names())
    names = [stable_config_name(i, n) for i in range(n)]
    assert names[0] == "00"
    assert names[10] == "10"
    assert names[35] == "35"
    # Unpadded "0".."35" would sort "10" before "2". Padded order is numeric.
    assert names == sorted(names)


def test_chunk_size_bytes_and_global_row():
    assert chunk_size_bytes(DTYPE, 256) == 8 * 1088
    assert chunk_size_bytes(DTYPE, 640) == 20 * 1088
    assert GLOBAL_ROW == 640
    assert nkv_per_device(16, 4) == 4
    assert nkv_per_device(4, 4) == 1


def test_default_layer_types_is_31b_pattern():
    types = default_layer_types(60)
    assert len(types) == 60
    assert types[0] == "sliding_attention"
    assert types[5] == "full_attention"
    assert types[59] == "full_attention"
    assert num_global_layers(types) == 10
    assert [i for i, t in enumerate(types) if t == "full_attention"] == list(range(5, 60, 6))


# ── NdShard math ─────────────────────────────────────────────────────────────


def test_shard_id_h1_matches_sequential_walk():
    n_batch, n_heads, seq_local = 6, 1, 128
    for b, h, pos, sid in _walk_ndshard(n_batch, n_heads, seq_local):
        assert shard_id(batch_idx=b, local_head=h, local_pos=pos, n_heads=n_heads, seq_local=seq_local) == sid


def test_shard_id_is_head_major_within_a_batch_row():
    n_batch, n_heads, seq_local = 2, 4, 64
    seen = {}
    for b, h, pos, sid in _walk_ndshard(n_batch, n_heads, seq_local):
        seen[(b, h, pos)] = sid
        assert shard_id(batch_idx=b, local_head=h, local_pos=pos, n_heads=n_heads, seq_local=seq_local) == sid
    # Head 1's first block is immediately after head 0's last block of the same batch.
    n_blocks = seq_local // TILE
    assert seen[(0, 1, 0)] == seen[(0, 0, 0)] + n_blocks
    assert seen[(1, 0, 0)] == n_heads * n_blocks


def test_noc_addr_packing():
    noc, bank, off = chunk_noc_addr(shard=17, base_addr=0x1000, chunk_bytes=8704, num_banks=8)
    assert bank == 17 % 8
    assert off == (17 // 8) * 8704
    assert noc == (bank << 32) | (0x1000 + off)


def test_sp1_local_pos_is_identity():
    for p in (0, 32, 1023, 2047):
        local_pos, sp_row = block_cyclic_local_pos(p, chunk_size=SEQ, sp=1)
        assert (local_pos, sp_row) == (p, 0)


def test_block_cyclic_sp2_splits_a_period():
    chunk = 128
    # First half of the period is row 0; second half is row 1.
    assert block_cyclic_local_pos(0, chunk_size=chunk, sp=2) == (0, 0)
    assert block_cyclic_local_pos(64, chunk_size=chunk, sp=2) == (0, 1)
    assert block_cyclic_local_pos(96, chunk_size=chunk, sp=2) == (32, 1)
    assert block_cyclic_local_pos(128, chunk_size=chunk, sp=2) == (64, 0)


# ── layer ownership / SWA extent ─────────────────────────────────────────────


def test_layer_ownership_splits_families():
    geom = _geom()
    k0 = _spec_by_name(geom, "k_h0")
    kv0 = _spec_by_name(geom, "kv_h0")
    for layer in WINDOWED:
        assert layer_owns_config(geom, k0, layer)
        assert not layer_owns_config(geom, kv0, layer)
    for layer in FULL:
        assert layer_owns_config(geom, kv0, layer)
        assert not layer_owns_config(geom, k0, layer)


def test_locate_refuses_the_wrong_family():
    geom = _geom()
    k0 = _spec_by_name(geom, "k_h0")
    kv0 = _spec_by_name(geom, "kv_h0")
    sliding, full = min(WINDOWED), min(FULL)
    with pytest.raises(ValueError, match="no cache on global layer"):
        locate(geom, k0, position=0, slot=0, layer=full)
    with pytest.raises(ValueError, match="no cache on local layer"):
        locate(geom, kv0, position=0, slot=0, layer=sliding)


def test_window_is_an_extent_not_a_folded_address():
    geom = _geom()
    k0 = _spec_by_name(geom, "k_h0")
    layer = min(WINDOWED)
    a = locate(geom, k0, position=0, slot=0, layer=layer)
    b = locate(geom, k0, position=SW // 2, slot=0, layer=layer)
    assert a[0] != b[0]
    with pytest.raises(ValueError, match="outside"):
        locate(geom, k0, position=SW, slot=0, layer=layer)
    # Do not treat p and p+sw as the same bank page.
    # (SW is the exclusive end; p+sw is rejected rather than wrapped.)


def test_windowed_extent_covers_exactly_the_ring():
    geom = _geom()
    k0 = _spec_by_name(geom, "k_h0")
    layer = min(WINDOWED)
    addrs = [locate(geom, k0, position=p, slot=0, layer=layer)[0] for p in range(0, SW, TILE)]
    assert len(set(addrs)) == SW // TILE


def test_global_index_is_dense_among_full_layers():
    assert global_layer_index(LAYER_TYPES, 5) == 0
    assert global_layer_index(LAYER_TYPES, 11) == 1
    with pytest.raises(ValueError, match="sliding"):
        global_layer_index(LAYER_TYPES, 0)


# ── head → chip ──────────────────────────────────────────────────────────────


def test_local_heads_share_a_chip_global_heads_do_not():
    geom = _geom()
    sliding, full = min(WINDOWED), min(FULL)
    k0 = locate(geom, _spec_by_name(geom, "k_h0"), position=0, slot=0, layer=sliding)
    k3 = locate(geom, _spec_by_name(geom, "k_h3"), position=0, slot=0, layer=sliding)
    k4 = locate(geom, _spec_by_name(geom, "k_h4"), position=0, slot=0, layer=sliding)
    # Four local heads per chip: 0..3 on chip 0, 4..7 on chip 1. Same chip
    # shares device coord and differs in shard (hence bank / offset).
    assert k0[3] == k3[3] == (0, 0)
    assert k4[3] == (0, 1)
    assert k0[0] != k3[0]
    # Same local_head on the next chip repeats the per-bank address (replicated alloc).
    assert (k0[0] & 0xFFFFFFFF) == (k4[0] & 0xFFFFFFFF)

    g0 = locate(geom, _spec_by_name(geom, "kv_h0"), position=0, slot=0, layer=full)
    g2 = locate(geom, _spec_by_name(geom, "kv_h2"), position=0, slot=0, layer=full)
    assert g0[3] == (0, 0) and g2[3] == (0, 2)
    assert g0[0] == g2[0]


def test_global_batch_uses_dense_index_not_semantic_layer():
    """Two full-attention layers must not share a packed global slot."""
    geom = _geom()
    kv0 = _spec_by_name(geom, "kv_h0")
    full = sorted(FULL)
    a = locate(geom, kv0, position=0, slot=0, layer=full[0])
    b = locate(geom, kv0, position=0, slot=0, layer=full[1])
    c = locate(geom, kv0, position=0, slot=1, layer=full[0])
    n_blocks = SEQ // TILE
    # gi 0 / slot 0 → batch 0, shard 0; gi 1 → batch 1; slot 1 / gi 0 → batch n_global.
    assert a[0] != b[0] != c[0]
    want_b = chunk_noc_addr(shard=n_blocks, base_addr=0, chunk_bytes=kv0.chunk_size_bytes, num_banks=NUM_BANKS)
    want_c = chunk_noc_addr(
        shard=num_global_layers(LAYER_TYPES) * n_blocks,
        base_addr=0,
        chunk_bytes=kv0.chunk_size_bytes,
        num_banks=NUM_BANKS,
    )
    assert (b[0], b[1], b[2]) == want_b
    assert (c[0], c[1], c[2]) == want_c


def test_mesh_coord_follows_sp_axis():
    """(4, 1) + sp_axis=1 is also TP=4; coord is (chip, 0), not the (1, 4) view."""
    geom = _geom(mesh_shape=(TP, 1), sp_axis=1)
    sliding = min(WINDOWED)
    k0 = locate(geom, _spec_by_name(geom, "k_h0"), position=0, slot=0, layer=sliding)
    k4 = locate(geom, _spec_by_name(geom, "k_h4"), position=0, slot=0, layer=sliding)
    assert k0[3] == (0, 0)
    assert k4[3] == (1, 0)
