# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged form of the QSA index tail on the held one-by-eight mesh: the program on the Index pool [NB, 1, bs / 4, 128] with a
page table writes the same rows as the unpaged program on a contiguous cache (the pool gathered back into logical order
is bitwise equal), and its other outputs (rotated query, ring) are bitwise equal."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_qsa_paged_device import MESH, PARAMS, host, up
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)

LOGICAL_TILES = 64  # logical index tile rows per lane (2048 compressed rows, tokens < 8192)
POSITIONS = {1: [2051], 4: [3, 2047, 4227, 8191]}  # tiles / blocks apart, P % 4 == 3 among them


def as_int16(tensor):
    return tensor.view(torch.int16)


def tile_rows(cache):
    """[N, 1, H, 128] -> [N * H / 32, 32, 128] tile rows."""

    return cache.reshape(-1, 32, 128)


def physical_tile(table, lane, tile, tpb):
    return int(table[lane][tile // tpb]) * tpb + tile % tpb


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [128, 2048])
@pytest.mark.parametrize("rows", [1, 4])
def test_index_tail_paged_matches_unpaged(mesh_device, rows, page_size):
    generator = torch.Generator().manual_seed(700 + rows + page_size)
    positions = torch.tensor(POSITIONS[rows])
    bf = lambda *shape: torch.randn(*shape, generator=generator).to(torch.bfloat16)
    tile = lambda t: up(mesh_device, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    hit = torch.zeros(1, rows, 32, 1, dtype=torch.bfloat16)
    for u in range(rows):
        hit[0, u, int(positions[u]) % 32, 0] = 1.0
    position = SimpleNamespace(
        kv_block_start=up(mesh_device, (positions & ~31).reshape(1, 1, 1, rows), ttnn.uint32), kv_row_hit=tile(hit)
    )
    inputs = dict(
        index_q_ws=tile(bf(1, 1, rows, 128)),
        raw_key_ws=tile(bf(1, 1, rows, 128)),
        position=position,
        index_q_norm=tile(bf(1, 1, 1, 128).abs() + 0.5),
        index_k_norm=tile(bf(1, 1, 1, 128).abs() + 0.5),
        cos=tile(torch.cos(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
        sin=tile(torch.sin(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
        block_cos=tile(torch.cos(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
        block_sin=tile(torch.sin(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
    )
    ring_host = bf(1, rows, 32, 128)
    cache_host = bf(rows, 1, 32 * LOGICAL_TILES, 128)

    tpb = page_size // 128
    blocks = -(-LOGICAL_TILES // tpb // 8) * 8
    table, num_blocks = ref.random_page_table(rows, blocks, generator)
    pool_host = bf(num_blocks, 1, page_size // 4, 128)
    pool_tiles = tile_rows(pool_host)
    for u in range(rows):
        for t in range(LOGICAL_TILES):
            pool_tiles[physical_tile(table, u, t, tpb)] = tile_rows(cache_host)[u * LOGICAL_TILES + t]

    ring = tile(ring_host)
    cache = tile(cache_host)
    plain_out = qsa_block.index_tail(**inputs, ring=ring, compressed_cache=cache)
    paged_ring = tile(ring_host)
    pool = tile(pool_host)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    paged_out = qsa_block.index_tail(
        **inputs, ring=paged_ring, compressed_cache=pool, page_table=d_pt, page_size=page_size
    )

    after = tile_rows(host(cache))
    gathered = torch.stack(
        [tile_rows(host(pool))[physical_tile(table, u, t, tpb)] for u in range(rows) for t in range(LOGICAL_TILES)]
    )
    assert torch.equal(as_int16(gathered), as_int16(after))
    assert not torch.equal(as_int16(after), as_int16(tile_rows(cache_host)))  # the rows were written
    # every pool tile outside the lanes' logical tiles is unchanged
    untouched = torch.ones(tile_rows(pool_host).shape[0], dtype=torch.bool)
    for u in range(rows):
        for t in range(LOGICAL_TILES):
            untouched[physical_tile(table, u, t, tpb)] = False
    assert torch.equal(as_int16(tile_rows(host(pool))[untouched]), as_int16(tile_rows(pool_host)[untouched]))
    assert torch.equal(as_int16(host(paged_out)), as_int16(host(plain_out)))
    assert torch.equal(as_int16(host(paged_ring)), as_int16(host(ring)))


@MESH
@PARAMS
def test_index_tail_paged_pool_check(mesh_device, expect_error):
    with expect_error(ValueError, "Index pool"):
        generator = torch.Generator().manual_seed(9)
        table, _ = ref.random_page_table(1, 8, generator)
        bf = lambda *shape: torch.randn(*shape, generator=generator).to(torch.bfloat16)
        tile = lambda t: up(mesh_device, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        position = SimpleNamespace(
            kv_block_start=up(mesh_device, torch.zeros(1, 1, 1, 1, dtype=torch.int64), ttnn.uint32),
            kv_row_hit=tile(torch.zeros(1, 1, 32, 1, dtype=torch.bfloat16)),
        )
        rope = lambda: tile(bf(1, 1, 1, 64))
        qsa_block.index_tail(
            tile(bf(1, 1, 1, 128)),
            tile(bf(1, 1, 1, 128)),
            position,
            tile(bf(1, 1, 1, 128)),
            tile(bf(1, 1, 1, 128)),
            rope(),
            rope(),
            rope(),
            rope(),
            tile(bf(1, 1, 32, 128)),
            tile(bf(4, 1, 64, 128)),  # 64 rows is not 512 / 4 for page_size 512
            page_table=up(mesh_device, table.reshape(1, 1, 32, 8), ttnn.int32),
            page_size=512,
        )
