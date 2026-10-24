# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged KV mode of the QSA fused programs on the held four-die mesh: selection_row (fused == composed == torch,
bitwise), main_tail (the 32-row blocks land at the page-table rows), kv_page_scatter (== torch == composed), and the
unpaged selection_row / main_tail unchanged."""

import os
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block
from models.demos.blackhole.qwen38_flash_next.ttnn.fused.kv_page_scatter import (
    kv_page_scatter,
    kv_page_scatter_composed,
)

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")
MESH = pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG


def up(mesh, tensor, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    if dtype == ttnn.uint32:
        tensor = tensor.to(torch.int64).to(torch.uint32)
    return ttnn.from_torch(
        tensor,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def host(tensor):
    replicas = [ttnn.to_torch(local) for local in ttnn.get_device_tensors(tensor)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0]


def u32(tensor):
    values = host(tensor).to(torch.int64) & 0xFFFFFFFF
    return values.reshape(-1, values.shape[-1])


def selection_inputs(generator, rows, capacity, lane_offsets):
    """ids, offsets, keep, fill, sentinel_pad as int64 host tensors; lane-local tokens < capacity."""

    ids = torch.randint(0, capacity // 4, (rows, 512), generator=generator)
    offsets = (torch.arange(2048) % 4).unsqueeze(0).repeat(rows, 1) + lane_offsets.unsqueeze(1)
    keep = torch.where(torch.rand(rows, 2080, generator=generator) < 0.8, 0xFFFFFFFF, 0)
    tail = torch.randint(0, capacity, (rows, 2080), generator=generator)
    tail[torch.rand(rows, 2080, generator=generator) < 0.5] = 0xFFFFFFFF
    fill = torch.where(keep == 0, tail, torch.zeros_like(tail))
    sentinel = torch.randint(0, capacity, (1, 32), generator=generator)
    sentinel[0, ::5] = 0xFFFFFFFF
    return ids, offsets, keep.to(torch.int64), fill, sentinel


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_selection_row_paged(mesh_device, rows, page_size):
    generator = torch.Generator().manual_seed(100 + rows + page_size)
    blocks = 16
    table, _ = ref.random_page_table(rows, blocks, generator)
    ids, _, keep, fill, sentinel = selection_inputs(
        generator, rows, blocks * page_size, torch.zeros(rows, dtype=torch.int64)
    )
    zero_offsets = (torch.arange(2048) % 4).unsqueeze(0)
    padded = lambda t, width: torch.cat([t, torch.zeros(32 - rows, width, dtype=torch.int64)])
    d_ids = up(mesh_device, ids.reshape(1, 1, rows, 512), ttnn.uint32)
    d_off = up(mesh_device, zero_offsets.reshape(1, 1, 1, 2048), ttnn.uint32)
    d_sent = up(mesh_device, sentinel.reshape(1, 1, 1, 32), ttnn.uint32)
    d_keep = up(mesh_device, padded(keep, 2080).reshape(1, 1, 32, 2080), ttnn.uint32)
    d_fill = up(mesh_device, padded(fill, 2080).reshape(1, 1, 32, 2080), ttnn.uint32)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    unpaged = ref.selection_row_reference(ids, zero_offsets, keep, fill, sentinel)
    expected = ref.selection_row_paged_reference(unpaged, table, page_size)
    fused = qsa_block.selection_row(d_ids, d_sent, d_off, d_keep, d_fill, d_pt, page_size)
    composed = qsa_block.selection_row_composed(d_ids, d_sent, d_off, d_keep, d_fill, d_pt, page_size)
    assert torch.equal(u32(fused), expected)
    assert torch.equal(u32(composed), expected)
    assert (expected == ref.MASKED).any() and (expected != ref.MASKED).any()
    # the unpaged program of the same inputs is unchanged (the same call without the page table)
    plain = qsa_block.selection_row(d_ids, d_sent, d_off, d_keep, d_fill)
    assert torch.equal(u32(plain), unpaged)


@MESH
@PARAMS
@pytest.mark.parametrize("rows", [1, 8])
def test_selection_row_unpaged_unchanged(mesh_device, rows):
    generator = torch.Generator().manual_seed(200 + rows)
    capacity = 4096
    lane_offsets = torch.arange(rows, dtype=torch.int64) * capacity
    ids, offsets, keep, fill, sentinel = selection_inputs(generator, rows, capacity, lane_offsets)
    padded = lambda t, width: torch.cat([t, torch.zeros(32 - rows, width, dtype=torch.int64)])
    d_ids = up(mesh_device, ids.reshape(1, 1, rows, 512), ttnn.uint32)
    d_off = up(mesh_device, padded(offsets, 2048).reshape(1, 1, 32, 2048), ttnn.uint32)
    d_sent = up(mesh_device, sentinel.reshape(1, 1, 1, 32), ttnn.uint32)
    d_keep = up(mesh_device, padded(keep, 2080).reshape(1, 1, 32, 2080), ttnn.uint32)
    d_fill = up(mesh_device, padded(fill, 2080).reshape(1, 1, 32, 2080), ttnn.uint32)
    fused = u32(qsa_block.selection_row(d_ids, d_sent, d_off, d_keep, d_fill))
    assert torch.equal(fused, ref.selection_row_reference(ids, offsets, keep, fill, sentinel))
    if rows == 1:
        one_row = lambda t: up(mesh_device, t.reshape(1, 1, 1, 2080), ttnn.uint32)  # the chain's one-row inputs
        composed = qsa_block.selection_row_composed(
            d_ids,
            d_sent,
            up(mesh_device, offsets.reshape(1, 1, 1, 2048), ttnn.uint32),
            one_row(keep),
            one_row(fill),
        )
        assert torch.equal(fused, u32(composed))


def main_tail_inputs(mesh, generator, rows, positions):
    heads, hd = qsa_block.LOCAL_HEADS, qsa_block.HEAD_DIM
    bf = lambda *shape: torch.randn(*shape, generator=generator).to(torch.bfloat16)
    tile = lambda t: up(mesh, t, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    hit = torch.zeros(1, rows, 32, 1, dtype=torch.bfloat16)
    for u in range(rows):
        hit[0, u, int(positions[u]) % 32, 0] = 1.0
    starts = (positions & ~31).reshape(1, 1, 1, rows)
    position = SimpleNamespace(kv_block_start=up(mesh, starts, ttnn.uint32), kv_row_hit=tile(hit))
    staging = tile(bf(1, rows, 32, 512))
    return dict(
        qg_ws=tile(bf(1, 1, rows, 2 * heads * hd)),
        k_ws=tile(bf(1, 1, rows, hd)),
        v_ws=tile(bf(1, 1, rows, hd)),
        position=position,
        q_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
        k_norm=tile(bf(1, 1, 1, hd).abs() + 0.5),
        cos=tile(torch.cos(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
        sin=tile(torch.sin(bf(1, 1, rows, 64).float()).to(torch.bfloat16)),
        staging=staging,
    )


def as_rows(tensor):
    return tensor.view(torch.int16)


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("rows", [1, 8, 32])
def test_main_tail_paged_writes_land_at_page_rows(mesh_device, rows, page_size):
    generator = torch.Generator().manual_seed(300 + rows + page_size)
    blocks = 16
    table, pages = ref.random_page_table(rows, blocks, generator)
    positions = torch.randint(0, blocks * page_size, (rows,), generator=generator)
    inputs = main_tail_inputs(mesh_device, generator, rows, positions)
    pool_host = torch.randn(1, 1, pages * page_size, 512, generator=generator).to(torch.bfloat16)
    pool = up(mesh_device, pool_host, ttnn.bfloat16)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    qsa_block.main_tail(**inputs, kv_cache=pool, page_table=d_pt, page_size=page_size)
    staged = host(inputs["staging"])[0]  # [rows, 32, 512]: the blocks the program wrote
    bases = ref.kv_block_base_reference(table, positions & ~31, page_size)
    expected = ref.pool_after_block_writes(pool_host[0, 0], staged, bases)
    assert torch.equal(as_rows(host(pool)[0, 0]), as_rows(expected))
    assert not torch.equal(as_rows(expected), as_rows(pool_host[0, 0]))


@MESH
@PARAMS
@pytest.mark.parametrize("rows", [1, 8])
def test_main_tail_unpaged_unchanged(mesh_device, rows):
    generator = torch.Generator().manual_seed(400 + rows)
    lane_rows = 512
    positions = torch.randint(0, lane_rows, (rows,), generator=generator)
    inputs = main_tail_inputs(mesh_device, generator, rows, positions)
    cache_host = torch.randn(1, 1, rows * lane_rows, 512, generator=generator).to(torch.bfloat16)
    cache = up(mesh_device, cache_host, ttnn.bfloat16)
    qsa_block.main_tail(**inputs, kv_cache=cache, lane_rows=lane_rows)
    staged = host(inputs["staging"])[0]
    bases = torch.arange(rows) * lane_rows + (positions & ~31)
    expected = ref.pool_after_block_writes(cache_host[0, 0], staged, bases)
    assert torch.equal(as_rows(host(cache)[0, 0]), as_rows(expected))


@MESH
@PARAMS
def test_main_tail_paged_pool_check(mesh_device, expect_error):
    with expect_error(ValueError, "multiple of page_size"):
        generator = torch.Generator().manual_seed(5)
        table, _ = ref.random_page_table(1, 8, generator)
        inputs = main_tail_inputs(mesh_device, generator, 1, torch.tensor([3]))
        pool = up(mesh_device, torch.zeros(1, 1, 96, 512, dtype=torch.bfloat16), ttnn.bfloat16)
        qsa_block.main_tail(
            **inputs, kv_cache=pool, page_table=up(mesh_device, table.reshape(1, 1, 32, 8), ttnn.int32), page_size=64
        )


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("rows", [1, 31, 32, 33, 1000, 4097])
def test_kv_page_scatter(mesh_device, rows, page_size):
    generator = torch.Generator().manual_seed(500 + rows + page_size)
    source_rows = 4128
    blocks = -(-source_rows // page_size // 8) * 8
    lane = 5
    table, pages = ref.random_page_table(8, blocks, generator)
    src_host = torch.randn(1, 1, source_rows, 512, generator=generator).to(torch.bfloat16)
    pool_host = torch.randn(1, 1, pages * page_size, 512, generator=generator).to(torch.bfloat16)
    src = up(mesh_device, src_host, ttnn.bfloat16)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    expected = ref.pool_after_scatter(pool_host[0, 0], src_host[0, 0], table[lane], rows, page_size)
    fused_pool = up(mesh_device, pool_host, ttnn.bfloat16)
    kv_page_scatter(src, fused_pool, d_pt, lane=lane, rows=rows, page_size=page_size)
    assert torch.equal(as_rows(host(fused_pool)[0, 0]), as_rows(expected))
    composed_pool = up(mesh_device, pool_host, ttnn.bfloat16)
    kv_page_scatter_composed(src, composed_pool, d_pt, lane=lane, rows=rows, page_size=page_size)
    assert torch.equal(as_rows(host(composed_pool)[0, 0]), as_rows(expected))


@MESH
@PARAMS
@pytest.mark.parametrize("page_size", [32, 64])
@pytest.mark.parametrize("lanes", [1, 8, 32])
def test_paged_lane_row_starts_composed(mesh_device, lanes, page_size):
    """The composed lane write's start rows (``_write_kv_slabs_lanes`` PAGED) == the page lookups of the block starts."""

    generator = torch.Generator().manual_seed(600 + lanes + page_size)
    blocks = 16
    table, _ = ref.random_page_table(lanes, blocks, generator)
    starts = torch.randint(0, blocks * page_size, (lanes,), generator=generator) & ~31
    d_starts = up(mesh_device, starts.reshape(1, 1, 1, lanes), ttnn.uint32)
    d_pt = up(mesh_device, table.reshape(1, 1, 32, blocks), ttnn.int32)
    per_lane = qsa_block.paged_lane_row_starts_composed(d_starts, d_pt, page_size, lanes)
    got = torch.cat([u32(t).reshape(1) for t in per_lane])
    assert torch.equal(got, ref.kv_block_base_reference(table, starts, page_size))
