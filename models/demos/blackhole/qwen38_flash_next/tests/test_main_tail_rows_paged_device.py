# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Paged KV mode of qsa_rows main_tail_rows on the held one-by-eight mesh: the unpaged call on a flat cache and the
paged call on a page pool with a shuffled table (zeros only as trailing padding) leave bitwise-equal caches (the pool
gathered back to logical order) and equal sparse queries, with the KV fast flag on and off."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_qsa_mt_row_device import bits, host, up
from models.demos.blackhole.qwen38_flash_next.tests.test_fused_qsa_paged_device import DEVICE_PARAMS, MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.tests.test_qsa_rows_kv_fast_device import ENV, inputs
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_rows

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
USED_BLOCKS, MAX_BLOCKS = 3, 8


def case_list(page_size):
    """(position, live): P % 32 in {0, 27, 31}, the next block in the same page, in the next page, and (the last
    case) the rows crossing out of page 0 into page 1 from P % 32 = 31."""

    return [(64, 32), (27, 8), (page_size - 5, 8), (page_size - 1, 2), (95, 3), (2 * page_size - 5, 8)]


@MESH
@PARAMS
@pytest.mark.parametrize("fast", ["0", "1"])
@pytest.mark.parametrize("page_size", [128, 2048])
def test_main_tail_rows_paged_matches_unpaged(mesh_device, monkeypatch, page_size, fast):
    mesh = mesh_device
    monkeypatch.setenv(ENV, fast)
    capacity = USED_BLOCKS * page_size
    for position, live in case_list(page_size):
        generator = torch.Generator().manual_seed(900 + page_size + position + live)
        common, rows_u32, _ = inputs(mesh, position, live, 900 + position + live)
        cache_host = torch.randn(1, 1, capacity, 512, generator=generator).to(torch.bfloat16)
        table = torch.zeros(32, MAX_BLOCKS, dtype=torch.int32)
        table[:, :USED_BLOCKS] = (1 + torch.randperm(USED_BLOCKS, generator=generator)).to(torch.int32)
        logical = torch.arange(capacity)
        physical = table[0].to(torch.int64)[logical // page_size] * page_size + logical % page_size
        pool_host = torch.zeros(1, 1, (1 + USED_BLOCKS) * page_size, 512, dtype=torch.bfloat16)
        pool_host[0, 0, physical] = cache_host[0, 0]

        flat = up(mesh, cache_host, ttnn.bfloat16)
        flat_query = host(qsa_rows.main_tail_rows(**common, kv_cache=flat, rows=rows_u32, single_row=False))
        pool = up(mesh, pool_host, ttnn.bfloat16)
        d_pt = up(mesh, table.reshape(1, 1, 32, MAX_BLOCKS), ttnn.int32)
        paged_query = host(
            qsa_rows.main_tail_rows(
                **common, kv_cache=pool, rows=rows_u32, single_row=False, page_table=d_pt, page_size=page_size
            )
        )
        flat_after = host(flat)
        gathered = host(pool)[:, :, physical]
        label = f"P={position} live={live} bs={page_size} fast={fast}"
        assert torch.equal(bits(gathered), bits(flat_after)), f"cache differs, {label}"
        assert not torch.equal(bits(flat_after), bits(cache_host)), f"cache unchanged, {label}"
        assert torch.equal(bits(paged_query), bits(flat_query)), f"query differs, {label}"
        # the pool rows outside the three logical blocks (block 0, the null block) are not live data; a padding write
        # may only touch block 0
        untouched = torch.ones(pool_host.shape[2], dtype=torch.bool)
        untouched[physical] = False
        untouched[:page_size] = False
        assert torch.equal(bits(host(pool)[0, 0, untouched]), bits(pool_host[0, 0, untouched])), label


@MESH
@PARAMS
def test_main_tail_rows_paged_validation(mesh_device, expect_error):
    mesh = mesh_device
    common, rows_u32, cache_h = inputs(mesh, 64, 4, 1)
    pool = up(mesh, torch.zeros(1, 1, 4 * 128, 512, dtype=torch.bfloat16), ttnn.bfloat16)
    table = up(mesh, torch.zeros(1, 1, 32, 8, dtype=torch.int32), ttnn.int32)
    narrow = up(mesh, torch.zeros(1, 1, 32, 4, dtype=torch.int32), ttnn.int32)
    for kwargs, message in (
        (dict(page_table=table, page_size=64), "page_size must be a power of two"),  # below the supported page sizes
        (dict(page_table=table, page_size=192), "page_size must be a power of two"),  # not a power of two
        (dict(page_table=narrow, page_size=128), "page_table must be int32"),  # max_blocks not a multiple of 8
    ):
        with expect_error(ValueError, message):
            qsa_rows.main_tail_rows(**common, kv_cache=pool, rows=rows_u32, **kwargs)
    odd_pool = up(mesh, torch.zeros(1, 1, 384, 512, dtype=torch.bfloat16), ttnn.bfloat16)
    with expect_error(ValueError, "must be a multiple of page_size"):
        qsa_rows.main_tail_rows(**common, kv_cache=odd_pool, rows=rows_u32, page_table=table, page_size=256)
