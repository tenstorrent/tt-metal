# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""indexer_score_dsa page-table mode with the local per-lane length and row-write args: the paged op equals the
dense op, bitwise, on every written region, and both equal the dense op without the local args."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests.test_indexer_score_paged_device import (
    MESH,
    PARAMS,
    USER_ROWS,
    Case,
    host,
    up,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held one-by-eight mesh"
)
GEOMETRY = pytest.mark.parametrize("block_rows,width", [(32, 264), (512, 24)])
COUNT = pytest.mark.parametrize("count", [4, 3])
LENGTHS = [100, 5000, 8190, 0]


def _wide_case(mesh, block_rows, width, count, seed):
    users, S = 4, USER_ROWS
    case = Case(mesh, users, width, block_rows, seed=seed, user_rows=USER_ROWS, wide=True)
    k = up(mesh, case.flat_logical(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    extra = {"kv_len": count * S} if count < users else {}
    return case, k, extra, count * S - ttnn.TILE_SIZE


@MESH
@PARAMS
@GEOMETRY
@COUNT
def test_paged_wide_len_rowwrite(mesh_device, block_rows, width, count):
    S = USER_ROWS
    case, k, extra, start = _wide_case(mesh_device, block_rows, width, count, seed=61 + count)
    lengths = torch.tensor(LENGTHS[:count], dtype=torch.int32).reshape(1, 1, 1, count)
    len_t = up(mesh_device, lengths, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    kwargs = dict(valid_length_tensor=len_t, valid_length_lane_rows=S, consumed_rows=1, **extra)
    dense_len = host(case.score(k, None, False, chunk_start_idx=start, **kwargs))
    paged_len = host(case.score(None, case.pool, True, chunk_start_idx=start, **kwargs))
    dense_full = host(case.score(k, None, False, chunk_start_idx=start, **extra))
    for u in range(count):
        bound = min((LENGTHS[u] >> 5) + 1, S // 32)
        sl = (..., u, slice(u * S, u * S + bound * 32))
        assert torch.equal(paged_len[sl], dense_len[sl]), f"paged != dense_len, lane {u}"
        assert torch.equal(dense_len[sl], dense_full[sl]), f"dense_len != dense_full, lane {u}"


@MESH
@PARAMS
@GEOMETRY
@COUNT
def test_paged_wide_rowwrite_rows4(mesh_device, block_rows, width, count):
    S = USER_ROWS
    case, k, extra, start = _wide_case(mesh_device, block_rows, width, count, seed=71 + count)
    kwargs = dict(valid_length_lane_rows=S, consumed_rows=4, **extra)
    dense_rw = host(case.score(k, None, False, chunk_start_idx=start, **kwargs))
    paged_rw = host(case.score(None, case.pool, True, chunk_start_idx=start, **kwargs))
    dense_full = host(case.score(k, None, False, chunk_start_idx=start, **extra))
    for u in range(count):
        sl = (..., slice(u * 4, u * 4 + 4), slice(u * S, (u + 1) * S))
        assert torch.equal(paged_rw[sl], dense_rw[sl]), f"paged != dense_rw, lane {u}"
        assert torch.equal(dense_rw[sl], dense_full[sl]), f"dense_rw != dense_full, lane {u}"


@MESH
@PARAMS
@GEOMETRY
def test_paged_window_rowwrite(mesh_device, block_rows, width):
    case = Case(mesh_device, 1, width, block_rows, seed=41 + 7, user_rows=USER_ROWS)
    k = up(mesh_device, case.logical, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    start = USER_ROWS - ttnn.TILE_SIZE
    kwargs = dict(consumed_rows=1)
    dense_rw = host(case.score(k, None, False, chunk_start_idx=start, **kwargs))
    paged_rw = host(case.score(None, case.pool, True, chunk_start_idx=start, **kwargs))
    dense_full = host(case.score(k, None, False, chunk_start_idx=start))
    sl = (..., 0, slice(None))
    assert torch.equal(paged_rw[sl], dense_rw[sl]), "paged != dense_rw"
    assert torch.equal(dense_rw[sl], dense_full[sl]), "dense_rw != dense_full"
