# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

L = ttnn.experimental.llama31_8b_v3
NKV, HD, BS, MAXB = 8, 128, 32, 1024


@pytest.fixture(autouse=True)
def p100_only(device):
    if device.dram_grid_size().x != 7:
        pytest.skip("Llama-3.1-8B P100 configs")


def heads_mem(rows, y):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, y), ttnn.CoreCoord(rows - 1, y))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (32, HD), ttnn.ShardOrientation.ROW_MAJOR),
    )


def make(device, rows, seed):
    g = torch.Generator().manual_seed(seed)
    dram = ttnn.DRAM_MEMORY_CONFIG
    put = lambda x, dt, lay, mem: ttnn.from_torch(x, dtype=dt, layout=lay, device=device, memory_config=mem)
    caches = [torch.randn(MAXB, NKV, BS, HD, generator=g) for _ in range(2)]
    heads = [torch.randn(1, rows, NKV, HD, generator=g) * 2 for _ in range(2)]
    pt = torch.randperm(MAXB, generator=g).to(torch.int32).reshape(1, -1).repeat(rows, 1)
    return {
        "caches": [[put(c, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, dram) for c in caches] for _ in range(2)],
        "k": put(heads[0], ttnn.bfloat16, ttnn.TILE_LAYOUT, heads_mem(rows, 1)),
        "v": put(heads[1], ttnn.bfloat16, ttnn.TILE_LAYOUT, heads_mem(rows, 0)),
        "pt": put(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, dram),
    }


def ints(device, values):
    return ttnn.from_torch(
        torch.tensor(values, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def bits(t):
    return ttnn.to_torch(t).float().view(torch.int32)


CASES = [
    (8, 40, ()),
    (8, 28, ()),
    (8, 60, ()),
    (8, 28, (5, 6, 7)),
    (8, 2040, (1, 2, 3, 4, 5, 6, 7)),
    (4, 30, ()),
    (2, 31, ()),
    (1, 40, ()),
    (1, 0, ()),
    (8, -1, ()),
    (8, 33, (0, 1, 2)),
    (8, 36, (0, 1, 2, 3, 4, 6, 7)),
    (8, 26, (0, 7)),
]


@pytest.mark.parametrize("rows, p0, holes", CASES)
def test_kv_rows_update_matches_per_slot_update(device, rows, p0, holes):
    x = make(device, rows, 23 + rows + max(p0, 0))
    slots = [-1 if p0 < 0 or j in holes else p0 + j for j in range(rows)]
    (ref_k, ref_v), (out_k, out_v) = x["caches"]
    idx = [ints(device, [p if i == j else -1 for i in range(rows)]) for j, p in enumerate(slots)]
    for t, p in zip(idx, slots):
        if p >= 0:
            ttnn.experimental.paged_fused_update_cache(
                ref_k, x["k"], ref_v, x["v"], update_idxs_tensor=t, page_table=x["pt"]
            )
    assert L.kv_rows_update_supported(out_k, x["k"], out_v, x["v"], idx, x["pt"])
    L.kv_rows_update(out_k, x["k"], out_v, x["v"], idx, x["pt"])
    entries = device.num_program_cache_entries()
    L.kv_rows_update(out_k, x["k"], out_v, x["v"], idx, x["pt"])
    assert device.num_program_cache_entries() == entries
    assert torch.equal(bits(out_k), bits(ref_k))
    assert torch.equal(bits(out_v), bits(ref_v))


def test_kv_rows_update_rejects_other_shapes(device, expect_error):
    x = make(device, 8, 5)
    (k_cache, v_cache), _ = x["caches"]
    idx = [ints(device, [j if i == j else -1 for i in range(8)]) for j in range(8)]
    assert not L.kv_rows_update_supported(k_cache, x["k"], v_cache, x["v"], idx[:3], x["pt"])
    short = [ints(device, [j if i == j else -1 for i in range(4)]) for j in range(8)]
    assert not L.kv_rows_update_supported(k_cache, x["k"], v_cache, x["v"], short, x["pt"])
    with expect_error(RuntimeError, "llama31_8b_v3.kv_rows_update"):
        L.kv_rows_update(k_cache, x["k"], v_cache, x["v"], short, x["pt"])
