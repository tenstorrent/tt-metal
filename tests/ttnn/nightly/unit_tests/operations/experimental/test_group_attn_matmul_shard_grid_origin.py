# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device({"l1_small_size": 16384})


def memory(kind, start, size, shard, col=False):
    grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(*start), ttnn.CoreCoord(start[0] + size[0] - 1, start[1] + size[1] - 1))}
    )
    return ttnn.MemoryConfig(
        getattr(ttnn.TensorMemoryLayout, kind),
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.COL_MAJOR if col else ttnn.ShardOrientation.ROW_MAJOR),
    )


def put(device, x, mem=ttnn.DRAM_MEMORY_CONFIG, rm=False):
    return ttnn.from_torch(
        x,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT if rm else ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=mem,
    )


def compare(actual, expected):
    got = (actual if isinstance(actual, torch.Tensor) else ttnn.to_torch(actual)).float()
    expected = expected.float()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    wrong = got != expected
    return dict(wrong=int(wrong.sum()), elements=got.numel(), max_abs=float((got - expected).abs().max()))


def values(shape, op, start, high=100, salt=0):
    # Distinct placement data prevents stale L1 contents from hiding misplaced reads.
    seed = 100000 * start[0] + 10000 * start[1] + 100 * salt + sum((i + 1) * ord(c) for i, c in enumerate(op))
    return torch.randint(-high, high + 1, shape, generator=torch.Generator().manual_seed(seed)).bfloat16()


def group_case(device, op, start):
    grid = device.compute_with_storage_grid_size()
    if op == "group_split_ranges":
        # Q's grid is stored as two row ranges; the output names the same cores as one rectangle.
        a = values((1, 8, 32, 32), op, start, high=3)
        weights = values((32, 32), op, start, high=3, salt=1).abs() + 1
        rows = [
            ttnn.CoreRange(ttnn.CoreCoord(start[0], start[1] + r), ttnn.CoreCoord(start[0] + 3, start[1] + r))
            for r in range(2)
        ]
        spec = ttnn.ShardSpec(ttnn.CoreRangeSet(rows), (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
        y = ttnn.experimental.group_attn_matmul(
            put(device, a, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)),
            put(device, torch.diag_embed(weights).unsqueeze(1)),
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            memory_config=memory("HEIGHT_SHARDED", start, (4, 2), (32, 32)),
        )
        return [compare(y, a * weights)]
    if op == "group_kv2":
        # Two KV heads and K = N = 64, so each core's B shard spans several blocks.
        if start[0] + 8 > grid.x or start[1] + 4 > grid.y:
            pytest.skip("Offset attention shards exceed the device grid")
        a = values((1, 32, 32, 64), op, start, high=1)
        b = values((32, 2, 64, 64), op, start, high=1, salt=1)
        y = ttnn.experimental.group_attn_matmul(
            put(device, a, memory("HEIGHT_SHARDED", start, (8, 4), (32, 64))),
            put(device, b, memory("HEIGHT_SHARDED", start, (8, 4), (128, 64))),
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ref = torch.einsum("hum,uhmn->hun", a[0].float(), b.float()[:, torch.arange(32) // 16]).unsqueeze(0)
        return [compare(y, ref)]
    heads = 32 if op == "group" else 2
    a = values((1, heads, 32, 32), op, start, high=3)
    weights = values((32, 32), op, start, high=3, salt=1).abs() + 1
    b = torch.diag_embed(weights).unsqueeze(1)
    am = memory("HEIGHT_SHARDED", start, (8, 4) if heads == 32 else (2, 1), (32, 32))
    bm = memory("HEIGHT_SHARDED", (2, 1) if op == "group_separate" else start, (8, 4), (32, 32))
    a_mem = am if op in ["group", "group_a", "group_separate"] else ttnn.DRAM_MEMORY_CONFIG
    b_mem = bm if op in ["group", "group_b", "group_separate", "group_output"] else ttnn.DRAM_MEMORY_CONFIG
    out_mem = am if op == "group_output" else ttnn.DRAM_MEMORY_CONFIG
    rectangles = [((0, 0), (8, 4)), (start, (8, 4) if heads == 32 else (2, 1))]
    if op != "group_a":
        rectangles.append(((2, 1) if op == "group_separate" else start, (8, 4)))
    if any(origin[0] + size[0] > grid.x or origin[1] + size[1] > grid.y for origin, size in rectangles):
        pytest.skip("Offset attention shards exceed the device grid")
    y = ttnn.experimental.group_attn_matmul(
        put(device, a, a_mem),
        put(device, b, b_mem),
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        memory_config=out_mem,
    )
    return [compare(y, a * weights)]


def prealloc_case(device, op, start, col, expect_error):
    def vals(shape):
        return values(shape, op, start, high=48)

    if op == "group_prealloc":
        grid = device.compute_with_storage_grid_size()
        if start[0] + 8 > grid.x or start[1] + 4 > grid.y:
            pytest.skip("Offset attention shards exceed the device grid")
        x = vals((1, 32, 32, 32))
        w = (torch.arange(32 * 32).reshape(32, 32) % 3 + 1).bfloat16()
        b = torch.diag_embed(w).unsqueeze(1)
        o = put(device, torch.zeros_like(x), memory("HEIGHT_SHARDED", start, (8, 4), (32, 32), col))
        y = ttnn.experimental.group_attn_matmul(
            put(device, x),
            put(device, b),
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            optional_output_tensor=o,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return [compare(y, x * w)]
    if op == "group_rm_out":
        # The compute writes output tiles, which a ROW_MAJOR output would hold as rows.
        x = vals((1, 32, 32, 32))
        b = torch.diag_embed((torch.arange(32 * 32).reshape(32, 32) % 3 + 1).bfloat16()).unsqueeze(1)
        with expect_error(RuntimeError, "needs TILE layout"):
            ttnn.experimental.group_attn_matmul(
                put(device, x),
                put(device, b),
                compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
                optional_output_tensor=put(device, torch.zeros_like(x), rm=True),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return []


CASES = (
    [
        (op, origin, False)
        for op in ["group", "group_a", "group_b", "group_separate", "group_output", "group_kv2"]
        for origin in [(0, 0), (1, 0), (2, 1)]
    ]
    + [(op, origin, False) for op in ["group_split_ranges"] for origin in [(0, 0), (2, 1)]]
    + [(op, origin, False) for op in ["group_rm_out"] for origin in [(0, 0), (2, 3)]]
    + [(op, origin, col) for op in ["group_prealloc"] for origin in [(0, 0), (1, 0), (2, 1)] for col in [False, True]]
)


@pytest.mark.parametrize("op,origin,col", CASES)
def test_group_attn_matmul_shard_grid_origin(device, op, origin, col, expect_error):
    if op in ["group", "group_a", "group_b", "group_separate", "group_output", "group_kv2", "group_split_ranges"]:
        outputs = group_case(device, op, origin)
    else:
        outputs = prealloc_case(device, op, origin, col, expect_error)
    assert all(output["wrong"] == 0 for output in outputs), outputs
