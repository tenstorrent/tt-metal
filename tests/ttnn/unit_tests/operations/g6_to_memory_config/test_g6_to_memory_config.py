# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_memory_config.

The op moves whole tile pages from the input's placement to the one memory_config
names; nothing is computed. Graded like the golden suite (helpers.TOLERANCES is
("exact", None) for every dtype): the values read back equal the input's stored
values bit for bit, the result carries exactly the requested memory configuration,
and shape, dtype and TILE layout are the input's.

Cases are production cases of eval/golden_tests/g6_to_memory_config/feature_spec.py
(model and case id in each id), plus one placement each for the two regimes no
Phase 0 production case selects (bank_push, staged), on production shapes.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_memory_config import dump_core_roles, g6_to_memory_config, select_plan

L1_BUDGET_BYTES = 1347776

_BUFFER = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_SCHEME = {
    "interleaved": ttnn.TensorMemoryLayout.INTERLEAVED,
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "block_sharded": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}


def interleaved(buffer):
    return {"buffer": buffer, "scheme": "interleaved"}


def sharded(buffer, scheme, grid, shard_shape, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    return {"buffer": buffer, "scheme": scheme, "grid": grid, "shard_shape": shard_shape, "orientation": orientation}


def make_memory_config(memory):
    """Core ranges keep the order they are written in: the placement check compares cores in order."""
    if memory["scheme"] == "interleaved":
        return ttnn.MemoryConfig(_SCHEME["interleaved"], _BUFFER[memory["buffer"]])
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)) for sx, sy, ex, ey in memory["grid"]]
    )
    shard_spec = ttnn.ShardSpec(grid, tuple(memory["shard_shape"]), memory["orientation"])
    return ttnn.MemoryConfig(_SCHEME[memory["scheme"]], _BUFFER[memory["buffer"]], shard_spec)


def skip_if_grid_outside_device(device, memory):
    if "grid" not in memory:
        return
    size = device.dram_grid_size() if memory["buffer"] == "dram" else device.compute_with_storage_grid_size()
    for _, _, ex, ey in memory["grid"]:
        if ex > size.x - 1 or ey > size.y - 1:
            pytest.skip(f"shard grid end ({ex},{ey}) outside the device grid ({size.x},{size.y})")


def ordered_cores(shard_spec):
    cores = []
    for core_range in shard_spec.grid.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append((x, y))
    return cores


def assert_placement(output, requested):
    got = output.memory_config()
    assert got.buffer_type == requested.buffer_type, (got.buffer_type, requested.buffer_type)
    assert got.memory_layout == requested.memory_layout, (got.memory_layout, requested.memory_layout)
    if requested.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        assert got.shard_spec is not None, "output has no shard spec"
        assert tuple(got.shard_spec.shape) == tuple(requested.shard_spec.shape)
        assert got.shard_spec.orientation == requested.shard_spec.orientation
        assert ordered_cores(got.shard_spec) == ordered_cores(requested.shard_spec)


def assert_metadata(output, tt_input):
    assert list(output.shape) == list(tt_input.shape)
    assert output.dtype == tt_input.dtype
    assert output.layout == ttnn.TILE_LAYOUT


def pytorch_g6_to_memory_config(stored_values):
    """Reference: the identity on the logical values (torch.Tensor.clone)."""
    return stored_values.clone()


def make_input(device, shape, memory, seed=42):
    torch.manual_seed(seed)
    torch_input = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    return ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=make_memory_config(memory),
    )


def run_and_check(device, shape, in_memory, out_memory, seed=42):
    skip_if_grid_outside_device(device, in_memory)
    skip_if_grid_outside_device(device, out_memory)
    tt_input = make_input(device, shape, in_memory, seed)
    out_config = make_memory_config(out_memory)
    expected = pytorch_g6_to_memory_config(ttnn.to_torch(tt_input))

    tt_output = g6_to_memory_config(tt_input, out_config)

    assert_metadata(tt_output, tt_input)
    assert_placement(tt_output, out_config)
    readback = ttnn.to_torch(tt_output)
    assert list(readback.shape) == list(expected.shape)
    differ = (readback != expected).sum().item()
    assert torch.equal(readback, expected), f"{differ} of {expected.numel()} values differ"
    return tt_input, out_config, tt_output


# The plan a placement pair must select. Exact where the placements leave one owner
# side (forced); for L1 <-> L1 pairs any owner regime is a valid choice of the design.
OWNER_REGIMES = {"shard_pull", "shard_push", "bank_pull", "bank_push"}

# (id, shape, input placement, target placement, accepted regimes)
CASES = [
    # Pull into an L1-sharded target.
    (
        "qwen2058-dram-to-l1height-16-single-core-ranges",
        [1, 16, 1, 256],
        interleaved("dram"),
        sharded(
            "l1",
            "height_sharded",
            [
                [0, 2, 0, 2],
                [1, 2, 1, 2],
                [2, 2, 2, 2],
                [3, 2, 3, 2],
                [4, 2, 4, 2],
                [5, 1, 5, 1],
                [5, 2, 5, 2],
                [6, 1, 6, 1],
                [6, 2, 6, 2],
                [7, 1, 7, 1],
                [7, 2, 7, 2],
                [8, 1, 8, 1],
                [8, 2, 8, 2],
                [9, 1, 9, 1],
                [9, 2, 9, 2],
                [10, 1, 10, 1],
            ],
            [32, 256],
        ),
        {"shard_pull"},
    ),
    (
        "qwen2154-dram-to-l1height-one-core-at-1-0",
        [1, 1, 1, 256],
        interleaved("dram"),
        sharded("l1", "height_sharded", [[1, 0, 1, 0]], [32, 256]),
        {"shard_pull"},
    ),
    (
        "flux313-dram-to-l1height-4-pages",
        [1, 1, 16, 128],
        interleaved("dram"),
        sharded("l1", "height_sharded", [[0, 0, 0, 0]], [32, 128]),
        {"shard_pull"},
    ),
    (
        "qwen2074-dram-to-l1width-40-cores",
        [1, 1, 16, 5120],
        interleaved("dram"),
        sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
        {"shard_pull"},
    ),
    (
        "qwen2168-rank3-dram-to-l1width-40-cores",
        [1, 1, 5120],
        interleaved("dram"),
        sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
        {"shard_pull"},
    ),
    (
        "qwen2064-l1interleaved-to-l1width-8-cores",
        [1, 1, 16, 4352],
        interleaved("l1"),
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 544]),
        OWNER_REGIMES,
    ),
    (
        "qwen2163-l1width-8-to-40-cores",
        [1, 1, 1, 5120],
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 640]),
        sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
        OWNER_REGIMES,
    ),
    (
        "qwen2158-l1height-to-l1block-same-core",
        [1, 1, 6, 256],
        sharded("l1", "height_sharded", [[0, 0, 0, 0]], [32, 256]),
        sharded("l1", "block_sharded", [[0, 0, 0, 0]], [32, 256]),
        OWNER_REGIMES,
    ),
    # Push from an L1-sharded source.
    (
        "qwen2062-l1height-two-ranges-to-dram",
        [1, 16, 6, 256],
        sharded("l1", "height_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 256]),
        interleaved("dram"),
        {"shard_push"},
    ),
    (
        "qwen2071-l1width-uneven-last-shard-to-dram",
        [1, 1, 16, 12928],
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 1664]),
        interleaved("dram"),
        {"shard_push"},
    ),
    (
        "qwen2072-l1width-64-pages-per-core-to-dram",
        [1, 1, 16, 16384],
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 2048]),
        interleaved("dram"),
        {"shard_push"},
    ),
    (
        "flux314-l1width-two-ranges-to-dram-aligned",
        [1, 1, 32, 2048],
        sharded("l1", "width_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 128]),
        interleaved("dram"),
        {"shard_push"},
    ),
    (
        "qwen2065-l1width-34-shards-on-36-cores-to-l1interleaved",
        [1, 1, 16, 8704],
        sharded("l1", "width_sharded", [[0, 0, 10, 2], [0, 3, 2, 3]], [32, 256]),
        interleaved("l1"),
        OWNER_REGIMES,
    ),
    (
        "qwen2156-l1block-to-l1interleaved",
        [1, 1, 6, 256],
        sharded("l1", "block_sharded", [[0, 0, 0, 0]], [32, 256]),
        interleaved("l1"),
        OWNER_REGIMES,
    ),
    (
        "qwen2070-l1width-40-to-10-cores",
        [1, 1, 16, 5120],
        sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
        sharded("l1", "width_sharded", [[0, 0, 9, 0]], [32, 512]),
        OWNER_REGIMES,
    ),
    # DRAM interleaved -> L1 interleaved: owners are the L1 banks.
    ("qwen2089-dram-to-l1interleaved-96-pages", [1, 64, 1536], interleaved("dram"), interleaved("l1"), {"bank_pull"}),
    (
        "qwen2105-rank3-nonsquare-dram-to-l1interleaved",
        [1, 77, 4352],
        interleaved("dram"),
        interleaved("l1"),
        {"bank_pull"},
    ),
    (
        "qwen2078-multibatch-dram-to-l1interleaved-10240-pages",
        [16, 128, 5120],
        interleaved("dram"),
        interleaved("l1"),
        {"bank_pull"},
    ),
    # Regime pins: no Phase 0 production case selects these two regimes.
    ("pin-bank_push-l1interleaved-to-dram", [1, 1, 16, 4352], interleaved("l1"), interleaved("dram"), {"bank_push"}),
    ("pin-staged-dram-to-dram", [16, 128, 1536], interleaved("dram"), interleaved("dram"), {"staged"}),
]


@pytest.mark.parametrize(
    "shape,in_memory,out_memory,regimes",
    [pytest.param(shape, src, dst, regimes, id=case_id) for case_id, shape, src, dst, regimes in CASES],
)
def test_g6_to_memory_config(device, shape, in_memory, out_memory, regimes):
    tt_input, out_config, _ = run_and_check(device, shape, in_memory, out_memory)
    plan = select_plan(tt_input, out_config)
    assert plan.regime in regimes, f"selected {plan.regime}, expected one of {sorted(regimes)}"
    assert len(dump_core_roles(tt_input, out_config)) > 0


@pytest.mark.parametrize(
    "shape,in_memory,out_memory",
    [
        pytest.param(
            [1, 1, 16, 5120],
            interleaved("dram"),
            sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
            id="qwen2074-shard_pull",
        ),
        pytest.param(
            [1, 1, 16, 4608],
            sharded("l1", "width_sharded", [[0, 0, 9, 0]], [32, 480]),
            interleaved("dram"),
            id="qwen2069-shard_push",
        ),
        pytest.param([1, 77, 1536], interleaved("dram"), interleaved("l1"), id="qwen2107-bank_pull"),
    ],
)
def test_repeat_call_is_bit_identical_and_reuses_the_program(device, shape, in_memory, out_memory):
    tt_input, out_config, first = run_and_check(device, shape, in_memory, out_memory, seed=42)
    entries = device.num_program_cache_entries()
    second = g6_to_memory_config(tt_input, out_config)
    assert torch.equal(ttnn.to_torch(second), ttnn.to_torch(first))

    # Same configuration, other buffers and values: the cached program must serve it.
    run_and_check(device, shape, in_memory, out_memory, seed=7)
    assert device.num_program_cache_entries() == entries


def test_refuses_host_tensor(device, expect_error):
    host = ttnn.from_torch(torch.randn(1, 1, 32, 256).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(host, ttnn.L1_MEMORY_CONFIG)


def test_refuses_row_major_input(device, expect_error):
    tt_input = ttnn.from_torch(
        torch.randn(1, 1, 32, 256).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)


def test_refuses_rank_5(device, expect_error):
    tt_input = make_input(device, [1, 1, 1, 32, 256], interleaved("dram"))
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)


def test_refuses_shard_shape_not_multiple_of_32(device, expect_error):
    tt_input = make_input(device, [1, 1, 16, 4352], interleaved("dram"))
    target = make_memory_config(sharded("l1", "width_sharded", [[0, 0, 7, 0]], [16, 544]))
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, target)


def test_refuses_grid_with_fewer_cores_than_shards(device, expect_error):
    # 16 tile rows need 16 height shards of one tile row; the grid has 8 cores.
    tt_input = make_input(device, [1, 16, 1, 256], interleaved("dram"))
    target = make_memory_config(sharded("l1", "height_sharded", [[0, 0, 7, 0]], [32, 256]))
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, target)


def test_refuses_col_major_target_shard_spec(device, expect_error):
    tt_input = make_input(device, [1, 1, 16, 4352], interleaved("dram"))
    target = make_memory_config(
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 544], ttnn.ShardOrientation.COL_MAJOR)
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, target)


def test_refuses_col_major_input_shard_spec(device, expect_error):
    tt_input = make_input(
        device,
        [1, 1, 16, 4352],
        sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 544], ttnn.ShardOrientation.COL_MAJOR),
    )
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, ttnn.DRAM_MEMORY_CONFIG)


def test_refuses_target_beyond_l1_budget(device, expect_error):
    # One 4096 x 256 bfloat16 shard is 1024 pages = 2097152 B on one core, over the
    # L1 the model leaves to the op; outside the golden INPUTS.
    tt_input = make_input(device, [1, 1, 4096, 256], interleaved("dram"))
    target = make_memory_config(sharded("l1", "height_sharded", [[0, 0, 0, 0]], [4096, 256]))
    assert 1024 * 2048 > L1_BUDGET_BYTES
    with expect_error((ValueError, RuntimeError), ""):
        g6_to_memory_config(tt_input, target)
