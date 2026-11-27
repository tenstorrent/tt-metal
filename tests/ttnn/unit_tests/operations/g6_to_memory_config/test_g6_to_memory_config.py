# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_to_memory_config.

Bfloat16 production cases of eval/golden_tests/g6_to_memory_config/feature_spec.py (shape and both
placements copied verbatim, case ids in the test ids), the regime pins op_design.md requires, and the
validation errors the spec names. Every case is graded by exact equality and by the placement of the
result; the op does no arithmetic, so no tolerance applies.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.g6_to_memory_config import g6_to_memory_config, select_plan

_BUFFER = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_SCHEME = {
    "interleaved": ttnn.TensorMemoryLayout.INTERLEAVED,
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "block_sharded": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}

DRAM = {"buffer": "dram", "scheme": "interleaved"}
L1 = {"buffer": "l1", "scheme": "interleaved"}


def _sharded(buffer, scheme, grid, shard_shape):
    return {"buffer": buffer, "scheme": scheme, "grid": grid, "shard_shape": shard_shape}


def torch_reference(x):
    return x.clone()


def _memory_config(mem, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    if mem["scheme"] == "interleaved":
        return ttnn.MemoryConfig(_SCHEME["interleaved"], _BUFFER[mem["buffer"]])
    grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(sx, sy), ttnn.CoreCoord(ex, ey)) for sx, sy, ex, ey in mem["grid"]]
    )
    shard_spec = ttnn.ShardSpec(grid, tuple(mem["shard_shape"]), orientation)
    return ttnn.MemoryConfig(_SCHEME[mem["scheme"]], _BUFFER[mem["buffer"]], shard_spec)


def _skip_if_grid_outside_device(device, mem):
    if "grid" not in mem:
        return
    size = device.dram_grid_size() if mem["buffer"] == "dram" else device.compute_with_storage_grid_size()
    for _, _, ex, ey in mem["grid"]:
        if ex > size.x - 1 or ey > size.y - 1:
            pytest.skip(f"shard grid end ({ex},{ey}) is outside this device's {mem['buffer']} grid")


def _tile_grid(shape):
    # Each image is tile-padded on its own, so leading dims multiply ceil(H/32), not H.
    rows = math.prod(shape[:-2]) * math.ceil(shape[-2] / 32)
    cols = math.ceil(shape[-1] / 32)
    return rows, cols


def _holders(mem, shape, num_l1_banks, l1_bank_owners):
    rows, cols = _tile_grid(shape)
    if mem["buffer"] == "dram":
        return 0
    if mem["scheme"] == "interleaved":
        return min(rows * cols, num_l1_banks) if l1_bank_owners else 0
    shard_rows = mem["shard_shape"][0] // 32
    shard_cols = mem["shard_shape"][1] // 32
    return math.ceil(rows / shard_rows) * math.ceil(cols / shard_cols)


def _expected_regime(holders_in, holders_out):
    if holders_in == 0 and holders_out == 0:
        return "bounce"
    return "pull" if holders_out >= holders_in else "push"


def _cores(shard_spec):
    cores = []
    for core_range in shard_spec.grid.ranges():
        for y in range(core_range.start.y, core_range.end.y + 1):
            for x in range(core_range.start.x, core_range.end.x + 1):
                cores.append((x, y))
    return cores


def _assert_placement(output, requested):
    got = output.memory_config()
    assert got.buffer_type == requested.buffer_type
    assert got.memory_layout == requested.memory_layout
    if requested.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        assert got.shard_spec is not None
        assert tuple(got.shard_spec.shape) == tuple(requested.shard_spec.shape)
        assert got.shard_spec.orientation == requested.shard_spec.orientation
        assert _cores(got.shard_spec) == _cores(requested.shard_spec)


def _assert_regime(device, tt_input, out_config, shape, in_mem, out_mem):
    plan = select_plan(tt_input, out_config)
    grid = device.compute_with_storage_grid_size()
    num_l1_banks = ttnn._ttnn.reports.get_device_info(device).l1_num_banks
    assert plan.l1_bank_owners == (num_l1_banks == grid.x * grid.y)
    holders_in = _holders(in_mem, shape, num_l1_banks, plan.l1_bank_owners)
    holders_out = _holders(out_mem, shape, num_l1_banks, plan.l1_bank_owners)
    assert (plan.holders_in, plan.holders_out) == (holders_in, holders_out)
    assert plan.regime == _expected_regime(holders_in, holders_out)
    return plan


def _make_input(device, shape, in_mem):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_memory_config(in_mem),
    )
    return torch_input, tt_input


def _run_and_check(device, shape, in_mem, out_mem):
    _skip_if_grid_outside_device(device, in_mem)
    _skip_if_grid_outside_device(device, out_mem)
    torch_input, tt_input = _make_input(device, shape, in_mem)
    out_config = _memory_config(out_mem)

    plan = _assert_regime(device, tt_input, out_config, shape, in_mem, out_mem)
    tt_output = g6_to_memory_config(tt_input, out_config)

    assert list(tt_output.shape) == list(shape)
    assert tt_output.dtype == ttnn.bfloat16
    assert tt_output.layout == ttnn.TILE_LAYOUT
    _assert_placement(tt_output, out_config)
    readback = ttnn.to_torch(tt_output)
    expected = torch_reference(torch_input)
    assert readback.shape == expected.shape
    assert torch.equal(readback, expected), f"{(readback != expected).sum().item()} of {expected.numel()} values differ"
    return plan, tt_input, tt_output


PRODUCTION_CASES = [
    # (test id, shape, input placement, target placement)
    (
        "flux313_dram_to_l1_height_1core_single_tile_row",
        [1, 1, 16, 128],
        DRAM,
        _sharded("l1", "height_sharded", [[0, 0, 0, 0]], [32, 128]),
    ),
    (
        "qwen2154_dram_to_l1_height_1core_off_origin",
        [1, 1, 1, 256],
        DRAM,
        _sharded("l1", "height_sharded", [[1, 0, 1, 0]], [32, 256]),
    ),
    (
        "qwen2058_dram_to_l1_height_16_single_core_ranges",
        [1, 16, 1, 256],
        DRAM,
        _sharded(
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
    ),
    (
        "qwen2121_dram_to_l1_height_8_single_core_ranges",
        [1, 8, 1, 256],
        DRAM,
        _sharded(
            "l1",
            "height_sharded",
            [
                [0, 1, 0, 1],
                [1, 1, 1, 1],
                [2, 1, 2, 1],
                [3, 1, 3, 1],
                [4, 1, 4, 1],
                [8, 0, 8, 0],
                [9, 0, 9, 0],
                [10, 0, 10, 0],
            ],
            [32, 256],
        ),
    ),
    (
        "qwen2061_l1_height_two_ranges_to_dram",
        [1, 16, 1, 256],
        _sharded("l1", "height_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 256]),
        DRAM,
    ),
    (
        "qwen2062_l1_height_two_ranges_to_dram_h6",
        [1, 16, 6, 256],
        _sharded("l1", "height_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 256]),
        DRAM,
    ),
    (
        "qwen2063_l1_width_10c_to_l1_interleaved",
        [1, 1, 16, 3584],
        _sharded("l1", "width_sharded", [[0, 0, 9, 0]], [32, 384]),
        L1,
    ),
    (
        "qwen2064_l1_interleaved_to_l1_width_8c",
        [1, 1, 16, 4352],
        L1,
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 544]),
    ),
    (
        "qwen2065_l1_width_34_shards_on_36_cores_to_l1_interleaved",
        [1, 1, 16, 8704],
        _sharded("l1", "width_sharded", [[0, 0, 10, 2], [0, 3, 2, 3]], [32, 256]),
        L1,
    ),
    (
        "qwen2067_l1_width_8c_to_l1_width_40c",
        [1, 1, 16, 5120],
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 640]),
        _sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
    ),
    (
        "qwen2070_l1_width_40c_to_l1_width_10c",
        [1, 1, 16, 5120],
        _sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
        _sharded("l1", "width_sharded", [[0, 0, 9, 0]], [32, 512]),
    ),
    (
        "qwen2068_dram_to_l1_width_8c",
        [1, 1, 16, 1536],
        DRAM,
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 192]),
    ),
    (
        "qwen2071_l1_width_uneven_last_shard_to_dram",
        [1, 1, 16, 12928],
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 1664]),
        DRAM,
    ),
    (
        "qwen2072_l1_width_8c_512_pages_to_dram",
        [1, 1, 16, 16384],
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 2048]),
        DRAM,
    ),
    ("qwen2078_rank3_multi_batch_dram_to_l1_interleaved", [16, 128, 5120], DRAM, L1),
    ("qwen2105_rank3_h77_dram_to_l1_interleaved", [1, 77, 4352], DRAM, L1),
    (
        "qwen2155_l1_block_1c_to_l1_interleaved",
        [1, 1, 1, 256],
        _sharded("l1", "block_sharded", [[0, 0, 0, 0]], [32, 256]),
        L1,
    ),
    (
        "qwen2158_l1_height_1c_to_l1_block_1c_same_core",
        [1, 1, 6, 256],
        _sharded("l1", "height_sharded", [[0, 0, 0, 0]], [32, 256]),
        _sharded("l1", "block_sharded", [[0, 0, 0, 0]], [32, 256]),
    ),
    (
        "qwen2163_l1_width_8c_to_l1_width_40c_one_row",
        [1, 1, 1, 5120],
        _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 640]),
        _sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
    ),
    (
        "qwen2168_rank3_dram_to_l1_width_40c",
        [1, 1, 5120],
        DRAM,
        _sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128]),
    ),
    (
        "flux314_l1_width_two_ranges_to_dram",
        [1, 1, 32, 2048],
        _sharded("l1", "width_sharded", [[0, 0, 10, 0], [0, 1, 4, 1]], [32, 128]),
        DRAM,
    ),
]


@pytest.mark.parametrize(
    "shape,in_mem,out_mem",
    [pytest.param(shape, in_mem, out_mem, id=case_id) for case_id, shape, in_mem, out_mem in PRODUCTION_CASES],
)
def test_g6_to_memory_config(device, shape, in_mem, out_mem):
    _run_and_check(device, shape, in_mem, out_mem)


def test_bounce_regime_dram_to_dram(device):
    # No bfloat16 production case moves DRAM to DRAM, but validate() admits it; this pins the
    # bounce regime on the shape of production weight Qwen 2178.
    shape = [1536, 5120]
    plan, _, _ = _run_and_check(device, shape, DRAM, DRAM)
    assert plan.regime == "bounce"


def test_repeat_call_is_bit_identical_and_reuses_the_program(device):
    shape = [1, 1, 1, 5120]
    in_mem = _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 640])
    out_mem = _sharded("l1", "width_sharded", [[0, 0, 9, 3]], [32, 128])
    _skip_if_grid_outside_device(device, in_mem)
    _skip_if_grid_outside_device(device, out_mem)
    torch_input, tt_input = _make_input(device, shape, in_mem)
    out_config = _memory_config(out_mem)

    first = ttnn.to_torch(g6_to_memory_config(tt_input, out_config))
    entries_after_first = device.num_program_cache_entries()
    second = ttnn.to_torch(g6_to_memory_config(tt_input, out_config))

    assert device.num_program_cache_entries() == entries_after_first
    assert torch.equal(first, second)
    assert torch.equal(second, torch_reference(torch_input))


# The spec fixes the error types, not their text, so any message is accepted.
_REJECTED = (ValueError, RuntimeError)


def test_rejects_host_tensor(expect_error):
    torch_input = torch.randn([1, 1, 16, 128], dtype=torch.bfloat16)
    host_tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    with expect_error(_REJECTED, ""):
        g6_to_memory_config(host_tensor, ttnn.L1_MEMORY_CONFIG)


def test_rejects_row_major_input(device, expect_error):
    torch_input = torch.randn([1, 1, 16, 128], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(_REJECTED, ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)


def test_rejects_rank_5(device, expect_error):
    torch_input = torch.randn([1, 1, 1, 16, 128], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(_REJECTED, ""):
        g6_to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "grid,shard_shape,orientation",
    [
        pytest.param([[0, 0, 7, 0]], [32, 200], ttnn.ShardOrientation.ROW_MAJOR, id="shard_width_not_multiple_of_32"),
        pytest.param([[0, 0, 7, 0]], [16, 192], ttnn.ShardOrientation.ROW_MAJOR, id="shard_height_not_multiple_of_32"),
        pytest.param([[0, 0, 3, 0]], [32, 192], ttnn.ShardOrientation.ROW_MAJOR, id="fewer_cores_than_shards"),
        pytest.param([[0, 0, 7, 0]], [32, 192], ttnn.ShardOrientation.COL_MAJOR, id="col_major_target"),
    ],
)
def test_rejects_target_shard_spec(device, expect_error, grid, shard_shape, orientation):
    # Qwen 2068's shape and input placement (48 tile columns, DRAM interleaved), with a broken target.
    _, tt_input = _make_input(device, [1, 1, 16, 1536], DRAM)
    out_config = _memory_config(_sharded("l1", "width_sharded", grid, shard_shape), orientation)
    with expect_error(_REJECTED, ""):
        g6_to_memory_config(tt_input, out_config)


def test_rejects_col_major_input(device, expect_error):
    in_mem = _sharded("l1", "width_sharded", [[0, 0, 7, 0]], [32, 192])
    torch_input = torch.randn([1, 1, 16, 1536], dtype=torch.bfloat16)
    try:
        tt_input = ttnn.from_torch(
            torch_input,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_memory_config(in_mem, ttnn.ShardOrientation.COL_MAJOR),
        )
    except RuntimeError as exc:
        pytest.skip(f"ttnn cannot build a col-major width-sharded input here: {exc}")
    with expect_error(_REJECTED, ""):
        g6_to_memory_config(tt_input, ttnn.DRAM_MEMORY_CONFIG)
