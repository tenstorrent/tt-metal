# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

pytestmark = [
    pytest.mark.timeout(60, method="thread"),
    pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only"),
]

# Output modes: (memory_config kind, preallocate). "implicit" passes no memory_config.
OUTPUTS = [
    ("implicit", False),
    ("sharded", False),
    ("sharded", True),
    ("l1", False),
    ("l1", True),
    ("dram", False),
    ("dram", True),
]
INTERLEAVED_OUTPUTS = [output for output in OUTPUTS if output[0] != "sharded"]


def _cores(x, y):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})


def _sharded(layout, grid, shard):
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))


def _route(call):
    """The route a call takes, from a NO_DISPATCH graph capture: the device operations it would launch and the
    scratchpads of their programs, recorded without running them. No profiler is needed, so the check holds on
    every device. The composite path unshards before it repeats; native launches the repeat first, and a
    preallocated output may be copied into after it. Only the staged copy gives a native repeat on a sharded
    tensor two scratchpads."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
    try:
        call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    ops, repeat_scratchpads = [], 0
    for node in graph:
        if node["node_type"] == "function_start" and node["params"]["name"].endswith("DeviceOperation"):
            ops.append(node["params"]["name"].rsplit("::", 1)[-1])
        elif node["node_type"] == "scratchpad_allocate" and ops[-1:] == ["RepeatDeviceOperation"]:
            repeat_scratchpads += 1
    if "RepeatCodegenDeviceOperation" in ops:
        return "codegen"
    if ops[:1] == ["RepeatDeviceOperation"] and set(ops[1:]) <= {"CopyDeviceOperation"}:
        return "staged" if repeat_scratchpads == 2 else "native"
    if ops and set(ops) == {"CopyDeviceOperation"}:
        return "copy"
    if ops and set(ops) == {"ReshardDeviceOperation"}:
        return "reshard"
    return "view" if not ops else "composite"


def _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route=None, dtype=ttnn.uint32):
    """Check every output value, placement, a preallocated output, and the unchanged input."""
    indices = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    values = ((indices * 1000003 + 7) % (1 << 31)).to(torch.int32)
    if dtype == ttnn.bfloat16:
        values = (indices % 251 - 125).to(torch.bfloat16)
    source = ttnn.from_torch(
        values, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    assert torch.equal(ttnn.to_torch(source).to(values.dtype), values)
    expected = values.repeat(*repeats)
    kwargs = {} if output_memory is None else {"memory_config": output_memory}
    if preallocate:
        target = ttnn.from_torch(
            torch.full(expected.shape, -1, dtype=torch.int32),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
        kwargs["optional_output_tensor"] = target
    if route is not None:
        assert _route(lambda: ttnn.repeat(source, repeats, **kwargs)) == route
    result = ttnn.repeat(source, repeats, **kwargs)
    assert torch.equal(ttnn.to_torch(result).to(values.dtype), expected)
    if preallocate:
        assert result.buffer_address() == target.buffer_address()
        assert torch.equal(ttnn.to_torch(target).to(values.dtype), expected)
    assert torch.equal(ttnn.to_torch(source).to(values.dtype), values)
    if output_memory is not None:
        assert result.memory_config() == output_memory
    return result


def _assert_derived_placement(result, input_memory):
    """With no memory_config, codegen and the composite path place an unaligned or padded row-major shard on a spec
    derived for the output shape, whose page is L1-aligned, or in L1 interleaved when no such spec exists."""
    memory = result.memory_config()
    if memory.is_sharded():
        assert memory.memory_layout == input_memory.memory_layout
        assert memory.shard_spec.shape[1] * result.element_size() % ttnn.get_l1_alignment() == 0
    else:
        assert memory == ttnn.L1_MEMORY_CONFIG


def _output_memory(kind, sharded):
    return {"implicit": None, "sharded": sharded, "l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}[kind]


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK, ttnn.ShardStrategy.HEIGHT])
@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_narrow_rows(device, rank, strategy, output, preallocate):
    # A 3-word row in a 5-word shard is one padded 20-byte page. Native repeats this single axis in place.
    if strategy == ttnn.ShardStrategy.WIDTH:
        layout, grid, shape = ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(1, 1), [1, 4, 3]
    elif strategy == ttnn.ShardStrategy.HEIGHT:
        layout, grid, shape = ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(1, 1), [1, 4, 3]
    else:
        layout, grid, shape = ttnn.TensorMemoryLayout.BLOCK_SHARDED, _cores(1, 2), [2, 4, 3]
    shape = [1] * (rank - 3) + shape
    repeats = [1] * rank
    repeats[-2] = 2
    sharded = _sharded(layout, grid, [8, 5])
    result = _repeat(
        device, shape, _sharded(layout, grid, [4, 5]), repeats, _output_memory(output, sharded), preallocate, "native"
    )
    assert result.memory_config() == (sharded if output == "implicit" else _output_memory(output, sharded))


@pytest.mark.parametrize(
    "layout, grid, shard, shape, repeats",
    [
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [1, 1, 2, 1]),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [1, 1, 3, 1]),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [1, 4, 3], [1, 1, 2, 1]),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [1, 4, 3], [3, 1, 1, 1]),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [4, 3], [1, 2, 1]),
        (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1), [4, 5], [1, 4, 3], [1, 1, 2, 1]),
    ],
    ids=["block_new_axis", "block_rows_x3", "width_new_axis", "width_outer_x3", "width_rank2", "height_new_axis"],
)
@pytest.mark.parametrize("output, preallocate", INTERLEAVED_OUTPUTS)
def test_repeat_narrow_rows_rank_expansion(device, layout, grid, shard, shape, repeats, output, preallocate):
    # A repeat vector longer than the input rank adds leading axes before routing. That view keeps the padded row
    # too: re-deriving the shard from the logical width throws for WIDTH and HEIGHT, and for BLOCK reads 12-byte pages
    # out of 16-byte slots. Each repeated group fits in one shard's rows, so native takes every case.
    input_memory = _sharded(layout, _cores(*grid), shard)
    result = _repeat(device, shape, input_memory, repeats, _output_memory(output, None), preallocate, "native")
    if output == "implicit":
        memory = result.memory_config()
        assert memory.memory_layout == layout and memory.shard_spec.grid == input_memory.shard_spec.grid
        assert memory.shard_spec.shape[1] == shard[1]
    else:
        assert result.memory_config() == _output_memory(output, None)


@pytest.mark.parametrize(
    "layout, grid, shard, shape, repeats, output_shard, preallocate, route",
    [
        # 3 words in a 16-byte page: the 6-row group does not fit in 3-row shards, so native declines and
        # sharded_to_interleaved unshards the aligned page for codegen.
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [2, 1, 1, 1], None, False, "codegen"),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [1, 4, 3], [1, 1, 2, 1], [8, 5], False, "native"),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [1, 4, 3], [1, 1, 2, 1], [8, 5], True, "native"),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [1, 1, 2, 1], [6, 4], False, "native"),
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [1, 1, 2, 1], [6, 4], True, "native"),
        # 7 words over two 5-word shards: the partial final shard keeps its padding through the view.
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (2, 1), [4, 5], [1, 4, 7], [1, 1, 2, 1], None, False, "codegen"),
    ],
    ids=[
        "block_aligned_outer_x2",
        "width_sharded",
        "width_sharded_pre",
        "block_sharded",
        "block_sharded_pre",
        "partial",
    ],
)
def test_repeat_rank_expansion_routes(device, layout, grid, shard, shape, repeats, output_shard, preallocate, route):
    input_memory = _sharded(layout, _cores(*grid), shard)
    output_memory = None if output_shard is None else _sharded(layout, _cores(*grid), output_shard)
    result = _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route)
    if output_memory is None:
        _assert_derived_placement(result, input_memory)
    else:
        assert result.memory_config() == output_memory


_NARROW_WIDTH = (ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(1, 1), [4, 5], [1, 4, 3])
_NARROW_BLOCK = (ttnn.TensorMemoryLayout.BLOCK_SHARDED, _cores(1, 2), [4, 5])


@pytest.mark.parametrize(
    "source, repeats, output",
    [
        (_NARROW_WIDTH, [1, 1, 1, 2], None),
        (_NARROW_WIDTH, [1, 1, 2], None),
        (_NARROW_WIDTH, [2, 2, 1], None),
        (_NARROW_WIDTH, [1, 1, 1], ttnn.DRAM_MEMORY_CONFIG),
        (_NARROW_WIDTH, [1, 1, 1, 1], ttnn.DRAM_MEMORY_CONFIG),
        (_NARROW_WIDTH, [1, 2, 1], _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _cores(1, 1), [8, 5])),
        (
            _NARROW_WIDTH,
            [1, 2, 1],
            _sharded(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))}),
                [8, 5],
            ),
        ),
        (
            _NARROW_WIDTH,
            [1, 2, 1],
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(_cores(1, 1), [8, 5], ttnn.ShardOrientation.ROW_MAJOR),
            ),
        ),
        ((*_NARROW_BLOCK, [1, 6, 3]), [1, 2, 1], None),
        ((*_NARROW_BLOCK, [2, 4, 3]), [2, 1, 1], None),
        ((*_NARROW_BLOCK, [2, 4, 3]), [2, 1, 1, 1], None),
        # A 16-byte page is L1-aligned, but sharded_to_interleaved takes only L1 shards.
        (
            (ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(1, 1), [4, 4], [1, 4, 3], ttnn.BufferType.DRAM),
            [1, 2, 1],
            None,
        ),
    ],
    ids=[
        "new_axis_last_dim",
        "last_dim",
        "two_axes",
        "all_ones_dram",
        "all_ones_new_axis_dram",
        "block_output",
        "other_grid_output",
        "dram_sharded_output",
        "uneven_block",
        "block_split_rows",
        "block_split_rows_new_axis",
        "dram_input_aligned_page",
    ],
)
def test_repeat_unaligned_narrow_rows_off_native(device, source, repeats, output):
    # These calls may unshard through the row-major copy. Its reader clamps both initial block ends to the row.
    layout, grid, shard, shape, *buffer = source
    memory = ttnn.MemoryConfig(
        layout,
        buffer[0] if buffer else ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    route = "copy" if all(count == 1 for count in repeats) else "codegen"
    _repeat(device, shape, memory, repeats, output, False, route)


def _bank(x):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x, 0), ttnn.CoreCoord(x, 0))})


@pytest.mark.parametrize(
    "shard, buffer, repeats",
    [
        ([4, 4], ttnn.BufferType.DRAM, [1, 1, 1]),
        ([4, 5], ttnn.BufferType.DRAM, [1, 1, 1]),
        ([4, 4], ttnn.BufferType.L1, [1, 1, 1]),
        ([4, 4], ttnn.BufferType.DRAM, [1, 1, 1, 1]),
    ],
    ids=["dram_to_other_bank", "dram_unaligned_to_other_bank", "dram_to_l1", "dram_new_axis_to_other_bank"],
)
def test_repeat_narrow_rows_identity_into_shards(device, shard, buffer, repeats):
    # These placements preserve the shard width. DRAM-to-DRAM uses reshard; DRAM-to-L1 uses copy.
    # Different-width destinations are covered separately and rely on the row-major copy clamps.
    layout, row = ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.ShardOrientation.ROW_MAJOR
    input_memory = ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, ttnn.ShardSpec(_bank(0), shard, row))
    output_bank = 1 if buffer == ttnn.BufferType.DRAM else 0
    output_memory = ttnn.MemoryConfig(layout, buffer, ttnn.ShardSpec(_bank(output_bank), shard, row))
    result = _repeat(
        device,
        [1, 4, 3],
        input_memory,
        repeats,
        output_memory,
        False,
        "reshard" if buffer == ttnn.BufferType.DRAM else "copy",
    )
    assert result.memory_config() == output_memory


_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


@pytest.mark.parametrize(
    "shape, grid, shard, buffer, repeats, output, route",
    [
        ([1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.L1, [2, 2, 1], None, "codegen"),
        ([1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.DRAM, [1, 2, 1], None, "codegen"),
        ([1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.L1, [1, 1, 1], ttnn.DRAM_MEMORY_CONFIG, "copy"),
        ([1, 1, 1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.L1, [1, 1, 2, 2, 1], None, "composite"),
        ([1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.L1, [1, 2, 1], _sharded(_HEIGHT, _cores(2, 1), [4, 5]), "codegen"),
        ([1, 4, 3], (1, 1), [4, 5], ttnn.BufferType.L1, [1, 1, 2], None, "codegen"),
        ([12, 2], (2, 1), [6, 4], ttnn.BufferType.L1, [1, 2], None, "codegen"),
    ],
    ids=[
        "two_axes",
        "dram_input",
        "all_ones_dram",
        "rank5_two_axes",
        "other_grid_output",
        "last_dim",
        "last_dim_2_cores",
    ],
)
def test_repeat_height_narrow_rows_off_native(device, shape, grid, shard, buffer, repeats, output, route):
    # The row-major unshard reads a HEIGHT page as the logical row, so a narrow HEIGHT row leaves its shards wherever
    # native declines it, as on main. A last-dimension repeat goes there too: native's last-dimension view drops the
    # row padding, and on main such a call threw or wrote misaligned NoC transfers.
    input_memory = ttnn.MemoryConfig(
        _HEIGHT, buffer, ttnn.ShardSpec(_cores(*grid), shard, ttnn.ShardOrientation.ROW_MAJOR)
    )
    result = _repeat(device, shape, input_memory, repeats, output, False, route)
    if output is not None:
        assert result.memory_config() == output


@pytest.mark.parametrize("rank", [3, 5])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_single_page_unaligned_rows(device, rank, strategy, output, preallocate):
    # A 5-word row on one 20-byte page (a full row per core) needs no staging. It takes main's route: codegen copies it
    # in place below rank five and native at rank five, where codegen does not apply. The output stays sharded on the
    # input grid, as on main.
    if strategy == ttnn.ShardStrategy.WIDTH:
        layout, shape, shard = ttnn.TensorMemoryLayout.WIDTH_SHARDED, [1, 7, 5], [7, 5]
    else:
        layout, shape, shard = ttnn.TensorMemoryLayout.BLOCK_SHARDED, [1, 4, 5], [4, 5]
    shape = [1] * (rank - 3) + shape
    repeats = [1] * rank
    repeats[-2] = 2
    sharded = _sharded(layout, _cores(1, 1), [2 * shard[0], shard[1]])
    output_memory = _output_memory(output, sharded)
    result = _repeat(
        device,
        shape,
        _sharded(layout, _cores(1, 1), shard),
        repeats,
        output_memory,
        preallocate,
        "native" if rank == 5 else "codegen",
    )
    assert result.memory_config() == (sharded if output == "implicit" else output_memory)


@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_narrow_row_wider_page(device, output, preallocate):
    # An 8-word row in a 9-word shard: the 36-byte page holds the 32-byte row and one padding word.
    layout, grid = ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(1, 1)
    sharded = _sharded(layout, grid, [4, 9])
    result = _repeat(
        device,
        [1, 2, 8],
        _sharded(layout, grid, [2, 9]),
        [1, 2, 1],
        _output_memory(output, sharded),
        preallocate,
        "native",
    )
    assert result.memory_config() == (sharded if output == "implicit" else _output_memory(output, sharded))


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("width, shard_width", [(7, 5), (30, 8)])
@pytest.mark.parametrize("output, preallocate", INTERLEAVED_OUTPUTS)
def test_repeat_partial_final_shard(device, rank, strategy, width, shard_width, output, preallocate):
    # A row split over shards whose last shard is partial (7 words over 5-word shards; 30 over 8-word, 32-byte,
    # aligned shards) stays on codegen or the composite path: repeat's view changes each of these shapes.
    cores = -(-width // shard_width)
    layout = (
        ttnn.TensorMemoryLayout.WIDTH_SHARDED
        if strategy == ttnn.ShardStrategy.WIDTH
        else ttnn.TensorMemoryLayout.BLOCK_SHARDED
    )
    input_memory = _sharded(layout, _cores(cores, 1), [4, shard_width])
    shape = [1] * (rank - 2) + [4, width]
    repeats = [1] * rank
    repeats[-2] = 2
    output_memory = _output_memory(output, None)
    route = "composite" if rank == 5 else "codegen"
    result = _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route)
    if output_memory is None:
        _assert_derived_placement(result, input_memory)
    else:
        assert result.memory_config() == output_memory


@pytest.mark.parametrize("rank", [3, 4, 5])
def test_repeat_partial_final_shard_into_unaligned_sharded_output(device, expect_error, rank):
    # 7 words over two 5-word shards, into 20-byte output pages. Ranks 3 and 4 take codegen, which writes the unaligned
    # output exactly. At rank 5 codegen does not apply, and the composite path's interleaved_to_sharded rejects the
    # page: the call throws, checked in a NO_DISPATCH capture.
    shape = [1] * (rank - 2) + [4, 7]
    repeats = [1] * rank
    repeats[-2] = 2
    input_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(2, 1), [4, 5])
    output_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(2, 1), [8, 5])
    if rank == 5:
        values = torch.arange(prod(shape), dtype=torch.int32).reshape(shape)
        tensor = ttnn.from_torch(
            values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
        )
        with expect_error(RuntimeError, "L1 aligned page size"):
            _route(lambda: ttnn.repeat(tensor, repeats, memory_config=output_memory))
        return
    result = _repeat(device, shape, input_memory, repeats, output_memory, False, "codegen")
    assert result.memory_config() == output_memory


@pytest.mark.parametrize("shape", [[1, 2, 4, 30], [2, 4, 30]], ids=["rank4", "new_axis"])
@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_partial_final_shard_unchanged_view(device, shape, output, preallocate):
    # 30 words over four 8-word shards, the last one partial. Repeating axis 1 of a rank-4 tensor views it as
    # [upper, repeat, lower, width], its own shape, so the direct kernel repeats it without changing the view.
    grid = _cores(4, 1)
    sharded = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [16, 8])
    output_memory = _output_memory(output, sharded)
    result = _repeat(
        device,
        shape,
        _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [8, 8]),
        [1, 2, 1, 1],
        output_memory,
        preallocate,
        "native",
    )
    assert result.memory_config() == (sharded if output == "implicit" else output_memory)


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_unaligned_split_rows(device, rank, strategy, output, preallocate):
    # 20-byte shard pages split each row. Codegen copies them, as it did before the staged kernel existed, wherever
    # it supports the call. At rank five it does not: the composite path takes them, except into an unaligned
    # sharded output, which its interleaved_to_sharded rejects; the staged kernel writes that one.
    if strategy == ttnn.ShardStrategy.WIDTH:
        shape, grid = [7, 40], ttnn.CoreGrid(y=4, x=2)
    else:
        shape, grid = [4, 10], ttnn.CoreGrid(y=1, x=2)
    shape = [1] * (rank - 2) + shape
    repeats = [1] * rank
    repeats[-2] = 2
    output_shape = [size * count for size, count in zip(shape, repeats)]
    input_memory = ttnn.create_sharded_memory_config(
        shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    sharded = ttnn.create_sharded_memory_config(
        output_shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    output_memory = _output_memory(output, sharded)
    if rank < 5:
        route = "codegen"
    else:
        route = "staged" if output == "sharded" else "composite"
    result = _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route)
    if output_memory is None:
        _assert_derived_placement(result, input_memory)
    else:
        assert result.memory_config() == output_memory


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_aligned_rows_into_unaligned_pages(device, rank, preallocate):
    # 32-byte input pages are native; the requested 20-byte output pages split each row, so the staged kernel
    # writes them.
    grid = _cores(2, 4)
    shape = [1] * (rank - 2) + [7, 40]
    repeats = [1] * rank
    repeats[-2] = 2
    output_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [14, 5])
    result = _repeat(
        device,
        shape,
        _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [7, 8]),
        repeats,
        output_memory,
        preallocate,
        "staged",
    )
    assert result.memory_config() == output_memory


@pytest.mark.parametrize("rank", [3, 5])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_nondividing_output_width(device, rank, preallocate):
    # 5-word input shards into 12-word output shards: 40 words fill three and a third of them.
    shape = [1] * (rank - 2) + [7, 40]
    repeats = [1] * rank
    repeats[-2] = 2
    input_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(2, 4), [7, 5])
    output_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(4, 1), [14, 12])
    route = "composite" if rank == 5 else "codegen"
    result = _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route)
    assert result.memory_config() == output_memory


@pytest.mark.parametrize("buffer", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["l1", "dram"])
@pytest.mark.parametrize(
    "output_layout, output_width",
    [(ttnn.TensorMemoryLayout.WIDTH_SHARDED, w) for w in [4, 6, 8]]
    + [(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, w) for w in [3, 5]],
    ids=["width4", "width6", "width8", "height3", "height5"],
)
def test_repeat_identity_different_narrow_pages(device, buffer, output_layout, output_width):
    row = ttnn.ShardOrientation.ROW_MAJOR
    source = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, buffer, ttnn.ShardSpec(_bank(0), [4, 5], row))
    output = ttnn.MemoryConfig(output_layout, buffer, ttnn.ShardSpec(_bank(1), [4, output_width], row))
    route = "reshard" if buffer == ttnn.BufferType.DRAM and output_width == 5 else "copy"
    _repeat(device, [1, 4, 3], source, [1, 1, 1], output, False, route)


def test_repeat_identity_aligned_dram_narrow_pages(device):
    row, layout = ttnn.ShardOrientation.ROW_MAJOR, ttnn.TensorMemoryLayout.WIDTH_SHARDED
    source = ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, ttnn.ShardSpec(_bank(0), [4, 4], row))
    output = ttnn.MemoryConfig(layout, ttnn.BufferType.DRAM, ttnn.ShardSpec(_bank(1), [4, 8], row))
    _repeat(device, [1, 4, 3], source, [1, 1, 1], output, False, "copy")


@pytest.mark.parametrize("input_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
@pytest.mark.parametrize("buffer", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["l1", "dram"])
def test_repeat_identity_into_narrow_pages(device, input_memory, buffer):
    output = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer,
        ttnn.ShardSpec(_bank(0), [4, 5], ttnn.ShardOrientation.ROW_MAJOR),
    )
    _repeat(device, [1, 4, 3], input_memory, [1, 1, 1], output, False, "copy")


@pytest.mark.parametrize("output", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG], ids=["l1", "dram"])
def test_repeat_identity_narrow_nd_shard(device, output):
    source = ttnn.MemoryConfig(
        ttnn.BufferType.L1,
        ttnn.NdShardSpec(ttnn.Shape([1, 4, 5]), _bank(0), orientation=ttnn.ShardOrientation.ROW_MAJOR),
    )
    assert source.shard_spec is None
    _repeat(device, [1, 4, 3], source, [1, 1, 1], output, False, "copy")


@pytest.mark.parametrize(
    "shape, shard, repeats",
    [([1, 4, 48], [4, 64], [1, 2, 1]), ([1, 1, 65536], [1, 65537], [1, 1, 1])],
    ids=["parallel_copy", "capped_subblock"],
)
def test_repeat_dram_narrow_pages_that_already_complete(device, shape, shard, repeats):
    source = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(_bank(0), shard, ttnn.ShardOrientation.ROW_MAJOR),
    )
    # The rank-preserving identity copies; the repeated small DRAM row takes codegen on Blackhole.
    route = "copy" if all(count == 1 for count in repeats) else "codegen"
    _repeat(device, shape, source, repeats, ttnn.DRAM_MEMORY_CONFIG, False, route)


@pytest.mark.parametrize("rank", [2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
def test_repeat_narrow_input_output_staging(device, rank, dtype):
    grid = _cores(2, 1)
    source = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [4, 5])
    output = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, grid, [8, 2])
    shape = [1] * (rank - 2) + [4, 3]
    repeats = [1] * rank
    repeats[-2] = 2
    _repeat(device, shape, source, repeats, output, False, "staged", dtype)


@pytest.mark.parametrize("rank", [3, 5])
@pytest.mark.parametrize("layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED])
def test_repeat_bfloat16_staged_pages(device, rank, layout):
    grid = _cores(8, 1)
    shape = [1] * (rank - 2) + [4, 40]
    repeats = [1] * rank
    repeats[-2] = 2
    source = _sharded(layout, grid, [4, 8])
    output = _sharded(layout, grid, [8, 5])
    # Aligned 16-byte input pages and split 10-byte output pages require 2-byte-offset staging.
    _repeat(device, shape, source, repeats, output, True, "staged", ttnn.bfloat16)


@pytest.mark.parametrize("layout", [ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED])
@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.bfloat16])
@pytest.mark.parametrize("rows", [1, 2])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_aligned_single_row_shards(device, layout, dtype, rows, preallocate):
    # With one row, shape squeezing removes the row axis. Page-aligned halves still need separate transfers.
    grid = _cores(2, 1)
    source = _sharded(layout, grid, [rows, 32])
    output = _sharded(layout, grid, [rows * 2, 32])
    result = _repeat(
        device,
        [1, rows, 64],
        source,
        [1, 2, 1],
        output if preallocate else None,
        preallocate,
        "staged" if rows == 1 else "native",
        dtype,
    )
    assert result.memory_config() == output


@pytest.mark.parametrize("height", [30, 32])
@pytest.mark.parametrize("leading_axes", [1, 2])
def test_repeat_rank_expansion_keeps_untilized_height_padding(device, height, leading_axes):
    values = (torch.arange(2 * height * 20).reshape(2, height, 20) % 251 - 125).to(torch.bfloat16)
    memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(2, 1), [32, 32])
    tiled = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    source = ttnn.untilize(tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(source), values)
    repeats = [1] * (len(values.shape) + leading_axes)
    assert _route(lambda: ttnn.repeat(source, repeats)) == "view"
    result = ttnn.repeat(source, repeats)
    assert list(result.padded_shape) == [1] * leading_axes + list(source.padded_shape)
    assert result.memory_config() == source.memory_config()
    assert torch.equal(ttnn.to_torch(result), values.repeat(*repeats))
    assert torch.equal(ttnn.to_torch(source), values)


@pytest.mark.parametrize(
    "height, width, repeats",
    [
        (30, 20, [1, 2, 1, 1]),
        (30, 20, [1, 1, 2, 1, 1]),
        (30, 20, [2, 1, 1, 1, 1]),
        (30, 20, [1, 1, 1, 2, 1]),
        (30, 20, [2, 1, 1]),
        (30, 20, [1, 2, 1]),
        (48, 32, [1, 1, 1, 2, 1]),
    ],
)
@pytest.mark.parametrize("padded_height", [False, True])
@pytest.mark.parametrize("output_mode, preallocate", INTERLEAVED_OUTPUTS)
def test_repeat_untilized_height_padding(
    device, expect_error, height, width, repeats, padded_height, output_mode, preallocate
):
    # untilize retains padding between batches. Real repeats must not read it as values from the next batch.
    shard_height = ((height + 31) // 32) * 32
    if not padded_height:
        height = shard_height
        # Rank-expanded controls retain width padding. Same-rank controls use full-width rows;
        # their existing codegen view rejects a 40-byte page in a tile-padded buffer.
        if len(repeats) == 3:
            width = 32
    values = (torch.arange(2 * height * width).reshape(2, height, width) % 251 - 125).to(torch.bfloat16)
    memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(2, 1), [shard_height, 32])
    tiled = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    source = ttnn.untilize(tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert list(source.padded_shape) == [2, shard_height, 32]
    assert torch.equal(ttnn.to_torch(source), values)
    expected = values.repeat(*repeats)
    kwargs = {}
    if output_mode != "implicit":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG if output_mode == "l1" else ttnn.DRAM_MEMORY_CONFIG
    if preallocate:
        sentinel = torch.full_like(expected, -7)
        target = ttnn.from_torch(
            sentinel,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=kwargs["memory_config"],
        )
        kwargs["optional_output_tensor"] = target
    if padded_height:
        message = "repeat does not support row-major tensors with padding between logical batches"
        with expect_error(RuntimeError, message):
            ttnn.repeat(source, repeats, **kwargs)
        if preallocate:
            assert torch.equal(ttnn.to_torch(target), sentinel)
    else:
        result = ttnn.repeat(source, repeats, **kwargs)
        assert torch.equal(ttnn.to_torch(result), expected)
        if output_mode != "implicit":
            assert result.memory_config() == kwargs["memory_config"]
        if preallocate:
            assert result.buffer_address() == target.buffer_address()
            assert torch.equal(ttnn.to_torch(target), expected)
    assert torch.equal(ttnn.to_torch(source), values)


@pytest.mark.parametrize("repeats", [[1, 0, 1], [1, 2, 0, 1]])
def test_repeat_zero_output_keeps_untilized_height_padding_supported(device, repeats):
    values = (torch.arange(1200).reshape(2, 30, 20) % 251 - 125).to(torch.bfloat16)
    memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(2, 1), [32, 32])
    tiled = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    source = ttnn.untilize(tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    result = ttnn.repeat(source, repeats)
    assert list(result.shape) == list(values.repeat(*repeats).shape)
    assert torch.equal(ttnn.to_torch(source), values)


@pytest.mark.parametrize("height", [30, 48])
@pytest.mark.parametrize(
    "repeats", [[1, 2, 1, 1], [1, 1, 2, 1, 1], [2, 1, 1, 1, 1], [1, 1, 1, 2, 1], [2, 1, 1], [1, 2, 1]]
)
@pytest.mark.parametrize("output_mode, preallocate", INTERLEAVED_OUTPUTS)
def test_repeat_untilized_trailing_height_padding(device, height, repeats, output_mode, preallocate):
    # A single logical batch has no padding gap before another batch's values.
    values = (torch.arange(height * 32).reshape(1, height, 32) % 251 - 125).to(torch.bfloat16)
    shard_height = ((height + 31) // 32) * 32
    memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(1, 1), [shard_height, 32])
    tiled = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    source = ttnn.untilize(tiled, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    assert list(source.padded_shape) == [1, shard_height, 32]
    assert torch.equal(ttnn.to_torch(source), values)
    expected = values.repeat(*repeats)
    kwargs = {}
    if output_mode != "implicit":
        kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG if output_mode == "l1" else ttnn.DRAM_MEMORY_CONFIG
    if preallocate:
        target = ttnn.from_torch(
            torch.full_like(expected, -7),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=kwargs["memory_config"],
        )
        kwargs["optional_output_tensor"] = target
    result = ttnn.repeat(source, repeats, **kwargs)
    assert torch.equal(ttnn.to_torch(result), expected)
    assert torch.equal(ttnn.to_torch(source), values)
    if output_mode != "implicit":
        assert result.memory_config() == kwargs["memory_config"]
    if preallocate:
        assert result.buffer_address() == target.buffer_address()
        assert torch.equal(ttnn.to_torch(target), expected)


def _untilize_with_batch_padding(device, values, output_memory):
    shard = [((values.shape[-2] + 31) // 32) * 32, ((values.shape[-1] + 31) // 32) * 32]
    memory = _sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, _cores(prod(values.shape[:-2]), 1), shard)
    tiled = ttnn.from_torch(values, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
    return ttnn.untilize(tiled, memory_config=output_memory)


@pytest.mark.parametrize("source_kind", ["interleaved", "width"])
@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("explicit_memory", [False, True])
@pytest.mark.parametrize("padded_output", [False, True])
def test_repeat_preallocated_batch_padding(
    device, expect_error, source_kind, output_memory, explicit_memory, padded_output
):
    shape = [1, 1, 30, 32] if source_kind == "interleaved" else [1, 1, 1, 64]
    values = (torch.arange(prod(shape)).reshape(shape) % 251 - 125).to(torch.bfloat16)
    source_memory = (
        ttnn.DRAM_MEMORY_CONFIG
        if source_kind == "interleaved"
        else _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(2, 1), [1, 32])
    )
    source = ttnn.from_torch(
        values, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=source_memory
    )
    repeats = [1, 2, 1, 1]
    expected = values.repeat(*repeats)
    sentinel = torch.full_like(expected, -7)
    if padded_output:
        target = _untilize_with_batch_padding(device, sentinel, output_memory)
        assert list(target.padded_shape) == [1, 2, 32, shape[-1]]
    else:
        target = ttnn.from_torch(
            sentinel,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
    assert torch.equal(ttnn.to_torch(source), values)
    assert torch.equal(ttnn.to_torch(target), sentinel)
    kwargs = {"optional_output_tensor": target}
    if explicit_memory:
        kwargs["memory_config"] = output_memory
    if padded_output:
        with expect_error(
            RuntimeError, "repeat does not support row-major tensors with padding between logical batches"
        ):
            ttnn.repeat(source, repeats, **kwargs)
        assert torch.equal(ttnn.to_torch(target), sentinel)
    else:
        result = ttnn.repeat(source, repeats, **kwargs)
        assert result.buffer_address() == target.buffer_address()
        assert result.memory_config() == output_memory
        assert torch.equal(ttnn.to_torch(result), expected)
        assert torch.equal(ttnn.to_torch(target), expected)
    assert torch.equal(ttnn.to_torch(source), values)


@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_repeat_identity_preallocated_batch_padding(device, expect_error, output_memory):
    # Identity repeats with a preallocated target still write the target, unlike an identity view.
    values = (torch.arange(1920).reshape(1, 2, 30, 32) % 251 - 125).to(torch.bfloat16)
    source = ttnn.from_torch(
        values,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    sentinel = torch.full_like(values, -7)
    target = _untilize_with_batch_padding(device, sentinel, output_memory)
    assert list(target.padded_shape) == [1, 2, 32, 32]
    with expect_error(RuntimeError, "repeat does not support row-major tensors with padding between logical batches"):
        ttnn.repeat(source, [1, 1, 1, 1], optional_output_tensor=target)
    assert torch.equal(ttnn.to_torch(source), values)
    assert torch.equal(ttnn.to_torch(target), sentinel)


@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_single_row_shards_interleaved_output(device, output_memory, preallocate):
    source_memory = _sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(2, 1), [1, 32])
    _repeat(device, [1, 1, 64], source_memory, [1, 2, 1], output_memory, preallocate, "staged")


@pytest.mark.parametrize("preallocate", [False, True])
def test_repeat_one_row_block_shards_multiple_rows(device, preallocate):
    source_memory = _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _cores(2, 2), [1, 32])
    output_memory = _sharded(ttnn.TensorMemoryLayout.BLOCK_SHARDED, _cores(2, 2), [2, 32])
    _repeat(device, [2, 1, 64], source_memory, [1, 2, 1], output_memory, preallocate, "staged")


@pytest.mark.parametrize("output_memory", [ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG])
def test_repeat_rank2_trailing_height_padding(device, output_memory):
    values = (torch.arange(960).reshape(30, 32) % 251 - 125).to(torch.bfloat16)
    source = _untilize_with_batch_padding(device, values, ttnn.DRAM_MEMORY_CONFIG)
    assert list(source.padded_shape) == [32, 32]
    result = ttnn.repeat(source, [2, 1], memory_config=output_memory)
    assert torch.equal(ttnn.to_torch(result), values.repeat(2, 1))
    assert result.memory_config() == output_memory
    assert torch.equal(ttnn.to_torch(source), values)
