# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from math import prod

import pytest
import torch
import ttnn

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
    return "composite"


def _repeat(device, shape, input_memory, repeats, output_memory, preallocate, route):
    """Repeats uint32 words that use the high bits; checks every output word and a preallocated output. The route is
    asserted before the call runs, so a case routed onto a path that hangs fails instead."""
    values = (torch.arange(prod(shape), dtype=torch.int64).reshape(shape) * 1000003 + 7).to(torch.int32)
    source = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    assert torch.equal(ttnn.to_torch(source).to(torch.int64), values.to(torch.int64))
    expected = values.to(torch.int64).repeat(*repeats)
    kwargs = {} if output_memory is None else {"memory_config": output_memory}
    if preallocate:
        target = ttnn.from_torch(
            torch.full(expected.shape, -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
        kwargs["optional_output_tensor"] = target
    assert _route(lambda: ttnn.repeat(source, repeats, **kwargs)) == route
    result = ttnn.repeat(source, repeats, **kwargs)
    assert torch.equal(ttnn.to_torch(result).to(torch.int64), expected)
    if preallocate:
        assert result.buffer_address() == target.buffer_address()
        assert torch.equal(ttnn.to_torch(target).to(torch.int64), expected)
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
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("output, preallocate", OUTPUTS)
def test_repeat_narrow_rows(device, rank, strategy, output, preallocate):
    # A 3-word row in a 5-word shard is one padded 20-byte page. Native is the only route: codegen and the
    # composite path unshard it first, and the row-major unshard does not take a page wider than the row.
    if strategy == ttnn.ShardStrategy.WIDTH:
        layout, grid, shape = ttnn.TensorMemoryLayout.WIDTH_SHARDED, _cores(1, 1), [1, 4, 3]
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
        (ttnn.TensorMemoryLayout.BLOCK_SHARDED, (1, 2), [3, 4], [2, 3, 3], [3, 1, 1, 1]),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [1, 4, 3], [1, 1, 2, 1]),
        (ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1), [4, 5], [4, 3], [1, 2, 1]),
    ],
    ids=["block_new_axis", "block_outer_x3", "width_new_axis", "width_rank2"],
)
@pytest.mark.parametrize("output, preallocate", INTERLEAVED_OUTPUTS)
def test_repeat_narrow_rows_rank_expansion(device, layout, grid, shard, shape, repeats, output, preallocate):
    # A repeat vector longer than the input rank adds leading axes before routing. That view keeps the padded row
    # too: re-deriving the shard from the logical width throws for WIDTH, and for BLOCK reads 12-byte pages out of
    # 16-byte slots.
    input_memory = _sharded(layout, _cores(*grid), shard)
    result = _repeat(device, shape, input_memory, repeats, _output_memory(output, None), preallocate, "native")
    if output == "implicit":
        memory = result.memory_config()
        assert memory.memory_layout == layout and memory.shard_spec.grid == input_memory.shard_spec.grid
        assert memory.shard_spec.shape[1] == shard[1]
    else:
        assert result.memory_config() == _output_memory(output, None)


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
    # aligned shards) stays on codegen or the composite path.
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
