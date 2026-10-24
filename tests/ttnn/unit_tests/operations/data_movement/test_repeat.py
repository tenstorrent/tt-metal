# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from functools import reduce
from math import prod

import pytest
import torch
import ttnn
from tt_py_test_utils_common.utility_functions import is_blackhole

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal

layouts = [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT]

dtypes = [
    (torch.float32, ttnn.float32),
    (torch.bfloat16, ttnn.bfloat16),
    (torch.bfloat16, ttnn.bfloat8_b),
    (torch.int32, ttnn.int32),
    (torch.int32, ttnn.uint32),
    (torch.int16, ttnn.uint16),
]
shapes = [(1,), (2,), (2, 3), (4, 16, 3, 1), (4, 3, 1, 2, 2)]
repeat_shapes = [
    (1,),
    (1, 1),
    (1, 2),
    (4, 3, 2, 1),
    (2, 3, 4, 5, 2),
    (2048,),
]


def _get_size(larger, smaller) -> int:
    return prod([a * b for a, b in zip(((1,) * (len(larger) - len(smaller)) + smaller), larger)])


def _get_final_size(shape, reshape):
    if len(shape) > len(reshape):
        return _get_size(shape, reshape)
    else:
        return _get_size(reshape, shape)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_repeat(device, layout, dtype, shape, repeat_shape):
    torch_dtype, ttnn_dtype = dtype

    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")

    if layout == ttnn.ROW_MAJOR_LAYOUT and ttnn_dtype == ttnn.bfloat8_b:
        pytest.skip("Illegal config")

    if layout == ttnn.TILE_LAYOUT and ttnn_dtype == ttnn.uint16:
        pytest.skip("UINT16 tensors cannot be tilized - only bfloat16/float32/int32/uint32 supported")

    mul = lambda x, y: x * y
    torch_input_tensor = torch.arange(0, reduce(mul, shape, 1), dtype=torch_dtype).reshape(shape)

    torch_result = torch_input_tensor.repeat(repeat_shape)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=ttnn_dtype)

    output = ttnn.repeat(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)
    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"

    if ttnn_dtype == ttnn.bfloat8_b:
        assert_with_pcc(torch_result, output, 0.9999)
    else:
        assert_equal(torch_result, output)


@pytest.mark.parametrize("layout", layouts)
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("repeat_shape", repeat_shapes)
def test_pc_repeat(device, layout, shape, repeat_shape):
    # trying to avoid the `buffer not divisible by page size` error. Does this make sense?
    if layout == ttnn.TILE_LAYOUT and (
        prod(shape) % ttnn.TILE_SIZE != 0 or _get_final_size(shape, repeat_shape) % ttnn.TILE_SIZE != 0
    ):
        pytest.skip("Tensor not suitable for tile layout")

    if len(repeat_shape) < len(shape):
        pytest.skip("PyTorch repeat dim must be >= tensor dim (although we can handle this).")
    num_iters = 3
    input_tensors = []
    torch_results = []
    for i in range(num_iters):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch_tensor.repeat(repeat_shape))
        input_tensors.append(ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16))
    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = ttnn.repeat(input_tensors[i], ttnn.Shape(repeat_shape))
        output = ttnn.to_torch(output)
        assert (
            output.shape == torch_results[i].shape
        ), f"Output shape {output.shape} does not match torch shape {torch_results[i].shape}"

        assert_equal(torch_results[i], output)
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


# 17975 test cases


# --- Codegen-path coverage ---
#
# ttnn.repeat routes gate-supported cases to codegen and the rest to native, and offers no way to
# ask for one: the verification-only entries below live in the private module for that reason (see
# repeat_force.hpp). These duplicate the correctness / program-cache checks above but pin the
# codegen path so the suite exercises it regardless of the gate's verdict.
#
# Every shape below satisfies the codegen gate (repeat_codegen_supported.cpp), so the forced entry resolves.
# Random inputs (not arange) keep bf16 exact: repeat copies values verbatim.
codegen_supported_cases = [
    # (shape, repeat_shape, layout) -- TILE
    ((1, 1, 32, 32), (2, 1, 1, 1), ttnn.TILE_LAYOUT),  # N (batch) repeat
    ((1, 1, 32, 32), (1, 3, 1, 1), ttnn.TILE_LAYOUT),  # C repeat
    ((1, 1, 32, 32), (1, 1, 2, 1), ttnn.TILE_LAYOUT),  # H repeat (tile-aligned)
    ((1, 1, 32, 64), (1, 1, 1, 2), ttnn.TILE_LAYOUT),  # W repeat (tile-aligned)
    ((32, 64), (2, 1), ttnn.TILE_LAYOUT),  # rank-2 TILE
    # ROW_MAJOR
    ((2, 3, 4, 8), (2, 1, 1, 1), ttnn.ROW_MAJOR_LAYOUT),  # N repeat (higher-dim)
    ((1, 2, 4, 8), (1, 1, 2, 1), ttnn.ROW_MAJOR_LAYOUT),  # H repeat (higher-dim)
    ((1, 1, 4, 8), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT),  # last-dim (within-stick), width >= 2
]

codegen_dtypes = [
    (torch.bfloat16, ttnn.bfloat16),
    (torch.float32, ttnn.float32),
]


_force_codegen = ttnn._ttnn.operations.data_movement.repeat_force_codegen


@pytest.mark.parametrize("shape, repeat_shape, layout", codegen_supported_cases)
@pytest.mark.parametrize("dtype", codegen_dtypes)
def test_repeat_codegen(device, shape, repeat_shape, layout, dtype):
    torch_dtype, ttnn_dtype = dtype

    torch_input_tensor = torch.rand(shape, dtype=torch_dtype)
    torch_result = torch_input_tensor.repeat(repeat_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device, dtype=ttnn_dtype)
    output = _force_codegen(input_tensor, ttnn.Shape(repeat_shape))
    output = ttnn.to_torch(output)

    assert (
        output.shape == torch_result.shape
    ), f"Output shape {output.shape} does not match torch shape {torch_result.shape}"
    assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "shape, repeat_shape, layout",
    [
        ((1, 1, 32, 32), (2, 1, 1, 1), ttnn.TILE_LAYOUT),
        ((1, 1, 4, 8), (1, 1, 1, 2), ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_pc_repeat_codegen(device, shape, repeat_shape, layout):
    num_iters = 3
    input_tensors = []
    torch_results = []
    for i in range(num_iters):
        torch_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch_tensor.repeat(repeat_shape))
        input_tensors.append(ttnn.from_torch(torch_tensor, layout=layout, device=device, dtype=ttnn.bfloat16))
    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = _force_codegen(input_tensors[i], ttnn.Shape(repeat_shape))
        output = ttnn.to_torch(output)
        assert (
            output.shape == torch_results[i].shape
        ), f"Output shape {output.shape} does not match torch shape {torch_results[i].shape}"

        assert_equal(torch_results[i], output)
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"


def test_pc_with_different_shapes_in_sequence(device):
    y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)
    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x = torch.zeros((64, 1, 256, 384), dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    num_iters = 4
    z_tt = x_tt + y_tt

    y = torch.rand((1, 1, 32, 32), dtype=torch.bfloat16)
    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    x = torch.zeros((4, 1, 32, 32), dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # codegen-supported (interleaved, tile-aligned N-axis repeat) -> exercise the codegen path
    _force_codegen(y_tt, [4, 1, 1, 1])
    z_tt = ttnn.add(x_tt, y_tt)
    z_tt = x_tt + y_tt

    for i in range(num_iters):
        z_torch = ttnn.to_torch(z_tt[i : i + 1])
        assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"
    for _ in range(num_iters):
        y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        x = torch.zeros((4, 1, 32, 32), dtype=torch.bfloat16)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        base_count = device.cache_entries_counter.total
        with device.cache_entries_counter.measure():
            _force_codegen(y_tt, [4, 1, 1, 1])
        assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"
        z_tt = ttnn.add(x_tt, y_tt)
        z_tt = x_tt + y_tt

        for i in range(num_iters):
            z_torch = ttnn.to_torch(z_tt[i : i + 1])
            assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"
    y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)

    y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.repeat(y_tt, ttnn.Shape([64, 1, 1, 1]))
    for i in range(64):
        z_torch = ttnn.to_torch(z_tt[i : i + 1])
        assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"

    for _ in range(num_iters):
        y = torch.rand((1, 1, 256, 384), dtype=torch.bfloat16)
        y_tt = ttnn.from_torch(y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        base_count = device.cache_entries_counter.total
        with device.cache_entries_counter.measure():
            z_tt = ttnn.repeat(y_tt, ttnn.Shape([64, 1, 1, 1]))

        assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"

        for i in range(64):
            z_torch = ttnn.to_torch(z_tt[i : i + 1])
            assert torch.allclose(z_torch, y, atol=1e-2), f"z_torch[{i}] != y"


def _assert_derived_placement(output, input_memory):
    # With no memory_config, an unaligned row-major shard page takes codegen or the composite path, which place the
    # output on a spec derived for its shape, with an L1-aligned page, or in L1 interleaved when none exists.
    memory = output.memory_config()
    if memory.is_sharded():
        assert memory.memory_layout == input_memory.memory_layout
        assert memory.shard_spec.shape[1] * output.element_size() % ttnn.get_l1_alignment() == 0
    else:
        assert memory == ttnn.L1_MEMORY_CONFIG


@pytest.mark.parametrize(
    "dtype, shape, grid, repeats",
    [
        (ttnn.uint32, [1, 7, 40], (4, 2), [1, 2, 1]),  # 20-byte shard rows
        (ttnn.bfloat16, [63, 16], (1, 4), [3, 1]),  # 8-byte shard rows
        (ttnn.bfloat16, [63, 16], (4, 2), [3, 1]),  # 4-byte shard rows
        (ttnn.uint32, [1, 8, 64], (4, 2), [1, 2, 1]),  # control: 32-byte shard rows
    ],
)
@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_width_sharded_row_major_subaligned_rows(device, dtype, shape, grid, repeats):
    words = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    if dtype == ttnn.uint32:
        torch_input = ((words * 1000003 + 7) % (2**32)).to(torch.int32)
    else:
        torch_input = (words % 251).to(torch.bfloat16)
    memory_config = ttnn.create_sharded_memory_config(
        shape,
        core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_tensor = ttnn.from_torch(
        torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )

    def read(tensor):
        result = ttnn.to_torch(tensor)
        return result.to(torch.int64) & 0xFFFFFFFF if dtype == ttnn.uint32 else result

    expected_input = torch_input.to(torch.int64) & 0xFFFFFFFF if dtype == ttnn.uint32 else torch_input
    assert torch.equal(read(input_tensor), expected_input)
    output = ttnn.repeat(input_tensor, ttnn.Shape(repeats))
    assert torch.equal(read(output), expected_input.repeat(*repeats))
    expected_memory = ttnn.create_sharded_memory_config(
        [extent * count for extent, count in zip(shape, repeats)],
        core_grid=ttnn.CoreGrid(y=grid[0], x=grid[1]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    if memory_config.shard_spec.shape[1] * input_tensor.element_size() % ttnn.get_l1_alignment():
        _assert_derived_placement(output, memory_config)
    else:
        assert output.memory_config() == expected_memory


@pytest.mark.parametrize("rank", [3, 4, 5])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("split_row", [False, True])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("output_mode", ["implicit", "memory_config", "preallocated", "both"])
@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_sharded_row_major_placement(device, rank, strategy, split_row, aligned, output_mode):
    shard_width = 8 if aligned else 5
    if strategy == ttnn.ShardStrategy.WIDTH:
        grid = ttnn.CoreGrid(y=4, x=2) if split_row else ttnn.CoreGrid(y=1, x=1)
        shape = [1, 7, shard_width * (8 if split_row else 1)]
    else:
        grid = ttnn.CoreGrid(y=1, x=2) if split_row else ttnn.CoreGrid(y=1, x=1)
        shape = [1, 4, shard_width * (2 if split_row else 1)]
    shape = [1] * (rank - 3) + shape
    repeats = [1] * rank
    repeats[1] = 2
    expected_shape = [size * count for size, count in zip(shape, repeats)]
    input_memory = ttnn.create_sharded_memory_config(
        shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    expected_memory = ttnn.create_sharded_memory_config(
        expected_shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    words = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    values = ((words * 1000003 + 7) % (2**32)).to(torch.int32)
    tensor = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    expected_input = values.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(ttnn.to_torch(tensor).to(torch.int64) & 0xFFFFFFFF, expected_input)
    kwargs = {}
    if output_mode in ("memory_config", "both"):
        kwargs["memory_config"] = expected_memory
    if output_mode in ("preallocated", "both"):
        preallocated = ttnn.from_torch(
            torch.full(expected_shape, -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=expected_memory,
        )
        kwargs["optional_output_tensor"] = preallocated
    output = ttnn.repeat(tensor, repeats, **kwargs)
    assert torch.equal(ttnn.to_torch(output).to(torch.int64) & 0xFFFFFFFF, expected_input.repeat(*repeats))
    if output_mode == "implicit" and not aligned and split_row:
        # Codegen places split unaligned rows on a derived spec; a row on one page stays native on the input grid.
        _assert_derived_placement(output, input_memory)
    else:
        assert output.memory_config() == expected_memory
    if output_mode in ("preallocated", "both"):
        assert output.buffer_address() == preallocated.buffer_address()
        assert torch.equal(ttnn.to_torch(preallocated).to(torch.int64) & 0xFFFFFFFF, expected_input.repeat(*repeats))


@pytest.mark.parametrize("input_shard_width, output_shard_width", [(5, 10), (10, 5), (5, 8), (8, 5)])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_sharded_row_major_different_page_widths(device, input_shard_width, output_shard_width, preallocate):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 3))})
    input_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [7, input_shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_memory = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [14, output_shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    values = (torch.arange(280, dtype=torch.int64).reshape(1, 1, 1, 7, 40) * 1000003 + 7).to(torch.int32)
    tensor = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=input_memory
    )
    expected = (values.to(torch.int64) & 0xFFFFFFFF).repeat(1, 2, 1, 1, 1)
    kwargs = {"memory_config": output_memory}
    if preallocate:
        target = ttnn.from_torch(
            torch.full(list(expected.shape), -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=output_memory,
        )
        kwargs["optional_output_tensor"] = target
    output = ttnn.repeat(tensor, [1, 2, 1, 1, 1], **kwargs)
    assert output.memory_config() == output_memory
    assert torch.equal(ttnn.to_torch(output).to(torch.int64) & 0xFFFFFFFF, expected)
    if preallocate:
        assert output.buffer_address() == target.buffer_address()


@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_large_split_row_scratch_space(device):
    width = 8 * 30005
    shape = [1, 1, width]
    repeats = [1, 2, 1]
    grid = ttnn.CoreGrid(y=4, x=2)

    def config(rows):
        return ttnn.create_sharded_memory_config(
            [1, rows, width],
            core_grid=grid,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )

    values = (torch.arange(width, dtype=torch.int64) * 1000003 + 7).to(torch.int32).reshape(shape)
    source = ttnn.from_torch(
        values, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config(1)
    )
    target = ttnn.from_torch(
        torch.full([1, 2, width], -1, dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=config(2),
    )
    result = ttnn.repeat(source, repeats, memory_config=config(2), optional_output_tensor=target)
    assert result.memory_config() == config(2)
    assert result.buffer_address() == target.buffer_address()
    assert torch.equal(
        ttnn.to_torch(result).to(torch.int64) & 0xFFFFFFFF, (values.to(torch.int64) & 0xFFFFFFFF).repeat(*repeats)
    )


@pytest.mark.parametrize("dtype", [ttnn.uint32, ttnn.int32, ttnn.float32, ttnn.bfloat16, ttnn.uint16])
@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.parametrize("aligned", [False, True])
@pytest.mark.parametrize("preallocate", [False, True])
@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_split_shards_supported_dtypes(device, dtype, strategy, aligned, preallocate):
    element_bytes = 2 if dtype in (ttnn.bfloat16, ttnn.uint16) else 4
    shard_width = 48 // element_bytes if aligned else 5
    shape = [1, 4, 2 * shard_width]
    repeats = [1, 2, 1]
    output_shape = [1, 8, 2 * shard_width]
    grid = ttnn.CoreGrid(y=1, x=2)

    def config(shape):
        return ttnn.create_sharded_memory_config(
            shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

    words = torch.arange(prod(shape), dtype=torch.int64).reshape(shape)
    if dtype == ttnn.float32:
        values = words.to(torch.float32) / 4 - 17.5
    elif dtype == ttnn.bfloat16:
        values = (words % 251 - 125).to(torch.bfloat16)
    elif dtype == ttnn.uint16:
        values = (words * 251 % 32749).to(torch.int16)
    elif dtype == ttnn.int32:
        values = (words * 1000003 - 100000007).to(torch.int32)
    else:
        values = (words * 100000003 + 7).to(torch.int32)

    def read(tensor):
        result = ttnn.to_torch(tensor)
        if dtype == ttnn.uint32:
            return result.to(torch.int64) & 0xFFFFFFFF
        return result.to(torch.int64) if dtype == ttnn.uint16 else result

    expected_input = values.to(torch.int64) & 0xFFFFFFFF if dtype == ttnn.uint32 else values
    if dtype == ttnn.uint16:
        expected_input = values.to(torch.int64)
    source = ttnn.from_torch(
        values, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config(shape)
    )
    assert torch.equal(read(source), expected_input)
    kwargs = {}
    if preallocate:
        target = ttnn.from_torch(
            torch.zeros(output_shape, dtype=values.dtype),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=config(output_shape),
        )
        kwargs = {"memory_config": config(output_shape), "optional_output_tensor": target}
    output = ttnn.repeat(source, repeats, **kwargs)
    assert torch.equal(read(output), expected_input.repeat(*repeats))
    if not preallocate and not aligned:
        _assert_derived_placement(output, config(shape))
    else:
        assert output.memory_config() == config(output_shape)
    if preallocate:
        assert output.buffer_address() == target.buffer_address()
        assert torch.equal(read(target), expected_input.repeat(*repeats))


@pytest.mark.parametrize("strategy", [ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK])
@pytest.mark.skipif(not is_blackhole(), reason="Shard-page repeat cases are validated on Blackhole only")
def test_repeat_sharded_multiple_axes_control(device, strategy):
    # Multiple repeated axes retain their existing route. A 48-byte shard row is aligned to 16, not 32.
    shape, output_shape, repeats = [1, 4, 24], [2, 8, 24], [2, 2, 1]
    grid = ttnn.CoreGrid(y=1, x=2)

    def config(shape):
        return ttnn.create_sharded_memory_config(
            shape, core_grid=grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )

    values = torch.arange(prod(shape), dtype=torch.float32).reshape(shape) / 4
    source = ttnn.from_torch(
        values, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=config(shape)
    )
    output = ttnn.repeat(source, repeats, memory_config=config(output_shape))
    assert output.memory_config() == config(output_shape)
    assert torch.equal(ttnn.to_torch(output), values.repeat(*repeats))
