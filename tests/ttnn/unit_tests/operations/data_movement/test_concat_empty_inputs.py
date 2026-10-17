# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

TORCH_DTYPE = {ttnn.bfloat16: torch.float32, ttnn.float32: torch.float32, ttnn.int32: torch.int32}


def _distinct_values(shape, base, dtype):
    """Small integers that every tested dtype holds exactly; each input gets its own base."""
    count = 1
    for extent in shape:
        count *= extent
    return (base + torch.arange(count) % 97).reshape(shape).to(TORCH_DTYPE[dtype])


def _empty_along(shape, dim):
    empty_shape = list(shape)
    empty_shape[dim] = 0
    return tuple(empty_shape)


def _torch_inputs(order, shape, dim, dtype):
    """`order` spells the input list: "E" is an input with no elements along `dim`, any other letter a regular one."""
    torch_inputs = []
    for position, kind in enumerate(order):
        if kind == "E":
            torch_inputs.append(torch.zeros(_empty_along(shape, dim), dtype=TORCH_DTYPE[dtype]))
        else:
            torch_inputs.append(_distinct_values(shape, 1 + 30 * position, dtype))
    return torch_inputs


def _assert_matches(result, expected):
    output = ttnn.to_torch(result).to(expected.dtype)
    shape, expected_shape = tuple(output.shape), tuple(expected.shape)
    assert shape == expected_shape, f"shape {shape} != expected {expected_shape}"
    mismatched = int((output != expected).sum())
    assert mismatched == 0, f"{mismatched} of {expected.numel()} elements differ from the torch result"


def _assert_empty_result(result, expected_shape, layout, memory_config):
    shape, expected_shape = tuple(result.shape), tuple(expected_shape)
    assert shape == expected_shape, f"shape {shape} != expected {expected_shape}"
    assert result.dtype == ttnn.bfloat16, f"dtype {result.dtype} != expected {ttnn.bfloat16}"
    assert result.layout == layout, f"layout {result.layout} != expected {layout}"
    config = result.memory_config()
    assert config == memory_config, f"memory config {config} != expected {memory_config}"
    shape = tuple(ttnn.to_torch(result).shape)
    assert shape == expected_shape, f"shape read back {shape} != expected {expected_shape}"


def _two_core_sharded_config(memory_layout, shard_shape, buffer_type=ttnn.BufferType.L1):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})
    return ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


def _height_sharded_config(shard_width):
    """Two shards of 32 rows on two cores, in L1."""
    return _two_core_sharded_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, [32, shard_width])


@pytest.mark.parametrize("width", [64, 50], ids=["tile_aligned", "not_tile_aligned"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32], ids=["bfloat16", "float32", "int32"])
@pytest.mark.parametrize("dim", [-1, -2, 1])
@pytest.mark.parametrize("order", ["EA", "AE", "EAB", "AEB", "ABE", "EEAB", "EE"])
def test_concat_with_empty_inputs(device, order, dim, dtype, layout, width):
    """An input that is empty along the concat dimension adds nothing to the result, as in torch.cat.

    "EAB" is concat([empty, a, b]). The inputs are deallocated before the result is read: the result
    must own its data, also when only one input has elements.
    """
    torch_inputs = _torch_inputs(order, (1, 2, 32, width), dim, dtype)
    expected = torch.cat(torch_inputs, dim=dim)

    inputs = [ttnn.from_torch(t, dtype=dtype, layout=layout, device=device) for t in torch_inputs]
    result = ttnn.concat(inputs, dim=dim)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    _assert_matches(result, expected)


@pytest.mark.parametrize(
    "output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram_output", "l1_output"]
)
@pytest.mark.parametrize(
    "input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram_inputs", "l1_inputs"]
)
@pytest.mark.parametrize("width", [64, 50], ids=["tile_aligned", "not_tile_aligned"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("order", ["EA", "AE"])
def test_concat_with_one_non_empty_input_across_memory_configs(
    device, order, layout, width, input_memory_config, output_memory_config
):
    """With one input left that has elements, the result is a copy of it in the requested memory config."""
    torch_inputs = _torch_inputs(order, (1, 2, 32, width), -1, ttnn.bfloat16)
    expected = torch.cat(torch_inputs, dim=-1)

    inputs = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=input_memory_config)
        for t in torch_inputs
    ]
    result = ttnn.concat(inputs, dim=-1, memory_config=output_memory_config)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    config = result.memory_config()
    assert config == output_memory_config, f"memory config {config} != expected {output_memory_config}"
    _assert_matches(result, expected)


@pytest.mark.parametrize("order", ["EA", "AE"])
def test_concat_with_one_non_empty_input_and_sharded_output(device, order):
    """With one input left that has elements, a sharded output memory config is honoured, as it is for two."""
    sharded_config = _height_sharded_config(64)
    torch_inputs = _torch_inputs(order, (1, 1, 64, 64), -1, ttnn.bfloat16)
    expected = torch.cat(torch_inputs, dim=-1)

    inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for t in torch_inputs]
    result = ttnn.concat(inputs, dim=-1, memory_config=sharded_config)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    config = result.memory_config()
    assert config == sharded_config, f"memory config {config} != expected {sharded_config}"
    _assert_matches(result, expected)


@pytest.mark.parametrize(
    "memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram_output", "l1_output"]
)
@pytest.mark.parametrize("width", [64, 50], ids=["tile_aligned", "not_tile_aligned"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dim", [-1, 1])
def test_concat_with_zero_sized_other_dimension(device, dim, layout, width, memory_config):
    """Inputs with a zero-sized dimension other than the concat one give an empty result of the right shape."""
    torch_inputs = [torch.zeros((1, 2, 0, width)), torch.zeros((1, 2, 0, width))]
    expected = torch.cat(torch_inputs, dim=dim)

    inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device) for t in torch_inputs]
    result = ttnn.concat(inputs, dim=dim, memory_config=memory_config)

    _assert_empty_result(result, expected.shape, layout, memory_config)


@pytest.mark.parametrize(
    "output_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram_output", "l1_output"]
)
@pytest.mark.parametrize(
    "input_memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG], ids=["dram_input", "l1_input"]
)
@pytest.mark.parametrize("shape", [(1, 2, 32, 0), (1, 2, 0, 64)], ids=["empty_last_dim", "empty_other_dim"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_concat_of_a_single_zero_sized_input(device, layout, shape, input_memory_config, output_memory_config):
    """A list holding one zero-sized input gives an empty result of that shape in the requested memory config."""
    torch_input = torch.zeros(shape)
    tensor = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=input_memory_config
    )
    result = ttnn.concat([tensor], dim=-1, memory_config=output_memory_config)

    _assert_empty_result(result, shape, layout, output_memory_config)


def test_concat_of_a_single_deallocated_zero_sized_input_is_rejected(device, expect_error):
    """A deallocated zero-sized input is rejected when it is the only input, as a deallocated regular input is."""
    tensor = ttnn.from_torch(torch.zeros((1, 2, 32, 0)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.deallocate(tensor)

    with expect_error(RuntimeError, "Input Tensor is not allocated"):
        ttnn.concat([tensor], dim=-1)


def test_concat_of_a_single_host_zero_sized_input_is_rejected(device, expect_error):
    """A zero-sized host input is rejected when it is the only input, as a regular host input is."""
    tensor = ttnn.from_torch(torch.zeros((1, 2, 32, 0)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    with expect_error(RuntimeError, "Device Operations expect device tensors as inputs"):
        ttnn.concat([tensor], dim=-1)


@pytest.mark.parametrize("order", ["AE", "EA", "EE", "AEB"])
def test_concat_with_zero_sized_input_keeps_dtype_rule(device, expect_error, order):
    """Inputs of two dtypes are rejected with a zero-sized input in the list, as they are without one."""
    inputs = []
    for position, torch_input in enumerate(_torch_inputs(order, (1, 2, 32, 64), -1, ttnn.bfloat16)):
        dtype = ttnn.float32 if position == 1 else ttnn.bfloat16
        inputs.append(ttnn.from_torch(torch_input, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device))

    with expect_error(RuntimeError, "All Tensors should have same dtypes"):
        ttnn.concat(inputs, dim=-1)


@pytest.mark.parametrize("order", ["AE", "EA", "EE", "AEB"])
def test_concat_with_zero_sized_input_keeps_device_rule(device, expect_error, order):
    """A host tensor is rejected with a zero-sized input in the list, as it is without one."""
    inputs = []
    for position, torch_input in enumerate(_torch_inputs(order, (1, 2, 32, 64), -1, ttnn.bfloat16)):
        on_device = {} if position == 1 else {"device": device}
        inputs.append(ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, **on_device))

    with expect_error(RuntimeError, "expected a tensor with device storage"):
        ttnn.concat(inputs, dim=-1)


@pytest.mark.parametrize("width", [64, 50], ids=["tile_aligned", "not_tile_aligned"])
@pytest.mark.parametrize("order", ["AE", "EA", "AEB"])
def test_concat_with_zero_sized_input_of_another_layout(device, order, width):
    """A zero-sized input is left out before the inputs are converted to one layout: its layout does not matter."""
    torch_inputs = _torch_inputs(order, (1, 2, 32, width), -1, ttnn.bfloat16)
    expected = torch.cat(torch_inputs, dim=-1)

    inputs = []
    for position, torch_input in enumerate(torch_inputs):
        layout = ttnn.ROW_MAJOR_LAYOUT if order[position] == "E" else ttnn.TILE_LAYOUT
        inputs.append(ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device))
    result = ttnn.concat(inputs, dim=-1)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    _assert_matches(result, expected)


@pytest.mark.parametrize(
    "layout, memory_layout, shape, shard_shape, dim",
    [
        (ttnn.TILE_LAYOUT, ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1, 1, 64, 64), [32, 64], -1),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1, 1, 32, 64), [32, 32], -2),
    ],
    ids=["height_sharded_width_concat", "width_sharded_height_concat"],
)
@pytest.mark.parametrize("order", ["AE", "EA", "AEB"])
def test_concat_with_interleaved_zero_sized_input_next_to_sharded_inputs(
    device, order, layout, memory_layout, shape, shard_shape, dim
):
    """A zero-sized input is left out before the device operation: that it is interleaved does not matter."""
    sharded_config = _two_core_sharded_config(memory_layout, shard_shape)
    torch_inputs = _torch_inputs(order, shape, dim, ttnn.bfloat16)
    expected = torch.cat(torch_inputs, dim=dim)

    inputs = []
    for position, torch_input in enumerate(torch_inputs):
        tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=layout, device=device)
        inputs.append(tensor if order[position] == "E" else ttnn.to_memory_config(tensor, sharded_config))
    result = ttnn.concat(inputs, dim=dim)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    _assert_matches(result, expected)


@pytest.mark.parametrize("buffer_type", [ttnn.BufferType.L1, ttnn.BufferType.DRAM], ids=["l1", "dram"])
@pytest.mark.parametrize("order", ["AE", "EA"])
def test_concat_with_one_sharded_input_left_and_its_memory_config(device, order, buffer_type):
    """With one sharded input left and its own memory config requested, the result is a sharded copy of it."""
    sharded_config = _two_core_sharded_config(ttnn.TensorMemoryLayout.WIDTH_SHARDED, [32, 32], buffer_type)
    torch_inputs = _torch_inputs(order, (1, 1, 32, 64), -2, ttnn.bfloat16)
    expected = torch.cat(torch_inputs, dim=-2)

    inputs = []
    for position, torch_input in enumerate(torch_inputs):
        tensor = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        inputs.append(tensor if order[position] == "E" else ttnn.to_memory_config(tensor, sharded_config))
    result = ttnn.concat(inputs, dim=-2, memory_config=sharded_config)
    for tensor in inputs:
        ttnn.deallocate(tensor)

    config = result.memory_config()
    assert config == sharded_config, f"memory config {config} != expected {sharded_config}"
    _assert_matches(result, expected)


@pytest.mark.parametrize("order", ["AE", "EE"])
def test_concat_with_zero_sized_input_keeps_allocation_rule(device, expect_error, order):
    """A deallocated zero-sized input is rejected, as a deallocated regular input is."""
    inputs = [
        ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for t in _torch_inputs(order, (1, 2, 32, 64), -1, ttnn.bfloat16)
    ]
    ttnn.deallocate(inputs[1])

    with expect_error(RuntimeError, "allocated"):
        ttnn.concat(inputs, dim=-1)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("shape", [(2, 0, 4), (2, 3, 0)], ids=["empty_middle_dim", "empty_last_dim"])
def test_stack_of_zero_sized_inputs(device, shape, dim, layout):
    """ttnn.stack is built on ttnn.concat: zero-sized inputs give an empty result of the torch.stack shape."""
    torch_inputs = [torch.zeros(shape), torch.zeros(shape)]
    expected = torch.stack(torch_inputs, dim=dim)

    inputs = [ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device) for t in torch_inputs]
    result = ttnn.stack(inputs, dim=dim)

    shape, expected_shape = tuple(result.shape), tuple(expected.shape)
    assert shape == expected_shape, f"shape {shape} != expected {expected_shape}"
    assert result.dtype == ttnn.bfloat16, f"dtype {result.dtype} != expected {ttnn.bfloat16}"
