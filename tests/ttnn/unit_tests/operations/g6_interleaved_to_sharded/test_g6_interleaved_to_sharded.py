# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_interleaved_to_sharded.

The op is an exact copy into a new placement, so a result is graded by
equality of the bfloat16 bit patterns AND by its memory configuration: values
read back to the host are right in any placement and core order. Cases are the
production placements and the ones the spec's rules or the design's regime pin
require (BB-22). A case outside the op's SUPPORTED is xfail-strict on its
support refusal, so it turns into a real test when the refinement lands.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from ttnn.operations.g6_interleaved_to_sharded import (
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    describe_plan,
    g6_interleaved_to_sharded,
)

# Device is module-scoped by conftest.py (use_module_device hook).

SEED = 42

_DRAM_INTERLEAVED = {"buffer": "dram", "scheme": "interleaved"}

# FLUX case 58 (p1), 486 of 498 traced calls: 32 width shards of 5 tile columns.
FLUX_58 = {
    "shape": [1, 5120],
    "dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "in_memory": dict(_DRAM_INTERLEAVED),
    "out_memory": {
        "buffer": "l1",
        "scheme": "width_sharded",
        "grid": [[0, 0], [7, 3]],
        "shard_shape": [32, 160],
        "orientation": "row_major",
    },
}

# FLUX case 57 (p2), 12 traced calls: one height shard of 4 tiles on one core.
FLUX_57 = {
    "shape": [1, 1, 1, 128],
    "dtype": ttnn.bfloat16,
    "layout": ttnn.TILE_LAYOUT,
    "in_memory": dict(_DRAM_INTERLEAVED),
    "out_memory": {
        "buffer": "l1",
        "scheme": "height_sharded",
        "grid": [[0, 0], [0, 0]],
        "shard_shape": [32, 128],
        "orientation": "row_major",
    },
}


def _variant(base, shape=None, **out_memory):
    scenario = copy.deepcopy(base)
    if shape is not None:
        scenario["shape"] = list(shape)
    scenario["out_memory"].update(out_memory)
    return scenario


# Second placement of the same tensor: a cache key without the shard spec confuses the two (BB-07).
FLUX_58_8_CORES = _variant(FLUX_58, grid=[[0, 0], [7, 0]], shard_shape=[32, 640])
# The core order must follow the orientation memory_config names (spec rule 8).
FLUX_58_COL_MAJOR = _variant(FLUX_58, orientation="col_major")
# Tile row 0 is fully valid and tile row 1 holds one valid row: both read spans in one block.
REGIME_PIN = _variant(FLUX_58, shape=(33, 5120), shard_shape=[64, 160])
# 2 images of 2 tile rows (the last one partial) over 4 height shards on 4 cores (spec rule 9).
HEIGHT_MULTI_SHARD = _variant(
    FLUX_58,
    shape=(2, 1, 33, 128),
    scheme="height_sharded",
    grid=[[0, 0], [3, 0]],
    shard_shape=[32, 128],
)

# bfloat16 bit patterns as int16: NaN payloads 0x7F81 and 0xFFA5, +inf, -inf, -0.0, smallest denormal.
SPECIAL_BITS = (0x7F81, 0xFFA5 - 0x10000, 0x7F80, 0xFF80 - 0x10000, 0x8000 - 0x10000, 0x0001)
SPECIAL_STRIDE = 7

_BUFFER = {"dram": ttnn.BufferType.DRAM, "l1": ttnn.BufferType.L1}
_SCHEME = {
    "height_sharded": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "width_sharded": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "block_sharded": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
}
_ORIENTATION = {"row_major": ttnn.ShardOrientation.ROW_MAJOR, "col_major": ttnn.ShardOrientation.COL_MAJOR}


def torch_reference(input_tensor):
    """The identity on the logical values: the placement never enters them."""
    return input_tensor.clone()


def _support_refusal(scenario):
    """Why the op's own registry refuses the scenario, or None when it is supported."""
    axes = {}
    for name, tagger in INPUT_TAGGERS.items():
        axes[name] = tagger((scenario,), axes)
    for axis, allowed in SUPPORTED.items():
        if axis in axes and axes[axis] not in allowed:
            return f"{axis}={axes[axis]!r} not in SUPPORTED {allowed}"
    for cell in EXCLUSIONS:
        if all(axes.get(key) == value for key, value in cell.items()):
            return f"excluded cell {cell}"
    return None


def _case(scenario, name):
    reason = _support_refusal(scenario)
    marks = [pytest.mark.xfail(reason=reason, strict=True, raises=NotImplementedError)] if reason else []
    return pytest.param(scenario, marks=marks, id=name)


def _memory_config(memory, device):
    if memory["scheme"] == "interleaved":
        return ttnn.DRAM_MEMORY_CONFIG if memory["buffer"] == "dram" else ttnn.L1_MEMORY_CONFIG
    (x0, y0), (x1, y1) = memory["grid"]
    grid_size = device.compute_with_storage_grid_size()
    if x1 >= grid_size.x or y1 >= grid_size.y:
        pytest.skip(f"shard grid end ({x1},{y1}) exceeds the compute grid ({grid_size.x},{grid_size.y})")
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    shard_spec = ttnn.ShardSpec(grid, tuple(memory["shard_shape"]), _ORIENTATION[memory["orientation"]])
    return ttnn.MemoryConfig(_SCHEME[memory["scheme"]], _BUFFER[memory["buffer"]], shard_spec)


def _torch_input(shape, seed=SEED):
    torch.manual_seed(seed)
    return torch.randn(tuple(shape), dtype=torch.float32).to(torch.bfloat16)


def _device_input(torch_input, device, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=None):
    return ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
    )


def _bits(tensor):
    return tensor.contiguous().view(torch.int16)


def _check(output, expected, shape, memory_config):
    """Shape, dtype, layout, the exact requested placement, then bit equality."""
    assert list(output.shape) == list(shape)
    assert output.dtype == ttnn.bfloat16
    assert output.layout == ttnn.TILE_LAYOUT
    got = output.memory_config()
    assert got.buffer_type == memory_config.buffer_type
    assert got.memory_layout == memory_config.memory_layout
    assert got.shard_spec is not None, "result carries no shard spec"
    assert got.shard_spec.grid == memory_config.shard_spec.grid
    assert list(got.shard_spec.shape) == list(memory_config.shard_spec.shape)
    assert got.shard_spec.orientation == memory_config.shard_spec.orientation
    actual = ttnn.to_torch(output)
    assert list(actual.shape) == list(shape)
    wrong = _bits(actual) != _bits(expected)
    if bool(wrong.any()):
        first = int(wrong.flatten().nonzero()[0])
        got_bits = int(_bits(actual).flatten()[first]) & 0xFFFF
        want_bits = int(_bits(expected).flatten()[first]) & 0xFFFF
        pytest.fail(
            f"{int(wrong.sum())} of {wrong.numel()} values differ in their bit pattern; "
            f"first at flat index {first}: got 0x{got_bits:04X}, expected 0x{want_bits:04X}"
        )


@pytest.mark.parametrize(
    "scenario",
    [
        _case(FLUX_58, "flux58_width_8x4_row_major"),
        _case(FLUX_57, "flux57_height_1core"),
        _case(FLUX_58_8_CORES, "flux58_width_8cores_640"),
        _case(FLUX_58_COL_MAJOR, "flux58_width_8x4_col_major"),
        _case(REGIME_PIN, "regime_pin_33x5120_width"),
        _case(HEIGHT_MULTI_SHARD, "height_multi_shard_2x1x33x128"),
    ],
)
def test_placement(scenario, device):
    torch_input = _torch_input(scenario["shape"])
    ttnn_input = _device_input(torch_input, device)
    memory_config = _memory_config(scenario["out_memory"], device)
    output = g6_interleaved_to_sharded(ttnn_input, memory_config)
    _check(output, torch_reference(torch_input), scenario["shape"], memory_config)
    assert not bool((_bits(ttnn.to_torch(ttnn_input)) != _bits(torch_input)).any()), "the op wrote its input"


def test_special_values_bit_exact(device):
    """NaN payloads, infinities, -0.0 and a denormal survive: no cast, no engine on the path."""
    shape = FLUX_58["shape"]
    bits = _bits(_torch_input(shape)).flatten().clone()
    picks = torch.arange(0, bits.numel(), SPECIAL_STRIDE)
    pattern = torch.tensor(SPECIAL_BITS, dtype=torch.int16)
    bits[picks] = pattern[torch.arange(picks.numel()) % len(SPECIAL_BITS)]
    torch_input = bits.reshape(shape).view(torch.bfloat16)
    ttnn_input = _device_input(torch_input, device)
    # Graded against the values the device holds: the host conversion may rewrite a NaN payload.
    expected = ttnn.to_torch(ttnn_input)
    memory_config = _memory_config(FLUX_58["out_memory"], device)
    _check(g6_interleaved_to_sharded(ttnn_input, memory_config), expected, shape, memory_config)


def test_poisoned_padding(device):
    """NaN in rows 1-31 of every input tile must not reach the logical row."""
    shape = FLUX_58["shape"]
    torch_input = _torch_input(shape)
    ttnn_input = _device_input(torch_input, device)
    poisoned = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    memory_config = _memory_config(FLUX_58["out_memory"], device)
    _check(g6_interleaved_to_sharded(poisoned, memory_config), torch_input, shape, memory_config)


def test_repeat_then_other_program(device):
    """Nothing in flight at kernel end (BB-03) and the same bits on every run (BB-06)."""
    shape = FLUX_58["shape"]
    torch_input = _torch_input(shape)
    ttnn_input = _device_input(torch_input, device)
    memory_config = _memory_config(FLUX_58["out_memory"], device)
    for _ in range(2):
        _check(g6_interleaved_to_sharded(ttnn_input, memory_config), torch_input, shape, memory_config)
    other = ttnn.add(ttnn_input, ttnn_input)
    ttnn.deallocate(other)
    _check(g6_interleaved_to_sharded(ttnn_input, memory_config), torch_input, shape, memory_config)


def test_program_cache_and_addresses(device):
    """A repeat call builds no program; the shard spec and orientation are in the key; a cached
    program reads the current input and writes the current output (BB-07, BB-16)."""
    shape = FLUX_58["shape"]
    row_major = _memory_config(FLUX_58["out_memory"], device)
    eight_cores = _memory_config(FLUX_58_8_CORES["out_memory"], device)
    col_major = _memory_config(FLUX_58_COL_MAJOR["out_memory"], device)

    first_torch = _torch_input(shape, seed=SEED)
    first = _device_input(first_torch, device)
    _check(g6_interleaved_to_sharded(first, row_major), first_torch, shape, row_major)

    entries = device.num_program_cache_entries()
    repeat = g6_interleaved_to_sharded(first, row_major)
    assert device.num_program_cache_entries() == entries, "a repeat call with the same signature built a program"
    _check(repeat, first_torch, shape, row_major)

    for memory_config in (eight_cores, col_major, row_major):
        _check(g6_interleaved_to_sharded(first, memory_config), first_torch, shape, memory_config)

    # Allocated while the first input is alive, so it sits at another DRAM address. A new address
    # must patch the cached program, not build another one.
    second_torch = _torch_input(shape, seed=SEED + 1)
    second = _device_input(second_torch, device)
    entries = device.num_program_cache_entries()
    moved = g6_interleaved_to_sharded(second, row_major)
    assert device.num_program_cache_entries() == entries, "a new input address built a program"
    _check(moved, second_torch, shape, row_major)
    ttnn.deallocate(first)


@pytest.mark.parametrize(
    "scenario,row_wise", [(FLUX_58, True), (FLUX_58_COL_MAJOR, False)], ids=["row_major", "col_major"]
)
def test_plan_report_core_order(scenario, row_wise, device):
    """The op reports its plan (BB-08, BB-21): owner-fetch, shard s on the s-th core of the grid
    in the orientation's order, one block of 1 x 5 tiles per owner."""
    ttnn_input = _device_input(_torch_input(scenario["shape"]), device)
    memory_config = _memory_config(scenario["out_memory"], device)
    plan = describe_plan(ttnn_input, memory_config)
    assert plan["regime"] == "owner_fetch"
    assert plan["num_shards"] == 32
    owners = {entry["shard"]: entry for entry in plan["cores"] if entry["role"] == "owner"}
    assert sorted(owners) == list(range(32))
    expected_order = ttnn.corerange_to_cores(memory_config.shard_spec.grid, None, row_wise)
    for shard, core in enumerate(expected_order[:32]):
        entry = owners[shard]
        assert tuple(entry["core"]) == (core.x, core.y), f"shard {shard} on {entry['core']}, expected {core}"
        assert (entry["block_tile_rows"], entry["block_tile_cols"]) == (1, 5)


def _p1_target(device):
    return _memory_config(FLUX_58["out_memory"], device)


def _target(device, **out_memory):
    return _memory_config({**FLUX_58["out_memory"], **out_memory}, device)


_MALFORMED = {
    "l1_interleaved_input": lambda d: (
        _device_input(_torch_input([1, 5120]), d, memory_config=ttnn.L1_MEMORY_CONFIG),
        _p1_target(d),
    ),
    "row_major_input": lambda d: (
        _device_input(_torch_input([1, 5120]), d, layout=ttnn.ROW_MAJOR_LAYOUT),
        _p1_target(d),
    ),
    "float32_input": lambda d: (
        _device_input(_torch_input([1, 5120]).to(torch.float32), d, dtype=ttnn.float32),
        _p1_target(d),
    ),
    "rank3_input": lambda d: (_device_input(_torch_input([1, 1, 5120]), d), _p1_target(d)),
    "host_input": lambda d: (
        ttnn.from_torch(_torch_input([1, 5120]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        _p1_target(d),
    ),
    "dram_interleaved_target": lambda d: (_device_input(_torch_input([1, 5120]), d), ttnn.DRAM_MEMORY_CONFIG),
    "l1_interleaved_target": lambda d: (_device_input(_torch_input([1, 5120]), d), ttnn.L1_MEMORY_CONFIG),
    "block_sharded_target": lambda d: (_device_input(_torch_input([1, 5120]), d), _target(d, scheme="block_sharded")),
    "dram_width_sharded_target": lambda d: (_device_input(_torch_input([1, 5120]), d), _target(d, buffer="dram")),
    "shard_not_tile_multiple": lambda d: (_device_input(_torch_input([1, 5120]), d), _target(d, shard_shape=[32, 176])),
    "fewer_cores_than_shards": lambda d: (_device_input(_torch_input([1, 5120]), d), _target(d, shard_shape=[32, 128])),
    "width_shard_shorter_than_tensor": lambda d: (_device_input(_torch_input([33, 5120]), d), _p1_target(d)),
    "height_shard_narrower_than_tensor": lambda d: (
        _device_input(_torch_input([1, 1, 1, 128]), d),
        _target(d, scheme="height_sharded", grid=[[0, 0], [1, 0]], shard_shape=[32, 64]),
    ),
}


@pytest.mark.parametrize("case", sorted(_MALFORMED))
def test_malformed_request_is_refused(case, device, expect_error):
    """Malformed requests raise ValueError or RuntimeError, never a support refusal (spec Validation)."""
    ttnn_input, memory_config = _MALFORMED[case](device)
    # The spec names the error types, not their text: any message is accepted.
    with expect_error((ValueError, RuntimeError), "") as refused:
        g6_interleaved_to_sharded(ttnn_input, memory_config)
    # A support refusal derives from NotImplementedError, itself a RuntimeError.
    assert not isinstance(refused.value, NotImplementedError), f"answered with a support refusal: {refused.value!r}"
